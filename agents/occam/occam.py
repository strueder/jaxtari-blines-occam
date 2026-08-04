"""
OCCAM: Object-Centric Attention via Masking  -  JAXtari baseline


A reimplementation of OCCAM as an observation
wrapper for JAXtari (https://github.com/k4ntz/JAXAtari), intended as a
neuro-symbolic baseline for https://github.com/remunds/jaxtari-blines.

Original method:
    "Deep Reinforcement Learning via Object-Centric Attention via Masking (OCCAM)"
    Bl"uml, Derstroff, Gregori, Dillies, Delfosse, Kersting (2025), arXiv:2504.03024
    Reference code: https://github.com/VanillaWhey/OCAtariWrappers
This file re-implements the *idea* (object-centric masking) natively in JAX.


THE FOUR OCCAM ABSTRACTION LEVELS (all implemented here, selected via
`mask_mode`); see Sec. 2.2 / Figure 3 of the paper:
    - "object" : keep the real (grayscale) pixels inside each object box,
                 zero out the background.                       -> (F, 84, 84)
    - "binary" : 255 inside any object box, 0 elsewhere (no identity / texture).
                                                                -> (F, 84, 84)
    - "class"  : each object box filled with a class-specific gray level
                 (category preserved, texture discarded).       -> (F, 84, 84)
    - "planes" : one plane per object category, stacked as channels.
                                                          -> (F * n_classes, 84, 84)

    All four are drawn at native 210x160 and area-downscaled to 84x84 afterwards, so
    "binary" and "planes" are NOT two-valued in the output: pixels on an object edge
    are only partially covered and carry the intermediate gray. Anything downstream
    that tests `== 255` or `astype(bool)` on a mask is wrong; test `> 0` for occupancy
    and treat the value as coverage.

The "class" of an object is its group in the game's observation PyTree
(e.g. Pong -> {player, enemy, ball}; Skiing -> {skier, flags, trees, moguls}).
This maps 1:1 to the object categories in the paper's classification step
(Figure 2) and does NOT rely on the optional `visual_id` field, so it works
uniformly across all games that expose `ObjectObservation`s.

JIT NOTE: the number of object groups (classes) and the number of instances
per group are *static* for a given game, so the whole pipeline is traced once
and stays JIT/vmap-compatible. The frame-stack and (for Planes) the per-class
planes are folded into the channel axis so the existing CNN in jaxtari-blines
(`agents/ppo/ppo.py::Network`) consumes the output unchanged.

CHECKPOINT COMPATIBILITY: checkpoints trained before the switch to exact area
downscaling are NOT comparable to ones trained after it. The observation
distribution changed in two ways: "binary"/"planes" went from two-valued 0/255 to
continuous coverage values, and the "class" palette moved from 90/172/255 to
85/170/255 (K=3). Shapes are unchanged, so an old checkpoint still loads and runs -
it just sees inputs it was never trained on. Retrain rather than fine-tune.
"""

from __future__ import annotations

import functools
import colorsys
from typing import Any, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from jaxatari.wrappers import JaxatariWrapper, AtariWrapper
from jaxatari.environment import ObjectObservation
from jaxatari import spaces


MASK_MODES = ("object", "binary", "class", "planes")

# same grayscale weights as PixelObsWrapper for comparability
_GRAY_W = jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32)


def _rgb_to_gray(frame_rgb: jnp.ndarray) -> jnp.ndarray:
    """(H, W, 3) uint8/float -> (H, W) float32 grayscale."""
    return jnp.dot(frame_rgb.astype(jnp.float32), _GRAY_W)


def _area_weights(src: int, dst: int) -> np.ndarray:
    """(dst, src) area-resampling weights, identical to cv2.INTER_AREA on downscale.

    Destination pixel i covers the source interval [i*s, (i+1)*s) with s = src/dst;
    weight[i, j] is the length of that interval's overlap with source pixel [j, j+1),
    normalized by s. Rows sum to 1, so the operator is an exact box average.

    Downscale only: on upscale cv2.INTER_AREA degenerates to INTER_NEAREST and this
    formula no longer describes it.
    """
    assert src >= dst, f"_area_weights is downscale-only, got src={src} < dst={dst}"
    s = src / dst
    edges = np.arange(dst + 1, dtype=np.float64) * s
    lo = np.arange(src, dtype=np.float64)[None, :]
    overlap = np.minimum(lo + 1.0, edges[1:, None]) - np.maximum(lo, edges[:-1, None])
    return np.clip(overlap, 0.0, None) / s


def _area_resize(img: jnp.ndarray, w_y: jnp.ndarray, w_x: jnp.ndarray) -> jnp.ndarray:
    """Area-downscale the last two axes of `img` via `w_y @ img @ w_x`."""
    return jnp.matmul(jnp.matmul(w_y, img.astype(jnp.float32)), w_x)


def _make_gray_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1,) uint8. Index 0 = background (0), rest are distinct grays.

    Same ladder as the reference ObjectTypeMaskWrapper: shade = 255 // K, class i
    gets (i + 1) * shade, so the brightest class is at most 255 and the spacing is
    uniform and independent of K.
    """
    shade = 255 // max(n_classes, 1)
    levels = [(i + 1) * shade for i in range(max(n_classes, 1))]
    return np.array([0] + levels, dtype=np.uint8)


def _make_color_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1, 3) uint8 RGB palette for visualization. Index 0 = black."""
    cols = [(0, 0, 0)]
    for i in range(max(n_classes, 1)):
        h = i / max(n_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)


def _planes_isometric_rgb(
    planes: jnp.ndarray,
    palette: jnp.ndarray,
    out_h: int,
    out_w: int,
    shear: float = 0.25,
    gap_frac: float = 0.38,
    rise_frac: float = 0.05,
    bg_alpha: float = 0.20,
    aspect: float | None = None,
) -> jnp.ndarray:
    """Render (K, h, w) binary planes as obliquely stacked translucent sheets.

    Visualization only -- this never touches the observation. Each plane becomes a
    sheared parallelogram, offset right and up from the one behind it, tinted with
    its class colour. Mirrors Figure 3(e) of the OCCAM paper, where "Planes" is
    drawn as a stack of sheets rather than one flattened image.

    `aspect` is the width/height ratio each sheet is drawn at, independent of the
    source shape. Pass the native frame ratio to undo the square 84x84 squash, so
    the sheets match the other mask panels; defaults to the source ratio.

    Everything is a closed-form inverse map plus a gather, so it stays jit- and
    vmap-compatible: K is static, and no plane is materialised at full canvas size.
    """
    K, h, w = planes.shape
    A = (w / h) if aspect is None else float(aspect)

    # fit the stack into the canvas: solve for the sheet height, width follows A
    ph_from_w = out_w / (A * (1.0 + (K - 1) * gap_frac))
    ph_from_h = out_h / (1.0 + (K - 1) * rise_frac + shear * A)
    ph = 0.94 * min(ph_from_w, ph_from_h)
    pw = A * ph
    sx, sy = pw / w, ph / h        # independent axis scales -> free aspect change
    dx = gap_frac * pw             # per-plane shift right
    dy = -rise_frac * ph           # per-plane shift up (negative y)

    need_w = pw + (K - 1) * dx
    need_h = ph + shear * pw - (K - 1) * dy
    ox = (out_w - need_w) / 2.0
    oy = -(K - 1) * dy + (out_h - need_h) / 2.0

    X = jnp.arange(out_w, dtype=jnp.float32)[None, :]
    Y = jnp.arange(out_h, dtype=jnp.float32)[:, None]

    canvas = jnp.zeros((out_h, out_w, 3), jnp.float32)
    for k in range(K - 1, -1, -1):                     # painter's order: back to front
        xr = X - ox - k * dx
        u = xr / sx
        v = (Y - oy - k * dy - shear * xr) / sy
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = jnp.clip(u, 0, w - 1).astype(jnp.int32)
        vi = jnp.clip(v, 0, h - 1).astype(jnp.int32)

        hit = (planes[k][vi, ui] > 0) & inside
        # thresholds in source units, chosen so the frame is ~1.5 output px on both
        # axes even when sx != sy
        eu, ev = 1.5 / sx, 1.5 / sy
        edge = inside & ((u < eu) | (u > w - 1 - eu) | (v < ev) | (v > h - 1 - ev))

        # sheet frame carries the class colour too, so each pane is identifiable
        # even when it holds no active object
        cls = palette[k + 1].astype(jnp.float32)
        col = jnp.where((hit | edge)[..., None], cls, jnp.float32(18.0))
        alpha = jnp.where(hit, 1.0, jnp.where(edge, 0.75, jnp.where(inside, bg_alpha, 0.0)))
        canvas = canvas * (1.0 - alpha[..., None]) + col * alpha[..., None]

    return jnp.clip(canvas, 0.0, 255.0).astype(jnp.uint8)


def _extract_object_groups(obs: Any) -> List[ObjectObservation]:
    """ObjectObservation leaves from obs PyTree, in deterministic PyTree order."""
    leaves = jax.tree_util.tree_leaves(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    return [leaf for leaf in leaves if isinstance(leaf, ObjectObservation)]


def _group_names(obs: Any) -> List[str]:
    """Names of the ObjectObservation leaves, in the same order as _extract_object_groups."""
    paths, _ = jax.tree_util.tree_flatten_with_path(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    names = []
    for path, leaf in paths:
        if isinstance(leaf, ObjectObservation):
            names.append(".".join(
                str(getattr(k, "name", getattr(k, "key", k))) for k in path
            ))
    return names


def _render_order(base_env: Any, names: List[str]) -> List[int]:
    """Back-to-front group indices. Reads OCCAM_RENDER_ORDER off the game if declared.

    A game may expose OCCAM_RENDER_ORDER as a tuple of group names in the order its
    own render() draws them. Without it the PyTree field order is used, which is not
    the draw order in general.
    """
    declared = getattr(base_env, "OCCAM_RENDER_ORDER", None)
    if not declared:
        return list(range(len(names)))
    pos = {n: i for i, n in enumerate(names)}
    order = [pos[n] for n in declared if n in pos]
    seen = set(order)
    return order + [i for i in range(len(names)) if i not in seen]


def _sheets_width(base_w: int, k: int, gap_frac: float = 0.38) -> int:
    """Panel width the isometric sheet stack needs for k planes."""
    return int(round(base_w * (1.0 + gap_frac * max(0, k - 1))))


def _group_box_arrays(group: ObjectObservation, img_h: int, img_w: int):
    """Normalize one ObjectObservation to (x, y, w, h, valid) arrays of shape (n,)."""
    x = jnp.atleast_1d(group.x).astype(jnp.int32)
    y = jnp.atleast_1d(group.y).astype(jnp.int32)
    w = jnp.atleast_1d(group.width).astype(jnp.int32)
    h = jnp.atleast_1d(group.height).astype(jnp.int32)

    n = x.shape[0]
    active = jnp.broadcast_to(jnp.atleast_1d(group.active).astype(bool), (n,))

    valid = (
        active
        & (w > 0)
        & (h > 0)
        & (x < img_w)
        & (y < img_h)
        & (x + w > 0)
        & (y + h > 0)
    )
    return x, y, w, h, valid


def _rasterize_group(x, y, w, h, valid, img_h: int, img_w: int) -> jnp.ndarray:
    """Boolean occupancy mask (img_h, img_w): True wherever any valid box covers the pixel.

    Stamps each box as +/-1 corners into a difference array and integrates with two
    cumsums, so cost is O(n + img_h * img_w) instead of the O(n * img_h * img_w) of an
    n-wide broadcast. Matters for grid games (Breakout bricks, MsPacman pills) where a
    single group holds hundreds of boxes.
    """
    x0 = jnp.clip(x, 0, img_w)
    x1 = jnp.clip(x + w, 0, img_w)
    y0 = jnp.clip(y, 0, img_h)
    y1 = jnp.clip(y + h, 0, img_h)
    inc = valid.astype(jnp.int32)

    diff = jnp.zeros((img_h + 1, img_w + 1), dtype=jnp.int32)
    diff = diff.at[y0, x0].add(inc)
    diff = diff.at[y0, x1].add(-inc)
    diff = diff.at[y1, x0].add(-inc)
    diff = diff.at[y1, x1].add(inc)

    counts = jnp.cumsum(jnp.cumsum(diff, axis=0), axis=1)
    return counts[:img_h, :img_w] > 0  # (H, W) bool


def _union(group_masks: List[jnp.ndarray]) -> jnp.ndarray:
    """Logical OR over a non-empty list of (H, W) boolean masks."""
    union = group_masks[0]
    for gm in group_masks[1:]:
        union = union | gm
    return union


def _group_masks(obs: Any, img_h: int, img_w: int) -> List[jnp.ndarray]:
    """One (H, W) boolean occupancy mask per ObjectObservation group."""
    return [
        _rasterize_group(*_group_box_arrays(g, img_h, img_w), img_h, img_w)
        for g in _extract_object_groups(obs)
    ]


def _class_map(group_masks: List[jnp.ndarray], order: List[int] | None = None) -> jnp.ndarray:
    """(H, W) int32 class ids, 1..K with 0 = background.

    Painted back to front, so a later group in `order` overwrites an earlier one.
    Without `order` the PyTree field order is used, matching the reference's
    sequential `state[...].fill(value)` loop.
    """
    idx = range(len(group_masks)) if order is None else order
    class_map = jnp.zeros(group_masks[0].shape, dtype=jnp.int32)
    for k in idx:
        class_map = jnp.where(group_masks[k], k + 1, class_map)
    return class_map


@struct.dataclass
class OCCAMState:
    # must be named `atari_state` so jaxtari-blines eval code can walk the state tree
    atari_state: Any
    mask_stack: jnp.ndarray  # (F, C, H, W) uint8


class OCCAMWrapper(JaxatariWrapper):
    """
    Object-Centric Attention via Masking wrapper. Apply after AtariWrapper, in place of PixelObsWrapper.
    Output shape: (frame_stack_size, 84, 84, 1) for object/binary/class; (frame_stack_size * n_classes, 84, 84, 1) for planes.

    Frame-skip, max-pooling, stacking and reward clipping live here because
    AtariWrapper deliberately leaves them to the observation wrapper (one emulator
    frame per AtariWrapper.step), so this mirrors PixelObsWrapper exactly.

    `max_pooling` is a deliberate deviation from the reference, which has none. It is
    kept for parity with PixelObsWrapper and only affects `mask_mode="object"`: the
    other three modes are drawn from object coordinates, never from pixels, so the
    max-pooled frame never reaches them.

    REWARD ACROSS TERMINATION: AtariWrapper has no autoreset, so the frame-skip scan
    runs its full `frame_skip` sub-steps even when the first one already terminated,
    and `jnp.sum(rewards)` therefore includes rewards emitted after termination. This
    is intentional parity, not an oversight: PixelObsWrapper and ObjectCentricWrapper
    reduce identically (fixed-length scan, `sum(rewards)`, `terminations.any()`), so
    changing it here alone would make OCCAM runs incomparable to the pixel baseline.
    """

    def __init__(
        self,
        env,
        mask_mode: str = "binary",
        frame_stack_size: int = 4,
        frame_skip: int = 4,
        max_pooling: bool = True,
        clip_reward: bool = True,
        out_size: Tuple[int, int] = (84, 84),
        game_name: str = "unknown",
    ):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "OCCAMWrapper must be applied after AtariWrapper"
        assert mask_mode in MASK_MODES, f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}"

        self.mask_mode = mask_mode
        self.frame_stack_size = int(frame_stack_size)
        self.frame_skip = int(frame_skip)
        self.max_pooling = bool(max_pooling)
        self.clip_reward = bool(clip_reward)
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        self.game_name = game_name

        self.base_env = env._env  # base game env; used for render + obs
        img_shape = self.base_env.image_space().shape  # (H, W, 3)
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])

        # probe object layout once to fix the static structure
        probe_obs = self.base_env._get_observation(self.base_env.reset(jax.random.PRNGKey(0))[1])
        self.num_classes = len(_extract_object_groups(probe_obs))

        if self.num_classes == 0:
            raise NotImplementedError(
                f"OCCAM: game '{game_name}' exposes no ObjectObservation groups, so no "
                f"object bounding boxes are available to build masks from."
            )

        self._gray_palette = jnp.asarray(_make_gray_palette(self.num_classes))   # (K+1,)
        self._color_palette = jnp.asarray(_make_color_palette(self.num_classes)) # (K+1, 3)

        # static area-resampling operators: (out_h, img_h) and (img_w, out_w)
        self._w_y = jnp.asarray(_area_weights(self.img_h, self.out_h), dtype=jnp.float32)
        self._w_x = jnp.asarray(_area_weights(self.img_w, self.out_w).T, dtype=jnp.float32)

        self.per_frame_channels = self.num_classes if mask_mode == "planes" else 1
        total_channels = self.frame_stack_size * self.per_frame_channels
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(total_channels, self.out_h, self.out_w, 1), dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _mask_single(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Build the OCCAM mask for one native-resolution frame. Returns (C, out_h, out_w) uint8."""
        group_masks = _group_masks(obs, self.img_h, self.img_w)

        # The reference draws every mask at native 210x160 and then area-downscales
        # once, with no thresholding, so partially covered output pixels keep their
        # continuous gray value. Downscale last, never threshold after.
        if self.mask_mode == "object":
            # real grayscale pixels inside boxes, zero background. Round first: the
            # reference reads an already-quantized uint8 screen from the emulator.
            gray = jnp.round(_rgb_to_gray(frame_rgb))                   # (H, W) float
            union = _union(group_masks)
            native = jnp.where(union, gray, 0.0)[None]                  # (1, H, W)

        elif self.mask_mode == "binary":
            native = _union(group_masks).astype(jnp.float32)[None] * 255.0

        elif self.mask_mode == "class":
            # class gray levels painted natively, later group wins on overlap
            native = self._gray_palette[_class_map(group_masks)].astype(jnp.float32)[None]

        else:  # "planes": one plane per class
            native = jnp.stack([gm.astype(jnp.float32) for gm in group_masks], axis=0) * 255.0

        out = _area_resize(native, self._w_y, self._w_x)                # (C, oh, ow)
        # round, don't truncate: cv2.resize rounds when it writes uint8, so a bare
        # cast would bias every partially covered pixel down by up to 1.
        return jnp.clip(jnp.round(out), 0.0, 255.0).astype(jnp.uint8)

    def _stack_to_obs(self, mask_stack: jnp.ndarray) -> jnp.ndarray:
        """(F, C, H, W) -> (F*C, H, W, 1) uint8."""
        f, c, h, w = mask_stack.shape
        return mask_stack.reshape(f * c, h, w)[..., None]

    def _reset_internal(self, key):
        # AtariWrapper.step already advances state.key on every sub-step, so the key
        # handed in on autoreset is never a literal repeat. Split anyway so the seed
        # consumed by reset is provably distinct from the key living in the state.
        reset_key, _ = jax.random.split(key)
        _, atari_state = self._env.reset(reset_key)
        frame = self.base_env.render(atari_state.env_state)
        obs = self.base_env._get_observation(atari_state.env_state)
        m = self._mask_single(frame, obs)                              # (C, H, W)
        mask_stack = jnp.stack([m] * self.frame_stack_size)            # (F, C, H, W)
        return mask_stack, OCCAMState(atari_state, mask_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[jnp.ndarray, OCCAMState]:
        mask_stack, state = self._reset_internal(key)
        return self._stack_to_obs(mask_stack), state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: OCCAMState, action: int):
        # frame_skip sub-steps
        def body_fn(carry, _):
            atari_state, action = carry
            _, new_atari_state, reward, terminated, truncated, info = self._env.step(
                atari_state, action
            )
            return (new_atari_state, action), (
                new_atari_state.env_state, reward, terminated, truncated, info
            )

        (atari_state, _), (env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn, (state.atari_state, action), None, length=self.frame_skip
        )

        last_env_state = jax.tree.map(lambda z: z[-1], env_states)

        # max-pooling anti-flicker, matching PixelObsWrapper
        if self.max_pooling and self.frame_skip > 1:
            img = self.base_env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda z: z[-2], env_states)
            prev_img = self.base_env.render(prev_env_state)
            frame = jnp.maximum(img, prev_img)
        else:
            frame = self.base_env.render(last_env_state)

        obs = self.base_env._get_observation(last_env_state)
        new_mask = self._mask_single(frame, obs)                        # (C, H, W)
        mask_stack = jnp.concatenate(
            [state.mask_stack[1:], new_mask[None]], axis=0
        )                                                               # (F, C, H, W)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()

        # autoreset on done
        mask_stack, occ_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),
            lambda: self._reset_internal(atari_state.key),
            lambda: (mask_stack, OCCAMState(atari_state, mask_stack)),
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        obs_out = self._stack_to_obs(mask_stack)
        return obs_out, occ_state, reward, terminated, truncated, info_dict


class _OCCAMViz:
    """Side-by-side [game | OCCAM mask] video helper; used only for eval logging."""

    def __init__(self, base_env, mask_mode: str, obs_res: bool = False,
                 out_size: Tuple[int, int] = (84, 84)):
        self.env = base_env
        self.mask_mode = mask_mode
        self.obs_res = bool(obs_res)
        img_shape = base_env.image_space().shape
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])

        probe = base_env._get_observation(base_env.reset(jax.random.PRNGKey(0))[1])
        self.num_classes = len(_extract_object_groups(probe))
        self.render_order = _render_order(base_env, _group_names(probe))
        self._gray_palette = jnp.asarray(_make_gray_palette(self.num_classes))
        self._color_palette = jnp.asarray(_make_color_palette(self.num_classes))
        self._w_y = jnp.asarray(_area_weights(self.img_h, self.out_h), dtype=jnp.float32)
        self._w_x = jnp.asarray(_area_weights(self.img_w, self.out_w).T, dtype=jnp.float32)

    def _to_obs_res(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """(H, W, 3) uint8 -> (out_h, out_w, 3) uint8, same area operator as the wrapper."""
        chw = jnp.transpose(rgb.astype(jnp.float32), (2, 0, 1))
        out = _area_resize(chw, self._w_y, self._w_x)
        out = jnp.clip(jnp.round(out), 0.0, 255.0).astype(jnp.uint8)
        return jnp.transpose(out, (1, 2, 0))

    def planes_rgb(self, obs: Any) -> jnp.ndarray:
        """(K, h, w) boolean planes at the active resolution, for the sheet stack."""
        masks = jnp.stack(_group_masks(obs, self.img_h, self.img_w))
        if not self.obs_res:
            return masks
        f = _area_resize(masks.astype(jnp.float32), self._w_y, self._w_x)
        return f > 0.0

    def _mask_rgb(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Native-resolution RGB visualization (H, W, 3) uint8 of the mask."""
        group_masks = _group_masks(obs, self.img_h, self.img_w)
        if self.mask_mode == "binary":
            union = _union(group_masks)
            rgb = jnp.where(union[..., None], jnp.uint8(255), jnp.uint8(0))
            rgb = jnp.broadcast_to(rgb, (self.img_h, self.img_w, 3))

        elif self.mask_mode == "object":
            gray = _rgb_to_gray(frame_rgb)
            union = _union(group_masks)
            masked = jnp.where(union, gray, 0.0).astype(jnp.uint8)
            rgb = jnp.repeat(masked[..., None], 3, axis=-1)

        elif self.mask_mode == "class":
            # same single-label rule as _mask_single: later group wins on overlap
            rgb = self._color_palette[_class_map(group_masks)]

        else:
            rgb = self._color_palette[_class_map(group_masks, self.render_order)]

        rgb = rgb.astype(jnp.uint8)
        return self._to_obs_res(rgb) if self.obs_res else rgb

    @functools.partial(jax.jit, static_argnums=(0,))
    def _frame(self, env_state) -> jnp.ndarray:
        clean = self.env.render(env_state).astype(jnp.uint8)
        obs = self.env._get_observation(env_state)
        mask_rgb = self._mask_rgb(clean, obs)
        panel = self._to_obs_res(clean) if self.obs_res else clean
        if self.mask_mode != "planes":
            return jnp.concatenate([panel, mask_rgb], axis=1)
        h, w = panel.shape[0], panel.shape[1]
        sheets = _planes_isometric_rgb(
            self.planes_rgb(obs), self._color_palette,
            h, _sheets_width(w, self.num_classes), aspect=w / h,
        )
        return jnp.concatenate([panel, mask_rgb, sheets], axis=1)

    def frames(self, env_states) -> jnp.ndarray:
        """(T, ...) base env states -> (T, H, W_row, 3) uint8; 3 panels in planes mode."""
        return jax.vmap(self._frame)(env_states)


def occam_comparison_frames(env_id: str, mask_mode: str, env_states, mods=None,
                            obs_res: bool = False):
    """Side-by-side [game | mask] frames (T, H, 2W, 3) uint8 for one eval rollout."""
    import jaxatari  # local import to avoid a hard dependency at module import time

    base_env = jaxatari.make(env_id, mods=mods)
    viz = _OCCAMViz(base_env, mask_mode, obs_res=obs_res)
    return np.asarray(viz.frames(env_states), dtype=np.uint8)         # (T, H, 2W, 3)


def _to_chw(frames_thwc: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) -> (T, 3, H, W) contiguous, for wandb.Video."""
    return np.ascontiguousarray(np.transpose(frames_thwc, (0, 3, 1, 2)))


def _load_font(size: int):
    """TrueType font with fallback to PIL bitmap default."""
    try:
        from PIL import ImageFont
        for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()
    except Exception:
        return None


def _text_banner(width: int, text: str, height: int, font_size: int = 16,
                 bg=(255, 255, 255), fg=(0, 0, 0)) -> np.ndarray:
    """(height, width, 3) banner with left-aligned text."""
    banner = np.full((height, width, 3), 0, np.uint8)
    banner[:] = np.array(bg, np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(banner)
        d = ImageDraw.Draw(im)
        font = _load_font(font_size)
        try:
            bbox = d.textbbox((0, 0), text, font=font)
            ty = max(0, (height - (bbox[3] - bbox[1])) // 2 - bbox[1])
        except Exception:
            ty = max(0, (height - font_size) // 2)
        d.text((6, ty), text, fill=tuple(fg), font=font)
        banner = np.asarray(im)
    except Exception:
        pass
    return banner


def _caption_clip(frames_thwc: np.ndarray, text: str, banner_h: int = 16) -> np.ndarray:
    """Prepend a caption banner on every frame; uses PIL if available."""
    T, H, W, C = frames_thwc.shape
    banner = np.zeros((banner_h, W, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(banner)
        ImageDraw.Draw(im).text((3, 2), text, fill=(255, 255, 255))
        banner = np.asarray(im)
    except Exception:
        pass
    banner = np.broadcast_to(banner, (T, banner_h, W, 3))
    return np.concatenate([banner, frames_thwc], axis=1)


def _write_video_file(path: str, frames_thwc: np.ndarray, fps: int = 30) -> str | None:
    """Write mp4 or gif; returns actual path written, or None."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio
        try:
            imageio.mimwrite(path, list(frames_thwc), fps=fps, macro_block_size=None)
            return path
        except Exception:
            gif = os.path.splitext(path)[0] + ".gif"
            imageio.mimwrite(gif, list(frames_thwc), duration=1.0 / fps)
            return gif
    except Exception:
        return None


def save_eval_frames(save_dir: str, mod_label: str, frames_thwc: np.ndarray,
                     fps: int = 30, write_mp4: bool = False):
    """Write frames to <save_dir>/eval_<mod_label>.npy; mp4 only if write_mp4=True."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"eval_{mod_label}.npy"), frames_thwc)
    if write_mp4:
        _write_video_file(os.path.join(save_dir, f"eval_{mod_label}.mp4"), frames_thwc, fps=fps)


def log_occam_comparison_video(
    env_id: str,
    mask_mode: str,
    env_states,
    mods=None,
    mod_label: str = "default",
    step: int = 0,
    fps: int = 30,
    save_dir: str | None = None,
    wandb_run=None,
    log_wandb: bool = True,
    obs_res: bool = False,
):
    """Render, optionally save (.npy), caption and W&B-log a [game | mask] eval clip."""
    frames = occam_comparison_frames(env_id, mask_mode, env_states, mods=mods,
                                     obs_res=obs_res)
    if save_dir is not None:
        save_eval_frames(save_dir, mod_label, frames, fps=fps)
    captioned = _caption_clip(frames, f"{env_id} | {mask_mode} | {mod_label} | step {step}")
    if log_wandb:
        import wandb
        key = f"eval/{env_id}/{mask_mode}/{mod_label}"
        (wandb_run or wandb).log({key: wandb.Video(_to_chw(captioned), fps=fps, format="mp4")}, step=step)
    return captioned


def build_occam_summary_video(
    env_id: str,
    save_root: str,
    mods=None,
    mask_modes=MASK_MODES,
    out_name: str | None = None,
    fps: int = 30,
    log_wandb: bool = False,
    max_frames: int | None = None,
    strip_top_px: int = 0,
    hold_last_seconds: float = 4.0,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags=None,
    wandb_run_name: str | None = None,
):
    """
    Stitch all saved eval clips into one summary video.
    Columns = 4 mask variants, rows = eval mods. Clips are memory-mapped; frames stream to the encoder.
    Returns (path_written_or_None, num_frames).
    """
    import os
    import glob

    grid_order = list(mask_modes)[:4]
    while len(grid_order) < 4:
        grid_order.append(None)

    # auto-discover mod labels from saved clips
    if not mods:
        found = set()
        for mm in grid_order:
            if mm is None:
                continue
            for p in glob.glob(os.path.join(save_root, env_id, mm, "eval_*.npy")):
                found.add(os.path.basename(p)[len("eval_"):-len(".npy")])
        mods = sorted(found) if found else ["default"]
    else:
        mods = list(mods)

    def open_clip(mask_mode, mod):
        if mask_mode is None:
            return None
        p = os.path.join(save_root, env_id, mask_mode, f"eval_{mod}.npy")
        if not os.path.exists(p):
            return None
        return np.load(p, mmap_mode="r")  # mmap: one frame at a time, avoids loading full clip

    mods = sorted(mods, key=lambda m: (m != "default", m))   # "default" first

    # open all clips as memmaps; track longest length
    clips = {}
    cell_h = None
    seen_w = {}
    max_len = 0
    for mod in mods:
        for mm in grid_order:
            c = open_clip(mm, mod)
            clips[(mod, mm)] = c
            if c is not None:
                cell_h = c.shape[1]
                seen_w[mm] = max(seen_w.get(mm, 0), c.shape[2])
                max_len = max(max_len, c.shape[0])
    if cell_h is None:
        return None, None
    fallback_w = max(seen_w.values())
    col_w = {mm: seen_w.get(mm, fallback_w) for mm in grid_order}
    flat = [w for mm, w in col_w.items() if mm != "planes"]
    base_w = (min(flat) // 2) if flat else (col_w.get("planes", fallback_w) // 3)

    # stride if max_frames set
    if max_frames and max_len > max_frames:
        out_idx = np.linspace(0, max_len - 1, max_frames).astype(np.int64)
    else:
        out_idx = np.arange(max_len, dtype=np.int64)
    T = len(out_idx)

    BG = (255, 255, 255)
    FG = (0, 0, 0)
    bw = 4        # cell border width
    col_gap = 26  # horizontal gap between cells
    row_gap = 18  # vertical gap between rows

    # pre-render static banners
    cap_h, title_h = 24, 34
    cell_banner = {
        (mod, mm): _text_banner(
            col_w[mm],
            f"{(mm or '-').upper()}  -  {mod}" + ("   (n/a)" if clips[(mod, mm)] is None else ""),
            cap_h, font_size=16, bg=BG, fg=FG,
        )
        for mod in mods for mm in grid_order
    }
    eff_h = cell_h - strip_top_px
    black_cell = {mm: np.zeros((eff_h, col_w[mm], 3), np.uint8) for mm in grid_order}
    row_w = sum(col_w[mm] for mm in grid_order) + col_gap * 3
    title = _text_banner(row_w,
                         f"{env_id}  -  OCCAM mask comparison", title_h, font_size=24, bg=BG, fg=FG)

    full_cell_h = cap_h + eff_h
    h_spacer = np.full((full_cell_h, col_gap, 3), 255, np.uint8)
    v_spacer = np.full((row_gap, row_w, 3), 255, np.uint8)

    def _decorate(frame, mm=None):
        """White border + panel divider lines."""
        f = np.array(frame, dtype=np.uint8)
        h, w = f.shape[:2]
        for mid in ([base_w, 2 * base_w] if mm == "planes" else [w // 2]):
            if 0 < mid < w:
                f[:, max(0, mid - bw // 2):mid + (bw - bw // 2)] = 255
        f[:bw, :] = 255
        f[-bw:, :] = 255
        f[:, :bw] = 255
        f[:, -bw:] = 255
        return f

    def grid_frame(t_src):
        row_imgs = []
        for mod in mods:
            cells = []
            for mm in grid_order:
                c = clips[(mod, mm)]
                if c is None:
                    frame = black_cell[mm]
                else:
                    frame = np.asarray(c[min(t_src, c.shape[0] - 1)])[strip_top_px:]
                    if frame.shape[1] != col_w[mm]:
                        pad = max(0, col_w[mm] - frame.shape[1])
                        frame = np.pad(frame, [(0, 0), (0, pad), (0, 0)])[:, :col_w[mm]]
                cell = np.concatenate([cell_banner[(mod, mm)], _decorate(frame, mm)], axis=0)
                cells.append(cell)
            row_parts = []
            for i, cell in enumerate(cells):
                if i:
                    row_parts.append(h_spacer)
                row_parts.append(cell)
            row_imgs.append(np.concatenate(row_parts, axis=1))
        grid_parts = []
        for i, ri in enumerate(row_imgs):
            if i:
                grid_parts.append(v_spacer)
            grid_parts.append(ri)
        grid = np.concatenate(grid_parts, axis=0)
        return np.concatenate([title, grid], axis=0)

    # stream frames to encoder; hold last frame for hold_last_seconds
    out_name = out_name or f"summary_{env_id}.mp4"
    out_path = os.path.join(save_root, env_id, out_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    hold_n = max(0, int(round(hold_last_seconds * fps)))

    written = None
    try:
        import imageio
        try:
            writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
            target = out_path
        except Exception:
            target = os.path.splitext(out_path)[0] + ".gif"
            writer = imageio.get_writer(target, duration=1.0 / fps)
        last = None
        for t in out_idx:
            last = grid_frame(int(t))
            writer.append_data(last)
        for _ in range(hold_n):
            if last is not None:
                writer.append_data(last)
        writer.close()
        written = target
        T += hold_n
    except Exception:
        written = None

    # upload as own W&B run for persistence across re-runs
    if wandb_project and written is not None:
        try:
            import wandb
            import time
            run = wandb.init(
                project=wandb_project,
                entity=(wandb_entity or None),
                name=wandb_run_name or f"summary_{env_id}_{int(time.time())}",
                tags=list(wandb_tags) if wandb_tags else ["summary", env_id],
                job_type="summary",
                reinit=True,
            )
            run.log({f"summary/{env_id}": wandb.Video(written, fps=fps, format="mp4")})
            run.finish()
        except Exception as e:
            print(f"[warn] W&B summary upload failed: {e}")
    elif log_wandb and written is not None:
        try:
            import wandb
            wandb.log({f"summary/{env_id}": wandb.Video(written, fps=fps, format="mp4")})
        except Exception:
            pass

    return written, T



JAXTARI_15 = [
    "asteroids", "beamrider", "breakout", "enduro", "freeway", "frostbite",
    "gravitar", "kangaroo", "montezumarevenge", "mspacman", "phoenix", "pong",
    "seaquest", "skiing", "tennis",
]


def probe_game(env_id: str):
    """Classify OCCAM support for one game.
    Returns (status, num_object_groups, num_grid_fields, error_or_None)."""
    import jaxatari
    try:
        env = jaxatari.make(env_id)
        obs = env._get_observation(env.reset(jax.random.PRNGKey(0))[1])
        leaves = jax.tree_util.tree_leaves(
            obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
        )
        groups = [l for l in leaves if isinstance(l, ObjectObservation)]
        grids = [
            l for l in leaves
            if (not isinstance(l, ObjectObservation)) and getattr(l, "ndim", 0) >= 2
        ]
        if len(groups) == 0:
            return "none", 0, len(grids), None
        return ("partial" if grids else "full"), len(groups), len(grids), None
    except Exception as e:  # pragma: no cover - depends on local install
        return "error", 0, 0, repr(e)


def print_support_table(games=None):  # pragma: no cover
    games = games or JAXTARI_15
    print(f"{'game':<20}{'status':<10}{'#obj groups':<13}{'#grid fields':<13}note")
    print("-" * 72)
    for g in games:
        status, ngroups, ngrids, err = probe_game(g)
        note = ""
        if status == "partial":
            note = "grid objects (bricks/ice/...) not masked"
        elif status == "none":
            note = "no ObjectObservation -> needs adapter"
        elif status == "error":
            note = err or ""
        print(f"{g:<20}{status:<10}{ngroups:<13}{ngrids:<13}{note}")


if __name__ == "__main__":  # pragma: no cover
    print_support_table()
