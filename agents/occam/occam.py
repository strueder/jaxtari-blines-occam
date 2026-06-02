"""
OCCAM: Object-Centric Attention via Masking  --  JAX / JAXtari baseline
=======================================================================

A single-file, JIT-compatible implementation of OCCAM as an observation
wrapper for JAXtari (https://github.com/k4ntz/JAXAtari), intended as a
neuro-symbolic baseline for https://github.com/remunds/jaxtari-blines.

Original method (PyTorch / OCAtari / HackAtari):
    "Deep Reinforcement Learning via Object-Centric Attention via Masking (OCCAM)"
    Bl"uml, Derstroff, Gregori, Dillies, Delfosse, Kersting (2025), arXiv:2504.03024
    Reference code: https://github.com/VanillaWhey/OCAtariWrappers
This file re-implements the *idea* (object-centric masking) natively in JAX.

KEY DIFFERENCE TO THE ORIGINAL (must be stated in the report!):
    The original OCCAM extracts object bounding boxes with a lightweight
    *detector* operating on the raw frames (motion / optical flow / a learned
    detector), because OCAtari/ALE does not hand out clean object boxes.
    JAXtari instead exposes ground-truth object bounding boxes for every game
    via the `ObjectObservation` dataclass (x, y, width, height, active,
    visual_id, ...). We therefore build the masks directly from these
    ground-truth boxes -- no frame-differencing / detector is needed. This is
    in the spirit of the paper (which treats the detector purely as overhead),
    but it removes the (noisy) detection step and should be disclosed.

WHY THIS IS A *NEURO-SYMBOLIC* BASELINE:
    OCCAM injects *symbolic, structured object knowledge* (which entities exist,
    where they are, and -- for Class/Planes masks -- which category each entity
    belongs to) as a hard-attention inductive bias into a *neural* (CNN) policy.
    The "symbolic" part is the object extraction (positions + categories); the
    "neural" part is the convolutional policy that learns on top of the masked
    input. The Class Masks and Planes variants are the most strongly
    neuro-symbolic (they encode symbolic object *categories*), while the paper
    deliberately shows you do NOT need a *full* symbolic state vector (à la
    OCAtari's semantic vector) to get the robustness benefit.

THE FOUR OCCAM ABSTRACTION LEVELS (all implemented here, selected via
`mask_mode`); see Sec. 2.2 / Figure 3 of the paper:
    - "object" : keep the real (grayscale) pixels inside each object box,
                 zero out the background.                       -> (F, 84, 84)
    - "binary" : 1 inside any object box, 0 elsewhere (no identity / texture).
                                                                -> (F, 84, 84)
    - "class"  : each object box filled with a class-specific gray level
                 (category preserved, texture discarded).       -> (F, 84, 84)
    - "planes" : one binary plane per object category, stacked as channels.
                                                          -> (F * n_classes, 84, 84)

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
"""

from __future__ import annotations

import functools
import warnings
import colorsys
from typing import Any, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from jaxatari.wrappers import JaxatariWrapper, AtariWrapper
from jaxatari.environment import ObjectObservation
from jaxatari import spaces


# --------------------------------------------------------------------------- #
#  constants                                                                    #
# --------------------------------------------------------------------------- #
MASK_MODES = ("object", "binary", "class", "planes")

# grayscale weights identical to JAXtari's PixelObsWrapper.preprocess_image,
# so the "object" mask matches the pixel baseline's grayscale exactly.
_GRAY_W = jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32)


# --------------------------------------------------------------------------- #
#  small pure helpers                                                           #
# --------------------------------------------------------------------------- #
def _rgb_to_gray(frame_rgb: jnp.ndarray) -> jnp.ndarray:
    """(H, W, 3) uint8/float -> (H, W) float32 grayscale."""
    return jnp.dot(frame_rgb.astype(jnp.float32), _GRAY_W)


def _resize(img: jnp.ndarray, out_hw: Tuple[int, int], method: str) -> jnp.ndarray:
    """Resize only the last two (spatial) axes of `img` to `out_hw`."""
    target = tuple(img.shape[:-2]) + tuple(out_hw)
    return jax.image.resize(img.astype(jnp.float32), target, method=method)


def _make_gray_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1,) uint8.  Index 0 == background (0). Distinct grays else."""
    if n_classes <= 1:
        levels = [255]
    else:
        levels = list(np.linspace(90, 255, n_classes).round().astype(int))
    return np.array([0] + levels, dtype=np.uint8)


def _make_color_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1, 3) uint8 RGB palette for *visualization only*. Index 0 = black."""
    cols = [(0, 0, 0)]
    for i in range(max(n_classes, 1)):
        h = i / max(n_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)


def _extract_object_groups(obs: Any) -> List[ObjectObservation]:
    """
    Return the list of `ObjectObservation` nodes contained in an observation
    PyTree, in deterministic (PyTree) order. Non-object leaves (scores, grids,
    timers, ...) are dropped. Works both eagerly (concrete arrays) and under
    `jax.jit` (tracer arrays), because it only relies on the PyTree *structure*,
    which is static for a given game.
    """
    leaves = jax.tree_util.tree_leaves(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    return [leaf for leaf in leaves if isinstance(leaf, ObjectObservation)]


def _group_box_arrays(group: ObjectObservation, img_h: int, img_w: int):
    """
    Normalize one ObjectObservation group to 1-D int arrays (x, y, w, h) of
    shape (n,) plus a boolean `valid` of shape (n,). `valid` is False for
    inactive objects and for boxes that are degenerate or fully off-screen.
    """
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
    """
    Vectorized box rasterization for one group.
    Returns a boolean occupancy mask of shape (img_h, img_w): True wherever any
    *valid* box of this group covers the pixel.
    """
    n = x.shape[0]
    ox = x.reshape(n, 1, 1)
    oy = y.reshape(n, 1, 1)
    ow = w.reshape(n, 1, 1)
    oh = h.reshape(n, 1, 1)
    col = jnp.arange(img_w).reshape(1, 1, img_w)
    row = jnp.arange(img_h).reshape(1, img_h, 1)

    inside = (col >= ox) & (col < ox + ow) & (row >= oy) & (row < oy + oh)
    inside = inside & valid.reshape(n, 1, 1)
    return inside.any(axis=0)  # (H, W) bool


def _union(group_masks: List[jnp.ndarray]) -> jnp.ndarray:
    """Logical OR over a non-empty list of (H, W) boolean masks."""
    union = group_masks[0]
    for gm in group_masks[1:]:
        union = union | gm
    return union


# --------------------------------------------------------------------------- #
#  wrapper state                                                                #
# --------------------------------------------------------------------------- #
@struct.dataclass
class OCCAMState:
    # NOTE: the field MUST be named `atari_state` so that the eval/video code in
    # jaxtari-blines (which walks `LogState.atari_state.atari_state.env_state`)
    # keeps working unchanged, exactly like PixelState / ObjectCentricState.
    atari_state: Any
    mask_stack: jnp.ndarray  # (F, C, H, W) uint8, internal stacked masks


# --------------------------------------------------------------------------- #
#  the wrapper                                                                  #
# --------------------------------------------------------------------------- #
class OCCAMWrapper(JaxatariWrapper):
    """
    Object-Centric Attention via Masking wrapper.

    Apply it AFTER `AtariWrapper`, in place of `PixelObsWrapper`:

        env = jaxatari.make(env_id)
        env = AtariWrapper(env, sticky_actions=0.0, episodic_life=not eval, ...)
        env = OCCAMWrapper(env, mask_mode="binary", frame_stack_size=4, frame_skip=4)
        env = LogWrapper(env)

    Output observation shape (drop-in for the CNN in ppo.py):
        object/binary/class : (frame_stack_size,                 84, 84, 1)
        planes              : (frame_stack_size * n_classes,      84, 84, 1)
    The trailing size-1 axis matches PixelObsWrapper, so ppo.py's `.squeeze()`
    turns it into (B, F*C, 84, 84) and the CNN handles F*C input channels.
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

        # base game env (AtariWrapper stores it in ._env); used for render + obs.
        self.base_env = env._env

        # native render resolution (e.g. 210 x 160).
        img_shape = self.base_env.image_space().shape  # (H, W, 3)
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])

        # ---- probe the object layout once (eager) to fix the static structure.
        probe_obs = self.base_env._get_observation(self.base_env.reset(jax.random.PRNGKey(0))[1])
        obj_groups = _extract_object_groups(probe_obs)
        self.num_classes = len(obj_groups)

        # ---- guard rails / informative diagnostics ----------------------------
        if self.num_classes == 0:
            raise NotImplementedError(
                f"OCCAM: game '{game_name}' exposes no ObjectObservation groups, so no "
                f"object bounding boxes are available to build masks from. This game "
                f"needs a small game-specific adapter (or upstream ObjectObservation "
                f"support in JAXtari) before OCCAM can be used. See the support table "
                f"printed by `python -m agents.occam.occam`."
            )

        # warn if the game also carries grid-structured objects in raw arrays
        # (e.g. Breakout `blocks`, Frostbite `ice_grid`): those are NOT masked.
        all_leaves = jax.tree_util.tree_leaves(
            probe_obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
        )
        grid_like = [
            l for l in all_leaves
            if (not isinstance(l, ObjectObservation)) and getattr(l, "ndim", 0) >= 2
        ]
        if grid_like:
            warnings.warn(
                f"OCCAM[{game_name}]: {len(grid_like)} grid-structured field(s) are "
                f"stored as raw arrays (not ObjectObservation) and will NOT appear in "
                f"the mask (e.g. Breakout bricks / Frostbite ice). Moving objects are "
                f"masked normally. Add a game-specific adapter for full fidelity.",
                stacklevel=2,
            )

        # palettes (numpy at init -> jnp constants at use time)
        self._gray_palette = jnp.asarray(_make_gray_palette(self.num_classes))      # (K+1,)
        self._color_palette = jnp.asarray(_make_color_palette(self.num_classes))    # (K+1, 3)

        # channels per single frame: 1 for object/binary/class, K for planes.
        self.per_frame_channels = self.num_classes if mask_mode == "planes" else 1

        total_channels = self.frame_stack_size * self.per_frame_channels
        if total_channels < 2:
            warnings.warn(
                f"OCCAM[{game_name}]: frame_stack_size * channels == {total_channels} < 2. "
                f"jaxtari-blines calls obs.squeeze(), which would collapse a size-1 "
                f"channel axis and break the CNN. Use frame_stack_size >= 2 (default 4).",
                stacklevel=2,
            )
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(total_channels, self.out_h, self.out_w, 1), dtype=jnp.uint8
        )

    # ----- spaces ----------------------------------------------------------- #
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    # ----- mask construction (called inside jit) ---------------------------- #
    def _mask_single(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """
        Build the OCCAM representation for a single (native-resolution) frame.
        Returns (C, out_h, out_w) uint8, with C == self.per_frame_channels.
        """
        groups = _extract_object_groups(obs)
        # per-group occupancy masks at native resolution (list of (H, W) bool)
        group_masks = []
        for g in groups:
            x, y, w, h, valid = _group_box_arrays(g, self.img_h, self.img_w)
            group_masks.append(_rasterize_group(x, y, w, h, valid, self.img_h, self.img_w))

        oh, ow = self.out_h, self.out_w

        if self.mask_mode == "object":
            # Keep real grayscale texture inside boxes, zero background, then
            # bilinear-resize EXACTLY like the pixel baseline's preprocess_image
            # (same grayscale weights + bilinear) so the two are comparable.
            gray = _rgb_to_gray(frame_rgb)                              # (H, W) float
            union = _union(group_masks)
            masked = jnp.where(union, gray, 0.0)[None]                 # (1, H, W)
            out = _resize(masked, (oh, ow), "bilinear")

        elif self.mask_mode == "binary":
            # 1 inside any box, 0 else. Downscale via linear + (>0) threshold so
            # that even sub-pixel objects (ball, shots) survive the 160->84 step
            # instead of being dropped by nearest-neighbour sampling.
            union = _union(group_masks).astype(jnp.float32)[None]      # (1, H, W)
            out = (_resize(union, (oh, ow), "linear") > 0.0).astype(jnp.float32) * 255.0

        elif self.mask_mode == "class":
            # Each box filled with a class-specific gray level. We resize the
            # per-class coverage with linear interpolation and take an argmax so
            # small objects survive; a tiny increasing per-class bias reproduces
            # the "later groups overwrite earlier ones" behaviour on overlaps.
            cov = jnp.stack(
                [_resize(gm.astype(jnp.float32)[None], (oh, ow), "linear")[0] for gm in group_masks],
                axis=0,
            )                                                           # (K, oh, ow)
            bias = (jnp.arange(self.num_classes, dtype=jnp.float32).reshape(-1, 1, 1) + 1.0) * 1e-3
            scored = jnp.where(cov > 0.0, cov + bias, 0.0)
            any_cov = (cov > 0.0).any(axis=0)                           # (oh, ow)
            cls = jnp.argmax(scored, axis=0) + 1                        # 1..K
            class_map = jnp.where(any_cov, cls, 0)                      # 0 == background
            out = self._gray_palette[class_map].astype(jnp.float32)[None]

        else:  # "planes": one binary plane per class, stacked on the channel axis
            planes = jnp.stack(
                [_resize(gm.astype(jnp.float32)[None], (oh, ow), "linear")[0] for gm in group_masks],
                axis=0,
            )                                                           # (K, oh, ow)
            out = (planes > 0.0).astype(jnp.float32) * 255.0

        return jnp.clip(out, 0.0, 255.0).astype(jnp.uint8)

    def _stack_to_obs(self, mask_stack: jnp.ndarray) -> jnp.ndarray:
        """(F, C, H, W) -> (F*C, H, W, 1) uint8 (drop-in for PixelObsWrapper)."""
        f, c, h, w = mask_stack.shape
        return mask_stack.reshape(f * c, h, w)[..., None]

    # ----- reset / step ----------------------------------------------------- #
    def _reset_internal(self, key):
        _, atari_state = self._env.reset(key)
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
        # advance `frame_skip` sub-steps, collecting env_states (like PixelObsWrapper)
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

        # render frame (with anti-flicker max-pooling, matching PixelObsWrapper)
        if self.max_pooling and self.frame_skip > 1:
            img = self.base_env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda z: z[-2], env_states)
            prev_img = self.base_env.render(prev_env_state)
            frame = jnp.maximum(img, prev_img)
        else:
            frame = self.base_env.render(last_env_state)

        # object geometry comes from the latest env_state's observation
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

        # autoreset (gym SAME_STEP): reset the whole stack on env_done/truncation
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


# --------------------------------------------------------------------------- #
#  visualization: side-by-side  game | mask  video frames                       #
# --------------------------------------------------------------------------- #
class _OCCAMViz:
    """
    Lightweight helper that turns a sequence of *base* env states into a
    side-by-side [clean game | OCCAM mask] video. Built fresh from a base env;
    not used during training, only for logging eval rollouts.
    """

    def __init__(self, base_env, mask_mode: str, out_native: bool = True):
        self.env = base_env
        self.mask_mode = mask_mode
        img_shape = base_env.image_space().shape
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])

        probe = base_env._get_observation(base_env.reset(jax.random.PRNGKey(0))[1])
        self.num_classes = len(_extract_object_groups(probe))
        self._gray_palette = jnp.asarray(_make_gray_palette(self.num_classes))
        self._color_palette = jnp.asarray(_make_color_palette(self.num_classes))

    def _mask_rgb(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Native-resolution RGB visualization (H, W, 3) uint8 of the mask."""
        groups = _extract_object_groups(obs)
        group_masks = [
            _rasterize_group(*_group_box_arrays(g, self.img_h, self.img_w), self.img_h, self.img_w)
            for g in groups
        ]
        if self.mask_mode == "binary":
            union = _union(group_masks)
            rgb = jnp.where(union[..., None], jnp.uint8(255), jnp.uint8(0))
            rgb = jnp.broadcast_to(rgb, (self.img_h, self.img_w, 3))

        elif self.mask_mode == "object":
            gray = _rgb_to_gray(frame_rgb)
            union = _union(group_masks)
            masked = jnp.where(union, gray, 0.0).astype(jnp.uint8)
            rgb = jnp.repeat(masked[..., None], 3, axis=-1)

        else:  # class / planes -> color-code categories (human-friendly)
            class_map = jnp.zeros((self.img_h, self.img_w), dtype=jnp.int32)
            for k, gm in enumerate(group_masks):
                class_map = jnp.where(gm, k + 1, class_map)
            rgb = self._color_palette[class_map]

        return rgb.astype(jnp.uint8)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _frame(self, env_state) -> jnp.ndarray:
        clean = self.env.render(env_state).astype(jnp.uint8)          # (H, W, 3)
        obs = self.env._get_observation(env_state)
        mask_rgb = self._mask_rgb(clean, obs)                          # (H, W, 3)
        return jnp.concatenate([clean, mask_rgb], axis=1)             # (H, 2W, 3)

    def frames(self, env_states) -> jnp.ndarray:
        """(T, ...) base env states -> (T, H, 2W, 3) uint8."""
        return jax.vmap(self._frame)(env_states)


def occam_comparison_frames(env_id: str, mask_mode: str, env_states, mods=None):
    """
    Build side-by-side [clean | mask] frames for an eval rollout.

    Returns a numpy array of shape (T, H, 2W, 3) uint8 (channel-LAST). Use
    `_to_chw(...)` to get the (T, 3, H, 2W) layout that `wandb.Video` expects.
    """
    import jaxatari  # local import to avoid a hard dependency at module import time

    base_env = jaxatari.make(env_id, mods=mods)
    viz = _OCCAMViz(base_env, mask_mode)
    return np.asarray(viz.frames(env_states), dtype=np.uint8)         # (T, H, 2W, 3)


def _to_chw(frames_thwc: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) -> (T, 3, H, W) contiguous, for wandb.Video."""
    return np.ascontiguousarray(np.transpose(frames_thwc, (0, 3, 1, 2)))


def _load_font(size: int):
    """Best-effort readable TrueType font; falls back to PIL's bitmap default."""
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


def _text_banner(width: int, text: str, height: int, font_size: int = 16) -> np.ndarray:
    """A black (height, width, 3) banner with centered-left white text, readable font."""
    banner = np.zeros((height, width, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(banner)
        d = ImageDraw.Draw(im)
        font = _load_font(font_size)
        # vertically center the text within the banner
        try:
            bbox = d.textbbox((0, 0), text, font=font)
            ty = max(0, (height - (bbox[3] - bbox[1])) // 2 - bbox[1])
        except Exception:
            ty = max(0, (height - font_size) // 2)
        d.text((6, ty), text, fill=(255, 255, 255), font=font)
        banner = np.asarray(im)
    except Exception:
        pass
    return banner


def _caption_clip(frames_thwc: np.ndarray, text: str, banner_h: int = 16) -> np.ndarray:
    """Prepend a black caption banner with `text` on top of every frame.
    Uses PIL if available; otherwise the banner stays blank (still aligns sizes)."""
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
    """Write a shareable video file. Tries mp4 (imageio/ffmpeg), falls back to gif.
    Returns the actual path written, or None if no writer is available."""
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
    """Persist one variant's eval clip so the final summary can pick it up later.
    Writes `<save_dir>/eval_<mod_label>.npy` (overwritten each call -> holds the
    LAST step).

    By default it writes ONLY the .npy (pure numpy, no subprocess), so this is
    safe to call from inside the multithreaded JAX training process: no
    `os.fork()` / ffmpeg, hence no fork-deadlock warning. The shareable mp4 is
    built once at the very end by `build_occam_summary_video`. Pass
    `write_mp4=True` only if you explicitly want a per-variant clip during
    training (and accept the fork warning)."""
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
):
    """
    Render a side-by-side [game | mask] clip for one eval rollout, log it to W&B
    (per-variant preview) and/or persist it for the final summary.

    Fork-safety note: the only step that can spawn an `os.fork()`/ffmpeg
    subprocess is the W&B mp4 encoding (`log_wandb=True`). Because the ppo patch
    only calls this on the LAST step(s) (gated by `VIDEO_EVERY`), that encode
    happens once at the very end of training, when JAX is no longer actively
    computing -> safe in practice. The `.npy` written via `save_dir` is pure
    numpy (no subprocess) and is what `build_occam_summary_video` reads later.

    - W&B key:  `eval/<env_id>/<mask_mode>/<mod_label>`  (game name included).
    - If `save_dir` is given, also writes `<save_dir>/eval_<mod_label>.npy`.
    - Set `log_wandb=False` to skip W&B entirely (then no fork at all).

    Returns the captioned numpy frames (T, H, 2W+banner, 3) uint8.
    """
    frames = occam_comparison_frames(env_id, mask_mode, env_states, mods=mods)   # (T,H,2W,3) RAW
    if save_dir is not None:
        # save the RAW (un-captioned) clip; the summary adds its own single, clean
        # caption later, so nothing is labelled twice.
        save_eval_frames(save_dir, mod_label, frames, fps=fps)
    captioned = _caption_clip(frames, f"{env_id} | {mask_mode} | {mod_label} | step {step}")
    if log_wandb:
        import wandb
        key = f"eval/{env_id}/{mask_mode}/{mod_label}"
        (wandb_run or wandb).log({key: wandb.Video(_to_chw(captioned), fps=fps, format="mp4")}, step=step)
    return captioned


def _pad_to_len(clip: np.ndarray, T: int) -> np.ndarray:
    """Repeat the last frame so `clip` reaches length T (for time-aligned tiling)."""
    if clip.shape[0] >= T:
        return clip[:T]
    pad = np.broadcast_to(clip[-1:], (T - clip.shape[0],) + clip.shape[1:])
    return np.concatenate([clip, pad], axis=0)


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
    hold_last_seconds: float = 2.0,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags=None,
    wandb_run_name: str | None = None,
):
    """
    Stitch the LAST saved eval clips of all variants into ONE shareable video,
    showing everything at once.

    Layout: columns = the 4 variants (object, binary, class, planes),
            rows    = the eval mods (`default` on top, `lazy_enemy` below).
    Each cell is the `[game | mask]` clip of that (variant, mod) with ONE readable
    caption. Missing combinations become a labelled placeholder. Shorter clips
    hold their last frame so everything stays time-synced; the output length
    equals the longest eval clip, and the final frame is then HELD for
    `hold_last_seconds` so every result is clearly visible at the end (instead of
    appearing to snap back to the start).

    Memory: clips are memory-mapped and the grid is streamed FRAME-BY-FRAME to the
    encoder, so peak RAM is ~one output frame regardless of episode length.

    - `strip_top_px`: crop this many rows off the top of each saved clip before
      tiling. Use it only for OLD clips that were saved WITH a baked-in caption
      (set 16); clips saved by the current code are raw, so leave it 0.
    - W&B upload: if `wandb_project` is given, the finished video is uploaded as
      its OWN new W&B run (named `wandb_run_name`, tagged `wandb_tags`,
      job_type="summary"), so repeated runs are preserved and comparable. This
      references the already-written mp4 file (no re-encode, no fork).

    Writes `<save_root>/<env_id>/<out_name or summary_<env_id>>.mp4` (falls back to
    .gif if no ffmpeg is available). Returns `(path_written_or_None, num_frames)`.
    """
    import os
    import glob

    grid_order = list(mask_modes)[:4]
    while len(grid_order) < 4:
        grid_order.append(None)

    # auto-discover the eval mod-labels from the saved clips if not given
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
        # memory-map: frames are pulled from disk one at a time during streaming,
        # so a multi-GB clip is never fully resident in RAM.
        return np.load(p, mmap_mode="r")

    mods = sorted(mods, key=lambda m: (m != "default", m))   # "default" first

    # open every (mod, variant) clip as a memmap; remember the longest length
    clips = {}
    cell_h = cell_w = None
    max_len = 0
    for mod in mods:
        for mm in grid_order:
            c = open_clip(mm, mod)
            clips[(mod, mm)] = c
            if c is not None:
                cell_h, cell_w = c.shape[1], c.shape[2]
                max_len = max(max_len, c.shape[0])
    if cell_h is None:
        return None, None

    # output length: full by default; `max_frames` (if set) strides it down
    if max_frames and max_len > max_frames:
        out_idx = np.linspace(0, max_len - 1, max_frames).astype(np.int64)
    else:
        out_idx = np.arange(max_len, dtype=np.int64)
    T = len(out_idx)

    # pre-render the static banners ONCE (text is constant per cell)
    cap_h, title_h = 22, 30
    cell_banner = {
        (mod, mm): _text_banner(
            cell_w,
            f"{(mm or '-').upper()}  -  {mod}" + ("   (n/a)" if clips[(mod, mm)] is None else ""),
            cap_h, font_size=15,
        )
        for mod in mods for mm in grid_order
    }
    eff_h = cell_h - strip_top_px
    black_cell = np.zeros((eff_h, cell_w, 3), np.uint8)
    title = _text_banner(
        cell_w * 4,
        f"{env_id}  -  OCCAM mask comparison    |    columns: object / binary / class / planes"
        f"    |    rows: {' / '.join(mods)}",
        title_h, font_size=20,
    )

    def grid_frame(t_src):
        row_imgs = []
        for mod in mods:
            cells = []
            for mm in grid_order:
                c = clips[(mod, mm)]
                if c is None:
                    frame = black_cell
                else:
                    frame = np.asarray(c[min(t_src, c.shape[0] - 1)])[strip_top_px:]   # one frame
                cells.append(np.concatenate([cell_banner[(mod, mm)], frame], axis=0))
            row_imgs.append(np.concatenate(cells, axis=1))
        grid = np.concatenate(row_imgs, axis=0)
        return np.concatenate([title, grid], axis=0)

    # stream the frames straight to the encoder -> peak RAM is ~one output frame.
    # After the last real frame, HOLD it for `hold_last_seconds` so the end state
    # of every cell stays on screen instead of appearing to jump back to start.
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

    # upload as its OWN tagged W&B run (preserved + comparable across re-runs)
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


# --------------------------------------------------------------------------- #
#  utility: per-game OCCAM support report (run:  python -m agents.occam.occam)  #
# --------------------------------------------------------------------------- #
JAXTARI_15 = [
    "asteroids", "beamrider", "breakout", "enduro", "freeway", "frostbite",
    "gravitar", "kangaroo", "montezumarevenge", "mspacman", "phoenix", "pong",
    "seaquest", "skiing", "tennis",
]


def probe_game(env_id: str):
    """
    Probe a single game and classify its OCCAM support:
        "full"    : >=1 ObjectObservation group, no raw grid objects
        "partial" : >=1 ObjectObservation group + raw grid object(s) (ignored)
        "none"    : no ObjectObservation groups (needs an adapter)
    Returns (status, num_object_groups, num_grid_fields, error_or_None).
    """
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


def print_support_table(games=None):  # pragma: no cover - convenience CLI
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