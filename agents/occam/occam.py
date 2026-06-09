"""
OCCAM: Object-Centric Attention via Masking  --  JAX / JAXtari baseline
=======================================================================

A single-file, JIT-compatible implementation of OCCAM as an observation
wrapper for JAXtari (https://github.com/k4ntz/JAXAtari), intended as a
neuro-symbolic baseline for https://github.com/remunds/jaxtari-blines.

Original method:
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


def _resize(img: jnp.ndarray, out_hw: Tuple[int, int], method: str) -> jnp.ndarray:
    """Resize the last two spatial axes of `img` to `out_hw`."""
    target = tuple(img.shape[:-2]) + tuple(out_hw)
    return jax.image.resize(img.astype(jnp.float32), target, method=method)


def _make_gray_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1,) uint8. Index 0 = background (0), rest are distinct grays."""
    if n_classes <= 1:
        levels = [255]
    else:
        levels = list(np.linspace(90, 255, n_classes).round().astype(int))
    return np.array([0] + levels, dtype=np.uint8)


def _make_color_palette(n_classes: int) -> np.ndarray:
    """(n_classes + 1, 3) uint8 RGB palette for visualization. Index 0 = black."""
    cols = [(0, 0, 0)]
    for i in range(max(n_classes, 1)):
        h = i / max(n_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)


def _extract_object_groups(obs: Any) -> List[ObjectObservation]:
    """ObjectObservation leaves from obs PyTree, in deterministic PyTree order."""
    leaves = jax.tree_util.tree_leaves(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    return [leaf for leaf in leaves if isinstance(leaf, ObjectObservation)]


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
    """Boolean occupancy mask (img_h, img_w): True wherever any valid box covers the pixel."""
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


@struct.dataclass
class OCCAMState:
    # must be named `atari_state` so jaxtari-blines eval code can walk the state tree
    atari_state: Any
    mask_stack: jnp.ndarray  # (F, C, H, W) uint8


class OCCAMWrapper(JaxatariWrapper):
    """
    Object-Centric Attention via Masking wrapper. Apply after AtariWrapper, in place of PixelObsWrapper.
    Output shape: (frame_stack_size, 84, 84, 1) for object/binary/class; (frame_stack_size * n_classes, 84, 84, 1) for planes.
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

        self.per_frame_channels = self.num_classes if mask_mode == "planes" else 1
        total_channels = self.frame_stack_size * self.per_frame_channels
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(total_channels, self.out_h, self.out_w, 1), dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _mask_single(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Build the OCCAM mask for one native-resolution frame. Returns (C, out_h, out_w) uint8."""
        groups = _extract_object_groups(obs)
        group_masks = []  # (H, W) bool per group
        for g in groups:
            x, y, w, h, valid = _group_box_arrays(g, self.img_h, self.img_w)
            group_masks.append(_rasterize_group(x, y, w, h, valid, self.img_h, self.img_w))

        oh, ow = self.out_h, self.out_w

        if self.mask_mode == "object":
            # real pixels inside boxes, zero background, bilinear-resize like PixelObsWrapper
            gray = _rgb_to_gray(frame_rgb)                              # (H, W) float
            union = _union(group_masks)
            masked = jnp.where(union, gray, 0.0)[None]                 # (1, H, W)
            out = _resize(masked, (oh, ow), "bilinear")

        elif self.mask_mode == "binary":
            # linear downsample + threshold so sub-pixel objects survive 160->84
            union = _union(group_masks).astype(jnp.float32)[None]      # (1, H, W)
            out = (_resize(union, (oh, ow), "linear") > 0.0).astype(jnp.float32) * 255.0

        elif self.mask_mode == "class":
            # argmax over per-class coverage; tiny bias breaks ties toward later groups
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

        else:  # "planes": one binary plane per class
            planes = jnp.stack(
                [_resize(gm.astype(jnp.float32)[None], (oh, ow), "linear")[0] for gm in group_masks],
                axis=0,
            )                                                           # (K, oh, ow)
            out = (planes > 0.0).astype(jnp.float32) * 255.0

        return jnp.clip(out, 0.0, 255.0).astype(jnp.uint8)

    def _stack_to_obs(self, mask_stack: jnp.ndarray) -> jnp.ndarray:
        """(F, C, H, W) -> (F*C, H, W, 1) uint8."""
        f, c, h, w = mask_stack.shape
        return mask_stack.reshape(f * c, h, w)[..., None]

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

    def __init__(self, base_env, mask_mode: str):
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

        else:  # class / planes -> color-code categories
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
    """Side-by-side [game | mask] frames (T, H, 2W, 3) uint8 for one eval rollout."""
    import jaxatari  # local import to avoid a hard dependency at module import time

    base_env = jaxatari.make(env_id, mods=mods)
    viz = _OCCAMViz(base_env, mask_mode)
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
):
    """Render, optionally save (.npy), caption and W&B-log a [game | mask] eval clip."""
    frames = occam_comparison_frames(env_id, mask_mode, env_states, mods=mods)   # (T,H,2W,3) RAW
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
            cell_w,
            f"{(mm or '-').upper()}  -  {mod}" + ("   (n/a)" if clips[(mod, mm)] is None else ""),
            cap_h, font_size=16, bg=BG, fg=FG,
        )
        for mod in mods for mm in grid_order
    }
    eff_h = cell_h - strip_top_px
    black_cell = np.zeros((eff_h, cell_w, 3), np.uint8)
    title = _text_banner(cell_w * 4 + col_gap * 3,
                         f"{env_id}  -  OCCAM mask comparison", title_h, font_size=24, bg=BG, fg=FG)

    full_cell_h = cap_h + eff_h
    h_spacer = np.full((full_cell_h, col_gap, 3), 255, np.uint8)
    row_w = cell_w * 4 + col_gap * 3
    v_spacer = np.full((row_gap, row_w, 3), 255, np.uint8)

    def _decorate(frame):
        """White border + game|mask divider line."""
        f = np.array(frame, dtype=np.uint8)
        h, w = f.shape[:2]
        mid = w // 2
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
                    frame = black_cell
                else:
                    frame = np.asarray(c[min(t_src, c.shape[0] - 1)])[strip_top_px:]
                cell = np.concatenate([cell_banner[(mod, mm)], _decorate(frame)], axis=0)
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
