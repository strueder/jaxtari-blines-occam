"""Deterministic King Kong sweep across all 4 OCCAM mask modes side by side.

Forces a set of representative game states (every floor, every ladder, all player
animation states, Kong at both anchors and at both off-screen teleport targets,
bombs on every floor, death/success stages) and renders
[clean | clean+boxes | object | binary | class | planes] in one row -- no trained
policy needed. Optionally appends frames from a real random rollout.

    uv run python verify_kingkong.py
    uv run python verify_kingkong.py --rollout 4000 --png
"""
import argparse
import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v2 as imageio

import jaxatari
from jaxatari.wrappers import AtariWrapper
from agents.occam.occam import (
    OCCAMWrapper, _OCCAMViz, _load_font, _make_color_palette,
    _planes_isometric_rgb, _area_weights, _sheets_width,
)

OUT_H, OUT_W = 84, 84

ENV_ID = "kingkong"
MODES = ["object", "binary", "class", "planes"]

# One palette for everything: box outlines, legend, class map and planes all use
# OCCAM's own class colours (_make_color_palette), so colour == class everywhere.
# Index 0 is background, so group i gets palette[i + 1]. Verified: _obs_groups()
# and occam._extract_object_groups() enumerate the groups in the same order.
REQUIRED = ("x", "y", "width", "height", "active")


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

TITLE_H = 22      # row title banner
HEAD_H = 16       # per-panel name banner
LEG_LINE_H = 15   # one legend line


def _strip(width, height, text, fill=(255, 255, 255), font_size=13):
    """Black banner of the given size with left-aligned text."""
    band = np.zeros((height, width, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(band)
        d = ImageDraw.Draw(im)
        d.text((4, max(0, (height - font_size) // 2 - 1)), text,
               fill=fill, font=_load_font(font_size))
        band = np.asarray(im)
    except Exception:
        pass
    return band


def _legend_strip(width, entries, font_size=13):
    """Colour key for the obs-box groups: [(name, n_active, rgb)] -> black banner.

    Uses a fixed slot grid derived only from the *number* of groups (constant for
    a game), so every frame of the video gets the exact same banner height.
    """
    n = max(1, len(entries))
    rows = 1 if n <= 6 else (n + 5) // 6   # height depends on the group *count* only
    band = np.zeros((rows * LEG_LINE_H + 4, width, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return band
    im = Image.fromarray(band)
    d = ImageDraw.Draw(im)
    font = _load_font(font_size)

    prefix = "obs boxes:"
    d.text((5, 3), prefix, fill=(150, 150, 150), font=font)
    x = 5 + int(d.textlength(prefix, font=font)) + 10
    y = 2
    per_row = (n + rows - 1) // rows
    for i, (name, n_act, col) in enumerate(entries):
        if i and i % per_row == 0:          # next line
            x, y = 5, y + LEG_LINE_H
        d.rectangle([x, y + 4, x + 7, y + 11], fill=col)
        text = f"{name} ({n_act})"
        d.text((x + 13, y + 1), text, fill=col, font=font)
        x += 13 + int(d.textlength(text, font=font)) + 16
    return np.asarray(im)


def _obs_groups(obs):
    """name -> dict of (n,) numpy arrays, for every ObjectObservation field."""
    out = {}
    names = ([f.name for f in dataclasses.fields(obs)] if dataclasses.is_dataclass(obs)
             else list(getattr(obs, "_fields", [])))
    for name in names:
        v = getattr(obs, name, None)
        if v is None or not all(hasattr(v, a) for a in REQUIRED):
            continue
        x = np.atleast_1d(np.asarray(v.x))
        n = x.shape[0]
        g = {"n": n}
        for a in REQUIRED:
            arr = np.atleast_1d(np.asarray(getattr(v, a)))
            g[a] = np.repeat(arr, n) if arr.shape[0] == 1 and n > 1 else arr
        out[name] = g
    return out


def _draw_boxes(frame, obs):
    """Overlay every active box on the clean frame, one colour per group.

    Returns (frame, [(group_name, n_active, rgb)]) -- the counts feed the legend
    banner above the row instead of being painted into the frame.
    """
    entries = []
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame, entries
    im = Image.fromarray(np.ascontiguousarray(frame))
    d = ImageDraw.Draw(im)
    H, W = frame.shape[0], frame.shape[1]
    groups = _obs_groups(obs)
    palette = _make_color_palette(len(groups))     # idx 0 = background
    for gi, (gname, g) in enumerate(groups.items()):
        col = tuple(int(c) for c in palette[gi + 1])
        n_act = 0
        for i in range(g["n"]):
            if int(g["active"][i]) == 0:
                continue
            x, y = int(g["x"][i]), int(g["y"][i])
            w, h = int(g["width"][i]), int(g["height"][i])
            if w <= 0 or h <= 0:
                continue
            n_act += 1
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W - 1, x + w - 1), min(H - 1, y + h - 1)
            if x1 < x0 or y1 < y0:
                continue  # box lies completely outside the frame
            d.rectangle([x0, y0, x1, y1], outline=col)
        entries.append((gname, n_act, col))
    return np.asarray(im), entries


# --------------------------------------------------------------------------
# real OCCAM masks (the ones the agent is actually trained on)
# --------------------------------------------------------------------------

def _downscale(img, out_h, out_w):
    """Area-downscale (h, w, 3) -> (out_h, out_w, 3), same operator as the wrapper."""
    wy = _area_weights(img.shape[0], out_h)
    wx = _area_weights(img.shape[1], out_w).T
    out = np.einsum("ij,jkc,kl->ilc", wy, img.astype(np.float64), wx)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _occam_rgb(wrapper, mask):
    """OCCAMWrapper._mask_single output (C, h, w) uint8 -> displayable (h, w, 3)."""
    mask = np.asarray(mask, dtype=np.uint8)
    pal = np.asarray(wrapper._color_palette).astype(np.int32)   # (K+1, 3), idx 0 = bg

    if wrapper.mask_mode == "planes":
        rgb = np.zeros(mask.shape[1:] + (3,), np.int32)
        for k in getattr(wrapper, "render_order", range(mask.shape[0])):
            rgb = np.where((mask[k] > 0)[..., None], pal[k + 1], rgb)

    elif wrapper.mask_mode == "class":
        # gray levels are class ids ((i+1) * 255//K), but area downscaling blends them
        # at object edges, so snap to the nearest palette level instead of an exact
        # lookup and dim the colour by how far the pixel fell short of that level.
        gray = np.asarray(wrapper._gray_palette).astype(np.int32)     # (K+1,)
        idx = np.abs(mask[0][..., None].astype(np.int32) - gray[None, None, :]).argmin(-1)
        level = np.maximum(gray[idx], 1)
        frac = np.clip(mask[0].astype(np.float32) / level, 0.0, 1.0)[..., None]
        rgb = (pal[idx] * frac).round().astype(np.int32)

    else:
        rgb = np.repeat(mask[0][..., None], 3, axis=-1)
    return rgb.astype(np.uint8)


def _row(env, maskers, state, title, obs_res=False):
    """One situation as [title banner / legend banner / named panels]."""
    clean = np.asarray(env.render(state), dtype=np.uint8)
    obs = env._get_observation(state)
    H, W = clean.shape[0], clean.shape[1]

    boxed, entries = _draw_boxes(clean, obs)
    if obs_res:
        clean, boxed = _downscale(clean, OUT_H, OUT_W), _downscale(boxed, OUT_H, OUT_W)
        H, W = OUT_H, OUT_W
    panels = [("clean", clean), ("obs boxes", boxed)]
    raw_frame = jnp.asarray(np.asarray(env.render(state), dtype=np.uint8))
    for m in MODES:
        if obs_res:
            raw = maskers[m]._mask_single(raw_frame, obs)
            panels.append((m, _occam_rgb(maskers[m], raw)))
        else:
            mask = np.asarray(maskers[m]._mask_rgb(raw_frame, obs), dtype=np.uint8)
            panels.append((m, mask))

    planes = np.asarray(maskers["planes"]._mask_single(raw_frame, obs)) > 0 if obs_res \
        else np.asarray(maskers["planes"].planes_rgb(obs))
    iso = _planes_isometric_rgb(
        jnp.asarray(planes), jnp.asarray(_make_color_palette(planes.shape[0])),
        H, _sheets_width(W, planes.shape[0]), aspect=W / H,
    )
    panels.append(("planes (sheets)", np.asarray(iso, np.uint8)))

    # panel name above each panel, never inside it
    # header width follows each panel -- the sheet view is wider than the rest
    cols = [np.concatenate([_strip(img.shape[1], HEAD_H, name), img], axis=0)
            for name, img in panels]
    grid = np.concatenate(cols, axis=1)
    total_w = grid.shape[1]
    row = np.concatenate([
        _strip(total_w, TITLE_H,
               f"{title}   -   {W}x{H} "
               + ("agent input" if obs_res else "native viz"), font_size=16),
        _legend_strip(total_w, entries),
        grid,
    ], axis=0)
    # h264/yuv420p needs even dimensions; banners can make the height odd
    return np.pad(row, [(0, row.shape[0] % 2), (0, row.shape[1] % 2), (0, 0)])


# --------------------------------------------------------------------------
# state surgery
# --------------------------------------------------------------------------

def rep(state, **kw):
    """state.replace() that keeps the original dtype of every field."""
    fields = {f.name for f in dataclasses.fields(state)}
    unknown = set(kw) - fields
    if unknown:
        raise KeyError(f"unknown state fields: {sorted(unknown)} "
                       f"(available: {sorted(fields)})")
    new = {}
    for k, v in kw.items():
        old = getattr(state, k)
        new[k] = jnp.asarray(v, dtype=jnp.asarray(old).dtype)
    return state.replace(**new)


def build_poses(env, base):
    """Yield (title, state) for every situation worth looking at."""
    c = env.consts
    W, H = int(c.WIDTH), int(c.HEIGHT)
    floors = np.asarray(c.FLOOR_LOCATIONS).tolist()
    ladders = np.asarray(c.LADDER_LOCATIONS).tolist()
    nb = int(c.MAX_BOMBS)
    zeros = np.zeros(nb, dtype=np.int32)

    gp = rep(base,
             gamestate=c.GAMESTATE_GAMEPLAY, stage_steps=0,
             kong_visible=1, princess_visible=1,
             kong_x=int(c.KONG_UPPER_LOCATION[0]), kong_y=int(c.KONG_UPPER_LOCATION[1]),
             princess_x=int(c.PRINCESS_START_LOCATION[0]),
             princess_y=int(c.PRINCESS_START_LOCATION[1]),
             bomb_active=zeros, death_type=c.DEATH_TYPE_NONE)

    # --- stages -----------------------------------------------------------
    for name, gs, steps in [("IDLE", c.GAMESTATE_IDLE, 60),
                            ("STARTUP", c.GAMESTATE_STARTUP, 240),
                            ("RESPAWN", c.GAMESTATE_RESPAWN, 90)]:
        yield f"stage {name}", rep(gp, gamestate=gs, stage_steps=steps)

    # --- player on every floor -------------------------------------------
    for fi, fy in enumerate(floors[:9]):
        yield (f"player floor {fi} (y={fy})",
               rep(gp, player_x=77, player_y=int(fy),
                   player_state=c.PLAYER_IDLE_RIGHT, player_floor=fi))

    # --- player animation states -----------------------------------------
    for label, st in [("idle-left", c.PLAYER_IDLE_LEFT),
                      ("idle-right", c.PLAYER_IDLE_RIGHT),
                      ("move-left", c.PLAYER_MOVE_LEFT),
                      ("move-right", c.PLAYER_MOVE_RIGHT),
                      ("jump-left", c.PLAYER_JUMP_LEFT),
                      ("jump-right", c.PLAYER_JUMP_RIGHT),
                      ("fall", c.PLAYER_FALL),
                      ("dead", c.PLAYER_DEAD)]:
        yield (f"player {label}",
               rep(gp, player_x=77, player_y=int(floors[0]), player_state=st))

    # --- player on ladders ------------------------------------------------
    for li in (0, 1, 4, 10):
        if li >= len(ladders):
            continue
        x1, y1, x2, y2 = ladders[li]
        yield (f"player on ladder {li}",
               rep(gp, player_x=int(x1) + 1, player_y=int((y1 + y2) // 2),
                   player_state=c.PLAYER_CLIMB_UP))

    # --- kong -------------------------------------------------------------
    for label, loc in [("upper", c.KONG_UPPER_LOCATION),
                       ("start/lower", c.KONG_START_LOCATION),
                       ("teleport upper (y=12)", c.KONG_UPPER_TELEPORT_LOCATION),
                       ("teleport lower (y=276)", c.KONG_LOWER_TELEPORT_LOCATION)]:
        yield (f"kong {label}",
               rep(gp, kong_x=int(loc[0]), kong_y=int(loc[1]), kong_visible=1))

    # --- bombs ------------------------------------------------------------
    bx = np.array([20 + 14 * i for i in range(nb)], dtype=np.int32)
    by = np.array([floors[i % 9] for i in range(nb)], dtype=np.int32)
    yield ("bombs: one per floor (normal)",
           rep(gp, bomb_positions_x=bx, bomb_positions_y=by,
               bomb_active=np.ones(nb, dtype=np.int32),
               bomb_is_magic=zeros, bomb_directions_y=np.ones(nb, dtype=np.int32)))
    yield ("bombs: one per floor (magic)",
           rep(gp, bomb_positions_x=bx, bomb_positions_y=by,
               bomb_active=np.ones(nb, dtype=np.int32),
               bomb_is_magic=np.ones(nb, dtype=np.int32),
               bomb_directions_y=np.ones(nb, dtype=np.int32)))
    yield ("bombs: inactive but stale positions",
           rep(gp, bomb_positions_x=bx, bomb_positions_y=by, bomb_active=zeros))

    # --- death / success --------------------------------------------------
    yield ("DEATH fall, player off-screen (y=H+50)",
           rep(gp, gamestate=c.GAMESTATE_DEATH, death_type=c.DEATH_TYPE_FALL,
               stage_steps=100, player_x=77, player_y=H + 50,
               player_state=c.PLAYER_FALL))
    yield ("DEATH bomb explode (flash)",
           rep(gp, gamestate=c.GAMESTATE_DEATH, death_type=c.DEATH_TYPE_BOMB_EXPLODE,
               stage_steps=20, player_x=77, player_y=int(floors[2]),
               player_state=c.PLAYER_DEAD))
    yield ("SUCCESS",
           rep(gp, gamestate=c.GAMESTATE_SUCCESS, stage_steps=40,
               player_x=int(c.PLAYER_SUCCESS_LOCATION[0]),
               player_y=int(c.PLAYER_SUCCESS_LOCATION[1]),
               princess_x=int(c.PRINCESS_SUCCESS_LOCATION[0]),
               princess_y=int(c.PRINCESS_SUCCESS_LOCATION[1])))


def rollout_states(env, key, steps, every):
    """Yield (title, state) from a random rollout: on stage change and every N steps."""
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    aspace = env.action_space()
    prev = None
    for t in range(steps):
        key, ak = jax.random.split(key)
        obs, state, r, done, info = step(state, aspace.sample(ak))
        gs = int(np.asarray(state.gamestate))
        if gs != prev:
            yield f"rollout t={t} stage={gs}", state
            prev = gs
        elif t % every == 0:
            yield f"rollout t={t} stage={gs}", state
        if bool(np.asarray(done)):
            key, rk = jax.random.split(key)
            obs, state = env.reset(rk)
            prev = None


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", type=int, default=0,
                   help="also sweep N steps of a random rollout (0 = poses only)")
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--hold", type=int, default=45, help="video frames per situation")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true", help="also write one PNG per situation")
    p.add_argument("--obs-res", action="store_true",
                   help=f"show the real OCCAMWrapper masks at {OUT_H}x{OUT_W} instead of "
                        "native-resolution visualizations; every panel shrinks")
    p.add_argument("--out", default=f"./models/{ENV_ID}/kingkong_sweep_all_modes.mp4")
    args = p.parse_args()

    env = jaxatari.make(ENV_ID)
    if args.obs_res:
        maskers = {m: OCCAMWrapper(AtariWrapper(jaxatari.make(ENV_ID)), mask_mode=m,
                                   game_name=ENV_ID) for m in MODES}
    else:
        maskers = {m: _OCCAMViz(env, m) for m in MODES}
    _, base_state = env.reset(jax.random.PRNGKey(args.seed))

    situations = list(build_poses(env, base_state))
    if args.rollout:
        situations += list(rollout_states(env, jax.random.PRNGKey(args.seed + 1),
                                          args.rollout, args.every))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    png_dir = os.path.splitext(args.out)[0] + "_png"
    if args.png:
        os.makedirs(png_dir, exist_ok=True)

    frames = []
    for i, (title, state) in enumerate(situations):
        row = _row(env, maskers, state, f"[{i:02d}] {title}", obs_res=args.obs_res)
        frames.extend([row] * args.hold)
        if args.png:
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)
            imageio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe}.png"), row)
        print(f"[{i:02d}] {title}")

    imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {args.out}  ({len(situations)} situations, {len(frames)} frames)")
    src = f"OCCAMWrapper._mask_single (real {OUT_H}x{OUT_W} training masks)" if args.obs_res \
          else "_OCCAMViz._mask_rgb (native res)"
    print(f"panels per row: clean | obs boxes | {' | '.join(MODES)} | planes sheets"
          f"   [mask source: {src}]")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()