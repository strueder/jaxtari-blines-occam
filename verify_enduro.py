import argparse
import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from jaxatari.wrappers import AtariWrapper
from agents.occam.occam import OCCAMWrapper, _OCCAMViz
from verify_kingkong import MODES, OUT_H, OUT_W, _row, _obs_groups

ENV_ID = "enduro"
N_SLOTS = 7


def put(state, **kw):
    fields = {f.name for f in dataclasses.fields(state)}
    unknown = set(kw) - fields
    if unknown:
        raise KeyError(f"unknown state fields: {sorted(unknown)}")
    new = {k: jnp.asarray(v, dtype=jnp.asarray(getattr(state, k)).dtype) for k, v in kw.items()}
    return state.replace(**new)


def lane_x(env, state, slot, lane, y):
    c = env.consts
    off = int(np.clip(np.floor(float(state.track_top_x_curve_offset)), -c.curve_offset_base, c.curve_offset_base))
    idx = off + c.curve_offset_base
    row = int(np.clip(y - c.sky_height, 0, c.track_height - 1))
    left = int(np.asarray(c.precomputed_left_curves)[idx, row]) + int(float(state.track_top_x))
    right = int(np.asarray(c.precomputed_right_curves)[idx, row]) + int(float(state.track_top_x))
    center = left + (right - left) * float(np.asarray(c.lane_ratios)[lane])
    return int(center - int(np.asarray(c.car_widths)[slot]) // 2)


def opponents(env, state, slots, lanes=None, ys=None):
    base_ys = np.asarray(env.consts.opponent_slot_ys).astype(int)
    pos = np.full((N_SLOTS, 3), -1, dtype=np.int32)
    pos[:, 2] = np.arange(N_SLOTS) + 1
    for i in slots:
        y = int(base_ys[i]) if ys is None or i not in ys else int(ys[i])
        lane = 1 if lanes is None else lanes.get(i, 1)
        pos[i, 0] = lane_x(env, state, i, lane, y)
        pos[i, 1] = y
    return pos


def build_poses(env, base):
    c = env.consts
    all_slots = list(range(N_SLOTS))
    lanes = {i: i % 3 for i in all_slots}
    gw = int(c.game_window_height)
    cutoff = gw - int(env.renderer.opponent_bottom_cutoff)

    yield "reset [player 1, enemies = spawn state]", base

    day = put(base, weather_index=int(c.day_weather_index))
    yield ("day, all slots [player 1, enemies 7/7]",
           put(day, visible_opponent_positions=opponents(env, day, all_slots, lanes)))

    yield ("no opponents (x=-1) [player 1, enemies 0/7]",
           put(day, visible_opponent_positions=opponents(env, day, [])))

    for y in (gw - 10, gw - 6, gw - 3, cutoff):
        exp = 0 if y >= cutoff else 1
        h = max(0, min(int(np.asarray(c.car_heights)[0]), gw - 1 - y))
        yield (f"slot0 y={y} at bottom bar [enemies slot0 {exp}, h={h if exp else '-'}]",
               put(day, visible_opponent_positions=opponents(env, day, [0], lanes={0: 0}, ys={0: y})))

    night = put(base, weather_index=int(c.night_weather_index))
    yield ("night sprites [player 1, enemies 7/7]",
           put(night, visible_opponent_positions=opponents(env, night, all_slots, lanes)))

    fog = put(base, weather_index=int(c.fog_weather_index))
    base_ys = np.asarray(c.opponent_slot_ys).astype(int)
    n_vis = int((base_ys >= int(c.fog_height)).sum())
    yield (f"fog, all slots [player 1, enemies {n_vis}/7, only y>={int(c.fog_height)}]",
           put(fog, visible_opponent_positions=opponents(env, fog, all_slots, lanes)))

    snow = put(base, weather_index=int(c.snow_weather_index))
    yield ("snow [player 1, enemies 7/7]",
           put(snow, visible_opponent_positions=opponents(env, snow, all_slots, lanes)))

    fast = put(day, player_speed=float(c.max_speed), player_y=float(c.player_y_start) - float(c.max_speed))
    yield ("max speed, player raised [player 1, enemies 3/7]",
           put(fast, visible_opponent_positions=opponents(env, fast, [1, 2, 3], lanes)))

    for label, curve in (("left", -float(c.track_max_top_x_offset)), ("right", float(c.track_max_top_x_offset))):
        curved = put(day, track_top_x_curve_offset=curve)
        yield (f"full {label} curve [player 1, enemies 7/7]",
               put(curved, visible_opponent_positions=opponents(env, curved, all_slots, lanes)))

    for label, px in (("left edge", 58.0), ("right edge", 94.0)):
        edge = put(day, player_x=px, track_top_x=84.0 + float(c.player_x_start) - px)
        yield (f"player at road {label} [player 1, enemies 7/7]",
               put(edge, visible_opponent_positions=opponents(env, edge, all_slots, lanes)))

    yield ("collision mode [player 1, enemies 2/7]",
           put(day, collision_mode=True, collision_steps=int(c.collision_duration),
               visible_opponent_positions=opponents(env, day, [0, 1], ys={0: int(c.player_y_start) - 2})))

    yield "game over [player 1]", put(day, game_over=True)


def rollout_states(env, key, steps, every):
    _, state = env.reset(key)
    step = jax.jit(env.step)
    aspace = env.action_space()
    for t in range(steps):
        key, ak = jax.random.split(key)
        _, state, _, done, _ = step(state, aspace.sample(ak))
        if t % every == 0:
            yield f"rollout t={t} weather={int(state.weather_index)}", state
        if bool(np.asarray(done)):
            key, rk = jax.random.split(key)
            _, state = env.reset(rk)


def _active_summary(env, state):
    groups = _obs_groups(env._get_observation(state))
    parts = []
    for name, g in groups.items():
        act = np.asarray(g["active"]).astype(int)
        parts.append(f"{name}={int(act.sum())}/{g['n']} {act.tolist()}" if g["n"] > 1 else f"{name}={int(act[0])}")
    return "  ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", type=int, default=0)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--hold", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true")
    p.add_argument("--obs-res", action="store_true")
    p.add_argument("--out", default=f"./models/{ENV_ID}/enduro_sweep_all_modes.mp4")
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
            iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe}.png"), row)
        print(f"[{i:02d}] {title}\n     {_active_summary(env, state)}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8),
                plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations)")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()