import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from agents.occam.occam import _OCCAMViz
from verify_kingkong import MODES, _obs_groups, _row

ENV_ID = "donkeykong"


def _put(obj, parts, v):
    old = getattr(obj, parts[0])
    if len(parts) > 1:
        return obj.replace(**{parts[0]: _put(old, parts[1:], v)})
    old = jnp.asarray(old)
    new = jnp.asarray(v, dtype=old.dtype)
    if new.shape != old.shape:
        new = jnp.broadcast_to(new, old.shape)
    return obj.replace(**{parts[0]: new})


def put(state, **kw):
    for path, v in kw.items():
        state = _put(state, path.split("__"), v)
    return state


def _mario_on_bar(env, stage, x, level):
    y = float(np.round(np.asarray(env.bar_linear_equation(stage, x, level)))) - env.consts.MARIO_HIT_BOX_Y - 2
    return dict(mario_x=float(x), mario_y=y, mario_stage=stage)


def _on_bar(env, stage, x, level, h):
    return int(np.round(np.asarray(env.bar_linear_equation(stage, x, level)))) - h


def build_poses(env, base):
    c = env.consts
    R, L, U = c.MOVING_RIGHT, c.MOVING_LEFT, c.MOVING_UP

    l1 = put(base, game_started=True)
    yield "L1 reset [mario 1, hammer 1, barrels 0, fires 0, traps 0, ladders 13]", l1

    s = put(l1, mario_view_direction=R, mario_walk_sprite=c.MARIO_WALK_SPRITE_0, **_mario_on_bar(env, 1, 60, 1))
    yield "L1 mario stand right [mario 1 h17]", s
    yield "L1 mario walk_1 right [mario 1 h17]", put(s, mario_walk_sprite=c.MARIO_WALK_SPRITE_1)
    yield "L1 mario walk_2 left [mario 1 h16 y+1]", put(s, mario_view_direction=L, mario_walk_sprite=c.MARIO_WALK_SPRITE_3)
    yield "L1 mario jump right [mario 1 h14]", put(s, mario_jumping=True, mario_y=float(np.asarray(s.mario_y)) - float(c.MARIO_JUMPING_HEIGHT))
    yield "L1 mario jump-wide left [mario 1 h14]", put(s, mario_view_direction=L, mario_jumping_wide=True)
    for cs in (c.MARIO_CLIMB_SPRITE_0, c.MARIO_CLIMB_SPRITE_1):
        yield (f"L1 mario climb sprite {cs} on ladder 0 [mario 1 h15]",
               put(l1, mario_view_direction=U, mario_climbing=True, mario_climb_sprite=cs,
                   mario_x=74.0, mario_y=42.0, mario_stage=5))

    bx = np.array([40, 70, 100, 76], dtype=np.int32)
    stages = [6, 5, 4, 3]
    by = np.array([_on_bar(env, st, x, 1, c.BARREL_HIT_BOX_Y) for st, x in zip(stages, bx)], dtype=np.int32)
    by[3] = by[3] - 12
    sprites = np.array([c.BARREL_SPRITE_RIGHT, c.BARREL_SPRITE_LEFT, c.BARREL_SPRITE_RIGHT, c.BARREL_SPRITE_FALL], dtype=np.int32)
    b = put(l1, barrels__barrel_x=bx, barrels__barrel_y=by, barrels__sprite=sprites,
            barrels__stage=np.array(stages, dtype=np.int32), barrels__reached_the_end=np.zeros(4, bool))
    yield "L1 barrels rolling + one falling [barrels 4, falling h7]", b
    yield "L1 barrels, two reached the end (stale pos) [barrels 2]", put(b, barrels__reached_the_end=np.array([True, False, True, False]))

    yield "L1 hammer lying [hammer 1 4x7 state 0]", l1
    hs = put(s, hammer_taken=True, mario_view_direction=R,
             hammer_x=int(np.asarray(s.mario_x)) + c.MARIO_HIT_BOX_X + 1, hammer_y=c.LEVEL_1_HAMMER_X)
    yield "L1 hammer carried up right [hammer 1 4x7 state 1]", put(hs, hammer_can_hit=False, hammer_y=c.LEVEL_1_HAMMER_X)
    yield "L1 hammer swing down right [hammer 1 6x6 state 1]", put(hs, hammer_can_hit=True, hammer_y=c.LEVEL_1_HAMMER_SWING_X)
    yield ("L1 hammer swing down left [hammer 1 6x6 state 1]",
           put(hs, mario_view_direction=L, hammer_can_hit=True, hammer_y=c.LEVEL_1_HAMMER_SWING_X,
               hammer_x=int(np.asarray(s.mario_x)) - c.HAMMER_SWING_HIT_BOX_X))
    yield "L1 hammer expired [hammer 0]", put(hs, hammer_usage_expired=True, hammer_taken=False)

    l2 = l1.replace(ladders=env.init_ladders_for_level(2),
                    invisible_wall_each_stage=env.init_invisible_wall_for_level(2))
    l2 = put(l2, level=2, mario_x=float(c.LEVEL_2_MARIO_START_Y), mario_y=float(c.LEVEL_2_MARIO_START_X),
             mario_stage=1, mario_view_direction=R, hammer_x=c.LEVEL_2_HAMMER_Y, hammer_y=c.LEVEL_2_HAMMER_X)
    fx = np.array([40.0, 60.0, 90.0, 110.0], dtype=np.float32)
    fy = np.array([_on_bar(env, st, int(x), 2, c.FIRE_HIT_BOX_Y) for st, x in zip([5, 4, 3, 2], fx)], dtype=np.float32)
    l2 = put(l2, fires__fire_x=fx, fires__fire_y=fy, fires__destroyed=np.zeros(4, bool))
    yield "L2 start [mario 1, hammer 1, barrels 0, fires 4, traps 8, ladders 16]", l2
    yield "L2 three traps triggered (holes) [traps 8, state 1 on 3]", put(l2, traps__triggered=np.array([True, False, False, True, False, True, False, False]))
    yield "L2 fires destroyed (stale pos) [fires 0]", put(l2, fires__destroyed=np.ones(4, bool))
    yield ("L2 mario climb ladder 12 [mario 1 h15]",
           put(l2, mario_view_direction=U, mario_climbing=True, mario_climb_sprite=c.MARIO_CLIMB_SPRITE_0,
               mario_x=38.0, mario_y=70.0, mario_stage=4))


def _active_line(obs):
    parts = []
    for name, g in _obs_groups(obs).items():
        parts.append(f"{name}={int(np.asarray(g['active']).sum())}/{g['n']}")
    return "  ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true")
    p.add_argument("--out", default=f"./models/{ENV_ID}/{ENV_ID}_sweep_all_modes.mp4")
    args = p.parse_args()

    env = jaxatari.make(ENV_ID)
    maskers = {m: _OCCAMViz(env, m) for m in MODES}
    _, base_state = env.reset(jax.random.PRNGKey(args.seed))
    situations = list(build_poses(env, base_state))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    png_dir = os.path.splitext(args.out)[0] + "_png"
    if args.png:
        os.makedirs(png_dir, exist_ok=True)

    frames = []
    for i, (title, state) in enumerate(situations):
        row = _row(env, maskers, state, f"[{i:02d}] {title}")
        frames.extend([row] * args.hold)
        if args.png:
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)
            iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe}.png"), row)
        print(f"[{i:02d}] {title}\n     active: {_active_line(env._get_observation(state))}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8), plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations)")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()