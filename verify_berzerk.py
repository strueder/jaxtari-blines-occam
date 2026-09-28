"""Deterministic Berzerk scenarios for the OCCAM mask check.

build_poses() is the pose provider verify_game.py loads for "berzerk". Run
standalone, it renders [clean | obs boxes | object | binary | class | planes]
per scenario, like verify_kingkong.py. Titles carry the expected `active`
state in brackets.

    uv run python verify_game.py berzerk
    uv run python verify_berzerk.py --png
"""
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
from verify_kingkong import MODES, OUT_H, OUT_W, _row

ENV_ID = "berzerk"

# room 0 layout (mid_walls_1): open floor between the two vertical walls,
# below the horizontal bar; enemies spread over the corridors
PLAYER_XY = [70, 110]
ENEMY_XY = np.array([[16, 40], [16, 120], [132, 40], [132, 120],
                     [60, 40], [90, 40], [100, 120]], dtype=np.float32)
OTTO_XY = [50, 125]


def _cast(old, value):
    return jnp.asarray(value, dtype=jnp.asarray(old).dtype)


def _sub(obj, **kw):
    names = {f.name for f in dataclasses.fields(obj)}
    unknown = set(kw) - names
    if unknown:
        raise KeyError(f"unknown fields {sorted(unknown)} in {type(obj).__name__}")
    return obj.replace(**{k: _cast(getattr(obj, k), v) for k, v in kw.items()})


def put(state, player=None, enemy=None, otto=None, **top):
    """state with top-level and nested fields replaced, each cast to its dtype."""
    new = {k: _cast(getattr(state, k), v) for k, v in top.items()}
    for name, kw in (("player", player), ("enemy", enemy), ("otto", otto)):
        if kw:
            new[name] = _sub(getattr(state, name), **kw)
    return state.replace(**new)


def _enemy_fields(n, **over):
    f = dict(pos=ENEMY_XY[:n], move_axis=np.full(n, -1), move_dir=np.zeros(n),
             alive=np.ones(n, bool), bullet_active=np.zeros(n, bool),
             death_timer=np.zeros(n), animation_counter=np.ones(n))
    f.update(over)
    return f


def build_poses(env, base):
    """Yield (title, state) for every situation worth looking at."""
    c = env.consts
    n = c.MAX_NUM_ENEMIES

    gp = put(base, room_counter=0, room_transition_timer=0, game_over_timer=0,
             entry_direction=3,
             player=dict(pos=PLAYER_XY, last_dir=[1, 0], animation_counter=0,
                         is_firing=False, death_timer=0, bullet_active=[False]),
             enemy=_enemy_fields(n),
             otto=dict(active=False, pos=[-100, -100], anim_counter=0))

    # --- every group active at least once ------------------------------------
    eb = np.zeros(n, bool)
    eb[:2] = True
    eb_pos = ENEMY_XY.copy()
    eb_pos[0] = [20, 70]
    eb_pos[1] = [28, 150]
    eb_dir = np.zeros((n, 2), np.float32)
    eb_dir[0] = [0, 1]
    eb_dir[1] = [1, 0]
    yield ("all groups: player, bullet, 7 enemies, 2 enemy bullets, otto "
           "[all active]",
           put(gp, player=dict(bullet=[[50, 100]], bullet_dir=[[1, 0]],
                               bullet_active=[True]),
               enemy=dict(bullets=eb_pos, bullet_dirs=eb_dir, bullet_active=eb),
               otto=dict(active=True, pos=OTTO_XY, anim_counter=15)))

    # --- player sprite: idle, walk frames, facing -----------------------------
    yield "player idle facing right (6x20)", gp
    yield "player idle facing left (6x20, flipped)", put(gp, player=dict(last_dir=[-1, 0]))
    for cnt, name in ((1, "walk 1, 8 wide"), (3, "walk 2, 5 wide")):
        yield (f"player {name}, facing right",
               put(gp, player=dict(animation_counter=cnt, last_dir=[1, 0])))
        yield (f"player {name}, facing left (flipped, box shifts)",
               put(gp, player=dict(animation_counter=cnt, last_dir=[-1, 0])))

    # --- player shooting: one sprite per direction, never flipped -------------
    for d, name in (([0, -1], "up"), ([1, 0], "right"), ([0, 1], "down"),
                    ([-1, 0], "left"), ([-1, -1], "up-left"), ([-1, 1], "down-left")):
        pb_dir = [[0, -1]] if d[0] == 0 else [[d[0], 0]]
        pb_xy = [[PLAYER_XY[0] + 3, PLAYER_XY[1] - 12]] if d[0] == 0 else \
                [[PLAYER_XY[0] + 12 * d[0], PLAYER_XY[1] + 6]]
        yield (f"player shooting {name} + player bullet [bullet active]",
               put(gp, player=dict(is_firing=True, last_dir=d, bullet=pb_xy,
                                   bullet_dir=pb_dir, bullet_active=[True])))

    # --- player death: blinks between idle (6x20) and death (8x24) -----------
    for dt, name in ((3, "death sprite 8x24"), (1, "idle sprite 6x20")):
        yield (f"player dying, {name} [player active]",
               put(gp, player=dict(death_timer=dt)))
    yield ("player dying facing left, death sprite flipped [player active]",
           put(gp, player=dict(death_timer=3, last_dir=[-1, 0])))

    # --- enemies: walk frames and the two taller vertical frames -------------
    axis = np.array([0, 0, 1, 1, 1, 1, -1])
    mdir = np.array([1, -1, 1, 1, -1, 1, 0])
    for cnt in (1, 8, 7, 19):
        yield (f"enemies walking, animation_counter={cnt} "
               f"(vertical walkers 8x18 at 7 and 19)",
               put(gp, enemy=dict(move_axis=axis, move_dir=mdir,
                                  animation_counter=np.full(n, cnt))))

    # --- dead and dying enemies ------------------------------------------------
    alive = np.ones(n, bool)
    alive[:3] = False
    pos = ENEMY_XY.copy()
    pos[:3] = -100
    dtimer = np.zeros(n)
    dtimer[0], dtimer[1] = 9, 3
    yield ("enemies 0-1 exploding, 2 dead [enemies active = 4]",
           put(gp, enemy=dict(alive=alive, pos=pos, death_pos=ENEMY_XY,
                              death_timer=dtimer)))
    yield ("only 1 enemy left [enemies active = 1]",
           put(gp, enemy=dict(alive=np.arange(n) == n - 1)))

    # --- evil otto: blink does not change active --------------------------------
    yield ("otto drawn [otto active]",
           put(gp, otto=dict(active=True, pos=OTTO_XY, anim_counter=15)))
    yield ("otto blink-off phase, not drawn [otto active]",
           put(gp, otto=dict(active=True, pos=OTTO_XY, anim_counter=0)))

    # --- room transition and game over ------------------------------------------
    full = put(gp, player=dict(bullet=[[50, 100]], bullet_dir=[[1, 0]],
                               bullet_active=[True]),
               enemy=dict(bullets=eb_pos, bullet_dirs=eb_dir, bullet_active=eb),
               otto=dict(active=True, pos=OTTO_XY, anim_counter=15))
    yield ("room transition, bars closing [all inactive except otto]",
           put(full, room_transition_timer=c.TRANSITION_ANIMATION_FRAMES // 2,
               entry_direction=0))
    yield ("game over, black screen [all inactive except otto]",
           put(full, room_transition_timer=c.GAME_OVER_FRAMES,
               game_over_timer=c.GAME_OVER_FRAMES // 2))

    # --- other room: new wall layout, recoloured enemies -------------------------
    yield "room 3 (other wall layout, enemy colour 2)", put(full, room_counter=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45, help="video frames per situation")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true", help="also write one PNG per situation")
    p.add_argument("--obs-res", action="store_true",
                   help=f"show the real OCCAMWrapper masks at {OUT_H}x{OUT_W}")
    p.add_argument("--out", default=f"./models/{ENV_ID}/berzerk_sweep_all_modes.mp4")
    args = p.parse_args()

    env = jaxatari.make(ENV_ID)
    if args.obs_res:
        maskers = {m: OCCAMWrapper(AtariWrapper(jaxatari.make(ENV_ID)), mask_mode=m,
                                   game_name=ENV_ID) for m in MODES}
    else:
        maskers = {m: _OCCAMViz(env, m) for m in MODES}
    _, base = env.reset(jax.random.PRNGKey(args.seed))
    situations = list(build_poses(env, base))

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
            iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe[:60]}.png"), row)
        obs = env._get_observation(state)
        print(f"[{i:02d}] {title}  | player={int(obs.player.active)}"
              f" bullet={int(obs.player_bullet.active)}"
              f" enemies={int(np.asarray(obs.enemies.active).sum())}"
              f" enemy_bullets={int(np.asarray(obs.enemy_bullets.active).sum())}"
              f" otto={int(obs.otto.active)}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8),
                plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations)")


if __name__ == "__main__":
    main()