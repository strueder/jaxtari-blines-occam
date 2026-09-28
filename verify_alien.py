"""Deterministic Alien sweep across all OCCAM mask modes side by side.

Titles carry the expected active count per group in [brackets]; the console
prints the actual counts.

    PYTHONPATH=../JAXAtari/src uv run python verify_alien.py --png
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from jaxatari.wrappers import AtariWrapper
from agents.occam.occam import OCCAMWrapper, _OCCAMViz
from verify_kingkong import MODES, OUT_H, OUT_W, _obs_groups, _row

ENV_ID = "alien"

UP, RIGHT, LEFT, DOWN = 2, 3, 4, 5
E = "enemies__multiple_enemies__"


def put(state, path, value):
    head, _, rest = path.partition(".")
    old = getattr(state, head)
    new = put(old, rest, value) if rest else jnp.asarray(value, dtype=jnp.asarray(old).dtype)
    return state.replace(**{head: new})


def puts(state, **kw):
    for k, v in kw.items():
        state = put(state, k.replace("__", "."), v)
    return state


def build_poses(env, base):
    g = puts(base, level__frame_count=61,
             **{E + "x": [14, 106, 14, 0, 0, 0], E + "y": [8, 8, 128, 0, 0, 0],
                E + "enemy_spawn_frame": [0, 0, 0, 0, 0, 0]})

    yield "reset [player 1, enemies 3, flame 0, eggs 106, kill_items 1, score_items 0]", base
    yield "enemies in the maze [enemies 3]", g
    yield ("enemy 0 hidden in spawn phase, enemy 2 descending [enemies 2]",
           puts(g, **{E + "enemy_spawn_frame": [5, 0, 15, 0, 0, 0]}))
    yield "enemies frightened [enemies 3]", puts(g, **{E + "killable": [1, 1, 1, 0, 0, 0]})
    for f in (55, 40, 20, 5):
        yield (f"enemy 0 death frame {f} [enemies 3]",
               puts(g, **{E + "enemy_death_frame": [f, 0, 0, 0, 0, 0]}))
    for x in (7, 10, 14, 120, 124, 127):
        yield (f"enemy 0 teleport x={x} [enemies 3]",
               puts(g, **{E + "x": [x, 106, 14, 0, 0, 0], E + "y": [80, 8, 128, 0, 0, 0]}))

    for name, o in (("LEFT", LEFT), ("RIGHT", RIGHT)):
        for fc in (61, 65, 69, 73):
            yield (f"player {name} frame_count={fc} [player 1]",
                   puts(g, player__orientation=o, player__last_horizontal_orientation=o,
                        level__frame_count=fc))
    for x in (7, 9, 11, 123, 125, 127):
        yield f"player teleport x={x} [player 1]", puts(g, player__x=x, player__y=80)
    yield "player death animation [player 1]", puts(g, level__death_frame_counter=30)
    yield "game over animation [player 1]", puts(g, level__death_frame_counter=-20)

    px, py = int(g.player.x), int(g.player.y)
    for name, o, fx in (("LEFT", LEFT, px - 6), ("RIGHT", RIGHT, px + 10)):
        yield (f"flame {name} [flame 1, kill_items 0]",
               puts(g, player__last_horizontal_orientation=o, player__flame__flame_flag=1,
                    player__flame__x=fx, player__flame__y=py + 6))

    eggs = np.asarray(base.eggs).copy()
    eggs[0, :, 2] = 0
    yield "left eggs eaten [eggs 53]", puts(g, eggs=eggs)
    eggs[1, 1:, 2] = 0
    yield "one egg left [eggs 1]", puts(g, eggs=eggs)

    items = np.asarray(base.items).copy()
    items[:3, 2] = [0, 1, 0]
    yield ("second kill item [kill_items 1]",
           puts(g, items=items, level__current_active_item_index=1))
    items[:3, 2] = 0
    yield ("all kill items eaten [kill_items 0]",
           puts(g, items=items, level__current_active_item_index=3))
    for t in range(1, 7):
        it = items.copy()
        it[3, 2:] = [1, t]
        yield f"score item type {t} [score_items 1]", puts(g, items=it)

    b = env.start_bonus_stage(puts(base, level__frame_count=61))
    yield "bonus stage [player 1, enemies 12, eggs 0, kill_items 0, score_items 1]", b
    yield ("bonus aliens pushed back at the right edge [enemies 12]",
           puts(b, **{E + "x": [155, 120, 150, 40, 159, 124],
                      E + "last_horizontal_orientation": [RIGHT, LEFT, RIGHT, LEFT, RIGHT, LEFT]}))
    yield "bonus prize collected [score_items 0]", puts(b, level__evil_item_frame_counter=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true")
    p.add_argument("--obs-res", action="store_true")
    p.add_argument("--out", default=f"./models/{ENV_ID}/alien_sweep_all_modes.mp4")
    args = p.parse_args()

    env = jaxatari.make(ENV_ID)
    if args.obs_res:
        maskers = {m: OCCAMWrapper(AtariWrapper(jaxatari.make(ENV_ID)), mask_mode=m,
                                   game_name=ENV_ID) for m in MODES}
    else:
        maskers = {m: _OCCAMViz(env, m) for m in MODES}
    _, base_state = env.reset(jax.random.PRNGKey(args.seed))
    situations = list(build_poses(env, base_state))

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
        groups = _obs_groups(env._get_observation(state))
        counts = "  ".join(f"{n}={int(np.asarray(g['active']).sum())}" for n, g in groups.items())
        print(f"[{i:02d}] {title}\n      active: {counts}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8),
                plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations, {len(frames)} frames)")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()