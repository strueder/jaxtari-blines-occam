import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from agents.occam.occam import _OCCAMViz
from verify_kingkong import MODES, _row

ENV_ID = "venture"
GROUPS = ("player", "monsters", "portals", "chest", "lasers", "chaser", "projectile")


def put(state, path, value):
    head, _, rest = path.partition(".")
    sub = getattr(state, head)
    if rest:
        return state.replace(**{head: put(sub, rest, value)})
    return state.replace(**{head: jnp.asarray(value, dtype=jnp.asarray(sub).dtype)})


def edit(state, **fields):
    for path, value in fields.items():
        state = put(state, path.replace("__", "."), value)
    return state


def level_range(c, world, level):
    g = (world - 1) * 5 + level
    offsets = np.asarray(c.LEVEL_OFFSETS)
    return int(offsets[g]), int(offsets[g + 1])


def entry_spawn(c, world, level):
    if level == 0:
        return float(c.PLAYER_INITIAL_X), float(c.PLAYER_INITIAL_Y)
    t = np.asarray(c.JAX_TRANSITIONS[world - 1, 0])
    row = t[t[:, 4].astype(int) == level][0]
    return float(row[5]), float(row[6])


def enter(env, base, world, level):
    c = env.consts
    n = int(c.TOTAL_MONSTERS)
    start, end = level_range(c, world, level)
    idx = np.arange(n)
    px, py = entry_spawn(c, world, level)
    return edit(
        base,
        world_level=world,
        current_level=level,
        last_level=level,
        player__x=px,
        player__y=py,
        monsters__x=np.asarray(c.ALL_MONSTER_SPAWNS[:, 0]),
        monsters__y=np.asarray(c.ALL_MONSTER_SPAWNS[:, 1]),
        monsters__active=(idx >= start) & (idx < end),
        monsters__dead_for=np.full(n, -1),
        chaser__active=False,
        projectile__active=False,
        lasers__positions=np.asarray(c.LASER_INITIAL_POSITIONS),
        collected_chest_in_current_visit=-1,
        level_timer=0,
    )


def build_poses(env, base):
    c = env.consts
    for world in (1, 2):
        for level in range(5):
            start, end = level_range(c, world, level)
            s = enter(env, base, world, level)
            chest = 1 if level else 0
            lasers = 4 if (world, level) == (1, 1) else 0
            portals = int((np.asarray(c.JAX_TRANSITIONS[world - 1, level])[:, 2] > 0).sum())
            yield (f"W{world} L{level} [monsters {end - start}, portals {portals}, chest {chest}, lasers {lasers}]", s)

    s = edit(enter(env, base, 1, 0), chests_active=np.array([True, False, True, True]))
    yield "W1 map, room 2 done [portals 6, both room-2 doors 0]", s
    s = edit(enter(env, base, 2, 0), chests_active=np.array([False, False, False, True]))
    yield "W2 map, rooms 1-3 done [portals 2]", s

    room = enter(env, base, 1, 2)
    start, end = level_range(c, 1, 2)
    px, py = float(np.asarray(room.player.x)), float(np.asarray(room.player.y))
    yield ("W1 L2 projectile mid-room [projectile 1]",
           edit(room, projectile__x=px + 20.0, projectile__y=py, projectile__active=True))
    yield ("W1 L2 projectile at left edge x=1 [projectile 1, trimmed w=3]",
           edit(room, projectile__x=1.0, projectile__y=py, projectile__active=True))
    yield ("W1 L2 projectile stale position, inactive [projectile 0]",
           edit(room, projectile__x=px + 20.0, projectile__y=py, projectile__active=False))
    dead = np.full(int(c.TOTAL_MONSTERS), -1)
    dead[start] = 50
    active = np.asarray(room.monsters.active).copy()
    active[start] = False
    yield (f"W1 L2 corpse slot {start} [monsters {end - start}, corpse state=1]",
           edit(room, monsters__active=active, monsters__dead_for=dead))
    active_all_dead = np.zeros_like(active)
    yield ("W1 L2 corpse expired, monster dead [monsters 0]",
           edit(room, monsters__active=active_all_dead))
    yield ("W1 L2 chaser at spawn [chaser 1]",
           edit(room, chaser__active=True,
                chaser__x=float(c.CHASER_SPAWN_POS[0]), chaser__y=float(c.CHASER_SPAWN_POS[1])))
    yield ("W1 L2 chaser next to player [chaser 1]",
           edit(room, chaser__active=True, chaser__x=px + 12.0, chaser__y=py))
    yield ("W1 L2 chest taken this visit [chest 0]",
           edit(room, collected_chest_in_current_visit=1))

    lasers_room = enter(env, base, 1, 1)
    bounds = np.asarray(c.LASER_BOUNDS)
    yield ("W1 L1 lasers at min bounds [lasers 4]",
           edit(lasers_room, lasers__positions=bounds.min(1)))
    yield ("W1 L1 lasers at max bounds [lasers 4]",
           edit(lasers_room, lasers__positions=bounds.max(1)))

    map_w1 = enter(env, base, 1, 0)
    mx, my = float(np.asarray(map_w1.player.x)), float(np.asarray(map_w1.player.y))
    yield ("W1 map projectile carried over from room [projectile 1]",
           edit(map_w1, projectile__x=mx, projectile__y=my - 20.0, projectile__active=True))


def active_counts(env, state):
    obs = env._get_observation(state)
    return {g: int(np.asarray(getattr(obs, g).active).sum()) for g in GROUPS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true")
    p.add_argument("--out", default=f"./models/{ENV_ID}/venture_sweep_all_modes.mp4")
    args = p.parse_args()

    env = jaxatari.make(ENV_ID)
    maskers = {m: _OCCAMViz(env, m) for m in MODES}
    _, base = env.reset(jax.random.PRNGKey(args.seed))
    poses = list(build_poses(env, base))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    png_dir = os.path.splitext(args.out)[0] + "_png"
    if args.png:
        os.makedirs(png_dir, exist_ok=True)

    frames = []
    for i, (title, state) in enumerate(poses):
        row = _row(env, maskers, state, f"[{i:02d}] {title}")
        frames.extend([row] * args.hold)
        if args.png:
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)
            iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe[:60]}.png"), row)
        counts = " ".join(f"{g}={n}" for g, n in active_counts(env, state).items())
        print(f"[{i:02d}] {title}\n     active: {counts}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8), plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(poses)} poses)")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()