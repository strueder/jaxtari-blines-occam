import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from jaxatari.wrappers import AtariWrapper
from agents.occam.occam import OCCAMWrapper, _OCCAMViz
from verify_kingkong import MODES, _row, _obs_groups

ENV_ID = "phoenix"


def put(obj, **kw):
    new = {}
    for k, v in kw.items():
        old = getattr(obj, k)
        if isinstance(v, dict):
            new[k] = put(old, **v)
        else:
            new[k] = jnp.asarray(v, dtype=jnp.asarray(old).dtype)
    return obj.replace(**new)


def build_poses(env, base):
    c = env.consts
    off8 = np.full(8, -1, dtype=np.int32)
    no8 = np.zeros(8, dtype=bool)

    def level(n, **kw):
        f = (n - 1) % 5
        ex = np.asarray(c.ENEMY_POSITIONS_X[f])
        ey = np.asarray(c.ENEMY_POSITIONS_Y[f])
        s = put(base, level=n, enemies_x=ex, enemies_y=ey, phoenix_original_y=ey,
                phoenix_do_attack=no8, phoenix_returning=no8, phoenix_dying=no8,
                bat_wings=np.full(8, 2, dtype=np.int32), bat_dying=no8,
                player_dying=False, player_respawn_timer=0, player_moving=False,
                invincibility=False, step_counter=0,
                projectile_x=-1, projectile_y=-1,
                enemy_projectile_x=off8, enemy_projectile_y=off8,
                boss={"active": True,
                      "x": float(c.BOSS_SPAWN_POSITION[0]),
                      "y": float(c.BOSS_SPAWN_POSITION[1]),
                      "blue_alive": np.ones((2, 20), dtype=bool),
                      "red_alive": np.ones(92, dtype=bool),
                      "green_alive": np.ones(30, dtype=bool)})
        return put(s, **kw) if kw else s

    px = int(c.PLAYER_POSITION[0])
    shots_x = np.array([20, 40, 60, 80, 100, 120, 140, 150], dtype=np.int32)
    shots_y = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=np.int32)

    yield "L1 phoenix idle [player=1 enemies=8 boss=0 boss_blocks=0]", level(1)

    ey1 = np.asarray(c.ENEMY_POSITIONS_Y[0])
    attack = no8.copy()
    attack[[0, 3]] = True
    yield ("L1 phoenix 0+3 attack frame [enemies=8]",
           level(1, phoenix_do_attack=attack, phoenix_attack_target_y=ey1 + 40))

    dying = no8.copy()
    dying[[1, 2]] = True
    yield ("L1 phoenix 1+2 dying [enemies=8]",
           level(1, phoenix_dying=dying,
                 phoenix_death_timer=np.where(dying, 20, 0).astype(np.int32)))

    yield "L2 green enemies [enemies=8]", level(2)

    wings = np.array([2, -1, 1, 0, 2, 2, 2, 2], dtype=np.int32)
    bdying = no8.copy()
    bdying[4] = True
    btimer = np.where(bdying, 25, 0).astype(np.int32)
    for n, col in [(3, "blue"), (4, "red")]:
        yield (f"L{n} {col} bats: 0 both, 1 left, 2 right, 3 none, 4 dying [enemies=7]",
               level(n, bat_wings=wings, bat_dying=bdying, bat_death_timer=btimer))
        yield f"L{n} {col} bats all wings [enemies=7]", level(n)

    yield "L5 boss full [boss=1 boss_blocks=162 enemies=0]", level(5)

    blue = np.ones((2, 20), dtype=bool)
    blue[:, ::2] = False
    red = np.ones(92, dtype=bool)
    red[::3] = False
    green = np.ones(30, dtype=bool)
    green[1::2] = False
    yield ("L5 boss damaged [boss=1 boss_blocks<162 enemies=0]",
           level(5, boss={"blue_alive": blue, "red_alive": red, "green_alive": green}))
    yield ("L5 boss lowest + shots [boss=1 enemy_projectiles=8]",
           level(5, boss={"y": float(c.BOSS_LOWEST_Y)},
                 enemy_projectile_x=shots_x, enemy_projectile_y=shots_y))
    yield "L5 boss destroyed [boss=0 boss_blocks=0 enemies=0]", level(5, boss={"active": False})

    yield "player moving frame [player=1]", level(1, player_moving=True, step_counter=0)
    yield ("player dying [player=1]",
           level(1, player_dying=True, player_death_timer=int(c.PLAYER_DEATH_DURATION) - 5))
    yield "player respawning [player=0]", level(1, player_respawn_timer=60)
    left = int(c.PLAYER_BOUNDS[0]) + int(c.PLAYER_EDGE_MARGIN_LEFT)
    right = int(c.PLAYER_BOUNDS[1]) - int(c.PLAYER_EDGE_MARGIN)
    yield "player at left stop [player=1]", level(1, player_x=left)
    yield "player at right stop [player=1]", level(1, player_x=right)

    yield ("shots in flight [player_projectile=1 enemy_projectiles=8]",
           level(1, projectile_x=px + 2, projectile_y=120,
                 enemy_projectile_x=shots_x, enemy_projectile_y=shots_y))
    yield ("player shot left screen [player_projectile=0, renderer may still draw at top]",
           level(1, projectile_x=px + 2, projectile_y=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true")
    p.add_argument("--obs-res", action="store_true")
    p.add_argument("--out", default=f"./models/{ENV_ID}/phoenix_sweep_all_modes.mp4")
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
            iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe}.png"), row)
        groups = _obs_groups(env._get_observation(state))
        act = " ".join(f"{k}={int(np.sum(g['active']))}" for k, g in groups.items())
        print(f"[{i:02d}] {title}\n      {act}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8),
                plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations)")
    if args.png:
        print(f"stills in {png_dir}/")


if __name__ == "__main__":
    main()