"""Deterministic Pacman scenarios for the OCCAM mask check.

build_poses() is the pose provider verify_game.py loads for "pacman". Run
standalone, it renders [clean | obs boxes | object | binary | class | planes]
per scenario, like verify_kingkong.py. Titles carry the expected `active`
state in brackets. Render flicker does not change `active`: a ghost or power
pellet the renderer skips in a frame keeps its box, as in the OCCAM paper,
where masks come from the game state (OCAtari RAM mode).

    uv run python verify_game.py pacman
    uv run python verify_pacman.py --png
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

import jaxatari
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_pacman import GhostMode
from jaxatari.wrappers import AtariWrapper
from agents.occam.occam import OCCAMWrapper, _OCCAMViz
from verify_kingkong import MODES, OUT_H, OUT_W, _row

ENV_ID = "pacman"

# corridor intersections (x % 4 == 1, y % 12 == 6), far enough apart that the
# renderer's ghost-proximity flicker stays off
SPREAD = np.array([[25, 30], [121, 30], [25, 162], [121, 162]], dtype=np.int32)
TUNNEL_X = 73
# step_count 128: pacman frame 0, render flicker shows only the left power pellets
T0 = 128


def _cast(old, value):
    return jnp.asarray(value, dtype=jnp.asarray(old).dtype)


def _sub(nt, **kw):
    unknown = set(kw) - set(nt._fields)
    if unknown:
        raise KeyError(f"unknown fields {sorted(unknown)} in {type(nt).__name__}")
    return nt._replace(**{k: _cast(getattr(nt, k), v) for k, v in kw.items()})


def put(state, level=None, player=None, ghosts=None, fruit=None, **top):
    """state with top-level and nested fields replaced, each cast to its dtype."""
    new = {k: _cast(getattr(state, k), v) for k, v in top.items()}
    for name, kw in (("level", level), ("player", player),
                     ("ghosts", ghosts), ("fruit", fruit)):
        if kw:
            new[name] = _sub(getattr(state, name), **kw)
    return state.replace(**new)


def _modes(*m):
    return np.array([int(x) for x in m], dtype=np.uint8)


def build_poses(env, base):
    """Yield (title, state) for every situation worth looking at."""
    c = env.consts
    chase = _modes(*[GhostMode.CHASE] * 4)
    left = np.full(4, int(Action.LEFT), dtype=np.uint8)
    start = np.asarray(c.INITIAL_PACMAN_POSITION, dtype=np.int32)
    jail = np.asarray(c.JAIL_POSITION, dtype=np.int32)
    fruit_xy = np.asarray(c.VITAMINS_POSITION, dtype=np.int32)

    gp = put(base, step_count=T0, freeze_timer=0, eat_freeze_timer=0,
             player=dict(position=start, action=int(Action.LEFT),
                         tunnel_timer=0, last_horiz_dir=2),
             ghosts=dict(positions=SPREAD, modes=chase, actions=left),
             fruit=dict(spawned=False))

    # --- every group active at least once ----------------------------------
    yield ("play: start position, ghosts spread (chase), all pellets", gp)

    # --- player sprite: animation frames and facing -------------------------
    for sc, frame in ((T0, 0), (T0 + 4, 1), (T0 + 8, 2)):
        yield (f"player frame {frame} facing right",
               put(gp, step_count=sc, player=dict(last_horiz_dir=1)))
    yield "player frame 0 facing left", put(gp, player=dict(last_horiz_dir=2))
    yield ("player at left edge x=1", put(gp, player=dict(position=[1, 54])))
    yield ("player at right edge x=145", put(gp, player=dict(position=[145, 54])))

    # --- vertical tunnel ----------------------------------------------------
    yield ("tunnel top, entering y=2 [player active]",
           put(gp, player=dict(position=[TUNNEL_X, 2], action=int(Action.UP))))
    yield ("tunnel top, inside y=0 timer>0 [player inactive]",
           put(gp, player=dict(position=[TUNNEL_X, 0], action=int(Action.UP),
                               tunnel_timer=c.TUNNEL_DELAY // 2)))
    yield ("tunnel bottom, entering y=188 [player active]",
           put(gp, player=dict(position=[TUNNEL_X, 188], action=int(Action.DOWN))))
    yield ("tunnel bottom y=190, below draw limit 189 [player inactive]",
           put(gp, player=dict(position=[TUNNEL_X, 190], action=int(Action.DOWN))))

    # --- vulnerable ghosts ---------------------------------------------------
    yield ("ghosts frightened x3 + blinking x1, one power pellet eaten",
           put(gp,
               level=dict(power_pellets=[False, True, True, True]),
               ghosts=dict(modes=_modes(GhostMode.FRIGHTENED, GhostMode.FRIGHTENED,
                                        GhostMode.FRIGHTENED, GhostMode.BLINKING))))

    # --- eaten ghost: no eyes sprite, the ghost is sent to the jail ----------
    pos = SPREAD.copy()
    pos[0] = jail
    pos[1] = jail + np.array([0, -12])
    yield ("eaten ghost 0 in jail (ENJAILED), ghost 1 leaving (RETURNING)",
           put(gp, ghosts=dict(
               positions=pos,
               modes=_modes(GhostMode.ENJAILED, GhostMode.RETURNING,
                            GhostMode.FRIGHTENED, GhostMode.FRIGHTENED))))

    # --- render flicker: sprites skipped, boxes stay active -----------------
    reset_ghosts = dict(positions=np.tile(jail, (4, 1)),
                        modes=_modes(GhostMode.RANDOM, GhostMode.ENJAILED,
                                     GhostMode.ENJAILED, GhostMode.ENJAILED))
    for sc in (T0, T0 + 1):
        dropped = [i for i in range(4) if ((sc + i) & 3) == 0]
        yield (f"4 ghosts crowded in jail, flicker skips ghost {dropped} "
               f"[all 4 active]",
               put(gp, step_count=sc, ghosts=reset_ghosts))
    yield ("power pellet flicker: right pair drawn [all 4 active]",
           put(gp, step_count=T0 + 8))

    # --- fruit ----------------------------------------------------------------
    yield ("fruit spawned [fruit active]",
           put(gp, fruit=dict(position=fruit_xy, type=0, spawned=True,
                              timer=c.VITAMINS_DURATION)))
    yield ("fruit gone, stale position [fruit inactive]",
           put(gp, fruit=dict(position=fruit_xy, spawned=False)))

    # --- death: the renderer has no death animation, sprites stay put -------
    pos = SPREAD.copy()
    pos[0] = start
    yield ("death freeze, ghost 0 on the player",
           put(gp, freeze_timer=c.RESET_TIMER,
               lives=int(np.asarray(base.lives)) - 1,
               ghosts=dict(positions=pos)))

    # --- nearly cleared level -------------------------------------------------
    pellets = np.zeros(np.asarray(base.level.pellets).shape, dtype=bool)
    for gx, gy in ((0, 0), (5, 3), (17, 7)):
        pellets[gx, gy] = True
    yield ("3 pellets + bottom-left power pellet left [pellets active = 3]",
           put(gp, level=dict(pellets=pellets,
                              power_pellets=[False, False, True, False],
                              eaten_pellets=126)))

    # --- second maze layout ---------------------------------------------------
    yield "level 2 (maze variant 2)", put(gp, level=dict(id=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=45, help="video frames per situation")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", action="store_true", help="also write one PNG per situation")
    p.add_argument("--obs-res", action="store_true",
                   help=f"show the real OCCAMWrapper masks at {OUT_H}x{OUT_W}")
    p.add_argument("--out", default=f"./models/{ENV_ID}/pacman_sweep_all_modes.mp4")
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
              f" ghosts={np.asarray(obs.ghosts.active).tolist()}"
              f" power={np.asarray(obs.power_pellets.active).tolist()}"
              f" fruit={int(obs.fruit.active)}"
              f" pellets={int(np.asarray(obs.pellets.active).sum())}")

    iio.imwrite(args.out, np.asarray(frames, dtype=np.uint8),
                plugin="pyav", codec="libx264", fps=args.fps)
    print(f"\nwrote {args.out}  ({len(situations)} situations)")


if __name__ == "__main__":
    main()