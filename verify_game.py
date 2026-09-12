"""Generic OCCAM verification sweep for any JAXAtari game.

Renders every scenario the game can reach as one row
[clean | obs boxes | object | binary | class | planes | planes (sheets)]
so the object decomposition can be checked by eye, and writes a Markdown report
with the checks that are hard to see by eye.

Checks (console/report only, the video stays untouched):
  offscreen   active boxes leaving the frame -- a few pixels are easy to miss
  classes     what the game exposes vs. what the object-centric wrapper emits
  ghost       active box covering nothing the renderer drew   (cf. PR #313)
  origin      observation emitting sprite-center instead of top-left coords
                                                              (cf. PR #309/#310)
  size        box extent vs. the extent of the sprite actually drawn
  uncovered   pixels the renderer draws that no box covers     (cf. PR #314)
  degenerate  active boxes with width/height <= 0

    uv run python verify_game.py kingkong
    uv run python verify_game.py gravitar --console-only
    uv run python verify_game.py all --console-only     # every game + SUMMARY.md
    uv run python verify_game.py --summarize            # rebuild SUMMARY.md only
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio
from scipy import ndimage

import jaxatari
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.wrappers import AtariWrapper

from agents.occam.occam import (
    OCCAMWrapper, _OCCAMViz, _extract_object_groups, _group_names,
)
# the row layout is game-agnostic -- reuse it instead of duplicating it
from verify_kingkong import MODES, OUT_H, OUT_W, _row

OUT_ROOT = "verification"

# games that ship a hand-written scenario list; everything else is mined
POSE_PROVIDERS = {"kingkong": ("verify_kingkong", "build_poses")}

# a state field is treated as a "phase" (gamestate / level / mode / lives) when it
# is a scalar integer taking at least 2 and at most this many distinct values
MAX_PHASE_VALUES = 12

# heuristics thresholds -- deliberately loose, these produce hints, not verdicts
MIN_SAMPLES = 5          # active instances a group needs before pixel checks apply
GHOST_WARN = 0.20        # >20% of active instances covering zero drawn pixels
ORIGIN_GAIN = 0.15       # shifted box must gain this much foreground coverage
ORIGIN_MAX_COV = 0.55    # ...and the reported box must be this bad to begin with
SIZE_TOL = 3             # px of median extent mismatch tolerated, absolute...
SIZE_REL = 0.20          # ...and relative to the box dimension
UNCOVERED_MIN_FRAC = 0.05  # blob must show up in >=5% of scenarios
UNCOVERED_MIN_AREA = 6     # px
STATIC_BLOB = 0.90       # present in >=90% of scenarios -> HUD / background art

# error categories the user tracks by hand; the value says how far this script can
# take each one. Rendered at the top of SUMMARY.md.
CATEGORIES = {
    "A": ("Koordinaten-Konvention", "full",
          "the `origin` check re-scores every box shifted by (−w/2, −h/2); a clear "
          "coverage gain is the sprite-center signature of PR #309/#310"),
    "B": ("Objektgruppe nicht exponiert", "heuristic",
          "the `uncovered` check finds drawn pixels no box covers and separates "
          "transient blobs (likely a real object) from static ones (HUD/background). "
          "A group hidden behind another object stays invisible to it"),
    "C": ("ObjectObservation leer", "full",
          "counted directly from the observation pytree"),
    "D": ("Größen/Skalierungs-Bug", "partial",
          "the `size` check compares the box extent against the extent of the ink "
          "actually drawn, but only for instances that do not touch a neighbour, and "
          "a sprite made of disconnected parts (ladder rungs, dashed lines) has no "
          "well-defined extent. It is therefore reported at info level: a 🔴 here means "
          "\"open the video and look\", not \"broken\". The `cov%` and `Δw/Δh` columns "
          "in the per-game report carry the numbers"),
    "E": ("Active-Defnition zu locker", "heuristic",
          "the `ghost` check flags active boxes covering zero drawn pixels; "
          "`always-on` additionally names groups that are never gated at all"),
    "F": ("Komplexe Geometrie (z.B. Terrain)", "none",
          "whether an object *needs* a multi-box decomposition instead of one bounding "
          "box is a design judgement. A single box around an L-shape looks exactly like "
          "a too-large box, so the two cannot be separated automatically — the `cov%` "
          "column is the closest hint: a group far below 50% either wastes area or "
          "wants to be split"),
}


# --------------------------------------------------------------------------
# report sink
# --------------------------------------------------------------------------

class Report:
    """print() that also keeps every line for the Markdown report file."""

    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, s: str = ""):
        print(s)
        self.lines.append(s)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


@dataclasses.dataclass
class Finding:
    level: str      # FAIL | WARN | INFO
    check: str
    group: str
    message: str

    def md(self) -> str:
        who = f"**{self.group}** — " if self.group else ""
        return f"- `{self.check}` {who}{self.message}"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# environment loading
# --------------------------------------------------------------------------

def all_game_names() -> List[str]:
    """Registered games plus every jax_<name>.py that is commented out of the registry."""
    names = set(jaxatari.list_available_games())
    try:
        import jaxatari.games as G
        d = os.path.dirname(G.__file__)
        names |= {f[4:-3] for f in os.listdir(d)
                  if f.startswith("jax_") and f.endswith(".py")}
    except Exception:
        pass
    return sorted(names)


def load_env(game: str):
    """(env, note). Falls back to a direct module import for unregistered games."""
    if game in jaxatari.list_available_games():
        return jaxatari.make(game), ""
    module = importlib.import_module(f"jaxatari.games.jax_{game}")
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            return obj(), (f"not in `GAME_MODULES` (commented out) — loaded directly "
                           f"from `jaxatari.games.jax_{game}`")
    raise ImportError(f"no JaxEnvironment subclass in jaxatari.games.jax_{game}")


# --------------------------------------------------------------------------
# observation plumbing
# --------------------------------------------------------------------------

def _groups_np(obs) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """[(name, {n, x, y, w, h, active})] in exactly the order OCCAM masks them."""
    out = []
    for name, g in zip(_group_names(obs), _extract_object_groups(obs)):
        x = np.atleast_1d(np.asarray(g.x)).astype(np.int64)
        n = int(x.shape[0])

        def fit(a):
            arr = np.atleast_1d(np.asarray(a)).astype(np.int64)
            return np.broadcast_to(arr, (n,)) if arr.shape[0] != n else arr

        out.append((name, {"n": n, "x": x, "y": fit(g.y), "w": fit(g.width),
                           "h": fit(g.height), "active": fit(g.active)}))
    return out


def _obs_field_kinds(obs) -> List[Tuple[str, str]]:
    """[(field_name, kind)] for every top-level observation field."""
    names = ([f.name for f in dataclasses.fields(obs)] if dataclasses.is_dataclass(obs)
             else list(getattr(obs, "_fields", [])))
    kinds = []
    for name in names:
        v = getattr(obs, name, None)
        if isinstance(v, ObjectObservation):
            n = int(np.atleast_1d(np.asarray(v.x)).shape[0])
            kinds.append((name, f"ObjectObservation(n={n})"))
        elif v is None:
            kinds.append((name, "None"))
        else:
            a = np.asarray(v)
            kinds.append((name, f"array{tuple(a.shape)} {a.dtype}"))
    return kinds


def _sprite_names(game: str) -> List[str]:
    try:
        from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir
        d = os.path.join(get_base_sprite_dir(), game)
        if not os.path.isdir(d):
            return []
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".npy"))
    except Exception:
        return []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _sprite_matches(sprite: str, group_names: List[str]) -> bool:
    """Loose name match: digits/underscores stripped, substring either way."""
    s = _norm(sprite)
    if not s:
        return False
    for g in group_names:
        gn = _norm(g.split(".")[-1])
        if not gn:
            continue
        for a, b in ((s, gn), (s, gn.rstrip("s")), (s.rstrip("s"), gn)):
            if a and b and (a in b or b in a):
                return True
    return False


# --------------------------------------------------------------------------
# scenario mining
# --------------------------------------------------------------------------

def _rollout(env, key, steps: int):
    """Stacked trajectory of states, leading axis = steps. Autoresets on done."""
    aspace = env.action_space()
    _, s0 = env.reset(key)

    def body(state, k):
        ak, rk = jax.random.split(k)
        _, ns, _, done, _ = env.step(state, aspace.sample(ak))
        ns = jax.lax.cond(jnp.asarray(done).any(), lambda: env.reset(rk)[1], lambda: ns)
        return ns, ns

    _, traj = jax.lax.scan(body, s0, jax.random.split(key, steps))
    return s0, traj


def _phase_columns(traj) -> List[Tuple[str, np.ndarray]]:
    """Scalar int state fields that behave like a discrete phase (gamestate, level...)."""
    paths, _ = jax.tree_util.tree_flatten_with_path(traj)
    cols = []
    for path, leaf in paths:
        try:
            a = np.asarray(leaf)               # typed PRNG keys are not convertible
        except Exception:
            continue
        if a.ndim != 1:                        # 0-d per state -> (steps,) stacked
            continue
        if not (np.issubdtype(a.dtype, np.integer) or a.dtype == bool):
            continue
        if 2 <= np.unique(a).size <= MAX_PHASE_VALUES:
            name = ".".join(str(getattr(k, "name", getattr(k, "key", k))) for k in path)
            cols.append((name.lstrip("."), a))
    return cols


def mine_scenarios(env, key, steps: int, max_scenarios: int, bg_frames: int):
    """(scenarios, background_frames).

    A scenario is kept the first time the tuple (phase fields, per-group active
    counts) is seen, so every stage, level and spawn/despawn combination the
    random rollout reaches shows up exactly once.
    """
    s0, traj = _rollout(env, key, steps)
    obs_traj = jax.vmap(env._get_observation)(traj)

    phase = _phase_columns(traj)
    counts = []
    for name, g in zip(_group_names(obs_traj), _extract_object_groups(obs_traj)):
        act = np.asarray(g.active)
        counts.append((name, act.reshape(act.shape[0], -1).sum(-1).astype(np.int64)))

    sigs = (np.stack([c for _, c in phase] + [c for _, c in counts], axis=-1)
            if (phase or counts) else np.zeros((steps, 1), np.int64))

    seen, picks = set(), []
    for t in range(steps):
        s = tuple(int(v) for v in sigs[t])
        if s in seen:
            continue
        seen.add(s)
        picks.append(t)
        if len(picks) >= max_scenarios:
            break

    def label(t):
        parts = [f"{n}={int(c[t])}" for n, c in phase]
        act = [f"{n.split('.')[-1]}={int(c[t])}" for n, c in counts if int(c[t])]
        return (f"t={t}" + ("  " + " ".join(parts) if parts else "")
                + ("  |  active: " + " ".join(act) if act else "  |  active: -"))

    at = lambda t: jax.tree.map(lambda a: a[t], traj)
    scenarios = [("reset", s0)] + [(label(t), at(t)) for t in picks]

    stride = max(1, steps // max(bg_frames, 1))
    bg = [np.asarray(env.render(at(t)), np.uint8) for t in range(0, steps, stride)][:bg_frames]
    return scenarios, np.asarray(bg, np.uint8)


def load_poses(env, game: str, base_state):
    """Hand-written scenarios for games that ship them, else []."""
    entry = POSE_PROVIDERS.get(game)
    if entry is None:
        return []
    try:
        mod = importlib.import_module(entry[0])
        return list(getattr(mod, entry[1])(env, base_state))
    except Exception as e:
        print(f"[warn] pose provider {entry[0]}.{entry[1]} failed: {e!r}")
        return []


# --------------------------------------------------------------------------
# pixel ground truth
# --------------------------------------------------------------------------

def backdrop_model(frames: np.ndarray) -> Tuple[np.ndarray, float]:
    """(backdrop RGB, share of pixels it covers).

    The single most frequent colour over the sampled frames is the empty playfield.
    Deliberately NOT a per-pixel background: that one absorbs static level geometry
    (ladders, platforms, walls) into the background and then reports every one of
    their boxes as a ghost. What the checks need is "did the renderer put ink here",
    which is exactly "differs from the backdrop colour".
    """
    if frames.size == 0:
        return np.zeros(3, np.uint8), 0.0
    codes = (frames[..., 0].astype(np.uint32) << 16
             | frames[..., 1].astype(np.uint32) << 8 | frames[..., 2].astype(np.uint32))
    vals, cnts = np.unique(codes, return_counts=True)
    c = int(vals[int(np.argmax(cnts))])
    return (np.array([(c >> 16) & 255, (c >> 8) & 255, c & 255], np.uint8),
            float(cnts.max()) / float(codes.size))


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(mask.astype(np.int32), 0), 1), ((1, 0), (1, 0)))


def _box_frac(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Share of True pixels inside the on-screen part of the box; -1 if fully outside."""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return -1.0
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return float(s) / float((x1 - x0) * (y1 - y0))


def _blob_extent(fg: np.ndarray, x: int, y: int, w: int, h: int):
    """(drawn_w, drawn_h) of the sprite around this box, or None if inconclusive.

    The box is padded by half its own size and the ink inside that window is split
    into connected components; the component covering most of the box is the sprite.
    Taking the raw window extent instead would measure whatever else crosses the
    window -- in a platformer nearly every window contains a floor line. If the
    chosen component still reaches a window border it has merged with a neighbour
    (player standing on a platform, two enemies touching) and no honest extent can
    be read -- those instances are dropped rather than guessed.
    """
    H, W = fg.shape
    px, py = max(6, w // 2 + 2), max(6, h // 2 + 2)   # the window must not be derived
    x0, y0 = max(0, x - px), max(0, y - py)           # from the box alone -- a box that
    x1, y1 = min(W, x + w + px), min(H, y + h + py)   # is too small would hide the bug
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    win = fg[y0:y1, x0:x1]
    if not win.any():
        return None
    lab, n = ndimage.label(win, structure=np.ones((3, 3), bool))
    if n == 0:
        return None
    # every component touching a window border has merged with something outside and
    # is dropped; the sprite is the union of what is left, which keeps sprites drawn
    # from several disconnected parts together
    border = set(np.unique(np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    keep = np.isin(lab, [k for k in range(1, n + 1) if k not in border])
    if not keep.any():
        return None
    bx0, by0 = max(x, x0) - x0, max(y, y0) - y0
    if not keep[by0:min(y + h, y1) - y0, bx0:min(x + w, x1) - x0].any():
        return None                                   # nothing kept overlaps the box
    rows, cols = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
    return int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def run_checks(env, scenarios, bg_frames) -> Tuple[List[Finding], Dict]:
    findings: List[Finding] = []
    img_h, img_w = (int(v) for v in env.image_space().shape[:2])
    gnames = [n for n, _ in _groups_np(env._get_observation(scenarios[0][1]))]

    backdrop, share = backdrop_model(bg_frames)
    fg_ok = share >= 0.30

    acc = {n: dict(slots=0, active=0, offscreen=0, clipped=0, degenerate=0,
                   ghost=0, ghost_of=0, cov=[], shift_cov={}, size=[],
                   ex_off=[], ex_ghost=[]) for n in gnames}
    uncovered_acc = np.zeros((img_h, img_w), np.float64)
    n_fg_scen = 0

    for si, (title, state) in enumerate(scenarios):
        tag = f"[{si:02d}] {title}"          # same label the video row carries
        obs = env._get_observation(state)
        frame = np.asarray(env.render(state), np.uint8)
        fg = (frame != backdrop).any(-1) if fg_ok else None
        ii = _integral(fg) if fg is not None else None
        union = np.zeros((img_h, img_w), bool)

        for gname, g in _groups_np(obs):
            a = acc[gname]
            a["slots"] += g["n"]
            for i in range(g["n"]):
                if int(g["active"][i]) == 0:
                    continue
                a["active"] += 1
                x, y = int(g["x"][i]), int(g["y"][i])
                w, h = int(g["w"][i]), int(g["h"][i])

                if w <= 0 or h <= 0:
                    a["degenerate"] += 1
                    continue

                fully_out = x + w <= 0 or y + h <= 0 or x >= img_w or y >= img_h
                clipped = x < 0 or y < 0 or x + w > img_w or y + h > img_h
                if fully_out or clipped:
                    a["offscreen" if fully_out else "clipped"] += 1
                    if len(a["ex_off"]) < 3:
                        a["ex_off"].append(f"`({x},{y},{w},{h})` @ {tag}")

                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(img_w, x + w), min(img_h, y + h)
                if x1 > x0 and y1 > y0:
                    union[y0:y1, x0:x1] = True

                if ii is None or fully_out:
                    continue
                c0 = _box_frac(ii, x, y, w, h)
                a["cov"].append(c0)
                a["ghost_of"] += 1
                if c0 == 0.0:
                    a["ghost"] += 1
                    if len(a["ex_ghost"]) < 3:
                        a["ex_ghost"].append(f"`({x},{y},{w},{h})` @ {tag}")
                    continue
                if w >= 2 and h >= 2:
                    for dx, dy in ((-w // 2, -h // 2), (-w // 2, 0), (0, -h // 2)):
                        c = _box_frac(ii, x + dx, y + dy, w, h)
                        a["shift_cov"].setdefault((dx == -w // 2, dy == -h // 2), []) \
                            .append(max(c, 0.0))
                    ext = _blob_extent(fg, x, y, w, h)
                    if ext is not None:
                        a["size"].append((ext[0] - w, ext[1] - h, w, h))

        if fg is not None:
            uncovered_acc += (fg & ~union)
            n_fg_scen += 1

    # ---- assemble findings ------------------------------------------------
    for n in gnames:
        a = acc[n]
        if a["offscreen"]:
            findings.append(Finding("FAIL", "offscreen", n,
                f"{a['offscreen']} active box(es) fully outside the {img_w}x{img_h} frame "
                f"→ invisible in every mask. e.g. {'; '.join(a['ex_off'][:2])}"))
        if a["clipped"]:
            findings.append(Finding("WARN", "offscreen", n,
                f"{a['clipped']} active box(es) cross the frame edge (partly clipped). "
                f"e.g. {'; '.join(a['ex_off'][:2])}"))
        if a["degenerate"]:
            findings.append(Finding("FAIL", "degenerate", n,
                f"{a['degenerate']} active box(es) with width ≤ 0 or height ≤ 0"))
        if a["active"] == 0:
            findings.append(Finding("WARN", "never-active", n,
                "never active in any scenario — group cannot be verified from a rollout"))
        elif a["slots"] and a["active"] == a["slots"]:
            findings.append(Finding("INFO", "always-on", n,
                "active in every scenario — check it is really always drawn "
                "(cf. PR #313: groups not gated by scene/mode)"))

        if fg_ok and a["ghost_of"] >= MIN_SAMPLES:
            frac = a["ghost"] / a["ghost_of"]
            if frac >= GHOST_WARN:
                findings.append(Finding("WARN", "ghost", n,
                    f"{frac:.0%} of active boxes ({a['ghost']}/{a['ghost_of']}) cover zero "
                    f"drawn pixels → active flag not gated by what the renderer draws "
                    f"(cf. PR #313). e.g. {'; '.join(a['ex_ghost'][:2])}"))

        if fg_ok and len(a["cov"]) >= MIN_SAMPLES:
            c0 = float(np.mean(a["cov"]))
            for key, lbl in (((True, True), "(-w/2, -h/2)"),
                             ((True, False), "(-w/2, 0)"),
                             ((False, True), "(0, -h/2)")):
                vals = a["shift_cov"].get(key)
                if not vals:
                    continue
                cs = float(np.mean(vals))
                if c0 <= ORIGIN_MAX_COV and cs >= c0 + ORIGIN_GAIN:
                    findings.append(Finding("WARN", "origin", n,
                        f"foreground coverage {c0:.0%} as reported, {cs:.0%} when shifted by "
                        f"{lbl} → observation looks like sprite-center coords, the convention "
                        f"is top-left (cf. PR #309/#310)"))
                    break

        if fg_ok and len(a["size"]) >= MIN_SAMPLES:
            d = np.asarray(a["size"])
            dw, dh = int(np.median(d[:, 0])), int(np.median(d[:, 1]))
            tw = max(SIZE_TOL, SIZE_REL * float(np.median(d[:, 2])))
            th = max(SIZE_TOL, SIZE_REL * float(np.median(d[:, 3])))
            if abs(dw) > tw or abs(dh) > th:
                verb = "larger" if (dw > 0 or dh > 0) else "smaller"
                # deliberately INFO: a sprite drawn from disconnected parts (ladder
                # rungs, dashed lines) or one that always touches its neighbours makes
                # "the drawn extent" genuinely ambiguous. This is a pointer to check
                # in the video, not a verdict.
                findings.append(Finding("INFO", "size", n,
                    f"drawn sprite is median {dw:+d} px wide / {dh:+d} px tall relative to "
                    f"the box ({len(a['size'])} isolated instances) → sprite may be {verb} "
                    f"than the box says (category **D**) — confirm in the video, "
                    f"disconnected sprites read falsely here"))

    if fg_ok and n_fg_scen:
        u = uncovered_acc / n_fg_scen
        lab, nlab = ndimage.label(u >= UNCOVERED_MIN_FRAC)
        blobs = []
        for sl, k in zip(ndimage.find_objects(lab), range(1, nlab + 1)):
            m = lab[sl] == k
            if int(m.sum()) >= UNCOVERED_MIN_AREA:
                blobs.append((int(m.sum()), sl, float(u[sl][m].mean())))
        for area, sl, pers in sorted(blobs, reverse=True, key=lambda b: b[0])[:8]:
            ys, xs = sl
            where = f"`x={xs.start} y={ys.start} w={xs.stop - xs.start} h={ys.stop - ys.start}`"
            static = pers >= STATIC_BLOB
            kind = ("static → most likely HUD/score or background art" if static else
                    "transient → most likely an object the observation never exposes "
                    "(cf. PR #314)")
            findings.append(Finding("INFO" if static else "WARN", "uncovered", "",
                f"{area} px drawn but covered by no box, present in {pers:.0%} of "
                f"scenarios, {where} — {kind}"))
    elif not fg_ok:
        findings.append(Finding("INFO", "pixel-checks", "",
            f"no dominant backdrop colour (best one covers only {share:.0%} of pixels) — "
            f"`ghost`/`origin`/`size`/`uncovered` disabled; `offscreen` and the class "
            f"inventory are unaffected"))

    return findings, dict(acc=acc, gnames=gnames, backdrop=backdrop, share=share,
                          fg_ok=fg_ok, img_h=img_h, img_w=img_w)


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

CHECKS = ["offscreen", "degenerate", "ghost", "origin", "size", "uncovered",
          "never-active", "always-on"]


def summary_status(findings: List[Finding]) -> Dict[str, str]:
    out = {}
    for c in CHECKS:
        hits = [f for f in findings if f.check == c]
        out[c] = ("FAIL" if any(f.level == "FAIL" for f in hits) else
                  "WARN" if any(f.level == "WARN" for f in hits) else
                  "info" if hits else "ok")
    return out


def write_summary_table(rep: Report, findings: List[Finding], classes: dict):
    status = summary_status(findings)
    icon = {"FAIL": "🔴 FAIL", "WARN": "🟠 WARN", "info": "🔵 info", "ok": "🟢 ok"}
    rep("## Summary")
    rep()
    rep("| check | status | detail |")
    rep("|---|---|---|")
    for c in CHECKS:
        hits = [f for f in findings if f.check == c]
        detail = ", ".join(sorted({f.group for f in hits if f.group})) or \
                 (f"{len(hits)} region(s)" if hits else "—")
        rep(f"| `{c}` | {icon[status[c]]} | {detail} |")
    nm, un = len(classes["not_masked"]), len(classes["unmatched_sprites"])
    rep(f"| `classes` | {icon['WARN'] if (nm or un) else icon['ok']} | "
        f"{nm} field(s) not masked, {un} sprite(s) unmatched |")
    rep()


def write_findings(rep: Report, findings: List[Finding]):
    rep("## Findings")
    rep()
    if not findings:
        rep("_none_")
        rep()
        return
    for level, head in (("FAIL", "### 🔴 FAIL"), ("WARN", "### 🟠 WARN"),
                        ("INFO", "### 🔵 info")):
        block = [f for f in findings if f.level == level]
        if not block:
            continue
        rep(head)
        rep()
        for f in sorted(block, key=lambda f: (f.check, f.group)):
            rep(f.md())
        rep()


def write_group_table(rep: Report, ctx):
    acc, gnames = ctx["acc"], ctx["gnames"]
    rep("## Per-group table")
    rep()
    rep("`cov%` = share of the box area the renderer actually filled. "
        "`Δw/Δh` = drawn sprite extent minus box extent, median over isolated instances.")
    rep()
    rep("| group | slots | active | cov% | ghost% | Δw/Δh | offscr | clip | degen |")
    rep("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    thin = False
    for n in gnames:
        a = acc[n]
        few = 0 < len(a["cov"]) < MIN_SAMPLES
        thin |= few
        cov = f"{np.mean(a['cov']) * 100:.1f}" + ("~" if few else "") if a["cov"] else "—"
        gh = f"{a['ghost'] / a['ghost_of'] * 100:.1f}" if a["ghost_of"] else "—"
        if len(a["size"]) >= MIN_SAMPLES:
            d = np.asarray(a["size"])
            sz = f"{int(np.median(d[:, 0])):+d}/{int(np.median(d[:, 1])):+d}"
            sz += f" ({len(a['size'])})"
        else:
            sz = "—"
        rep(f"| `{n}` | {a['slots']} | {a['active']} | {cov} | {gh} | {sz} | "
            f"{a['offscreen']} | {a['clipped']} | {a['degenerate']} |")
    rep()
    if thin:
        rep(f"`~` fewer than {MIN_SAMPLES} active instances — `ghost`/`origin`/`size` "
            f"skipped for that group; raise `--steps` or add poses.")
        rep()


def write_classes(rep: Report, classes: dict):
    rep("## Object classes")
    rep()
    rep("### Observation fields the game exposes")
    rep()
    rep("| field | kind | masked by OCCAM |")
    rep("|---|---|---|")
    for n, k in classes["fields"]:
        rep(f"| `{n}` | `{k}` | {'yes' if k.startswith('ObjectObservation') else '**no**'} |")
    rep()
    rep(f"### Groups the object-centric wrapper emits ({len(classes['occam_groups'])})")
    rep()
    rep(", ".join(f"`{n}`" for n in classes["occam_groups"]) or "_none_")
    rep()
    rep("### Difference — exposed but **not** masked")
    rep()
    rep("No `ObjectObservation` → the agent cannot see these at all:")
    rep()
    rep(", ".join(f"`{n}`" for n in classes["not_masked"]) or "_none_")
    rep()
    if classes["sprites"]:
        rep(f"### Difference — sprites with no name-matching group ({len(classes['unmatched_sprites'])}"
            f" of {len(classes['sprites'])})")
        rep()
        rep("Drawn by the renderer, no group whose name matches. Score digits and "
            "background art land here legitimately — decide per entry:")
        rep()
        rep(", ".join(f"`{s}`" for s in classes["unmatched_sprites"]) or "_none_")
        rep()


def collect_classes(game: str, obs) -> dict:
    fields = _obs_field_kinds(obs)
    occam = _group_names(obs)
    sprites = _sprite_names(game)
    return dict(
        fields=fields,
        occam_groups=occam,
        not_masked=[n for n, k in fields if not k.startswith("ObjectObservation")],
        sprites=sprites,
        unmatched_sprites=[s for s in sprites if not _sprite_matches(s, occam)],
    )


# --------------------------------------------------------------------------
# per-game run
# --------------------------------------------------------------------------

def run_game(args) -> int:
    rep = Report()
    stamp = _dt.datetime.now()
    ts = stamp.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.out_root, args.game, f"{args.game}_{ts}")

    try:
        env, note = load_env(args.game)
    except Exception as e:
        print(f"!!! {args.game}: cannot load environment: {e!r}")
        return 1
    img_h, img_w = (int(v) for v in env.image_space().shape[:2])

    rep(f"# OCCAM verification — `{args.game}`")
    rep()
    rep(f"| | |")
    rep(f"|---|---|")
    rep(f"| run | {stamp.strftime('%Y-%m-%d %H:%M:%S')} |")
    rep(f"| frame | {img_w}×{img_h} |")
    rep(f"| seed / steps | {args.seed} / {args.steps} |")
    if note:
        rep(f"| note | {note.strip()} |")

    _, base_state = env.reset(jax.random.PRNGKey(args.seed))
    probe_obs = env._get_observation(base_state)
    classes = collect_classes(args.game, probe_obs)
    empty = not _extract_object_groups(probe_obs)

    result = dict(game=args.game, timestamp=stamp.isoformat(), frame=[img_w, img_h],
                  no_object_observation=empty, classes=classes,
                  backdrop=dict(rgb=[0, 0, 0], share=0.0, pixel_checks=False),
                  findings=[], summary={}, n_scenarios=0)

    if empty:
        rep(f"| scenarios | — |")
        rep()
        rep("## Summary")
        rep()
        rep("🔴 **FAIL** — the observation contains no `ObjectObservation`. OCCAM cannot "
            "mask this game at all; the objects have to be exposed first "
            "(category **C**).")
        rep()
        write_classes(rep, classes)
        result["summary"] = {c: "n/a" for c in CHECKS}
        _write_outputs(args, run_dir, rep, result, None)
        return 0

    scenarios, bg_frames = mine_scenarios(
        env, jax.random.PRNGKey(args.seed), args.steps, args.max_scenarios, args.bg_frames)
    poses = [] if args.no_poses else load_poses(env, args.game, base_state)
    scenarios = poses + scenarios
    rep(f"| scenarios | {len(poses)} hand-written + {len(scenarios) - len(poses)} mined |")

    findings, ctx = run_checks(env, scenarios, bg_frames)
    rep(f"| backdrop | `rgb{tuple(int(v) for v in ctx['backdrop'])}` on {ctx['share']:.0%} "
        f"of pixels → pixel checks **{'on' if ctx['fg_ok'] else 'off'}** |")
    rep()

    write_summary_table(rep, findings, classes)
    write_findings(rep, findings)
    write_group_table(rep, ctx)
    write_classes(rep, classes)

    result.update(
        n_scenarios=len(scenarios),
        backdrop=dict(rgb=[int(v) for v in ctx["backdrop"]], share=ctx["share"],
                      pixel_checks=bool(ctx["fg_ok"])),
        findings=[f.as_dict() for f in findings],
        summary=summary_status(findings),
        groups={n: dict(slots=ctx["acc"][n]["slots"], active=ctx["acc"][n]["active"],
                        cov=(float(np.mean(ctx["acc"][n]["cov"])) if ctx["acc"][n]["cov"]
                             else None))
                for n in ctx["gnames"]},
    )
    _write_outputs(args, run_dir, rep, result, (env, scenarios))
    return 0


def _write_outputs(args, run_dir, rep: Report, result: dict, video_ctx):
    if video_ctx is not None and not args.console_only:
        env, scenarios = video_ctx
        if args.obs_res:
            maskers = {m: OCCAMWrapper(AtariWrapper(load_env(args.game)[0]),
                                       mask_mode=m, game_name=args.game) for m in MODES}
        else:
            maskers = {m: _OCCAMViz(env, m) for m in MODES}
        os.makedirs(run_dir, exist_ok=True)
        png_dir = os.path.join(run_dir, "png")
        if args.png:
            os.makedirs(png_dir, exist_ok=True)

        frames = []
        for i, (title, state) in enumerate(scenarios):
            row = _row(env, maskers, state, f"[{i:02d}] {title}", obs_res=args.obs_res)
            frames.extend([row] * args.hold)
            if args.png:
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)
                iio.imwrite(os.path.join(png_dir, f"{i:02d}_{safe[:60]}.png"), row)
        path = os.path.join(run_dir, f"{args.game}_sweep.mp4")
        iio.imwrite(path, np.asarray(frames, np.uint8),
                    plugin="pyav", codec="libx264", fps=args.fps)
        rep("## Artifacts")
        rep()
        rep(f"- video: `{path}` ({len(scenarios)} scenarios, {len(frames)} frames)")
        if args.png:
            rep(f"- stills: `{png_dir}/`")
        rep()

    if args.no_report:
        return
    os.makedirs(run_dir, exist_ok=True)
    md = os.path.join(run_dir, "report.md")
    with open(md, "w") as fh:
        fh.write(rep.text())
    with open(os.path.join(run_dir, "report.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(f"\nreport: {md}")


# --------------------------------------------------------------------------
# cross-game summary
# --------------------------------------------------------------------------

def _latest_reports(out_root: str) -> List[dict]:
    """Newest report.json per game."""
    out = []
    for game_dir in sorted(os.listdir(out_root)) if os.path.isdir(out_root) else []:
        d = os.path.join(out_root, game_dir)
        if not os.path.isdir(d):
            continue
        runs = sorted(os.listdir(d), reverse=True)
        for r in runs:
            p = os.path.join(d, r, "report.json")
            if os.path.isfile(p):
                try:
                    with open(p) as fh:
                        rec = json.load(fh)
                    rec["_path"] = os.path.join(d, r)
                    out.append(rec)
                except Exception:
                    pass
                break
    return out


def classify(r: dict) -> Dict[str, str]:
    """category letter -> 'yes' | 'no' | '?' | 'n/a'."""
    if r.get("no_object_observation"):
        return dict(A="n/a", B="n/a", C="yes", D="n/a", E="n/a", F="n/a")
    f = r.get("findings", [])
    pix = r.get("backdrop", {}).get("pixel_checks", False)
    unknown = "no" if pix else "?"
    hit = lambda c: any(x["check"] == c and x["level"] in ("FAIL", "WARN") for x in f)
    transient = any(x["check"] == "uncovered" and x["level"] == "WARN" for x in f)
    # `size` is reported at INFO on purpose (see run_checks), so it is matched on the
    # check name alone -- D is a "go look at it" flag, not a verdict
    return dict(
        A="yes" if hit("origin") else unknown,
        B="yes" if transient else unknown,
        C="no",
        D="yes" if any(x["check"] == "size" for x in f) else unknown,
        E="yes" if hit("ghost") else unknown,
        F="n/a",
    )


def other_flags(r: dict) -> str:
    f = r.get("findings", [])
    out = []
    for c in ("offscreen", "degenerate", "never-active"):
        hits = [x for x in f if x["check"] == c and x["level"] in ("FAIL", "WARN")]
        if hits:
            groups = sorted({x["group"] for x in hits if x["group"]})
            out.append(f"{c}({', '.join(groups)})" if groups else c)
    return ", ".join(out) or "—"


def write_overall_summary(out_root: str) -> str | None:
    recs = _latest_reports(out_root)
    if not recs:
        print(f"no report.json found under {out_root}/")
        return None

    mark = {"yes": "🔴", "no": "🟢", "?": "⚪", "n/a": "▪"}
    L = []
    L.append("# OCCAM verification — overall summary")
    L.append("")
    L.append(f"{len(recs)} game report(s), newest run each, generated "
             f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    L.append("")

    L.append("## What this summary can and cannot tell you")
    L.append("")
    L.append("Read this first — the table below is only as good as these limits.")
    L.append("")
    L.append("### Not readable from the reports")
    L.append("")
    for k, (name, cover, why) in CATEGORIES.items():
        if cover == "none":
            L.append(f"- **{k} — {name}** — {why}. Always shown as ▪ below.")
    L.append("")
    L.append("### Only partially readable")
    L.append("")
    for k, (name, cover, why) in CATEGORIES.items():
        if cover == "partial":
            L.append(f"- **{k} — {name}** — {why}.")
    L.append("")
    L.append("### Readable, but heuristic (pixel evidence, not proof)")
    L.append("")
    for k, (name, cover, why) in CATEGORIES.items():
        if cover == "heuristic":
            L.append(f"- **{k} — {name}** — {why}.")
    L.append("")
    L.append("### Readable directly")
    L.append("")
    for k, (name, cover, why) in CATEGORIES.items():
        if cover == "full":
            L.append(f"- **{k} — {name}** — {why}.")
    L.append("")
    L.append("A ⚪ means the game has no dominant backdrop colour, so every pixel-based "
             "check was switched off and the category is simply unknown — not clean.")
    L.append("")

    L.append("## Per game")
    L.append("")
    L.append("| game | A coords | B not exposed | C obs empty | D size | E active | "
             "F geometry | other | verdict |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|:--:|---|---|")
    counts = {k: 0 for k in CATEGORIES}
    clean = []
    for r in sorted(recs, key=lambda r: r["game"]):
        c = classify(r)
        for k, v in c.items():
            counts[k] += (v == "yes")
        f = r.get("findings", [])
        bad = any(x["level"] in ("FAIL", "WARN") for x in f)
        if r.get("no_object_observation"):
            verdict = "**C — nothing exposed**"
        elif not bad:
            verdict = "🟢 Keine Probleme"
            clean.append(r["game"])
        else:
            verdict = ", ".join(k for k in "ABDE" if c[k] == "yes") or "see `other`"
        L.append(f"| `{r['game']}` | " + " | ".join(mark[c[k]] for k in "ABCDEF")
                 + f" | {other_flags(r)} | {verdict} |")
    L.append("")
    L.append("Legend: 🔴 detected · 🟢 clean · ⚪ not checkable for this game · "
             "▪ category not derivable at all")
    L.append("")

    L.append("## Category totals")
    L.append("")
    L.append("| category | games flagged | confidence |")
    L.append("|---|--:|---|")
    for k, (name, cover, _) in CATEGORIES.items():
        n = "—" if cover == "none" else str(counts[k])
        L.append(f"| **{k}** {name} | {n} | {cover} |")
    L.append(f"| Keine Probleme | {len(clean)} | full |")
    L.append("")
    if clean:
        L.append("Clean: " + ", ".join(f"`{g}`" for g in clean))
        L.append("")

    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "SUMMARY.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {path}")
    return path


# --------------------------------------------------------------------------

def run_all(args) -> int:
    games = all_game_names()
    print(f"### sweeping {len(games)} games: {', '.join(games)}\n")
    failed = []
    for i, g in enumerate(games, 1):
        print(f"\n{'#' * 70}\n### [{i}/{len(games)}] {g}\n{'#' * 70}")
        cmd = [sys.executable, os.path.abspath(__file__), g,
               "--steps", str(args.steps), "--max-scenarios", str(args.max_scenarios),
               "--bg-frames", str(args.bg_frames), "--seed", str(args.seed),
               "--hold", str(args.hold), "--fps", str(args.fps),
               "--out-root", args.out_root]
        for flag in ("console_only", "no_report", "png", "obs_res", "no_poses"):
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))
        try:
            if subprocess.run(cmd, timeout=args.timeout).returncode != 0:
                failed.append(g)
        except subprocess.TimeoutExpired:
            print(f"!!! {g}: timeout after {args.timeout}s")
            failed.append(g)
    print(f"\n{'#' * 70}")
    if failed:
        print(f"### {len(failed)} game(s) failed: {', '.join(failed)}")
    if not args.no_report:
        print("### building overall summary\n")
        write_overall_summary(args.out_root)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("game", nargs="?",
                   help="jaxatari game id, or 'all' for every game found")
    p.add_argument("--summarize", action="store_true",
                   help="only rebuild SUMMARY.md from existing reports")
    p.add_argument("--steps", type=int, default=2000, help="rollout steps to mine")
    p.add_argument("--max-scenarios", type=int, default=60)
    p.add_argument("--bg-frames", type=int, default=96,
                   help="frames used to estimate the backdrop colour")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-poses", action="store_true",
                   help="ignore a game's hand-written scenario list")
    p.add_argument("--hold", type=int, default=45, help="video frames per scenario")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--png", action="store_true", help="also write one PNG per scenario")
    p.add_argument("--obs-res", action="store_true",
                   help=f"show the real {OUT_H}x{OUT_W} agent input instead of native res")
    p.add_argument("--console-only", action="store_true",
                   help="produce only the report, no video/PNGs")
    p.add_argument("--no-report", action="store_true",
                   help="do not save the console output to a file")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-game timeout in seconds, only for 'all'")
    p.add_argument("--out-root", default=OUT_ROOT)
    args = p.parse_args()

    if args.summarize:
        write_overall_summary(args.out_root)
        return
    if not args.game:
        p.error("give a game id, 'all', or --summarize")
    if args.game == "all":
        raise SystemExit(run_all(args))
    raise SystemExit(run_game(args))


if __name__ == "__main__":
    main()
