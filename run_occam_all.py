#!/usr/bin/env python3
"""
Run ALL FOUR OCCAM mask variants for one game, then stitch a single shareable
summary video that shows everything we did for that game.

Why a launcher (and not one training run): the four variants
(object / binary / class / planes) have *different observation shapes* and
therefore train *different policies* -> they are four separate runs. This script
runs each variant as its own `uv run python main.py` subprocess (clean JAX/wandb
state per run), then combines their final eval clips with
`build_occam_summary_video`.

Usage (the env var caps VRAM and is inherited by every subprocess):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python run_occam_all.py ENV_ID=pong

Any extra KEY=VALUE pairs are forwarded to main.py as Hydra overrides, e.g.:
    ... uv run python run_occam_all.py ENV_ID=seaquest TOTAL_TIMESTEPS=2000000 NUM_ENVS=16
    ... uv run python run_occam_all.py ENV_ID=pong VARIANTS=binary,class   # subset

Outputs:
    <SAVE_ROOT>/<ENV_ID>/<variant>/      per-variant checkpoints + eval clips
    <SAVE_ROOT>/<ENV_ID>/summary_<ENV_ID>.mp4   <-- the shareable video
"""
import os
import sys
import subprocess
import time

from agents.occam.occam import build_occam_summary_video, MASK_MODES

# keys handled by THIS script (not forwarded verbatim to main.py)
_LAUNCHER_KEYS = {"ENV_ID", "SAVE_ROOT", "VARIANTS", "MASK_MODE", "SAVE_PATH", "EXP_NAME"}


def parse_overrides(argv):
    ov = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            ov[k] = v
    return ov


def main():
    ov = parse_overrides(sys.argv[1:])
    env_id = ov.get("ENV_ID", "pong")
    save_root = ov.get("SAVE_ROOT", "./models")
    variants = [m.strip() for m in ov.get("VARIANTS", ",".join(MASK_MODES)).split(",") if m.strip()]
    forwarded = [f"{k}={v}" for k, v in ov.items() if k not in _LAUNCHER_KEYS]

    for mm in variants:
        save_path = os.path.join(save_root, env_id, mm)
        cmd = [
            "uv", "run", "python", "main.py", "--config-name", "occam",
            f"ENV_ID={env_id}", f"MASK_MODE={mm}",
            f"SAVE_PATH={save_path}", f"EXP_NAME=occam_{env_id}_{mm}",
            *forwarded,
        ]
        print("\n=== training variant:", mm, "===")
        print(" ", " ".join(cmd), flush=True)
        result = subprocess.run(cmd)  # inherits env (incl. XLA_PYTHON_CLIENT_MEM_FRACTION)
        if result.returncode != 0:
            print(f"[warn] variant '{mm}' exited with code {result.returncode}; continuing.", flush=True)

    print("\n=== building shareable summary video (+ W&B summary run) ===", flush=True)

    # resolve the W&B project/entity for the summary run (CLI override > occam.yaml)
    project, entity = ov.get("PROJECT"), ov.get("ENTITY")
    if project is None or entity is None:
        try:
            import yaml
            with open(os.path.join("config", "occam.yaml")) as f:
                cfg = yaml.safe_load(f) or {}
            project = project if project is not None else cfg.get("PROJECT", "occam-jaxtari")
            entity = entity if entity is not None else cfg.get("ENTITY", "")
        except Exception:
            project = project or "occam-jaxtari"
            entity = entity or ""

    steps = ov.get("TOTAL_TIMESTEPS", "default")
    path, nframes = build_occam_summary_video(
        env_id, save_root, mods=None,            # auto-discovers eval mods
        wandb_project=project, wandb_entity=entity,
        wandb_tags=["summary", env_id, f"steps:{steps}"],
        wandb_run_name=f"summary_{env_id}_{int(time.time())}",
    )
    if path:
        print(f"summary written to: {path}  ({nframes} frames)")
        print(f"uploaded to W&B project '{project}' as a 'summary'-tagged run")
    else:
        print("[warn] no per-variant eval clips found - did the runs save them? "
              "(check the save_dir wiring in save_and_eval / OCCAM video block)")


if __name__ == "__main__":
    main()