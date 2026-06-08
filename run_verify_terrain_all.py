"""Deterministic terrain sweep across all 4 mask modes side by side.
Forces level mode for every terrain bank and renders
[clean | object | binary | class | planes] in one row, so all worlds'
terrain decompositions can be verified in every mode — no trained policy needed."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import imageio.v2 as imageio
import jaxatari
from agents.occam.occam import _OCCAMViz

ENV_ID = "gravitar"
MODES  = ["object", "binary", "class", "planes"]
BANKS  = [1, 2, 3, 4, 5]          # 0 = map (no terrain)
HOLD   = 45                        # frames held per bank (~1.5 s @ 30 fps)
OUT    = f"./models/{ENV_ID}/terrain_sweep_all_modes.mp4"

base_env = jaxatari.make(ENV_ID)
vizzes = {m: _OCCAMViz(base_env, m) for m in MODES}
_, base_state = base_env.reset(jax.random.PRNGKey(0))

def _label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.ascontiguousarray(img))
        ImageDraw.Draw(im).text((3, 2), text, fill=(255, 255, 0))
        return np.asarray(im)
    except Exception:
        return img

frames = []
for b in BANKS:
    state_b = base_state.replace(mode=jnp.int32(1), terrain_bank_idx=jnp.int32(b))
    clean = np.asarray(base_env.render(state_b), dtype=np.uint8)        # (H, W, 3)
    obs = base_env._get_observation(state_b)
    panels = [_label(clean, f"bank {b} | clean")]
    for m in MODES:
        mask = np.asarray(vizzes[m]._mask_rgb(jnp.asarray(clean), obs), dtype=np.uint8)
        panels.append(_label(mask, m))
    row = np.concatenate(panels, axis=1)                               # (H, 5W, 3)
    frames.extend([row] * HOLD)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
imageio.mimwrite(OUT, frames, fps=30, macro_block_size=1)
print(f"wrote {OUT}  ({len(frames)} frames, banks {BANKS}, modes {MODES})")