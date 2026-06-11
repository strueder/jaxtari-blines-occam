"""Quick mask sanity-check for Breakout.
Resets the env, renders [clean | object | binary | class | planes]
side by side as a single PNG — no trained policy needed."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import imageio.v2 as imageio
import jaxatari
from agents.occam.occam import _OCCAMViz

ENV_ID = "breakout"
MODES  = ["object", "binary", "class", "planes"]
OUT    = f"./models/{ENV_ID}/mask_check.png"

# --- setup ---
base_env = jaxatari.make(ENV_ID)
vizzes   = {m: _OCCAMViz(base_env, m) for m in MODES}
_, state = base_env.reset(jax.random.PRNGKey(0))

def _label(img, text):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.ascontiguousarray(img))
        ImageDraw.Draw(im).text((3, 2), text, fill=(255, 255, 0))
        return np.asarray(im)
    except Exception:
        return img

# --- render all modes ---
clean = np.asarray(base_env.render(state), dtype=np.uint8)
obs   = base_env._get_observation(state)

print(f"observation: {obs}")          # dump structure for quick sanity-check

panels = [_label(clean, "clean")]
for m in MODES:
    mask = np.asarray(
        vizzes[m]._mask_rgb(jnp.asarray(clean), obs),
        dtype=np.uint8,
    )
    panels.append(_label(mask, m))

row = np.concatenate(panels, axis=1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
imageio.imwrite(OUT, row)
print(f"wrote {OUT}  (shape {row.shape}, {len(panels)} panels)")