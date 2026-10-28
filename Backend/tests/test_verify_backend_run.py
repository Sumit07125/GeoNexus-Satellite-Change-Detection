import base64
import numpy as np
from PIL import Image
import io

from spectral_cd import run_spectral_inference

data = np.load("scratch_tensors.npz")
t1_raw = data["t1"]
t2_raw = data["t2"]

# Load t2_rgb_b64 from scratch_t2.png
img2 = Image.open("scratch_t2.png")
buf = io.BytesIO()
img2.save(buf, format="PNG")
t2_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

result = run_spectral_inference(
    t1_tensor=t1_raw,
    t2_tensor=t2_raw,
    t2_rgb_b64=t2_b64,
)

print("=== FINAL SPECTRAL ENGINE TEST RUN ===")
print("Overall Change %:", result["stats"]["change_pct"], "%")
print("Changed Pixels:  ", result["stats"]["changed_pixels"])
print("Dynamics breakdown:")
for k, v in result["dynamics"].items():
    print(f"  - {v['label']} ({k}): {v['pct']}% ({v['pixels']} px, {v['ha']} ha) [color: {v['color']}]")

# Save overlay to scratch_final_result.png
overlay_bytes = base64.b64decode(result["overlay_b64"])
Image.open(io.BytesIO(overlay_bytes)).save("scratch_final_result.png")
print("Saved scratch_final_result.png successfully!")
