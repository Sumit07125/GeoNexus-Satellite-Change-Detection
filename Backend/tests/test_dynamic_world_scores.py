import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_opening

t1 = np.array(Image.open("scratch_t1.png")).astype(np.float32) / 255.0
t2 = np.array(Image.open("scratch_t2.png")).astype(np.float32) / 255.0

r1, g1, b1 = t1[..., 0], t1[..., 1], t1[..., 2]
r2, g2, b2 = t2[..., 0], t2[..., 1], t2[..., 2]

# In Sentinel-2 RGB preview, optical channels are directly R, G, B
d_red = r2 - r1
d_green = g2 - g1
d_blue = b2 - b1
mag_vis = np.sqrt(d_red**2 + d_green**2 + d_blue**2)

bright1 = (r1 + g1 + b1) / 3.0
bright2 = (r2 + g2 + b2) / 3.0
d_bright = bright2 - bright1

# Redness & Flatness
redness1 = (r1 - b1) / (r1 + b1 + 1e-6)
redness2 = (r2 - b2) / (r2 + b2 + 1e-6)
d_redness = redness1 - redness2

flatness2 = np.maximum(np.abs(r2 - g2), np.maximum(np.abs(g2 - b2), np.abs(r2 - b2))) / (bright2 + 1e-6)

# Detect changes:
# Any pixel with significant optical change
is_changed = mag_vis >= 0.11

# Multi-class scoring (Dynamic World style):
# 1. Vegetation Score: Green dominance, high green relative to red/blue
score_veg = np.where(
    (g2 > r2 * 1.05) & (g2 > b2 * 1.02),
    (g2 - np.maximum(r2, b2)) / (bright2 + 1e-6) * 3.0 + (g2 - g1) * 2.0,
    -1.0
)

# 2. Urban / Structural Score: High brightness, brightness gain, neutral color (flatness), loss of soil redness
score_urban = np.where(
    (bright2 >= 0.36) & (d_bright >= 0.08) & (redness2 < 0.20),
    (d_bright * 3.0) + (bright2 * 1.5) + (1.0 - flatness2) * 1.0 + np.clip(d_redness * 2.0, 0, 2.0),
    -1.0
)

# 3. Soil Score: High redness (red dominant over blue), brightened or fallow
score_soil = np.where(
    (redness2 >= 0.20) & (d_bright >= 0.08),
    (redness2 * 3.0) + (d_bright * 2.0),
    -1.0
)

# Classify based on argmax among candidate pixels:
veg_raw = is_changed & (score_veg > score_urban) & (score_veg > score_soil) & (score_veg > 0.10)
urban_raw = is_changed & (score_urban > score_veg) & (score_urban > score_soil) & (score_urban > 0.30)
soil_raw = is_changed & (score_soil > score_veg) & (score_soil > score_urban) & (score_soil > 0.30)

def clean_spatial(mask, min_size=20):
    opened = binary_opening(mask, structure=np.ones((3, 3)))
    lbl, num = label(opened)
    if num == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    cnt = np.bincount(lbl.ravel())
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num + 1):
        if cnt[i] >= min_size:
            out[lbl == i] = 1
    return out

veg_clean = clean_spatial(veg_raw, min_size=20)
urban_clean = clean_spatial(urban_raw, min_size=20)
soil_clean = clean_spatial(soil_raw, min_size=25)

tot = veg_clean.sum() + urban_clean.sum() + soil_clean.sum()
print("=== DYNAMIC WORLD MULTI-CLASS RESULTS ===")
print("Vegetation:", veg_clean.sum(), f"({veg_clean.sum()/max(tot,1)*100:.1f}%)")
print("Urban:     ", urban_clean.sum(), f"({urban_clean.sum()/max(tot,1)*100:.1f}%)")
print("Soil:      ", soil_clean.sum(), f"({soil_clean.sum()/max(tot,1)*100:.1f}%)")

overlay = np.copy(t2)
alpha = 0.70
overlay[veg_clean > 0] = overlay[veg_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
overlay[urban_clean > 0] = overlay[urban_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
overlay[soil_clean > 0] = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_dynamic_world_overlay.png")
print("Saved scratch_dynamic_world_overlay.png successfully!")
