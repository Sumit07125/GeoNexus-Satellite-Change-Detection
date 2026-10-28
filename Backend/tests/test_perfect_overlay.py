import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_opening

t1 = np.array(Image.open("scratch_t1.png")).astype(np.float32) / 255.0
t2 = np.array(Image.open("scratch_t2.png")).astype(np.float32) / 255.0

r1, g1, b1 = t1[..., 0], t1[..., 1], t1[..., 2]
r2, g2, b2 = t2[..., 0], t2[..., 1], t2[..., 2]

d_red = r2 - r1
d_green = g2 - g1
d_blue = b2 - b1
mag_vis = np.sqrt(d_red**2 + d_green**2 + d_blue**2)

bright1 = (r1 + g1 + b1) / 3.0
bright2 = (r2 + g2 + b2) / 3.0
d_bright = bright2 - bright1

redness1 = (r1 - b1) / (r1 + b1 + 1e-6)
redness2 = (r2 - b2) / (r2 + b2 + 1e-6)
d_redness = redness1 - redness2

flatness2 = np.maximum(np.abs(r2 - g2), np.maximum(np.abs(g2 - b2), np.abs(r2 - b2))) / (bright2 + 1e-6)

# Significant physical change
mu = float(mag_vis.mean())
sigma = float(mag_vis.std())
# Cutoff: 1.8 standard deviations above regional mean
cutoff = max(mu + 1.8 * sigma, 0.15)
is_cand = (mag_vis >= cutoff)

# 1. Vegetation Dynamics (Green: #22c55e)
# Strong green dominance: lawn, grass, agricultural bloom
is_veg = is_cand & (
    (g2 > r2 * 1.08) & (g2 > b2 * 1.05) & (d_green >= 0.08)
)

# 2. Urban & Structural Dynamics (Yellow: #eab308)
# Concrete, metal roofing, building expansion, paved surfaces:
# High brightness in T2, significant brightness increase, neutral spectral profile (not red, not green)
is_urban = is_cand & (~is_veg) & (
    (bright2 >= 0.44) &
    (d_bright >= 0.14) &
    (flatness2 < 0.20) &
    (redness2 < 0.16)
)

# 3. Bare Soil & Agriculture (Warm Terracotta: #f97316)
# Soil exposure, newly plowed land, earthworks:
is_soil = is_cand & (~is_veg) & (~is_urban) & (
    (redness2 >= 0.20) &
    (d_bright >= 0.12) &
    (bright2 >= 0.35)
)

def clean_spatial(mask, min_size=30):
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

veg_clean = clean_spatial(is_veg, min_size=25)
urban_clean = clean_spatial(is_urban, min_size=30)
soil_clean = clean_spatial(is_soil, min_size=35)

tot = veg_clean.sum() + urban_clean.sum() + soil_clean.sum()
print("=== PERFECT BALANCED RESULTS ===")
print("Vegetation:", veg_clean.sum(), f"({veg_clean.sum()/max(tot,1)*100:.1f}%)")
print("Urban:     ", urban_clean.sum(), f"({urban_clean.sum()/max(tot,1)*100:.1f}%)")
print("Soil:      ", soil_clean.sum(), f"({soil_clean.sum()/max(tot,1)*100:.1f}%)")

overlay = np.copy(t2)
alpha = 0.70
overlay[veg_clean > 0] = overlay[veg_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
overlay[urban_clean > 0] = overlay[urban_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
overlay[soil_clean > 0] = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_perfect_overlay.png")
print("Saved scratch_perfect_overlay.png successfully!")
