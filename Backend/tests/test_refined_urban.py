import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, generate_binary_structure

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

# Color saturation / neutrality in T2:
# For concrete, metal, white roofs: Red, Green, Blue are very close to each other:
# spectral flatness = max(|R-G|, |G-B|, |R-B|) / (Brightness + 1e-6)
flatness2 = np.maximum(
    np.abs(r2 - g2),
    np.maximum(np.abs(g2 - b2), np.abs(r2 - b2))
) / (bright2 + 1e-6)

# Soil in T1 had high redness:
redness1 = (r1 - b1) / (r1 + b1 + 1e-6)
redness2 = (r2 - b2) / (r2 + b2 + 1e-6)

# Lawn (Vegetation growth):
is_lawn = (g2 > r2 * 1.08) & (g2 > b2 * 1.05) & (mag_vis > 0.12)

# Structural / Urban emergence:
# 1. New roofs / concrete must be bright in T2 (bright2 >= 0.40)
# 2. Significant brightness increase (d_bright >= 0.14)
# 3. Neutral spectral profile (flatness2 < 0.22 - not red soil, not green tree!)
# 4. Redness must have dropped from soil (redness2 < 0.15)
is_building = (
    (~is_lawn) &
    (bright2 >= 0.42) &
    (d_bright >= 0.14) &
    (flatness2 < 0.22) &
    (redness2 < 0.18)
)

# Soil / Agriculture (bare soil exposure or fallow ground that brightened but remains red/brown):
is_soil = (
    (~is_lawn) &
    (~is_building) &
    (d_bright >= 0.12) &
    (redness2 >= 0.18) &
    (bright2 >= 0.30)
)

def filter_mmu(mask, min_size=12):
    lbl, num = label(mask)
    if num == 0:
        return mask
    cnt = np.bincount(lbl.ravel())
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num + 1):
        if cnt[i] >= min_size:
            out[lbl == i] = 1
    return out

lawn_clean = filter_mmu(is_lawn, min_size=15)
bldg_clean = filter_mmu(is_building, min_size=15)
soil_clean = filter_mmu(is_soil, min_size=20)

print("Lawn px:", lawn_clean.sum())
print("Building px:", bldg_clean.sum())
print("Soil px:", soil_clean.sum())

overlay = np.copy(t2)
alpha = 0.70
# Green for lawn [34, 197, 94]
overlay[lawn_clean > 0] = overlay[lawn_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
# Yellow for building [234, 179, 8]
overlay[bldg_clean > 0] = overlay[bldg_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
# Warm Terracotta for soil [249, 115, 22]
overlay[soil_clean > 0] = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_overlay2.png")
print("Saved scratch_overlay2.png")
