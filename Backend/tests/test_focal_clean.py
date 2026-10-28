import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_opening, binary_closing

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

# Color flatness
flatness2 = np.maximum(
    np.abs(r2 - g2),
    np.maximum(np.abs(g2 - b2), np.abs(r2 - b2))
) / (bright2 + 1e-6)

redness1 = (r1 - b1) / (r1 + b1 + 1e-6)
redness2 = (r2 - b2) / (r2 + b2 + 1e-6)

# Lawn
is_lawn = (g2 > r2 * 1.08) & (g2 > b2 * 1.05) & (mag_vis > 0.12)

# Building (Concrete, metal roof, structural expansion)
is_building = (
    (~is_lawn) &
    (bright2 >= 0.44) &
    (d_bright >= 0.15) &
    (flatness2 < 0.20) &
    (redness2 < 0.16)
)

# Soil (exposed ground, earthwork)
is_soil = (
    (~is_lawn) &
    (~is_building) &
    (d_bright >= 0.14) &
    (redness2 >= 0.18) &
    (bright2 >= 0.35)
)

def clean_spatial(mask, min_size=35):
    # 3x3 structuring element removes thin line noise / road boundaries
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

lawn_clean = clean_spatial(is_lawn, min_size=25)
bldg_clean = clean_spatial(is_building, min_size=35)
soil_clean = clean_spatial(is_soil, min_size=40)

print(f"Lawn: {lawn_clean.sum()} px")
print(f"Building: {bldg_clean.sum()} px")
print(f"Soil: {soil_clean.sum()} px")

overlay = np.copy(t2)
alpha = 0.70
overlay[lawn_clean > 0] = overlay[lawn_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
overlay[bldg_clean > 0] = overlay[bldg_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
overlay[soil_clean > 0] = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_focal_overlay.png")
print("Saved scratch_focal_overlay.png")
