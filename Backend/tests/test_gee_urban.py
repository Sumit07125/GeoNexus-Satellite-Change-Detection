import numpy as np
from PIL import Image

t1 = np.array(Image.open("scratch_t1.png")).astype(np.float32) / 255.0
t2 = np.array(Image.open("scratch_t2.png")).astype(np.float32) / 255.0

r1, g1, b1 = t1[..., 0], t1[..., 1], t1[..., 2]
r2, g2, b2 = t2[..., 0], t2[..., 1], t2[..., 2]

# Multi-spectral vector difference on visible reflectance
d_red = r2 - r1
d_green = g2 - g1
d_blue = b2 - b1

mag_vis = np.sqrt(d_red**2 + d_green**2 + d_blue**2)

# Brightness
bright1 = (r1 + g1 + b1) / 3.0
bright2 = (r2 + g2 + b2) / 3.0
d_bright = bright2 - bright1

# Redness ratio (Soil is red/brown: r1 > 1.35 * b1; concrete/metal is neutral/grey: r2 < 1.30 * b2)
redness1 = r1 / (b1 + 1e-4)
redness2 = r2 / (b2 + 1e-4)
d_redness = redness1 - redness2  # positive when red soil converts to grey building

# Vegetation greenness in T2 (e.g. lawn)
# Lawn has high green relative to red and blue
is_lawn = (g2 > r2 * 1.05) & (g2 > b2 * 1.05) & (mag_vis > 0.12)

# Building / Structural emergence:
# 1. High brightness increase (d_bright >= 0.10) OR high visible magnitude (mag_vis >= 0.18)
# 2. Brightness in T2 is substantial (bright2 >= 0.35)
# 3. Not vegetation (not is_lawn, and g2 not dominant)
# 4. Color neutrality or redness reduction (soil to concrete/roof)
is_building = (
    (~is_lawn) &
    (
        ((d_bright >= 0.10) & (bright2 >= 0.35)) |
        ((mag_vis >= 0.16) & (bright2 >= 0.32) & (d_redness >= 0.15))
    )
)

print("Lawn detected pixels:", is_lawn.sum())
print("Building detected pixels:", is_building.sum())

total = is_lawn.sum() + is_building.sum()
print("Lawn pct:", round(is_lawn.sum() / total * 100, 1), "%")
print("Building pct:", round(is_building.sum() / total * 100, 1), "%")
