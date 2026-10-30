"""
inference.py — GeoNexus model inference engine
Loads the mh_fewshot_best_k30 checkpoint and runs bi-temporal
change detection on 17-channel numpy tensors from GEE.

Returns:
  - change_prob:  [H, W] float32 in [0,1]
  - change_mask:  [H, W] bool (thresholded)
  - overlay_b64:  base64 PNG of T2 RGB with color-coded change overlay
  - change_map_b64: base64 PNG of multi-class change colormap
"""

import os
import io
import sys
import math
import base64
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Add model directory to path so we can import p3_geonexus_model
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Model", "geonexus_v3_2_k30")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from p3_geonexus_model import GeoNexusCD  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Globals (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_norm_stats = None
_manifest = None
_device = None
EXPECTED_CHANNEL_ORDER = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]


def _load_model():
    global _model, _norm_stats, _manifest, _device

    if _model is not None:
        return

    model_dir = os.path.join(os.path.dirname(__file__),
                              os.environ.get("MODEL_DIR", "Model/geonexus_v3_2_k30"))

    checkpoint_name = os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth")
    norm_stats_name = os.environ.get("NORM_STATS_FILE", "norm_stats_trainonly.json")
    manifest_name = os.environ.get("MODEL_MANIFEST", "manifest.json")

    ckpt_path = os.path.join(model_dir, checkpoint_name)
    norm_path = os.path.join(model_dir, norm_stats_name)
    manifest_path = os.path.join(model_dir, manifest_name)

    # Load norm stats
    with open(norm_path) as f:
        _norm_stats = json.load(f)

    # Load manifest
    with open(manifest_path) as f:
        _manifest = json.load(f)

    if _manifest.get("input_channels") != 17 or _manifest.get("channel_order") != EXPECTED_CHANNEL_ORDER or _manifest.get("patch_size") != PATCH:
        raise ValueError("Model manifest does not match the runtime's 17-channel, 128-pixel input contract.")
    if len(_norm_stats.get("mean", [])) != 17 or len(_norm_stats.get("std", [])) != 17:
        raise ValueError("Model normalization file must provide 17 train-only channel means and standard deviations.")

    # Determine device
    device_pref = os.environ.get("DEVICE", "cpu").lower()
    if device_pref == "cuda" and torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    # Build and load model
    model = GeoNexusCD(use_derived=True, decoder="lka")
    state = torch.load(ckpt_path, map_location=_device)

    # Handle various checkpoint formats
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.set_mode("gated")
    model.eval()
    model.to(_device)

    _model = model
    print(f"[Inference] Model loaded: {checkpoint_name} on {_device}")
    print(f"[Inference] Threshold from manifest: {_manifest.get('threshold', 0.28)}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(tensor: np.ndarray) -> np.ndarray:
    """
    Convert the acquisition's float32 [17, H, W] tensor to normalized model input.

    Channels 0-10:  optical raw, stored as ρ×10000 → divide by 10000 then z-score
    Channels 11-12: NDVI, NDBI — already in [-1,1] — pass through
    Channels 13-15: SAR VV/VH/VH_VV, stored as dB×100 → divide by 100, then z-score
    Channel 16:     Q (clip(n_clear / 8, 0, 1)) — pass through
    """
    arr = tensor.astype(np.float32)
    scale = float(_norm_stats.get("array_scale", 10000.0))
    sar_scale = float(_norm_stats.get("sar_extra_scale", 100.0))
    mu = np.array(_norm_stats["mean"], dtype=np.float32)   # [17]
    sigma = np.array(_norm_stats["std"], dtype=np.float32)  # [17]
    if mu.shape != (17,) or sigma.shape != (17,) or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Model normalization statistics must contain 17 finite means and positive standard deviations.")

    # Rescale from stored integer
    arr[:11] = arr[:11] / scale          # optical: DN → reflectance
    arr[13:16] = arr[13:16] / sar_scale  # SAR: ×100 stored → dB

    # Z-score normalise (channels 11,12,16 have mu=0, std=1 → no-op)
    arr = (arr - mu[:, None, None]) / (sigma[:, None, None] + 1e-6)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Tiled inference for arbitrary image sizes
# ─────────────────────────────────────────────────────────────────────────────

PATCH = 128
STRIDE = 64


def _infer_tiled(t1: torch.Tensor, t2: torch.Tensor) -> np.ndarray:
    """
    Run sliding-window inference. t1/t2: [1, 17, H, W] on _device.
    Returns probability map [H, W] float32.
    """
    _, _, H, W = t1.shape
    prob_sum = torch.zeros(H, W, device=_device)
    count = torch.zeros(H, W, device=_device)

    ys = list(range(0, H - PATCH + 1, STRIDE))
    xs = list(range(0, W - PATCH + 1, STRIDE))

    # Ensure we cover the entire image
    if not ys or ys[-1] + PATCH < H:
        ys.append(max(0, H - PATCH))
    if not xs or xs[-1] + PATCH < W:
        xs.append(max(0, W - PATCH))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                p1 = t1[:, :, y:y+PATCH, x:x+PATCH]
                p2 = t2[:, :, y:y+PATCH, x:x+PATCH]
                out = _model(p1, p2)
                prob = torch.sigmoid(out["logits"]).squeeze()  # [PATCH, PATCH]
                prob_sum[y:y+PATCH, x:x+PATCH] += prob
                count[y:y+PATCH, x:x+PATCH] += 1.0

    count = count.clamp(min=1.0)
    return (prob_sum / count).cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

# Change intensity colour palette: dark blue → cyan → yellow → orange → red
CHANGE_PALETTE = np.array([
    [8,   48,  107],   # 0.0  – deep blue (no change)
    [8,   81,  156],   # 0.15
    [33,  113, 181],   # 0.30
    [66,  146, 198],   # 0.40
    [107, 174, 214],   # 0.50
    [255, 255, 102],   # 0.60  – yellow (moderate change)
    [253, 174, 97],    # 0.70  – orange
    [244, 109, 67],    # 0.80
    [215, 48,  39],    # 0.90
    [165, 15,  21],    # 1.0   – deep red (high change)
], dtype=np.float32)


def _prob_to_colormap(prob: np.ndarray) -> np.ndarray:
    """Map [H, W] probability [0,1] to [H, W, 3] uint8 RGB colormap."""
    idxf = prob * (len(CHANGE_PALETTE) - 1)
    idx0 = np.clip(np.floor(idxf).astype(int), 0, len(CHANGE_PALETTE) - 2)
    idx1 = idx0 + 1
    frac = (idxf - idx0)[..., None]
    rgb = (1 - frac) * CHANGE_PALETTE[idx0] + frac * CHANGE_PALETTE[idx1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _classify_surface_dynamics(
    t1_tensor: np.ndarray,
    t2_tensor: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Classify detected change pixels into 4 segregated physical surface dynamics:
      - Water Dynamics (Blue: [37, 99, 235] / #2563eb)
      - Vegetation Dynamics (Green: [34, 197, 94] / #22c55e)
      - Urban & Built-up (Vivid Yellow: [234, 179, 8] / #eab308) - runways, concrete, asphalt, buildings
      - Bare Soil & Agriculture (Warm Terracotta / Orange: [249, 115, 22] / #f97316) - earthworks, sand, exposed soil
    Returns:
      semantic_rgb: [H, W, 3] uint8
      dynamics_stats: dict
    """
    H, W = mask.shape
    t1_f = t1_tensor.astype(np.float32)
    t2_f = t2_tensor.astype(np.float32)

    # Channels: 0=Blue, 1=Green, 2=Red, 6=NIR, 9=SWIR1
    blue2, green2, red2 = t2_f[0], t2_f[1], t2_f[2]
    nir1, nir2 = t1_f[6], t2_f[6]
    red1 = t1_f[2]
    swir1 = t1_f[9] if t1_f.shape[0] > 9 else red1
    swir2 = t2_f[9] if t2_f.shape[0] > 9 else red2

    # Spectral indices
    ndwi2 = (green2 - nir2) / (green2 + nir2 + 1e-6)
    mndwi2 = (green2 - swir2) / (green2 + swir2 + 1e-6)
    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-6)
    ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-6)
    d_ndvi = ndvi2 - ndvi1

    ndbi1 = (swir1 - nir1) / (swir1 + nir1 + 1e-6)
    ndbi2 = (swir2 - nir2) / (swir2 + nir2 + 1e-6)
    d_ndbi = ndbi2 - ndbi1

    # Spectral redness / soil ratio & brightness
    bright1 = (t1_f[0] + t1_f[1] + t1_f[2]) / 3.0
    bright2 = (blue2 + green2 + red2) / 3.0
    d_bright = bright2 - bright1

    flatness2 = np.maximum(
        np.abs(red2 - green2),
        np.maximum(np.abs(green2 - blue2), np.abs(red2 - blue2))
    ) / (bright2 + 1e-6)
    redness2 = (red2 - blue2) / (red2 + blue2 + 1e-6)

    # Initialize semantic RGB with dark background [10, 15, 29]
    semantic_rgb = np.full((H, W, 3), [10, 15, 29], dtype=np.uint8)

    # Exclude invalid/nodata border pixels from change detection
    valid_mask = (t1_f[:3].sum(0) > 30) & (t2_f[:3].sum(0) > 30)
    change_bool = (mask > 0) & valid_mask
    changed_count = int(change_bool.sum())

    if changed_count == 0:
        return semantic_rgb, {
            "water": {"pct": 0.0, "ha": 0.0, "color": "#2563eb", "label": "Water Dynamics"},
            "vegetation": {"pct": 0.0, "ha": 0.0, "color": "#22c55e", "label": "Vegetation Dynamics"},
            "urban": {"pct": 0.0, "ha": 0.0, "color": "#eab308", "label": "Urban & Structural"},
            "bare_soil": {"pct": 0.0, "ha": 0.0, "color": "#f97316", "label": "Soil & Agriculture"},
        }

    # 1. Water dynamics: real water bodies, rivers, reservoirs, flood
    water_mask = change_bool & (
        ((ndwi2 > 0.04) & (nir2 < 1400)) | (mndwi2 > 0.08)
    )

    # 2. Vegetation dynamics: strict directional transition (new lawn/growth or deforestation)
    # Never flag a permanent tree that was already present in T1!
    veg_gain = (
        ((d_ndvi >= 0.16) & (ndvi2 >= 0.25)) |
        ((green2 > red2 * 1.08) & (green2 > blue2 * 1.05))
    )
    veg_loss = (d_ndvi <= -0.18) & (ndvi1 >= 0.32)
    veg_mask = change_bool & (~water_mask) & (veg_gain | veg_loss)

    # 3. Urban & Structural: concrete runways, asphalt, tarmac, buildings, white/metal roofs
    # Note: in inference.py, t1_f/t2_f are int16 (reflectance * 10000)
    built_up_gain = (
        # Brightness increase (>= 350 DN or >= 0.035 reflectance) with neutral profile
        ((d_bright >= 350) & (bright2 >= 1200) & (flatness2 < 0.32) & (redness2 < 0.22)) |
        # High optical difference on low vegetation with neutral spectrum
        ((bright2 >= 1200) & (flatness2 < 0.28) & (redness2 < 0.20) & (ndvi2 < 0.26)) |
        # Classical NDBI built-up gain
        ((d_ndbi >= 0.08) & (d_bright >= 200) & (ndvi2 < 0.24))
    )
    urban_mask = change_bool & (~water_mask) & (~veg_mask) & (ndvi2 < 0.28) & built_up_gain

    # 4. Bare Soil & Agriculture: mineral soils, excavated earth, sand, fallow land
    soil_mask = change_bool & (~water_mask) & (~veg_mask) & (~urban_mask)

    # Colorize each segregated dynamic
    semantic_rgb[water_mask] = [37, 99, 235]   # Royal Blue
    semantic_rgb[veg_mask]   = [34, 197, 94]   # Vivid Green
    semantic_rgb[urban_mask] = [234, 179, 8]   # Vivid Yellow (Urban & Structural, Runways, Concrete, Asphalt)
    semantic_rgb[soil_mask]  = [249, 115, 22]  # Warm Terracotta / Earth Orange (Bare Soil & Agriculture, Earth, Sand)

    w_cnt = int(water_mask.sum())
    v_cnt = int(veg_mask.sum())
    u_cnt = int(urban_mask.sum())
    s_cnt = int(soil_mask.sum())

    pixel_area_ha = 0.01  # 10m x 10m = 0.01 ha
    dynamics_stats = {
        "water": {
            "pct": round(w_cnt / changed_count * 100, 1),
            "ha": round(w_cnt * pixel_area_ha, 1),
            "pixels": w_cnt,
            "color": "#2563eb",
            "label": "Water Dynamics",
        },
        "vegetation": {
            "pct": round(v_cnt / changed_count * 100, 1),
            "ha": round(v_cnt * pixel_area_ha, 1),
            "pixels": v_cnt,
            "color": "#22c55e",
            "label": "Vegetation Dynamics",
        },
        "urban": {
            "pct": round(u_cnt / changed_count * 100, 1),
            "ha": round(u_cnt * pixel_area_ha, 1),
            "pixels": u_cnt,
            "color": "#eab308",
            "label": "Urban & Structural",
        },
        "bare_soil": {
            "pct": round(s_cnt / changed_count * 100, 1),
            "ha": round(s_cnt * pixel_area_ha, 1),
            "pixels": s_cnt,
            "color": "#f97316",
            "label": "Soil & Agriculture",
        },
    }

    return semantic_rgb, dynamics_stats


def _make_binary_mask(mask: np.ndarray) -> str:
    """Generate high-contrast binary change mask (white=change, black=no change)."""
    bin_u8 = (mask.astype(np.uint8) * 255)
    return _arr_to_b64(bin_u8)


def _make_overlay(rgb_b64: str, semantic_rgb: np.ndarray, mask: np.ndarray, prob: np.ndarray) -> str:
    """
    Overlay the multicolor semantic change dynamics onto the clear T2 optical image.
    Unchanged pixels: 100% natural, clear satellite image.
    Changed pixels: vivid semantic dynamic color blended with the underlying satellite texture.
    """
    img_bytes = base64.b64decode(rgb_b64)
    rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    H, W = mask.shape
    rgb_resized = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS)).astype(np.float32)
    semantic_f = semantic_rgb.astype(np.float32)

    # Dynamic alpha on changed pixels: weighted by model confidence
    alpha = np.clip(prob * 0.45 + 0.40, 0.45, 0.85) * mask.astype(np.float32)
    alpha3 = alpha[..., None]

    overlay = rgb_resized * (1 - alpha3) + semantic_f * alpha3
    return _arr_to_b64(np.clip(overlay, 0, 255).astype(np.uint8))


def _arr_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    t1_tensor: np.ndarray,
    t2_tensor: np.ndarray,
    t2_rgb_b64: str,
    threshold: float = None,
) -> dict:
    """
    Run GeoNexus change detection and physical dynamics decomposition.

    Args:
        t1_tensor:   [17, H, W] int16 numpy array (before)
        t2_tensor:   [17, H, W] int16 numpy array (after)
        t2_rgb_b64:  base64 PNG of T2 RGB image (for overlay)
        threshold:   binarization threshold (default: from manifest)

    Returns dict with:
        change_mask_b64:   base64 PNG of multicolor semantic change mask
        binary_mask_b64:   base64 PNG of binary change mask (black/white)
        change_prob_b64:   base64 PNG of smooth probability heatmap
        overlay_b64:       base64 PNG of T2 with change overlay
        dynamics:          dict with surface dynamics classification breakdown
        stats:             dict with change_pct, mean_prob, etc.
    """
    _load_model()

    for name, tensor in (("t1_tensor", t1_tensor), ("t2_tensor", t2_tensor)):
        if not isinstance(tensor, np.ndarray) or tensor.ndim != 3 or tensor.shape[0] != 17:
            raise ValueError(f"{name} must have shape [17, H, W] to match the trained model contract.")
        if not np.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")
    if t1_tensor.shape != t2_tensor.shape or min(t1_tensor.shape[1:]) < PATCH:
        raise ValueError("T1 and T2 inputs must have identical spatial dimensions of at least 128×128.")

    threshold_override = threshold is not None
    env_threshold = os.environ.get("MODEL_THRESHOLD")
    if threshold is None:
        threshold = float(env_threshold if env_threshold is not None else _manifest.get("threshold", 0.28))
    if not np.isfinite(threshold) or not 0 < float(threshold) < 1:
        raise ValueError("AI probability threshold must be between 0 and 1.")

    # Normalize
    t1_norm = _normalize(t1_tensor)  # [17, H, W] float32
    t2_norm = _normalize(t2_tensor)

    # To torch tensor [1, 17, H, W]
    t1_t = torch.from_numpy(t1_norm).unsqueeze(0).to(_device)
    t2_t = torch.from_numpy(t2_norm).unsqueeze(0).to(_device)

    # Tiled inference
    prob = _infer_tiled(t1_t, t2_t)   # [H, W] float32

    # Threshold (strictly excluding nodata / swath border pixels)
    valid_mask = (t1_tensor[:3].sum(0) > 30) & (t2_tensor[:3].sum(0) > 30)
    mask = ((prob >= threshold) & valid_mask).astype(np.uint8)  # [H, W] 0/1

    # ── Visual outputs ────────────────────────────────────────────────────────

    # 1. Semantic typing / physical dynamics multicolor mask
    semantic_rgb, dynamics_stats = _classify_surface_dynamics(t1_tensor, t2_tensor, mask, prob)
    multicolor_mask_b64 = _arr_to_b64(semantic_rgb)

    # 2. Binary Change Mask (pure black & white)
    binary_mask_b64 = _make_binary_mask(mask)

    # 3. T2 + Multicolor Semantic Overlay (T2 visible, change areas blended)
    overlay_b64 = _make_overlay(t2_rgb_b64, semantic_rgb, mask, prob)

    # 4. Probability heatmap (full-image continuous intensity)
    heat_rgb = _prob_to_colormap(prob)
    alpha_ch = np.clip(prob * 255, 30, 255).astype(np.uint8)[..., None]
    heat_rgba = np.concatenate([heat_rgb, alpha_ch], axis=-1)
    change_prob_b64 = _arr_to_b64(heat_rgba)

    # ── Statistics ────────────────────────────────────────────────────────────
    total_px = int(valid_mask.sum()) if valid_mask.sum() > 0 else mask.size
    changed_px = int(mask.sum())
    change_pct = round(changed_px / total_px * 100, 2)

    # Area estimate (rough, at 10m resolution)
    H, W = prob.shape
    pixel_area_ha = 0.01  # 10m × 10m = 100m² = 0.01 ha
    changed_area_ha = round(changed_px * pixel_area_ha, 1)

    stats = {
        "change_pct": change_pct,
        "changed_pixels": changed_px,
        "total_pixels": total_px,
        "changed_area_ha": changed_area_ha,
        "mean_prob": round(float(prob.mean()), 4),
        "max_prob": round(float(prob.max()), 4),
        "threshold": threshold,
        "output_size": list(prob.shape),
    }

    return {
        "model_version": _manifest.get("version", "unknown"),
        "threshold": float(threshold),
        "threshold_source": "request override" if threshold_override else "MODEL_THRESHOLD environment setting" if env_threshold is not None else _manifest.get("threshold_source", "model manifest"),
        "change_prob_b64": change_prob_b64,
        "change_mask_b64": multicolor_mask_b64,
        "binary_mask_b64": binary_mask_b64,
        "overlay_b64": overlay_b64,
        "dynamics": dynamics_stats,
        "stats": stats,
    }

