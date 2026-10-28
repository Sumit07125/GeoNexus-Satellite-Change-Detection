#!/usr/bin/env python3
"""
Geo-Nexus / MH-DAPT-CD v3.2
Robust OSCD -> 128x128 bi-temporal 17-channel patch preparation.

Project contract
----------------
Per timestamp the model input is exactly 17 channels:
  0..10  : B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
  11     : NDVI = (B8 - B4) / (B8 + B4 + eps)
  12     : NDBI = (B11 - B8) / (B11 + B8 + eps)
  13     : VV placeholder = 0 (OSCD is optical-only in Geo-Nexus)
  14     : VH placeholder = 0
  15     : cross-ratio placeholder = 0
  16     : quality q = 1

OSCD-specific rules
-------------------
- Official OSCD has 24 bi-temporal Sentinel-2 city pairs and 13 bands per date.
- The Geo-Nexus frozen supervised split is 11 train / 3 validation / 10 test.
- Validation is carved only from the official 14 training cities.
- Official test cities remain test; in particular, Montpellier is NOT validation.
- Prefer imgs_1_rect / imgs_2_rect only when all 13 bands exist.
- Otherwise fall back to imgs_1 / imgs_2 and explicitly resample every band to
  the B4 (10 m) reference grid.
- Native 20 m and 60 m bands are therefore aligned before stacking.
- Original labels may be 1=no-change, 2=change; convert to binary 0/1.
- Store model tensors as int16 after multiplying float32 values by 10000.
- Train patches use stride 64; validation/test use stride 128.
- Only training metadata contains a 3x oversampling index for patches with
  change fraction > 1%. Validation and test are never oversampled.

This file is intentionally self-contained. It does not import project-local
modules, so it can be copied into the Geo_Watch pipeline directory safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# -----------------------------------------------------------------------------
# Frozen Geo-Nexus OSCD contract
# -----------------------------------------------------------------------------
PATCH = 128
TRAIN_STRIDE = 64
EVAL_STRIDE = 128
ARRAY_SCALE = 10000.0
EPS = 1e-6

BANDS_13 = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12"
)
KEEP = (
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
)

OFFICIAL_TRAIN_14 = {
    "abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux",
    "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa", "rennes", "saclay_e"
}
OFFICIAL_TEST_10 = {
    "brasilia", "chongqing", "dubai", "lasvegas", "milano", "montpellier",
    "norcia", "rio", "saclay_w", "valencia"
}
EXPECTED_ALL_CITIES = OFFICIAL_TRAIN_14 | OFFICIAL_TEST_10

EXPECTED_PROJECT = {
    "train_11": {
        "abudhabi", "aguasclaras", "beihai", "bercy", "bordeaux",
        "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa"
    },
    "val_3": {"beirut", "rennes", "saclay_e"},
    "test_10": set(OFFICIAL_TEST_10),
}

CHANNEL_ORDER = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
    "NDVI", "NDBI", "VV_placeholder", "VH_placeholder", "cross_ratio_placeholder", "quality_q"
]

BAND_RE = re.compile(r"(?:^|[_\-.])B(0*\d+|0*8A)$", re.IGNORECASE)


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    crs_wkt: str | None
    transform: Affine


@dataclass(frozen=True)
class SceneInfo:
    city: str
    timestamp: int
    folder: Path
    rectified: bool
    reference_band: Path
    grid: Grid


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def canonical_band(name: str) -> str | None:
    """Normalize B01/B1/B08/B8/B8A/B08A-like filenames."""
    stem = Path(name).stem.strip().upper()
    if stem == "B8A" or stem == "B08A":
        return "B8A"

    # Allow names such as TILE_B04 or *_B04.
    match = re.search(r"(?:^|[_\-.])B(0*\d+)$", stem)
    if not match:
        return None
    token = match.group(1).lstrip("0") or "0"
    return f"B{token}"


def find_band(folder: Path, band: str) -> Path | None:
    """Find a band despite zero-padding, case, or common filename prefixes."""
    if not folder.is_dir():
        return None

    wanted = band.upper()
    # Fast exact candidates first.
    for name in (f"{wanted}.tif", f"{wanted}.TIF"):
        p = folder / name
        if p.is_file():
            return p

    # Case-insensitive / zero-padding tolerant scan.
    candidates: list[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            if canonical_band(p.name) == wanted:
                candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name.lower())[0]


def list_scene_dirs(city_root: Path, timestamp: int) -> tuple[Path | None, bool]:
    """Prefer complete *_rect; otherwise use native directory."""
    rect = city_root / f"imgs_{timestamp}_rect"
    plain = city_root / f"imgs_{timestamp}"

    if rect.is_dir():
        missing = [b for b in BANDS_13 if find_band(rect, b) is None]
        if not missing:
            return rect, True

    if plain.is_dir():
        return plain, False

    return None, False


def read_grid(path: Path) -> Grid:
    with rasterio.open(path) as src:
        return Grid(
            height=int(src.height),
            width=int(src.width),
            crs_wkt=src.crs.to_wkt() if src.crs else None,
            transform=src.transform,
        )


def same_grid(a: Grid, b: Grid, atol: float = 1e-7) -> bool:
    if a.height != b.height or a.width != b.width:
        return False
    if (a.crs_wkt or "") != (b.crs_wkt or ""):
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a.transform, b.transform))


def grid_from_dataset(src: rasterio.io.DatasetReader) -> Grid:
    return Grid(
        height=int(src.height),
        width=int(src.width),
        crs_wkt=src.crs.to_wkt() if src.crs else None,
        transform=src.transform,
    )


def finite_stats(a: np.ndarray) -> dict:
    finite = np.isfinite(a)
    if not finite.any():
        return {"finite_fraction": 0.0}
    vals = a[finite]
    return {
        "finite_fraction": float(finite.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# OSCD roots and split manifest
# -----------------------------------------------------------------------------
def resolve_oscd_subdirs(root: Path) -> tuple[Path, Path, Path]:
    """
    Accept both observed layouts:

      root/Images
      root/Train Labels
      root/Test Labels

    and the official archive-directory names:

      root/Onera Satellite Change Detection dataset - Images
      root/Onera Satellite Change Detection dataset - Train Labels
      root/Onera Satellite Change Detection dataset - Test Labels
    """
    root = root.resolve()
    if not root.is_dir():
        fail(f"OSCD root does not exist or is not a directory: {root}")

    image_candidates = [
        root / "Images",
        root / "images",
        root / "Onera Satellite Change Detection dataset - Images",
        root / "Onera Satellite Change Detection Dataset - Images",
    ]
    train_candidates = [
        root / "Train Labels",
        root / "TrainLabels",
        root / "train_labels",
        root / "Onera Satellite Change Detection dataset - Train Labels",
        root / "Onera Satellite Change Detection Dataset - Train Labels",
    ]
    test_candidates = [
        root / "Test Labels",
        root / "TestLabels",
        root / "test_labels",
        root / "Onera Satellite Change Detection dataset - Test Labels",
        root / "Onera Satellite Change Detection Dataset - Test Labels",
    ]

    def first_existing(candidates: Sequence[Path]) -> Path | None:
        for p in candidates:
            if p.is_dir():
                return p
        return None

    images = first_existing(image_candidates)
    train_labels = first_existing(train_candidates)
    test_labels = first_existing(test_candidates)

    missing = []
    if images is None:
        missing.append("Images")
    if train_labels is None:
        missing.append("Train Labels")
    if test_labels is None:
        missing.append("Test Labels")
    if missing:
        fail(
            "Could not resolve the OSCD subdirectories under:\n"
            f"  {root}\n"
            f"Missing: {', '.join(missing)}\n"
            "Expected either Images/Train Labels/Test Labels or the official archive names."
        )

    return images, train_labels, test_labels


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        fail(f"Split manifest not found: {path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8"))

    # Current Geo-Nexus schema.
    if isinstance(data.get("project"), dict):
        p = data["project"]
        keys = ("train_11", "val_3", "test_10")
        if all(k in p for k in keys):
            splits = {
                "train": list(p["train_11"]),
                "val": list(p["val_3"]),
                "test": list(p["test_10"]),
            }
        else:
            fail("Manifest has a 'project' section but not train_11/val_3/test_10.")
    # Compatibility with the older generated manifest used earlier.
    elif all(k in data for k in ("train", "val", "test")):
        splits = {
            "train": list(data["train"]),
            "val": list(data["val"]),
            "test": list(data["test"]),
        }
    else:
        fail(
            "Unsupported split manifest schema. Expected either project.train_11/val_3/test_10 "
            "or top-level train/val/test."
        )

    for key in ("train", "val", "test"):
        if not isinstance(splits[key], list) or not all(isinstance(x, str) for x in splits[key]):
            fail(f"Manifest split '{key}' must be a list of city names.")
        splits[key] = [x.strip().lower() for x in splits[key] if x.strip()]

    return splits


def validate_project_split(splits: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in splits.items()}
    if not (len(splits["train"]) == 11 and len(splits["val"]) == 3 and len(splits["test"]) == 10):
        fail(
            "Geo-Nexus v3.2 requires exactly 11 train / 3 validation / 10 test cities, "
            f"but got {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}."
        )

    if any(len(sets[k]) != len(splits[k]) for k in sets):
        fail("Split manifest contains duplicate city names.")

    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        fail("Split manifest contains train/val/test city overlap.")

    expected = EXPECTED_PROJECT
    matches_expected = (
        sets["train"] == expected["train_11"]
        and sets["val"] == expected["val_3"]
        and sets["test"] == expected["test_10"]
    )
    if not matches_expected:
        details = {
            "train_extra": sorted(sets["train"] - expected["train_11"]),
            "train_missing": sorted(expected["train_11"] - sets["train"]),
            "val_extra": sorted(sets["val"] - expected["val_3"]),
            "val_missing": sorted(expected["val_3"] - sets["val"]),
            "test_extra": sorted(sets["test"] - expected["test_10"]),
            "test_missing": sorted(expected["test_10"] - sets["test"]),
        }
        fail(
            "The supplied split manifest does not match the frozen Geo-Nexus v3.2 11/3/10 split:\n"
            + json.dumps(details, indent=2)
        )


# -----------------------------------------------------------------------------
# Reading images and labels
# -----------------------------------------------------------------------------
def describe_scene(city_root: Path, timestamp: int) -> SceneInfo:
    folder, rectified = list_scene_dirs(city_root, timestamp)
    if folder is None:
        contents = sorted(p.name for p in city_root.iterdir()) if city_root.is_dir() else []
        fail(
            f"{city_root.name}: missing imgs_{timestamp}_rect and imgs_{timestamp}.\n"
            f"City directory: {city_root}\n"
            f"Contents: {contents}"
        )

    ref = find_band(folder, "B4")
    if ref is None:
        found = sorted(p.name for p in folder.iterdir() if p.is_file())
        fail(f"{city_root.name}: B4 reference band missing in {folder}. Found: {found[:30]}")

    missing = [b for b in BANDS_13 if find_band(folder, b) is None]
    if missing:
        fail(
            f"{city_root.name}: missing required Sentinel-2 bands for t{timestamp}: {missing}\n"
            f"Directory: {folder}"
        )

    return SceneInfo(
        city=city_root.name,
        timestamp=timestamp,
        folder=folder,
        rectified=rectified,
        reference_band=ref,
        grid=read_grid(ref),
    )


def read_band_to_grid(path: Path, target_grid: Grid) -> np.ndarray:
    """Read one band and align it exactly to target_grid."""
    with rasterio.open(path) as src:
        src_grid = grid_from_dataset(src)

        # Fast path: exact grid match.
        if same_grid(src_grid, target_grid):
            data = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            return data

        # A CRS is required for a true reprojection. If both are missing, a pure
        # shape mismatch is still safely resampled only when this is an identity-like
        # raster (e.g. PNG-derived TIFF with no georeferencing).
        if src.crs is None or target_grid.crs_wkt is None:
            if src.height == target_grid.height and src.width == target_grid.width:
                data = src.read(1, masked=True).astype(np.float32).filled(np.nan)
                return data
            fail(
                f"Cannot align non-georeferenced raster {path} from {src.height}x{src.width} "
                f"to {target_grid.height}x{target_grid.width}; both CRS/grid metadata are required."
            )

        destination = np.full((target_grid.height, target_grid.width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs_wkt,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return destination


def load_optical(city_root: Path, timestamp: int, target_grid: Grid | None = None) -> tuple[np.ndarray, SceneInfo, Grid]:
    scene = describe_scene(city_root, timestamp)
    grid = target_grid or scene.grid
    arrays = []
    for band in KEEP:
        p = find_band(scene.folder, band)
        assert p is not None
        a = read_band_to_grid(p, grid)
        if not np.isfinite(a).all():
            stats = finite_stats(a)
            fail(
                f"{city_root.name}: non-finite values after alignment for t{timestamp} {band}. "
                f"Stats: {stats}"
            )
        arrays.append(a / ARRAY_SCALE)

    stack = np.stack(arrays, axis=0).astype(np.float32)
    return stack, scene, grid


def find_label_file(label_root: Path, city: str) -> Path:
    cm = label_root / city / "cm"
    if not cm.is_dir():
        fail(f"{city}: label directory missing: {cm}")

    tifs = sorted([p for p in cm.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])
    pngs = sorted([p for p in cm.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if tifs:
        return tifs[0]
    if pngs:
        return pngs[0]
    fail(f"{city}: no .tif/.tiff/.png label found in {cm}")


def read_label(label_root: Path, city: str, target_grid: Grid) -> np.ndarray:
    path = find_label_file(label_root, city)
    with rasterio.open(path) as src:
        src_grid = grid_from_dataset(src)
        values = src.read(1, masked=True)
        source = values.astype(np.float32).filled(np.nan)

        if same_grid(src_grid, target_grid):
            a = source
        elif src.crs is not None and target_grid.crs_wkt is not None:
            destination = np.full((target_grid.height, target_grid.width), np.nan, dtype=np.float32)
            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=target_grid.transform,
                dst_crs=target_grid.crs_wkt,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            a = destination
        elif source.shape == (target_grid.height, target_grid.width):
            a = source
        else:
            fail(
                f"{city}: label {path.name} is {source.shape}, target image grid is "
                f"{target_grid.height, target_grid.width}, and georeferencing is unavailable for reprojection."
            )

    if not np.isfinite(a).all():
        fail(f"{city}: label contains non-finite/nodata pixels after alignment: {path}")

    vals = set(int(v) for v in np.unique(a))
    if vals.issubset({1, 2}):
        binary = (a - 1).astype(np.uint8)
    elif vals.issubset({0, 1}):
        binary = a.astype(np.uint8)
    elif vals.issubset({0, 255}):
        binary = (a > 0).astype(np.uint8)
    else:
        fail(
            f"{city}: unsupported label values {sorted(vals)} in {path}. "
            "Expected {1,2}, {0,1}, or {0,255}."
        )

    if not np.isin(binary, [0, 1]).all():
        fail(f"{city}: normalized label is not binary after conversion.")
    return binary


# -----------------------------------------------------------------------------
# Feature construction and tiling
# -----------------------------------------------------------------------------
def to_17_channels(optical11: np.ndarray) -> np.ndarray:
    if optical11.ndim != 3 or optical11.shape[0] != 11:
        fail(f"Expected optical array shape [11,H,W], got {optical11.shape}")
    if not np.isfinite(optical11).all():
        fail("Optical tensor contains non-finite values before feature construction.")

    b4 = optical11[2]
    b8 = optical11[6]
    b11 = optical11[9]

    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndbi = (b11 - b8) / (b11 + b8 + EPS)

    sar = np.zeros((3, optical11.shape[1], optical11.shape[2]), dtype=np.float32)
    q = np.ones((1, optical11.shape[1], optical11.shape[2]), dtype=np.float32)

    out = np.concatenate(
        [optical11, ndvi[None], ndbi[None], sar, q],
        axis=0,
    ).astype(np.float32)

    if out.shape[0] != 17 or not np.isfinite(out).all():
        fail(f"17-channel construction failed: shape={out.shape}")
    return out


def starts(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        return []
    return list(range(0, length - patch + 1, stride))


def count_patches(height: int, width: int, stride: int) -> int:
    return len(starts(height, PATCH, stride)) * len(starts(width, PATCH, stride))


def iter_patches(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    city: str,
    stride: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, dict]]:
    H, W = y.shape
    if x1.shape != (17, H, W) or x2.shape != (17, H, W):
        fail(f"{city}: image/label grid mismatch: x1={x1.shape}, x2={x2.shape}, y={y.shape}")

    for row in starts(H, PATCH, stride):
        for col in starts(W, PATCH, stride):
            p1 = x1[:, row:row + PATCH, col:col + PATCH]
            p2 = x2[:, row:row + PATCH, col:col + PATCH]
            py = y[row:row + PATCH, col:col + PATCH]
            if p1.shape != (17, PATCH, PATCH) or p2.shape != (17, PATCH, PATCH) or py.shape != (PATCH, PATCH):
                fail(f"{city}: unexpected patch shape at row={row}, col={col}")
            if not np.isfinite(p1).all() or not np.isfinite(p2).all():
                fail(f"{city}: non-finite values in patch row={row}, col={col}")

            meta = {
                "city": city,
                "row": int(row),
                "col": int(col),
                "patch": PATCH,
                "stride": int(stride),
                "chg_frac": float(py.mean()),
            }
            yield np.stack([p1, p2], axis=0), py.astype(np.uint8), meta


def scaled_int16(x: np.ndarray) -> tuple[np.ndarray, int, int, float, float]:
    """Serialize a patch using the frozen int16 x10000 storage contract.

    OSCD source TIFFs can contain a small number of numerically large pixels.
    The project contract is still int16 after x10000, so saturation is applied
    only at the final storage gate and is explicitly audited in the manifest.
    No clipping is used during feature construction.
    """
    scaled = np.rint(x * ARRAY_SCALE)
    if not np.isfinite(scaled).all():
        fail("Non-finite value encountered during int16 conversion.")
    low = scaled < -32768.0
    high = scaled > 32767.0
    clipped = int(low.sum() + high.sum())
    total = int(scaled.size)
    min_v = float(scaled.min()) if scaled.size else 0.0
    max_v = float(scaled.max()) if scaled.size else 0.0
    if clipped:
        scaled = np.clip(scaled, -32768.0, 32767.0)
    return scaled.astype(np.int16), clipped, total, min_v, max_v


# -----------------------------------------------------------------------------
# Two-pass split preparation (memory-safe)
# -----------------------------------------------------------------------------
def inspect_city(city_root: Path, label_root: Path, city: str) -> dict:
    """Load enough data to validate alignment and get exact output grid size."""
    root = city_root / city
    if not root.is_dir():
        fail(f"Split references city '{city}', but directory is missing: {root}")

    opt1, scene1, grid1 = load_optical(root, 1)
    opt2, scene2, grid2 = load_optical(root, 2, target_grid=grid1)
    label = read_label(label_root, city, grid1)

    if opt1.shape != opt2.shape:
        fail(f"{city}: t1/t2 optical shapes differ after alignment: {opt1.shape} vs {opt2.shape}")
    if opt1.shape[1:] != label.shape:
        fail(f"{city}: image shape {opt1.shape[1:]} does not match label shape {label.shape}")

    x1 = to_17_channels(opt1)
    x2 = to_17_channels(opt2)

    return {
        "city": city,
        "grid": {
            "height": int(label.shape[0]),
            "width": int(label.shape[1]),
        },
        "rect_t1": bool(scene1.rectified),
        "rect_t2": bool(scene2.rectified),
        "change_fraction": float(label.mean()),
        "patch_count_train": count_patches(label.shape[0], label.shape[1], TRAIN_STRIDE),
        "patch_count_eval": count_patches(label.shape[0], label.shape[1], EVAL_STRIDE),
        "_x1": x1,
        "_x2": x2,
        "_label": label,
    }


def prepare_split(
    split: str,
    cities: list[str],
    images_root: Path,
    label_root: Path,
    staging: Path,
) -> dict:
    stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE
    print(f"\n{'=' * 78}\nPROCESSING {split.upper()} | cities={len(cities)} | stride={stride}\n{'=' * 78}")

    infos = []
    total = 0
    for city in cities:
        info = inspect_city(images_root, label_root, city)
        n = info["patch_count_train"] if split == "train" else info["patch_count_eval"]
        total += n
        print(
            f"  {city:12s} grid={tuple(info['grid'].values())} "
            f"change={info['change_fraction'] * 100:7.3f}% "
            f"rect(t1,t2)=({info['rect_t1']},{info['rect_t2']}) patches={n}"
        )
        infos.append(info)

    if total == 0:
        fail(f"No {split} patches would be produced. Check scene dimensions and PATCH={PATCH}.")

    x_path = staging / f"oscd_{split}.npy"
    y_path = staging / f"oscd_{split}_label.npy"
    meta_path = staging / f"oscd_{split}_meta.json"

    # open_memmap writes a real .npy file incrementally instead of holding the
    # entire split in RAM.
    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.int16,
        shape=(total, 2, 17, PATCH, PATCH),
    )
    Y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, PATCH, PATCH),
    )

    metadata: list[dict] = []
    cursor = 0
    clip_pixels = 0
    clip_values = 0
    split_raw_min = float("inf")
    split_raw_max = float("-inf")
    for info in infos:
        city = info["city"]
        x1, x2, label = info["_x1"], info["_x2"], info["_label"]
        for patch_x, patch_y, meta in iter_patches(x1, x2, label, city, stride):
            stored_x, n_clipped, n_values, raw_min, raw_max = scaled_int16(patch_x)
            X[cursor] = stored_x
            Y[cursor] = patch_y
            clip_pixels += n_clipped
            clip_values += n_values
            split_raw_min = min(split_raw_min, raw_min)
            split_raw_max = max(split_raw_max, raw_max)
            metadata.append(meta)
            cursor += 1

        # Release large per-city arrays before moving to the next city.
        del info["_x1"], info["_x2"], info["_label"]

    X.flush()
    Y.flush()
    del X, Y

    if cursor != total or len(metadata) != total:
        fail(f"Internal patch-count error for {split}: wrote {cursor}, expected {total}")

    write_json(meta_path, metadata)

    if split == "train":
        pos = [i for i, m in enumerate(metadata) if m["chg_frac"] > 0.01]
        neg = [i for i, m in enumerate(metadata) if m["chg_frac"] <= 0.01]
        sampler = {
            "sample_index": pos * 3 + neg,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "oversample_factor": 3,
            "positive_rule": "chg_frac > 0.01",
            "train_only": True,
        }
        write_json(staging / "oscd_train_sampler.json", sampler)
    else:
        pos = [i for i, m in enumerate(metadata) if m["chg_frac"] > 0.01]

    print(f"  -> wrote {total} patches")
    print(f"  -> change-containing patches (>1%): {len(pos)}")
    print(f"  -> int16 saturation: {clip_pixels:,}/{clip_values:,} values ({(100.0 * clip_pixels / clip_values if clip_values else 0.0):.6f}%)")
    print(f"  -> X: {x_path.name}  shape={(total, 2, 17, PATCH, PATCH)} dtype=int16")
    print(f"  -> y: {y_path.name}  shape={(total, PATCH, PATCH)} dtype=uint8")

    return {
        "cities": cities,
        "patch_count": total,
        "stride": stride,
        "x_shape": [total, 2, 17, PATCH, PATCH],
        "y_shape": [total, PATCH, PATCH],
        "change_patch_count_gt_1pct": len(pos),
        "storage_saturation": {
            "clipped_values": int(clip_pixels),
            "total_values": int(clip_values),
            "clipped_fraction": float(clip_pixels / clip_values) if clip_values else 0.0,
            "scaled_raw_min": None if split_raw_min == float("inf") else float(split_raw_min),
            "scaled_raw_max": None if split_raw_max == float("-inf") else float(split_raw_max),
            "policy": "final_int16_clip_and_audit",
        },
    }


# -----------------------------------------------------------------------------
# Output verification and CLI
# -----------------------------------------------------------------------------
def verify_outputs(staging: Path, summary: dict) -> None:
    expected = [
        "oscd_train.npy", "oscd_train_label.npy", "oscd_train_meta.json", "oscd_train_sampler.json",
        "oscd_val.npy", "oscd_val_label.npy", "oscd_val_meta.json",
        "oscd_test.npy", "oscd_test_label.npy", "oscd_test_meta.json",
        "oscd_preprocess_manifest.json",
    ]
    missing = [name for name in expected if not (staging / name).exists()]
    if missing:
        fail(f"Output integrity check failed; missing files: {missing}")

    for split in ("train", "val", "test"):
        X = np.load(staging / f"oscd_{split}.npy", mmap_mode="r")
        Y = np.load(staging / f"oscd_{split}_label.npy", mmap_mode="r")
        if X.shape != tuple(summary["splits"][split]["x_shape"]):
            fail(f"{split}: X shape mismatch after write: {X.shape}")
        if Y.shape != tuple(summary["splits"][split]["y_shape"]):
            fail(f"{split}: y shape mismatch after write: {Y.shape}")
        if X.dtype != np.int16 or Y.dtype != np.uint8:
            fail(f"{split}: dtype mismatch: X={X.dtype}, Y={Y.dtype}")
        if X.shape[2] != 17 or X.shape[3:] != (PATCH, PATCH):
            fail(f"{split}: channel/patch shape invalid: {X.shape}")
        if not np.all(np.isin(Y, [0, 1])):
            fail(f"{split}: label output contains values other than 0/1")

    sampler = json.loads((staging / "oscd_train_sampler.json").read_text(encoding="utf-8"))
    if not sampler["train_only"] or sampler["oversample_factor"] != 3:
        fail("Train sampler metadata failed contract check.")

    # Check every stored file can be hashed before promotion.
    for p in staging.glob("*"):
        if p.is_file():
            sha256_file(p)


def promote(staging: Path, out: Path, overwrite: bool) -> None:
    out = out.resolve()
    staging = staging.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        if not overwrite:
            fail(
                f"Output directory already exists: {out}\n"
                "Use --overwrite to replace it after a verified run."
            )
        shutil.rmtree(out)

    staging.rename(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare OSCD for the Geo-Nexus v3.2 optical-only supervised stage."
    )
    parser.add_argument("oscd_root", type=Path, help="OSCD root containing Images/Train Labels/Test Labels")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/oscd/splits/oscd_splits.json"),
        help="Geo-Nexus frozen 11/3/10 split JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/oscd/prepared"),
        help="Output directory for prepared arrays",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and count patches without writing arrays")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()

    root = args.oscd_root.expanduser().resolve()
    manifest_path = args.split_manifest.expanduser().resolve()
    out = args.out.expanduser().resolve()

    print("=" * 78)
    print("GEO-NEXUS / MH-DAPT-CD v3.2 — OSCD PREPARATION")
    print("=" * 78)
    print(f"OSCD root      : {root}")
    print(f"Split manifest : {manifest_path}")
    print(f"Output         : {out}")
    print(f"Mode           : {'DRY RUN' if args.dry_run else 'WRITE'}")

    images_root, train_labels, test_labels = resolve_oscd_subdirs(root)
    splits = load_split_manifest(manifest_path)
    validate_project_split(splits)

    print("\nResolved dataset layout:")
    print(f"  Images       : {images_root}")
    print(f"  Train labels : {train_labels}")
    print(f"  Test labels  : {test_labels}")
    print(f"\nFrozen split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Resolve all city directories before doing any large IO.
    all_cities = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    if all_cities != EXPECTED_ALL_CITIES:
        fail("Split city set does not equal the 24-city OSCD benchmark set.")

    if args.dry_run:
        for split in ("train", "val", "test"):
            label_root = test_labels if split == "test" else train_labels
            stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE
            total = 0
            print(f"\n{'=' * 78}\nDRY-RUN {split.upper()} | stride={stride}\n{'=' * 78}")
            for city in splits[split]:
                root_city = images_root / city
                info = inspect_city(images_root, label_root, city)
                n = info["patch_count_train"] if split == "train" else info["patch_count_eval"]
                total += n
                print(
                    f"  {city:12s} grid={tuple(info['grid'].values())} "
                    f"change={info['change_fraction'] * 100:7.3f}% patches={n} "
                    f"rect=({info['rect_t1']},{info['rect_t2']})"
                )
                del info
            if total == 0:
                fail(f"Dry-run found zero {split} patches.")
            print(f"TOTAL {split}: {total} patches")
        print("\nDRY RUN COMPLETE — no output arrays were written.")
        return 0

    # Work in a temporary sibling directory. This prevents half-written results
    # from being mistaken for a valid prepared dataset if a later city fails.
    staging = out.with_name(out.name + ".__staging__")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)

    try:
        summary = {
            "version": "geonexus-oscd-v3.2",
            "patch": PATCH,
            "train_stride": TRAIN_STRIDE,
            "eval_stride": EVAL_STRIDE,
            "array_scale": ARRAY_SCALE,
            "channel_order": CHANNEL_ORDER,
            "splits": {},
            "cities": {},
            "roots": {
                "images": str(images_root),
                "train_labels": str(train_labels),
                "test_labels": str(test_labels),
            },
        }

        for split in ("train", "val", "test"):
            label_root = test_labels if split == "test" else train_labels
            result = prepare_split(split, splits[split], images_root, label_root, staging)
            summary["splits"][split] = result

        # City-level summary is lightweight and paper/reproducibility friendly.
        for split in ("train", "val", "test"):
            label_root = test_labels if split == "test" else train_labels
            for city in splits[split]:
                root_city = images_root / city
                s1 = describe_scene(root_city, 1)
                s2 = describe_scene(root_city, 2)
                with rasterio.open(find_label_file(label_root, city)) as src:
                    label_shape = [int(src.height), int(src.width)]
                summary["cities"][city] = {
                    "split": split,
                    "label_shape_source": label_shape,
                    "rect_t1": s1.rectified,
                    "rect_t2": s2.rectified,
                    "t1_dir": str(s1.folder),
                    "t2_dir": str(s2.folder),
                }

        write_json(staging / "oscd_preprocess_manifest.json", summary)
        verify_outputs(staging, summary)

        # Add hashes after manifest validation. Hashes are written as a separate
        # file so the manifest remains deterministic with respect to data layout.
        hashes = {
            p.name: sha256_file(p)
            for p in sorted(staging.iterdir())
            if p.is_file() and p.name != "oscd_file_hashes.json"
        }
        write_json(staging / "oscd_file_hashes.json", hashes)

        promote(staging, out, overwrite=args.overwrite)
        print("\n" + "=" * 78)
        print("PREPARATION COMPLETE")
        print("=" * 78)
        print(f"Output directory: {out}")
        print("Verified files:")
        for name in sorted(p.name for p in out.iterdir() if p.is_file()):
            print(f"  - {name}")
        return 0

    except Exception:
        # Never leave a misleading partial dataset behind.
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
