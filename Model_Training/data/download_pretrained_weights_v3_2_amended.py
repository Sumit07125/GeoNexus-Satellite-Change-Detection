#!/usr/bin/env python3
"""
Geo-Nexus v3.2 explicit weight-source amendment:

Optical branch:
  SSL4EO-S12 Sentinel-2 L1C, ResNet-18, 13-band MoCo-v2
  (TorchGeo / HuggingFace: torchgeo/resnet18_sentinel2_all_moco)

SAR branch:
  BigEarthNet v2.0 Sentinel-1, ResNet-18, 2-band pretrained classifier
  (HuggingFace: BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0)
  NOTE: The official SSL4EO-S12 release only trained S1 MoCo for ResNet-50.
  BigEarthNet v2.0 provides the official, peer-reviewed 2-band ResNet-18 for SAR.

The SAR checkpoint is converted to a clean ResNet-18 trunk state_dict so the
existing Geo-Nexus SplitStem / BranchEncoder loader can use it without changing
feature dimensions.

Run:
  python data/download_pretrained_weights_v3_2_amended.py --out data/weights

Requires:
  pip install huggingface_hub safetensors timm
  (optional: pip install torchgeo)
"""
# pyrefly: ignore [missing-import]
from torchgeo.models import ResNet18_Weights
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file
except ImportError as exc:
    raise RuntimeError("Install safetensors: pip install safetensors") from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

S2_OUT = "resnet18_s2c_moco.pth"
S1_OUT = "resnet18_s1_bigearthnet.pth"
S1_REPO = "BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0"
S2_REPO = "torchgeo/resnet18_sentinel2_all_moco"


def normalize_wrappers(sd):
    """Normalize common checkpoint wrappers without guessing model semantics."""
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]

    out = {}
    prefixes = (
        "model.vision_encoder.",
        "model.visual.",
        "module.",
        "encoder_q.",
        "backbone.",
        "encoder.",
    )
    for k, v in sd.items():
        new_k = k
        for pre in prefixes:
            if new_k.startswith(pre):
                new_k = new_k[len(pre):]
        out[new_k] = v
    return out


def require_resnet18_trunk(sd, label):
    """Return a clean ResNet-18 trunk if the actual tensors prove it exists."""
    conv_candidates = []
    for key, tensor in sd.items():
        if key.endswith("conv1.weight") and tuple(tensor.shape) == (64, 2, 7, 7):
            conv_candidates.append(key)

    if not conv_candidates:
        raise RuntimeError(
            f"{label}: could not find a [64, 2, 7, 7] conv1.weight tensor. "
            "Refusing to guess the SAR checkpoint mapping."
        )

    required_suffixes = [
        "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean",
        "bn1.running_var", "layer1.0.conv1.weight", "layer2.0.conv1.weight",
        "layer3.0.conv1.weight", "layer4.0.conv1.weight",
    ]

    best = None
    for conv_key in conv_candidates:
        prefix = conv_key[: -len("conv1.weight")]
        score = sum(1 for suffix in required_suffixes if prefix + suffix in sd)
        if best is None or score > best[0]:
            best = (score, prefix)

    score, prefix = best
    if score < len(required_suffixes):
        raise RuntimeError(
            f"{label}: found conv1 candidate '{prefix}conv1.weight' but only "
            f"{score}/{len(required_suffixes)} required ResNet-18 trunk keys. "
            "Refusing to produce a partially mapped checkpoint."
        )

    trunk = {}
    for key, tensor in sd.items():
        if key.startswith(prefix):
            trunk[key[len(prefix):]] = tensor

    # Exact structural gates needed by Geo-Nexus BranchEncoder.load_state_dict().
    expected_shapes = {
        "conv1.weight": (64, 2, 7, 7),
        "layer1.0.conv1.weight": (64, 64, 3, 3),
        "layer2.0.conv1.weight": (128, 64, 3, 3),
        "layer3.0.conv1.weight": (256, 128, 3, 3),
        "layer4.0.conv1.weight": (512, 256, 3, 3),
    }
    for key, shape in expected_shapes.items():
        got = tuple(trunk[key].shape)
        if got != shape:
            raise RuntimeError(f"{label}: {key} has shape {got}, expected {shape}")

    return trunk


def get_ssl4eo_s2():
    """Download S2 weights from TorchGeo if available, else directly from HuggingFace."""
    try:

        from torchgeo.models import ResNet18_Weights
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        print("  S2 source: torchgeo.models.ResNet18_Weights.SENTINEL2_ALL_MOCO")
        return weights.get_state_dict(progress=True)
    except Exception as exc:
        print(f"  torchgeo unavailable ({exc}); downloading directly from HuggingFace ({S2_REPO})...")
        from huggingface_hub import list_repo_files, hf_hub_download
        files = [f for f in list_repo_files(S2_REPO) if f.endswith((".pth", ".ckpt"))]
        if not files:
            raise RuntimeError(f"No checkpoint file found in {S2_REPO}")
        p = hf_hub_download(S2_REPO, files[0])
        return torch.load(p, map_location="cpu")


def main(out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "amendment": "Geo-Nexus v3.2 explicit SAR initializer source correction",
        "s2_source": "SSL4EO-S12 / TorchGeo SENTINEL2_ALL_MOCO",
        "s1_source": S1_REPO,
    }

    # ---------------- S2: official SSL4EO-S12 ----------------
    print("Downloading S2 SSL4EO-S12 ResNet-18 MoCo...")
    s2 = normalize_wrappers(get_ssl4eo_s2())
    if "conv1.weight" not in s2 or tuple(s2["conv1.weight"].shape) != (64, 13, 7, 7):
        raise RuntimeError("S2 checkpoint is not the expected 13-band ResNet-18 checkpoint.")

    s2_path = out_dir / S2_OUT
    torch.save(s2, s2_path)
    s2_sha = hashlib.sha256(s2_path.read_bytes()).hexdigest()[:16]
    manifest["s2"] = {
        "file": S2_OUT,
        "in_channels": 13,
        "n_keys": len(s2),
        "sha256_16": s2_sha,
        "size_mb": round(s2_path.stat().st_size / 1e6, 2),
    }
    print(f"S2 PASS: conv1={tuple(s2['conv1.weight'].shape)} -> {s2_path}")

    # ---------------- S1: BigEarthNet v2.0 ----------------
    print(f"Downloading S1 BigEarthNet ResNet-18 from {S1_REPO}...")
    from huggingface_hub import hf_hub_download
    s1_file = hf_hub_download(S1_REPO, "model.safetensors")
    raw_s1 = load_file(s1_file)
    s1 = normalize_wrappers(raw_s1)
    trunk = require_resnet18_trunk(s1, "BigEarthNet S1")

    s1_path = out_dir / S1_OUT
    torch.save(trunk, s1_path)
    s1_sha = hashlib.sha256(s1_path.read_bytes()).hexdigest()[:16]
    manifest["s1"] = {
        "file": S1_OUT,
        "source_repo": S1_REPO,
        "initializer_type": "supervised BigEarthNet v2.0 Sentinel-1 classifier trunk",
        "in_channels": 2,
        "n_keys": len(trunk),
        "sha256_16": s1_sha,
        "size_mb": round(s1_path.stat().st_size / 1e6, 2),
    }
    print(f"S1 PASS: extracted ResNet-18 trunk with conv1={tuple(trunk['conv1.weight'].shape)} -> {s1_path}")

    manifest_path = out_dir / "pretrained_weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    # Stem gate is project-specific and must run against the S2 checkpoint.
    try:
        from models.stem import verify_stem_surgery
        cos = verify_stem_surgery(s2["conv1.weight"])
        print(f"STEM SURGERY GATE: cosine={cos:.4f} {'PASS' if cos > 0.98 else 'FAIL'}")
        if cos <= 0.98:
            raise RuntimeError("Stem surgery gate failed.")
    except ImportError as exc:
        print(f"WARNING: project models.stem unavailable; S2 stem gate not executed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/weights")
    args = parser.parse_args()
    main(args.out)
