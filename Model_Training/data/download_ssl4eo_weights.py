#!/usr/bin/env python3
"""
Fetch SSL4EO-S12 MoCo ResNet-18 weights for BOTH modalities.

  SENTINEL2_ALL_MOCO : ResNet-18, 13-band S2 L1C, MoCo-v2   -> optical branch
  SENTINEL1_ALL_MOCO : ResNet-18,  2-band S1 GRD, MoCo-v2   -> SAR branch

Matched architectures for both modalities are exactly what the two-branch
design needs, and the reason ResNet-18 was chosen over a ViT foundation model
(Part 2.2): Prithvi / Clay / TerraMind are optical-first and would leave the
SAR branch at random init.

Usage:  python data/download_ssl4eo_weights.py --out data/weights/
Per FINAL_ARCH_3.md v3.2 (Part 28.1).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_NAMES = {'s2': 'resnet18_s2c_moco.pth', 's1': 'resnet18_s1_moco.pth'}
EXPECT_IN_CH = {'s2': 13, 's1': 2}


def via_torchgeo():
    """Primary path. pip install torchgeo"""
    from torchgeo.models import ResNet18_Weights
    return {'s2': ResNet18_Weights.SENTINEL2_ALL_MOCO,
            's1': ResNet18_Weights.SENTINEL1_ALL_MOCO}


def via_hf_hub(modality):
    """Fallback. pip install huggingface_hub safetensors"""
    from huggingface_hub import list_repo_files, hf_hub_download
    if modality == 's2':
        repo = 'torchgeo/resnet18_sentinel2_all_moco'
        files = [f for f in list_repo_files(repo) if f.endswith(('.pth', '.ckpt'))]
        if not files:
            raise RuntimeError(f'no checkpoint found in {repo}')
        return torch.load(hf_hub_download(repo, files[0]), map_location='cpu')
    else:
        # Sentinel-1 ResNet-18 (2 channels) from BIFOLD BigEarthNet S1
        repo = 'BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0'
        p = hf_hub_download(repo, 'model.safetensors')
        try:
            from safetensors.torch import load_file
            return load_file(p)
        except ImportError:
            return torch.load(p, map_location='cpu')


def normalise(sd):
    """Strip the wrappers different releases add around the same tensors."""
    for k in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    out = {}
    for k, v in sd.items():
        for pre in ('model.vision_encoder.', 'module.', 'encoder_q.', 'backbone.', 'encoder.'):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def main(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    try:
        weights = via_torchgeo()
        get = lambda m: weights[m].get_state_dict(progress=True)
        print('source: torchgeo')
    except Exception as e:
        print(f'torchgeo unavailable ({e}); falling back to HF hub')
        get = via_hf_hub

    for mod, fname in OUT_NAMES.items():
        sd = normalise(get(mod))

        if 'conv1.weight' not in sd:
            raise RuntimeError(f'{mod}: no conv1.weight after normalisation; '
                               f'keys start with {list(sd)[:5]}')
        c_in = sd['conv1.weight'].shape[1]
        exp  = EXPECT_IN_CH[mod]
        print(f'  {mod}: conv1 {tuple(sd["conv1.weight"].shape)}  '
              f'({c_in} input channels, expected {exp})')
        assert c_in == exp, (
            f'{mod} checkpoint has {c_in} input channels, not {exp}. '
            f'You have the RGB variant. Part 17 stem surgery assumes the '
            f'ALL-bands variant and will silently mis-map bands.')

        # the trunk keys Part 17 actually loads
        need = [k for k in sd if k.split('.')[0] in
                ('bn1','layer1','layer2','layer3','layer4')]
        assert len(need) > 50, f'{mod}: trunk looks incomplete ({len(need)} keys)'

        p = out_dir / fname
        torch.save(sd, p)
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        manifest[mod] = {'file': fname, 'in_channels': c_in,
                         'n_keys': len(sd), 'sha256_16': h,
                         'size_mb': round(p.stat().st_size / 1e6, 2)}
        print(f'  -> {p}  ({manifest[mod]["size_mb"]} MB, sha {h})')

    json.dump(manifest, open(out_dir / 'ssl4eo_manifest.json', 'w'), indent=2)

    # Gate: the stem surgery unit test from Part 17 must pass on the real file.
    from models.stem import verify_stem_surgery
    s2 = torch.load(out_dir / OUT_NAMES['s2'], map_location='cpu')
    cos = verify_stem_surgery(s2['conv1.weight'])
    print(f'\nSTEM SURGERY GATE: cosine = {cos:.4f}  '
          f'{"PASS" if cos > 0.98 else "FAIL"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/weights')
    main(ap.parse_args().out)
