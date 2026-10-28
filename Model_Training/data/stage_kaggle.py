#!/usr/bin/env python3
"""
Create/update the three private Kaggle datasets for Geo-Nexus v3.2.
Kaggle limits: 200 GB per dataset, 50 top-level files. We are far inside both.

Datasets managed:
  1. oscd-onera-v1:   OSCD Onera prepared 17-channel patches (train, val, test)
  2. ssl4eo-weights:  S2 SSL4EO-S12 MoCo + S1 BigEarthNet ResNet-18 weights
  3. geonexus-mh-v3:  Maharashtra processed training arrays and metadata

Usage:
  python data/stage_kaggle.py --user YOUR_KAGGLE_USERNAME --new
  python data/stage_kaggle.py --user YOUR_KAGGLE_USERNAME --dataset ssl4eo-weights --new
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SETS = {
    'ssl4eo-weights': {
        'title': 'Geo-Nexus v3.2 Pretrained Foundation Weights',
        'folder': REPO_ROOT / 'data' / 'weights',
        'license': 'other',  # Mixed: CC-BY-4.0 (SSL4EO-S12 S2) and MIT (BigEarthNet S1)
        'description': 'Pretrained ResNet-18 weights for Sentinel-2 (SSL4EO-S12 MoCo-v2, CC-BY-4.0) and Sentinel-1 (BigEarthNet v2.0, MIT) for Geo-Nexus v3.2.'
    },
    'oscd-onera-v1': {
        'title': 'OSCD Onera Prepared 17ch Patches',
        'folder': REPO_ROOT / 'data' / 'OSCD' / 'prepared',
        'license': 'CC-BY-NC-SA-4.0',  # IEEE Dataport OSCD benchmark license
        'description': 'Onera Satellite Change Detection (OSCD) dataset converted to 128x128 17-channel patches (11-3-10 split).'
    },
    'geonexus-mh-v3': {
        'title': 'Geo-Nexus Maharashtra CD v3.2 Training Data',
        'folder': REPO_ROOT / 'data' / 'processed',
        'license': 'other',  # Mixed: Copernicus Sentinel Open Access + Google DW (CC-BY-4.0) + Google OB (CC-BY-4.0 or ODbL-1.0) + Hansen GFC (CC-BY-4.0) + JRC GSW (EC)
        'description': 'Multi-modal optical (Sentinel-2) and SAR (Sentinel-1) bi-temporal change detection training arrays and evidence auto-labels for Maharashtra, India. Derived from Copernicus Sentinel open access data, with upstream multi-modal evidence layers from Google Dynamic World (CC-BY-4.0), Google Open Buildings Temporal V1 (CC-BY-4.0 or ODbL-1.0), Hansen Global Forest Change (CC-BY-4.0), and JRC Global Surface Water (EC open data / Copernicus programme acknowledgement).'
    },
}


def stage(slug: str, cfg: dict, username: str, new: bool = False):
    folder = cfg['folder']
    assert folder.exists(), f"Folder {folder} does not exist."
    files = [f for f in folder.iterdir() if not f.name.startswith('.')]
    n_top = len(files)
    assert n_top <= 50, f"{n_top} top-level entries in {folder}; Kaggle allows max 50. Use subdirectories."
    assert n_top > 0, f"Folder {folder} is empty; nothing to stage."

    meta_file = folder / 'dataset-metadata.json'
    meta = {
        'title': cfg['title'],
        'id': f'{username}/{slug}',
        'licenses': [{'name': cfg['license']}],
        'description': cfg.get('description', '')
    }
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    cmd = (['kaggle', 'datasets', 'create', '-p', str(folder), '--dir-mode', 'zip'] if new else
           ['kaggle', 'datasets', 'version', '-p', str(folder), '-m', 'v3.2 update', '--dir-mode', 'zip'])
    
    print(f"\n--- Staging {slug} ({folder}) ---")
    print("Command:", ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"SUCCESS: {slug} will mount at /kaggle/input/{slug}/")
    except FileNotFoundError:
        print("ERROR: 'kaggle' CLI not found. Install with: pip install kaggle")
        print("Ensure kaggle.json is in ~/.kaggle/kaggle.json (or C:\\Users\\<user>\\.kaggle\\kaggle.json)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Stage Geo-Nexus datasets to private Kaggle datasets.")
    parser.add_argument('--user', default=None, help="Kaggle username (or set in kaggle.json)")
    parser.add_argument('--dataset', choices=list(SETS.keys()) + ['all'], default='all',
                        help="Specific dataset to stage (default: all)")
    parser.add_argument('--new', action='store_true', help="Create new dataset (omit for updating existing)")
    args = parser.parse_args()

    username = args.user
    if not username:
        # Try to infer username from ~/.kaggle/kaggle.json
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            try:
                username = json.loads(kaggle_json.read_text()).get('username')
            except Exception:
                pass
    if not username:
        print("ERROR: Kaggle username required. Pass --user <username> or configure ~/.kaggle/kaggle.json")
        sys.exit(1)

    targets = SETS if args.dataset == 'all' else {args.dataset: SETS[args.dataset]}

    for slug, cfg in targets.items():
        if not cfg['folder'].exists():
            print(f"Skipping {slug}: directory {cfg['folder']} does not exist yet (will be created in later phase).")
            continue
        stage(slug, cfg, username, new=args.new)


if __name__ == '__main__':
    main()
