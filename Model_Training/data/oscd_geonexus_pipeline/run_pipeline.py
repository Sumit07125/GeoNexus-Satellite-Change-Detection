#!/usr/bin/env python3
"""Run the complete local Geo-Nexus OSCD pipeline with an overall progress bar."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from tqdm import tqdm


def run(cmd,cwd):
    print('\n$ '+' '.join(map(str,cmd)))
    subprocess.run(cmd,cwd=cwd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--upload',action='store_true')
    ap.add_argument('--create',action='store_true')
    ap.add_argument('--version',action='store_true')
    args=ap.parse_args()
    if args.upload and args.create==args.version: raise SystemExit('With --upload, choose exactly one: --create or --version')
    root=args.root.resolve(); pipe=root/'data/oscd_geonexus_pipeline'; oscd=root/'data/OSCD'; manifest=root/'data/oscd/splits/oscd_splits.json'; out=root/'data/oscd/prepared'
    steps=[
      ('BUILD MANIFEST',[sys.executable,str(pipe/'build_oscd_manifest.py')]),
      ('VERIFY RAW OSCD',[sys.executable,str(pipe/'verify_oscd.py'),str(oscd)]),
      ('AUDIT VALUE RANGE',[sys.executable,str(pipe/'audit_oscd_values.py'),str(oscd)]),
      ('DRY RUN',[sys.executable,str(pipe/'prepare_oscd.py'),str(oscd),'--split-manifest',str(manifest),'--out',str(out),'--dry-run']),
      ('WRITE PREPARED DATA',[sys.executable,str(pipe/'prepare_oscd.py'),str(oscd),'--split-manifest',str(manifest),'--out',str(out),'--overwrite']),
      ('VALIDATE RELEASE',[sys.executable,str(pipe/'validate_prepared.py'),str(out)]),
    ]
    bar=tqdm(total=len(steps),desc='LOCAL PIPELINE',unit='step',bar_format='{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for i,(label,cmd) in enumerate(steps,1):
        print(f'\n===== {i}/{len(steps)} {label} =====')
        run(cmd,root); bar.update(1)
    bar.close()
    if args.upload:
        print('\n===== 7/7 KAGGLE RELEASE =====')
        cmd=[sys.executable,str(pipe/'stage_kaggle_oscd.py'),'--prepared',str(out),'--staging',str(root/'data/oscd/kaggle_oscd'), '--create' if args.create else '--version']
        run(cmd,root)
        print('\nPIPELINE COMPLETE: 7/7 including Kaggle upload')
    else:
        print('\nPIPELINE COMPLETE: 6/6 local stages. Kaggle upload is intentionally separate.')
if __name__=='__main__': main()
