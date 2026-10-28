#!/usr/bin/env python3
"""Validate, stage, and optionally upload the prepared OSCD dataset to Kaggle."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from tqdm import tqdm

DEFAULT_SLUG='oscd-onera-v1'
REQUIRED=[
'oscd_train.npy','oscd_train_label.npy','oscd_train_meta.json','oscd_train_sampler.json',
'oscd_val.npy','oscd_val_label.npy','oscd_val_meta.json',
'oscd_test.npy','oscd_test_label.npy','oscd_test_meta.json',
'oscd_preprocess_manifest.json','oscd_file_hashes.json']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,default=Path('data/oscd/prepared'))
    ap.add_argument('--staging',type=Path,default=Path('data/oscd/kaggle_oscd'))
    ap.add_argument('--dataset-id',default=None,help='OWNER/SLUG; auto-detected from kaggle.json if omitted')
    ap.add_argument('--create',action='store_true',help='Create a new private dataset')
    ap.add_argument('--version',action='store_true',help='Create a new version of an existing dataset')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--public',action='store_true')
    args=ap.parse_args()

    if args.create and args.version: raise SystemExit('Choose only one: --create or --version')
    if not args.dry_run and not (args.create or args.version): raise SystemExit('For an actual upload choose --create or --version')
    prepared=args.prepared.resolve(); staging=args.staging.resolve(); staging.mkdir(parents=True,exist_ok=True)
    missing=[x for x in REQUIRED if not (prepared/x).is_file()]
    if missing: raise SystemExit('Prepared dataset missing: '+', '.join(missing))

    for name in tqdm(REQUIRED,desc='STAGE [1/3] copy validated files',unit='file'):
        dst=staging/name; dst.write_bytes((prepared/name).read_bytes())

    username=None
    kj=Path.home()/'.kaggle'/'kaggle.json'
    if kj.exists():
        try: username=json.loads(kj.read_text(encoding='utf-8')).get('username')
        except Exception: pass
    dataset_id=args.dataset_id or (f'{username}/{DEFAULT_SLUG}' if username else None)
    if not dataset_id: raise SystemExit('Kaggle dataset ID unknown. Use --dataset-id OWNER/oscd-onera-v1 or configure ~/.kaggle/kaggle.json')

    metadata={
        'title':'OSCD Onera Prepared 17ch Patches',
        'subtitle':'Geo-Nexus v3.2 OSCD 11/3/10 patch dataset',
        'description':'Processed Onera Satellite Change Detection dataset for Geo-Nexus v3.2: 128x128 bi-temporal 17-channel tensors; 11 train, 3 validation, 10 test cities; train-only 3x oversampling metadata. Source images are not re-uploaded.',
        'id':dataset_id,
        'licenses':[{'name':'CC-BY-NC-SA-4.0'}],
        'keywords':['satellite-imagery','change-detection','sentinel-2','oscd','pytorch']
    }
    (staging/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (staging/'README.md').write_text(
        '# OSCD Onera Prepared 17ch Patches\n\n'
        '- Geo-Nexus v3.2\n- 128x128 patches\n- 17 channels per timestamp\n'
        '- 11 train / 3 validation / 10 test\n- train stride 64\n- eval stride 128\n'
        '- int16 x10000 images; uint8 binary labels\n- train-only 3x change-patch sampler\n',encoding='utf-8')

    print('\nSTAGE [2/3] metadata + release audit: PASS')
    staged=sorted(p.name for p in staging.iterdir() if p.is_file())
    unexpected=[x for x in staged if x not in set(REQUIRED+['dataset-metadata.json','README.md'])]
    if unexpected: raise SystemExit('Unexpected staging files: '+str(unexpected))
    print(f'Stage directory: {staging}')
    print(f'Release files : {len(REQUIRED)} data/manifest files + metadata + README')

    if args.dry_run:
        print('STAGE [3/3] DRY RUN — no network upload performed.')
        return 0

    cmd=['kaggle','datasets','create' if args.create else 'version','-p',str(staging),'--dir-mode','zip']
    if args.version: cmd += ['-m','Geo-Nexus v3.2 OSCD prepared dataset update']
    if args.public: cmd += ['--public']
    print('\nSTAGE [3/3] KAGGLE UPLOAD')
    print('$ '+' '.join(map(str,cmd)))
    subprocess.run(cmd,check=True)
    print('\nKAGGLE UPLOAD COMPLETE')
    print('Dataset:',dataset_id)
    print('Mount : /kaggle/input/'+dataset_id.split('/')[-1]+'/')
    return 0

if __name__=='__main__': raise SystemExit(main())
