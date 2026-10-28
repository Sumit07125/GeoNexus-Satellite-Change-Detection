#!/usr/bin/env python3
"""Validate prepared Geo-Nexus OSCD arrays, metadata, sampler and hashes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
PATCH=128
EXPECTED={'train':11,'val':3,'test':10}

def sha(path,chunk=16*1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('prepared',type=Path); args=ap.parse_args(); root=args.prepared.resolve()
    required=['oscd_train.npy','oscd_train_label.npy','oscd_train_meta.json','oscd_train_sampler.json','oscd_val.npy','oscd_val_label.npy','oscd_val_meta.json','oscd_test.npy','oscd_test_label.npy','oscd_test_meta.json','oscd_preprocess_manifest.json','oscd_file_hashes.json']
    missing=[x for x in required if not (root/x).is_file()]
    if missing: raise SystemExit('FAIL: missing '+', '.join(missing))
    manifest=json.loads((root/'oscd_preprocess_manifest.json').read_text())
    for split in EXPECTED:
        X=np.load(root/f'oscd_{split}.npy',mmap_mode='r'); Y=np.load(root/f'oscd_{split}_label.npy',mmap_mode='r')
        if X.shape[1:]!=(2,17,PATCH,PATCH): raise SystemExit(f'FAIL {split}: X shape {X.shape}')
        if Y.shape!=(X.shape[0],PATCH,PATCH): raise SystemExit(f'FAIL {split}: Y shape {Y.shape}')
        if X.dtype!=np.int16 or Y.dtype!=np.uint8: raise SystemExit(f'FAIL {split}: dtype {X.dtype}/{Y.dtype}')
        if not np.all(np.isin(Y,[0,1])): raise SystemExit(f'FAIL {split}: non-binary labels')
        if manifest['splits'][split]['patch_count']!=int(X.shape[0]): raise SystemExit(f'FAIL {split}: manifest count mismatch')
        print(f'  {split.upper():5s}: X={tuple(X.shape)} Y={tuple(Y.shape)} dtype={X.dtype}/{Y.dtype} PASS')
    samp=json.loads((root/'oscd_train_sampler.json').read_text())
    if samp.get('oversample_factor')!=3 or not samp.get('train_only'): raise SystemExit('FAIL: sampler contract')
    hashes=json.loads((root/'oscd_file_hashes.json').read_text())
    bad=[name for name,digest in hashes.items() if (not (root/name).is_file()) or sha(root/name)!=digest]
    if bad: raise SystemExit('FAIL: SHA256 mismatch '+str(bad))
    print('  SHA256: all recorded files PASS')
    print('VALIDATION COMPLETE — PASS')
if __name__=='__main__': main()
