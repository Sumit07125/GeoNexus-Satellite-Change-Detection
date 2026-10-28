#!/usr/bin/env python3
"""Audit raw OSCD value ranges before serialization; never alters source data."""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
import numpy as np, rasterio

KEEP=("B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")
BANDS_13=("B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12")

def canon(n):
    s=Path(n).stem.upper()
    if s in {'B8A','B08A'}: return 'B8A'
    m=re.search(r'(?:^|[_\-.])B(0*\d+)$',s)
    return None if not m else 'B'+(m.group(1).lstrip('0') or '0')
def find(folder,b):
    if not folder.is_dir(): return None
    for p in folder.glob('*'):
        if p.is_file() and p.suffix.lower() in {'.tif','.tiff'} and canon(p.name)==b: return p
    return None
def resolve(city,t):
    r=city/f'imgs_{t}_rect'; p=city/f'imgs_{t}'
    if r.is_dir() and all(find(r,b) for b in BANDS_13): return r,True
    if p.is_dir(): return p,False
    raise FileNotFoundError(f'{city.name}: no image directory for t{t}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('oscd_root',type=Path); ap.add_argument('--out',type=Path,default=None); args=ap.parse_args(); root=args.oscd_root.resolve(); warnings.filterwarnings('ignore',category=rasterio.errors.NotGeoreferencedWarning)
    images=root/'Images'
    if not images.is_dir(): images=root/'Onera Satellite Change Detection dataset - Images'
    cities=sorted([p for p in images.iterdir() if p.is_dir()])
    rows=[]
    for city in cities:
        for t in (1,2):
            folder,rect=resolve(city,t)
            for b in KEEP:
                p=find(folder,b)
                with rasterio.open(p) as src:
                    a=src.read(1,masked=True).astype(np.float32).filled(np.nan)
                finite=np.isfinite(a); v=a[finite]
                if not v.size: continue
                rows.append({'city':city.name,'timestamp':t,'band':b,'rectified':rect,'min_dn':float(v.min()),'max_dn':float(v.max()),'mean_dn':float(v.mean()),'gt_10000':int(np.count_nonzero(v>10000)),'gt_32767':int(np.count_nonzero(v>32767)),'pixels':int(a.size),'nodata_or_nonfinite':int(a.size-finite.sum())})
    report={'version':'geonexus-oscd-v3.2-value-audit','scale':10000.0,'storage_dtype':'int16','storage_policy':'clip only at final serialization; audit counts retained','bands':list(KEEP),'rows':rows}
    out=args.out or (root/'oscd_value_audit.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    extreme=[r for r in rows if r['gt_32767']]
    print(f'VALUE AUDIT COMPLETE | rows={len(rows)} | rows with DN>32767={len(extreme)}')
    for r in extreme[:20]: print(f"  {r['city']} t{r['timestamp']} {r['band']}: max={r['max_dn']:.1f}, >32767={r['gt_32767']}")
    print('Report:',out)
if __name__=='__main__': main()
