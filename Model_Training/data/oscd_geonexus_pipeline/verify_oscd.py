#!/usr/bin/env python3
"""Verify the raw OSCD tree and the frozen 24-city benchmark contents."""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
import numpy as np
import rasterio

BANDS_13=["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
TRAIN_14=["abudhabi","aguasclaras","beihai","beirut","bercy","bordeaux","cupertino","hongkong","mumbai","nantes","paris","pisa","rennes","saclay_e"]
TEST_10=["brasilia","chongqing","dubai","lasvegas","milano","montpellier","norcia","rio","saclay_w","valencia"]
ALL=TRAIN_14+TEST_10

def canon(name):
    s=Path(name).stem.upper()
    if s in {'B8A','B08A'}: return 'B8A'
    m=re.search(r'(?:^|[_\-.])B(0*\d+)$',s)
    return None if not m else 'B'+(m.group(1).lstrip('0') or '0')

def find_band(folder,band):
    if not folder.is_dir(): return None
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {'.tif','.tiff'} and canon(p.name)==band:
            return p
    return None

def resolve(root,t):
    rect=root/f'imgs_{t}_rect'; plain=root/f'imgs_{t}'
    if rect.is_dir() and all(find_band(rect,b) for b in BANDS_13): return rect,True
    if plain.is_dir(): return plain,False
    return None,False

def label_file(root,city):
    cm=root/city/'cm'
    candidates=sorted([p for p in cm.glob('*') if p.suffix.lower() in {'.tif','.tiff','.png'}]) if cm.is_dir() else []
    return candidates[0] if candidates else None

def resolve_subdirs(base):
    variants=[('Images','Train Labels','Test Labels'),('Onera Satellite Change Detection dataset - Images','Onera Satellite Change Detection dataset - Train Labels','Onera Satellite Change Detection dataset - Test Labels')]
    for a,b,c in variants:
        pa,pb,pc=base/a,base/b,base/c
        if pa.is_dir() and pb.is_dir() and pc.is_dir(): return pa,pb,pc
    raise FileNotFoundError(f'Could not find Images/Train Labels/Test Labels under {base}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('oscd_root',type=Path); args=ap.parse_args(); base=args.oscd_root.resolve()
    img,tr,te=resolve_subdirs(base); warnings.filterwarnings('ignore',category=rasterio.errors.NotGeoreferencedWarning)
    report={'overall':'PASS','city_count':len(ALL),'report':{},'actual_image_city_dirs':[]}
    errors=[]
    for city in ALL:
        cr=img/city; entry={'city':city,'official_split':'train' if city in TRAIN_14 else 'test'}
        if not cr.is_dir(): errors.append(f'missing city {city}'); continue
        report['actual_image_city_dirs'].append(city)
        for t in (1,2):
            d,rect=resolve(cr,t)
            if d is None: errors.append(f'{city}: missing t{t} image directory'); continue
            missing=[b for b in BANDS_13 if not find_band(d,b)]
            if missing: errors.append(f'{city}: t{t} missing bands {missing}')
            ref=find_band(d,'B4')
            with rasterio.open(ref) as src: shape=(src.height,src.width)
            entry[f'imgs_{t}']={'dir':str(d),'rect':rect,'shape':list(shape),'bands_found':13-len(missing),'missing':missing}
        lab=label_file(tr if city in TRAIN_14 else te,city)
        if lab is None: errors.append(f'{city}: missing label'); entry['label']='MISSING'
        else:
            with rasterio.open(lab) as src: a=src.read(1)
            vals=sorted(set(np.unique(a).tolist())); entry['label']={'file':lab.name,'shape':list(a.shape),'values':vals}
            if set(vals)-{0,1,2,255}: errors.append(f'{city}: unsupported label values {vals}')
        report['report'][city]=entry
    actual=sorted(report['actual_image_city_dirs']); report['actual_image_city_count']=len(actual)
    if set(actual)!=set(ALL): errors.append(f'actual city set mismatch: {sorted(set(ALL)-set(actual))}')
    if errors:
        report['overall']='FAIL'; report['errors']=errors
    out=base/'oscd_verification_report.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'overall':report['overall'],'city_count':report['actual_image_city_count'],'project_train':sorted([c for c in ALL if c not in {'beirut','rennes','saclay_e'} and c in TRAIN_14]),'project_val':['beirut','rennes','saclay_e'],'project_test':sorted(TEST_10),'report':str(out)},indent=2))
    return 0 if report['overall']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
