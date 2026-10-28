#!/usr/bin/env python3
"""Freeze the Geo-Nexus v3.2 OSCD 11/3/10 scene split."""
from __future__ import annotations
import argparse, json
from pathlib import Path

OFFICIAL_TRAIN_14 = ["abudhabi","aguasclaras","beihai","beirut","bercy","bordeaux","cupertino","hongkong","mumbai","nantes","paris","pisa","rennes","saclay_e"]
OFFICIAL_TEST_10 = ["brasilia","chongqing","dubai","lasvegas","milano","montpellier","norcia","rio","saclay_w","valencia"]
VAL = ["beirut","rennes","saclay_e"]
TRAIN = sorted(c for c in OFFICIAL_TRAIN_14 if c not in VAL)
TEST = sorted(OFFICIAL_TEST_10)

def build(out: Path) -> dict:
    assert len(OFFICIAL_TRAIN_14)==14 and len(OFFICIAL_TEST_10)==10
    assert len(TRAIN)==11 and len(VAL)==3 and len(TEST)==10
    assert set(TRAIN).isdisjoint(set(VAL) | set(TEST))
    assert set(VAL).isdisjoint(TEST)
    m={"version":"geonexus-oscd-v3.2","frozen":"2026-09-20","official":{"train_14":OFFICIAL_TRAIN_14,"test_10":OFFICIAL_TEST_10},"project":{"train_11":TRAIN,"val_3":VAL,"test_10":TEST},"notes":["Validation is carved only from official OSCD train cities.","Montpellier remains official test.","OSCD L1C/TOA versus Maharashtra L2A/BOA is a documented domain shift.","Saclay_e validation and saclay_w test are adjacent halves; report both 10-city and 9-city-excluding-saclay_w test views.","Labels are normalized to binary 0/1 during preparation."]}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(out)
    return m

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('data/oscd/splits/oscd_splits.json'))
    args=ap.parse_args()
    build(args.out)
