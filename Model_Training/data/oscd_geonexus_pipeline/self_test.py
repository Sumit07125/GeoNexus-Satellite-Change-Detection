#!/usr/bin/env python3
"""Fast integrity self-test for the Geo-Nexus OSCD release scripts."""
from __future__ import annotations
import py_compile
from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent

def load_prepare():
    spec=importlib.util.spec_from_file_location('prepare_oscd',ROOT/'prepare_oscd.py')
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def main():
    scripts=sorted(ROOT.glob('*.py'))
    for p in scripts: py_compile.compile(str(p),doraise=True)
    mod=load_prepare()
    # Exercise the exact failure seen in the user's run.
    x=np.zeros((2,17,128,128),dtype=np.float32)
    x[0,2,0,0]=3.3758
    y, clipped, total, raw_min, raw_max=mod.scaled_int16(x)
    assert y.dtype==np.int16
    assert int(y[0,2,0,0])==32767
    assert clipped==1 and total==2*17*128*128
    assert abs(raw_max-33758.0)<1e-6
    print('PY_COMPILE: PASS')
    print('33758 OVERFLOW CASE: PASS (final int16 saturation audited)')
    print('SELF-TEST: PASS')
if __name__=='__main__': main()
