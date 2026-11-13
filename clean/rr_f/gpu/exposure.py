# SPDX-License-Identifier: MIT
"""Synthetic D3D12 spec-58 qualification, independent CPU reference, no game/model data."""
import argparse
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

# lanes per workgroup of each exposure kernel: the partial pass is one wave per 256-pixel block (RR_EX_WAVE)
EX_BLOCK = {'partial': 32, 'finish': 256}

from . import build
from ..reference import exposure as ref


def symbols():
    # Original toy fixture values, not converter values or player defaults.
    return ref.ExposureSymbols(*map(np.float32, (.25,.5,65504,.25,.001,.00001,4,-4,.25)))


def cases():
    rng=np.random.default_rng(58018)
    c=rng.uniform(-2,8,(23,31,4)).astype(np.float32)
    rows=[]
    for i,prev in enumerate((0,1,2.5,-.5,np.nan,np.inf,2**-140)):
        rows.append((f"crop-prev-{i}",c,(19,13),(3,5),np.float32(prev),False))
    rows.append(("reset-ignores-prev",c,(19,13),(3,5),np.float32(np.nan),True))
    for name,value in (("dark",0),("bright",1e6),("negative",-8),("subnormal",2**-140)):
        rows.append((name,np.full((8,9,3),value,np.float32),(9,8),(0,0),np.float32(1),False))
    special=np.array([[np.nan,1,2],[np.inf,1,2],[-np.inf,1,2],[np.inf,-np.inf,0]],np.float32).reshape(2,2,3)
    rows.append(("nonfinite",special,(2,2),(0,0),np.float32(1),False))
    edge=np.zeros((1,1,3),np.float32)
    edge[0,0,0]=np.array(0x017fffff,np.uint32).view(np.float32)
    rows.append(("underflow-roundup",edge,(1,1),(0,0),np.float32(1),False))
    mean_edge=np.zeros((1,3,3),np.float32)
    mean_edge[0,:2,0]=np.array([0x01800000,0x01ffffff],np.uint32).view(np.float32)
    rows.append(("mean-roundup",mean_edge,(3,1),(0,0),np.float32(1),False))
    wide=np.full((257,257,3),2**-12,np.float32)
    wide[0,0]=32768
    rows.append(("sparse-highlight",wide,(257,257),(0,0),np.float32(1),False))
    rows=[(*row,symbols()) for row in rows]
    negative_zero=replace(symbols(),X_KEY=np.float32(.125),X_MFL=np.float32(.25),X_NRM=np.float32(2**-126))
    rows.append(('negative-zero',np.zeros((1,1,3),np.float32),(1,1),(0,0),np.float32(.75),False,negative_zero))
    return rows


def exact_mean(image,size,origin,k):
    w,h=size; x,y=origin
    c=image[y:y+h,x:x+w,:3]
    # Reference operation boundaries are explicit; exact summation is independently
    # reconstructed from a histogram of the resulting f32 luminances.
    v=ref._add(ref._add(ref._mul(k.X_LWrb,c[...,0]),ref._mul(k.X_LWg,c[...,1])),ref._mul(k.X_LWrb,c[...,2]))
    v=np.fmin(np.fmax(v,np.float32(0)),k.X_LMAX)
    values,counts=np.unique(v,return_counts=True)
    return sum((Fraction(float(a))*int(n) for a,n in zip(values,counts)),Fraction())/(w*h)


def within_ulp(a,b,n=1):
    a,b=np.float32(a),np.float32(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if a==b:
        return n!=0 or bool(a.view(np.uint32)==b.view(np.uint32))
    if not np.isfinite(a) or not np.isfinite(b): return False
    lo=hi=b
    for _ in range(n):
        lo=np.nextafter(lo,np.float32(-np.inf)); hi=np.nextafter(hi,np.float32(np.inf))
    return bool(lo<=a<=hi)


def qualify(img, records, k):
    rows=[]
    for record in records:
        name,off,expected,mstar,reset=record[:5]
        config=record[5] if len(record)>5 else k
        actual=img[off:off+24].copy().view('<f4')
        mean64=img[off+24:off+32].copy().view('<f8')[0]
        stored=int(img[off+32:off+36].copy().view('<u4')[0])
        delta=abs(Fraction(float(mean64))-mstar) if np.isfinite(mean64) else None
        mean_ok=delta is not None and delta <= mstar/Fraction(1<<29)
        e,ep,inv,x3,raw,mean=actual
        half=np.float16(e)
        target=np.float16(expected.E)
        e_ok=bool(e==np.float32(half) and np.nextafter(target,np.float16(-np.inf))<=half<=np.nextafter(target,np.float16(np.inf)))
        ep_ok=within_ulp(ep,expected.Ep,0)
        inv_ok=within_ulp(inv,ref._rcp(e))
        # The exposure/shader selects a correctly rounded reciprocal. log2 alone
        # may differ by one ulp; propagate that allowance through clamp/multiply.
        if reset:
            x_ok=bool(x3==0)
        else:
            ratio=ref._mul(np.fmax(e,config.X_EFL),ref._rcp(np.fmax(ep,config.X_EFL)))
            lg=ref._log2(ratio)
            candidates=[lg]
            if np.isfinite(lg):
                candidates += [np.nextafter(lg,np.float32(-np.inf)),np.nextafter(lg,np.float32(np.inf))]
            permitted=[ref._mul(np.fmin(np.fmax(v,config.X_CLn),config.X_CL),config.X_NRM) for v in candidates]
            x_ok=any(within_ulp(x3,v,0) for v in permitted)
        store_ok=stored==int(np.float16(raw).view(np.uint16))
        own_e=np.float32(np.float16(raw)); own_e=np.float32(1) if own_e==0 else own_e
        raw_ok=within_ulp(raw,ref._mul(config.X_KEY,ref._rcp(np.fmax(mean,config.X_MFL))),0)
        rules=dict(mean_bound=mean_ok,mean_round=within_ulp(mean,ref._ftz(np.float32(mean64)),0),E=e_ok,
                   E_from_raw=within_ulp(e,own_e,0),Ep=ep_ok,invE=inv_ok,X3=x_ok,stored=store_ok,raw=raw_ok,
                   guard=bool(np.all(img[off+36:off+100]==0xA5)))
        rows.append(dict(case=name,passed=all(rules.values()),rules=rules,actual=[float(v) for v in actual],
                         expected=[float(getattr(expected,n)) for n in ('E','Ep','invE','X3','E_raw','mean')],
                         mean_relative_error=float(delta/mstar) if delta is not None and mstar else (0 if delta==0 else None)))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json',required=True)
    ap.add_argument('--mutation',action='store_true',help='skip sparse-highlight partial producer; must fail')
    a=ap.parse_args()
    if 'WG_RR_BUILD' not in os.environ: ap.error('set WG_RR_BUILD to a private clean-root output directory')
    k=symbols(); arena=build.Arena(); dispatches=[]; records=[]
    kernels={name:(build.compile_kernel('rr_exposure.hlsl','rr_exposure_'+name,{'WG_BLOCK':EX_BLOCK[name]}),23)
             for name in ('partial','finish')}
    for name,c,size,origin,prev,reset,k in cases():
        expected=ref.auto_exposure(c,symbols=k,render_size=size,Eprev=prev,reset=reset,origin=origin)
        w,h=size; blocks=(w*h+255)//256
        cp=arena.add(c); pp=arena.reserve(blocks*8); out=arena.reserve(100,fill=0xA5)
        ilo,ihi=struct.unpack('<II',struct.pack('<d',1.0/(w*h)))
        args=[cp,pp,out,c.shape[1],c.shape[2],*origin,w,h,blocks,ilo,ihi,int(prev.view(np.uint32)),int(reset)]
        args += [int(getattr(k,f.name).view(np.uint32)) for f in fields(k)]
        if not(a.mutation and name=='sparse-highlight'): dispatches.append(('partial',blocks,args))
        dispatches.append(('finish',1,args))
        records.append((name,out,expected,exact_mean(c,size,origin,k),reset,k))
    img,log=build.run(arena,kernels,dispatches,require_double=True,
                      workdir=str(Path(build.BUILD)/('exposure-mutant' if a.mutation else 'exposure')))
    rows=qualify(img,records,k)
    report=dict(passed=all(r['passed'] for r in rows),mutation=a.mutation,cases=rows,host_log=log,
                shaders={n:hashlib.sha256(Path(v[0]).read_bytes()).hexdigest() for n,v in kernels.items()},
                scope='synthetic API-decoded f32 buffers; no image views, temporal integration or player constants')
    Path(a.json).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS' if report['passed'] else 'FAIL',flush=True)
    for r in rows:
        if not r['passed']: print(r['case'],r['rules'],flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
