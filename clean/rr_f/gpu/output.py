# SPDX-License-Identifier: MIT
"""Synthetic D3D12 spec-54 qualification through CH/Q/PH1; no game resolve.

Two explicit boundaries: API-decoded values widened to f32, or packed head
Cnet/PREC half4 and GATES/CANDW byte4. Motion/history buffers remain decoded f32.
Caller bindings can connect packed head outputs directly in the same GPU arena.
All K symbols and TR are supplied by caller; fixtures are invented, not learned.
"""
import argparse
from dataclasses import fields
import hashlib
import itertools
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import build
from ..reference import output as ref
from ..tests.test_output import inputs, symbols

NARGS=39
BLOCK=32
# Fast path (kernels/rr_output_fast.hlsli, packed head inputs): prep (one lane per render pixel), fast (32 x 8 history
# pixels per workgroup, PH1 included), fallback (rr_output_main's / rr_output_pool's code for the listed 2 x 2 quads;
# fixed grid).
FAST_BLOCK=256
PREP_BLOCK=256
FALLBACK_BLOCK=64
FALLBACK_GROUPS=64
FAST_KERNELS=('prep','fast','fallback')
# the runtime's fast head writes the render records (rr_d0r.hlsl RR_OUT_PREP); prepu = prep's uniform part (OF_PREPU)
MAIN_KERNELS=('main','pool')
FAST_WAVE=64   # rr_output_fast's wave size (OF_FAST_WAVE define overrides; 64: -3 % on F-B1 1440p Q, same bits)
_KBLOCK={'prep':PREP_BLOCK,'fast':FAST_BLOCK,'fallback':FALLBACK_BLOCK}


def scratch_bytes(wr,hr,wh,hh):
    """Scratch of the fast path: counters + fallback list, then the render records (rr_output_fast.hlsli)."""
    lst=(256+wh*hh*4+255)//256*256
    return lst+wr*hr*16+(wr*hr*4+15)//16*16+wr*hr*8+64


def kernel_set(defines=None,fast=True,names=None,spec=False):
    """{name: (dxil, NARGS)} of the output stage's kernels (fast path or rr_output_main)."""
    defines=dict(defines or {})
    # spec: the runtime's fixed fast-path flags (kOutFastFlags) at compile time (OFX_SPEC, rr_output_fast.hlsli; same bits)
    if spec and fast:
        defines.setdefault('OFX_SPEC',1)
    # opt-in, NOT exact variants of the fast kernel (tier B), e.g. WG_RR_OUT_DEFINES='{"OF_REPROJ_MAX": 1}'; never default
    if os.environ.get('WG_RR_OUT_DEFINES'):
        defines.update(json.loads(os.environ['WG_RR_OUT_DEFINES']))
    # clean/rr-f-a14-tail, A/B builds only: extra defines of Fast mode's output set (recognised by its OF_HWRCP)
    if defines.get('OF_HWRCP') and os.environ.get('WG_RR_OUTF_DEFINES'):
        defines.update(json.loads(os.environ['WG_RR_OUTF_DEFINES']))
    out={}
    for name in names or (FAST_KERNELS if fast else MAIN_KERNELS):
        d={**defines,'WG_BLOCK':_KBLOCK.get(name,defines.get('WG_BLOCK',BLOCK))}
        if name=='prepu':
            d['WG_BLOCK']=PREP_BLOCK; d['OF_PREPU']=1
        if name=='fast':
            d['WG_WAVE']=int(d.pop('OF_FAST_WAVE',FAST_WAVE))
        else:
            d.pop('OF_FAST_WAVE',None)
        out[name]=(build.compile_kernel('rr_output.hlsl','rr_output_'+('prep' if name=='prepu' else name),d),NARGS)
    return out


def dispatch_list(args,fast=True,prefix=''):
    """The output stage's dispatches for prepare()'s arguments (fast path needs packed inputs and the scratch)."""
    wr,hr,wt,wh,hh=args[14:19]
    pool=(prefix+'pool',(((wh+1)//2)*((hh+1)//2)+BLOCK-1)//BLOCK,args)
    if not fast:
        return [(prefix+'main',(wh*hh+BLOCK-1)//BLOCK,args),pool]
    if not (args[37]&1) or not args[38]:
        raise ValueError('the fast output path needs packed head inputs and a scratch area')
    # prep: at least 513 x 3 lanes (the TR table checks)
    return [(prefix+'prep',max(wr*hr,513*3)//PREP_BLOCK+1,args),
            (prefix+'fast',((wh+31)//32)*((hh+7)//8),args),
            (prefix+'fallback',FALLBACK_GROUPS,args)]


def fixtures():
    rows=[]
    a=inputs(); rows.append(('uniform-odd-history',a))
    a=inputs((3,2),(3,2)); a['P'][:]=[1,1,0]; a['Cnet'][0,:3,:3]=1; a['Cnet'][1,:3,:3]=8; a['constants']=symbols(K1=16); rows.append(('vertical-pair',a))
    a=inputs((2,1),(3,3)); a['G'][:]=[0,0,1]; a['Hprev'][...,:3]=.125; rows.append(('distinct-gates-even-history',a))
    a=inputs((1,1),(1,1)); a['W4'][:]=0; a['G'][:]=0; rows.append(('zero-weights-guard',a))
    rng=np.random.default_rng(541018)
    for idx,(render,outsz) in enumerate([((3,3),(4,4)),((4,2),(5,3)),((2,3),(2,3)),((1,1),(1,1))]):
        a=inputs(render,outsz); hr,wr=a['G'].shape[:2]
        a['Cnet'][:]=np.nan; a['Cnet'][:hr,:wr,:3]=rng.uniform(-.5,8,(hr,wr,3)).astype(np.float16)
        a['P'][:]=rng.uniform(.1,2,a['P'].shape).astype(np.float16); a['P'][...,2]*=.125
        a['G']=ref.decode_unorm8(rng.integers(0,256,a['G'].shape,dtype=np.uint8))
        a['W4']=ref.decode_unorm8(rng.integers(0,256,a['W4'].shape,dtype=np.uint8))
        for name in ['PM','Pe','Pn']: a[name][:]=rng.uniform(-2,2,a[name].shape).astype(np.float16)
        a['Hprev'][...,:3]=rng.uniform(0,.75,a['Hprev'][...,:3].shape).astype(np.float16)
        a['jitter']=(-.49,.49); a['E']=np.float32(1.25); a['Ep']=np.float32(.75); a['invE']=np.float32(.8)
        table=np.tile(np.array([.125,.75,.125,.25],np.float32),(513,1)); table[:,3]=np.linspace(0,.5,513,dtype=np.float32)
        a['constants']=symbols(TR=table,K3=.75)
        rows.append((f'random-border-{idx}',a))
    for poison in (False,True):
        a=inputs((2,1),(3,1)); a['reset']=True; a['G'][:]=ref.decode_unorm8(np.array([0,64,191],np.uint8)); a['P'][:]=[1,1,0]; a['PM'][...,0]=np.nan
        if poison: a['Hprev'][:]=[np.nan,np.inf,-np.inf,4]
        rows.append(('reset-poison' if poison else 'reset-clean',a))
    a=inputs(); a['G'][:]=0; a['Hprev'][...,:3]=[np.nan,np.inf,-np.inf]; rows.append(('history-nan-guard',a))
    a=inputs(); a['Cnet'][0,0,:3]=[np.nan,np.inf,-np.inf]; a['Cnet'][0,1,:3]=[-1,-2,0]; a['P'][:]=[1,1,0]; rows.append(('nonfinite-current',a))
    a=inputs(); a['P'][:]=[-32,-32,4]; a['G'][:]=0; rows.append(('growing-gaussian-overflow',a))
    a=inputs(); a['G'][:]=0; a['Hprev'][...,:3]=.125; a['PM'][:]=[np.inf,-np.inf,-80,80]; a['Pe'][:]=[80,-80]; a['Pn'][:]=[-80,80]; rows.append(('out-of-history',a))
    a=inputs(); a['G'][:]=0; a['Hprev'][...,:3]=np.arange(3)[None,:,None]/8; a['PM'][...,0]=np.nan; a['W4'][:]=[1,0,0,0]; rows.append(('nan-motion-is-sampled',a))
    a=inputs(); a['G'][:]=0; a['W4'][:]=ref.decode_unorm8(np.array([0,128,128,128],np.uint8)); a['PM'][:]=[0,0,1,0]; a['Pe'][:]=[-1,0]; a['Hprev'][...,:3]=np.arange(3)[None,:,None]/8; a['constants']=symbols(K3=.5); rows.append(('tie-loop-sensitive',a))
    return rows


def oracle_with_trace(a):
    traces=[]; active=None
    original_order,original_sample,original_blend=ref.candidate_order,ref.reconstruct_history,ref.blend_guard
    def order(w):
        nonlocal active
        o=original_order(w); active=[int(sum(int(o[i])<<(2*i) for i in range(4))),0,0,0]; traces.append(active); return o
    def sample(history,x,y,motion,fallback,reset,c):
        index=active[1].bit_count(); active[1]|=1<<index
        px,py=ref.add(x+.5,motion[0]),ref.add(y+.5,motion[1]); hh,wh=history.shape[:2]
        if reset or px<0 or px>wh or py<0 or py>hh: active[2]|=1<<index
        return original_sample(history,x,y,motion,fallback,reset,c)
    guard_call=0
    def blend(current,hist,alpha,c):
        nonlocal guard_call
        b=ref.fma(alpha,ref.fma(current,c.K12,-np.asarray(hist)),hist)
        if np.isnan(ref.add(b[2],ref.add(b[0],b[1]))): active[3]|=1<<(guard_call%2)
        guard_call+=1
        return original_blend(current,hist,alpha,c)
    with patch.object(ref,'candidate_order',side_effect=order),patch.object(ref,'reconstruct_history',side_effect=sample),patch.object(ref,'blend_guard',side_effect=blend): result=ref.output_stage(**a)
    return result,np.array(traces,np.uint32)


def prepare(arena,a,packed=False,*,input_bindings=None,half_inputs=False):
    """Allocate a spec54 call; returns (38 root constants, output offsets, byte sizes).

    input_bindings maps Cnet/G/P/W4/PM/Pe/Pn/Hprev to existing arena offsets.
    Bound inputs allocate/upload no oracle data; supplied arrays supply shape only.
    In packed mode Cnet/P are half4 and G/W4 byte4 (including ignored channel 3).
    PM/Pe/Pn/Hprev carry decoded f32 with 4/2/2/4 channels respectively, or binary16 with half_inputs
    (packed_head bits 2 and 5; the fast path, and rr_output_main used as its fallback).
    Output offsets/sizes correspond to linear_h f32 RGB, CH half4, Q half3,
    PH1 uint16 RGBA channel codes, and trace uint32[4] per history pixel.
    Launch main ceil(Wh*Hh/BLOCK), pool ceil(ceil(Wh/2)*ceil(Hh/2)/BLOCK),
    with a UAV barrier between (build.run supplies it), require_double=True;
    packed: dispatch_list() (prep, fast, fallback, pool) with the scratch area (last argument).
    """
    input_bindings={} if input_bindings is None else input_bindings
    names=('Cnet','G','P','W4','PM','Pe','Pn','Hprev')
    if set(input_bindings)-set(names): raise ValueError('unknown output-stage input binding')
    offsets=[]
    for n in names:
        if n in input_bindings:
            offsets.append(int(input_bindings[n])); continue
        if packed and n in ('Cnet','P'):
            v=np.zeros((*a[n].shape[:2],4),dtype='<f2'); v[...,:a[n].shape[2]]=a[n]
        elif packed and n in ('G','W4'):
            v=np.zeros((*a[n].shape[:2],4),dtype=np.uint8)
            decoded=ref.decode_unorm8(np.arange(256,dtype=np.uint8))
            v[...,:a[n].shape[2]]=np.searchsorted(decoded,a[n])
        elif half_inputs and n in ('PM','Pe','Pn','Hprev'): v=np.asarray(a[n],dtype='<f2')
        else: v=np.asarray(a[n],dtype='<f4')
        offsets.append(arena.add(v))
    offsets.append(arena.add(np.asarray(a['constants'].TR,dtype='<f4')))
    hh,wh=a['Hprev'].shape[:2]; hr,wr=a['G'].shape[:2]; wt=a['Cnet'].shape[1]
    sizes=[hh*wh*12,hh*wh*8,hh*wh*6,((hh+1)//2)*((wh+1)//2)*8,hh*wh*16]
    outputs=[arena.reserve(n+64,fill=0xA5) for n in sizes]
    fbits=lambda v:int(np.asarray(v,np.float32).view(np.uint32))
    args=offsets+outputs+[wr,hr,wt,wh,hh]+[fbits(v) for v in (*a['jitter'],a['E'],a['Ep'],a['invE'])]+[int(a['reset'])]
    args += [fbits(getattr(a['constants'],f.name)) for f in fields(a['constants']) if f.name!='TR']
    args.append(int(packed)|(36 if half_inputs else 0))
    args.append(arena.reserve(scratch_bytes(wr,hr,wh,hh)) if packed else 0)
    assert len(args)==NARGS,len(args)
    return args,outputs,sizes


def tolerated_half(actual,expected):
    actual,expected=np.asarray(actual,np.float16),np.asarray(expected,np.float16)
    lo=hi=expected.copy()
    with np.errstate(over='ignore',invalid='ignore'):
        for _ in range(2): lo=np.nextafter(lo,np.float16(-np.inf)); hi=np.nextafter(hi,np.float16(np.inf))
        ok=(np.isnan(actual)&np.isnan(expected))|((actual>=lo)&(actual<=hi))|(np.abs(actual.astype(np.float64)-expected.astype(np.float64))<=2**-9)
    # Infinity is a special value, not a finite half neighbour admitted through nextafter.
    ok=np.where(np.isinf(expected),actual==expected,ok)
    ok=np.where(np.isinf(actual)&~np.isinf(expected),False,ok)
    return ok


def read_outputs(img,a,offs,sizes):
    """Read output arrays in prepare() order; copy storage out of the arena."""
    hh,wh=a['Hprev'].shape[:2]
    shapes=[(hh,wh,3),(hh,wh,4),(hh,wh,3),((hh+1)//2,(wh+1)//2,4),(hh*wh,4)]
    dtypes=['<f4','<f2','<f2','<u2','<u4']
    return [img[p:p+n].copy().view(dt).reshape(shape) for p,n,dt,shape in zip(offs,sizes,dtypes,shapes)]


def same_storage(a,b):
    """Exact encoded bytes, including signed zeros and NaN payloads."""
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


def qualify(img,records,*,reset_pairs=(('reset-clean','reset-poison'),('reset-clean-packed','reset-poison-packed'))):
    """Check records (name, inputs, offsets, sizes, CPU result, CPU trace).

    Returns (case reports, exact reset-pair equality). Custom callers explicitly
    pass reset_pairs=() when no paired reset scenario is being evaluated.
    """
    rows=[]; actuals={}
    for name,a,offs,sizes,expected,traces in records:
        linear,ch,q,ph,trace=read_outputs(img,a,offs,sizes); actuals[name]=(ch,q,ph)
        tests={}
        for key,value,target in [('CH',ch,expected.CH),('Q',q,expected.Q),('PH1',ph,expected.PH1)]:
            ok=tolerated_half(value,target) if key!='PH1' else abs(value.astype(int)-target.astype(int))<=1
            tests[key]=dict(values=int(ok.size),outside=int(np.count_nonzero(~ok)),passed=bool(np.count_nonzero(~ok)*1000<=ok.size))
        exact=dict(order=bool(np.array_equal(trace[:,0],traces[:,0])),used=bool(np.array_equal(trace[:,1],traces[:,1])),fallback=bool(np.array_equal(trace[:,2],traces[:,2])),nan_guard=bool(np.array_equal(trace[:,3],traces[:,3])),CH_alpha=bool(np.all(ch[...,3].view(np.uint16)==0)),PH1_alpha=bool(np.all(ph[...,3]==3)),PH1_RGB_range=bool(np.all(ph[...,:3]<=1023)),guards=all(np.all(img[p+n:p+n+64]==0xA5) for p,n in zip(offs,sizes)))
        # Pooling and conversion must implement their exact rules on the actual linear intermediate.
        own_pool=ref.pool_history(linear,a['constants'])
        tests['PH1_from_GPU_linear']=dict(values=int(ph.size),outside=int(np.count_nonzero(abs(ph.astype(int)-own_pool.astype(int))>1)),passed=bool(np.all(abs(ph.astype(int)-own_pool.astype(int))<=1)))
        rows.append(dict(case=name,passed=all(t['passed'] for t in tests.values()) and all(exact.values()),comparison=tests,exact=exact))
    reset_equal=all(same_storage(a,b) for left,right in reset_pairs for a,b in zip(actuals[left],actuals[right]))
    return rows,reset_equal


def arithmetic_cases():
    rng=np.random.default_rng(540018)
    # Diverse finite f32 exponents plus the normal/subnormal rounding boundary.
    v=rng.integers(0,0xffffffff,(1024,4),dtype=np.uint32).view(np.float32)
    finite=np.isfinite(v); v=np.where(finite,v,np.float32(0))
    edge=[]
    for sign in (1,-1):
        for u in range(0x00fffff0,0x01000010):
            edge.append([sign*np.array(u,np.uint32).view(np.float32),.5,0,.5])
        for p in (np.float32(2**-126),np.float32(2**-100)):
            for b in [np.float32(-2**-24),np.nextafter(np.float32(-2**-24),np.float32(-1)),np.nextafter(np.float32(-2**-24),np.float32(0))]:
                edge.append([sign*p,b,sign*np.float32(2**-126),.5])
    for sign in (1,-1):
        # Regression: old scaled-f32 repair rounded this product up incorrectly.
        for bbits in (0x3f7ffffc,0x3f7ffffd,0x3f7ffffe,0x3f7fffff):
            edge.append([sign*np.array(0x00800001,np.uint32).view(np.float32),np.array(bbits,np.uint32).view(np.float32),0,.5])
    for x in (np.nan,-np.inf,np.inf,-0.,0.,.5,np.nextafter(np.float32(.5),np.float32(0)),np.nextafter(np.float32(.5),np.float32(1)),1): edge.append([x,1,0,x])
    v=np.concatenate([v,np.array(edge,np.float32)])
    # A separate sample near every RGB10 tie exercises exact scaled-product rounding.
    ties=np.array([(k+.5)/1023 for k in range(1023)],np.float32)
    tv=np.concatenate([np.nextafter(ties,np.float32(0)),ties,np.nextafter(ties,np.float32(1))])
    extra=np.zeros((len(tv),4),np.float32); extra[:,3]=tv; v=np.concatenate([v,extra])
    e=np.zeros((len(v),8),np.uint32)
    with np.errstate(all='ignore'):
        e[:,0]=ref.fma(v[:,0],v[:,1],v[:,2]).view(np.uint32)
        e[:,1]=ref.mul(v[:,0],v[:,1]).view(np.uint32)
        e[:,2]=ref.add(v[:,0],v[:,2]).view(np.uint32)
        e[:,3]=ref.unorm_codes(v[:,3],10)
        h=ref.round_f16(v[:,:3])
        e[:,4]=ref.fma16(h[:,0],h[:,1],h[:,2]).astype(np.float16).view(np.uint16)
        e[:,5]=ref.mul16(h[:,0],h[:,1]).astype(np.float16).view(np.uint16)
    return v,e


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--json',required=True); ap.add_argument('--mutation',choices=['order','reset','code']); args=ap.parse_args()
    if 'WG_RR_BUILD' not in os.environ: ap.error('set WG_RR_BUILD to a private clean-root directory')
    arena=build.Arena(); dispatches=[]; records=[]
    defines={'WG_BLOCK':BLOCK,'BAD_ORDER':int(args.mutation=='order'),'BAD_RESET':int(args.mutation=='reset'),'BAD_CODE':int(args.mutation=='code')}
    kernels=kernel_set(defines,names=('main','pool','order','arith','prep','fast','fallback'))
    for name,a in fixtures():
        print('oracle',name,flush=True)
        expected,traces=oracle_with_trace(a)
        for packed in (False,True):
            kargs,offs,sizes=prepare(arena,a,packed,half_inputs=packed); wh,hh=kargs[17:19]
            # decoded inputs: rr_output_main; packed: the fast path (packed_decoded_exact compares their bytes)
            dispatches += dispatch_list(kargs,fast=packed)
            records.append((name+('-packed' if packed else ''),a,offs,sizes,expected,traces))
    weights=np.array(list(itertools.product([0,.5,1,np.nan],repeat=4)),np.float32)
    wo=arena.add(weights); oo=arena.reserve(len(weights)*4+64,fill=0xA5)
    probe=[0]*NARGS; probe[0]=wo; probe[13]=oo; probe[14]=len(weights)
    dispatches.append(('order',(len(weights)+BLOCK-1)//BLOCK,probe))
    expected_order=np.array([sum(int(o)<<(2*i) for i,o in enumerate(ref.candidate_order(w))) for w in weights],np.uint32)
    arithmetic,arith_expected=arithmetic_cases()
    aptr=arena.add(arithmetic); aout=arena.reserve(len(arithmetic)*32+64,fill=0xA5)
    probe=[0]*NARGS; probe[0]=aptr; probe[13]=aout; probe[14]=len(arithmetic)
    dispatches.append(('arith',(len(arithmetic)+BLOCK-1)//BLOCK,probe))
    img,log=build.run(arena,kernels,dispatches,require_double=True,workdir=str(Path(build.BUILD)/('run-'+(args.mutation or 'normal'))))
    rows,reset_equal=qualify(img,records)
    actual_order=img[oo:oo+len(weights)*4].copy().view('<u4')
    probe_ok=bool(np.array_equal(actual_order,expected_order) and np.all(img[oo+len(weights)*4:oo+len(weights)*4+64]==0xA5))
    arith_actual=img[aout:aout+len(arithmetic)*32].copy().view('<u4').reshape(-1,8)
    arith_ok=(arith_actual==arith_expected)
    for col in (0,1,2):
        arith_ok[:,col]|=np.isnan(arith_actual[:,col].copy().view('<f4'))&np.isnan(arith_expected[:,col].copy().view('<f4'))
    for col in (4,5):
        arith_ok[:,col]|=((arith_actual[:,col]&0x7fff)>0x7c00)&((arith_expected[:,col]&0x7fff)>0x7c00)
    arithmetic_passed=bool(np.all(arith_ok) and np.all(img[aout+len(arithmetic)*32:aout+len(arithmetic)*32+64]==0xA5))
    arith_mismatches=[dict(index=int(i),column=int(j),input=arithmetic[i].tolist(),actual=int(arith_actual[i,j]),expected=int(arith_expected[i,j])) for i,j in zip(*np.where(~arith_ok))][:30]
    # every output byte of the fast path (packed records) against rr_output_main (decoded records), incl. linear / trace
    packed_diff={}
    for r,p in zip(records[::2],records[1::2]):
        for name,x,y in zip(('linear','CH','Q','PH1','trace'),read_outputs(img,r[1],r[2],r[3]),read_outputs(img,p[1],p[2],p[3])):
            if not same_storage(x,y):
                xb=np.ascontiguousarray(x).view(np.uint8).reshape(x.shape[0],-1); yb=np.ascontiguousarray(y).view(np.uint8).reshape(y.shape[0],-1)
                packed_diff[r[0]+':'+name]=int(np.count_nonzero(np.any(xb!=yb,axis=1)))
    packed_equal=not packed_diff
    report=dict(packed_decoded_exact=packed_equal,packed_decoded_differences=packed_diff,passed=packed_equal and all(r['passed'] for r in rows) and reset_equal and probe_ok and arithmetic_passed,arithmetic=dict(cases=len(arithmetic),passed=arithmetic_passed,mismatches=arith_mismatches),mutation=args.mutation,cases=rows,reset_independence=reset_equal,tie_probe=dict(cases=len(weights),passed=probe_ok,mismatches=int(np.count_nonzero(actual_order!=expected_order))),host_log=log,
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(build.KDIR)/'rr_output.hlsl',Path(build.RR)/'tests/test_gpu_output.py')},shaders={n:hashlib.sha256(Path(k[0]).read_bytes()).hexdigest() for n,k in kernels.items()},scope='synthetic spec54 CH/Q/PH1; decoded-buffer and packed-head-layout fixtures; no live head connection, image views, learned data, resolve, game, or performance claim')
    Path(args.json).parent.mkdir(parents=True,exist_ok=True); Path(args.json).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS' if report['passed'] else 'FAIL',flush=True)
    for row in rows:
        if not row['passed']: print(row,flush=True)
    print('packed (fast path) == decoded (rr_output_main) bytes:',packed_equal,packed_diff,flush=True)
    print('reset_independence',reset_equal,'tie_probe',probe_ok,'arithmetic',arithmetic_passed,arith_mismatches,flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
