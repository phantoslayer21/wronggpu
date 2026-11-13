# SPDX-License-Identifier: MIT
"""Connected synthetic GPU body/head -> spec54 history outputs; no input stage or resolve.

All 19 dispatches execute in one arena/submission. CPU references are evaluated
for comparison only. The four head bindings directly consume packed GPU output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import build, network, output
from ..reference import output as output_ref
from ..tests.test_output import symbols


HEAD_LINKS={'Cnet':'cnet','G':'gates','P':'prec','W4':'candw'}


def output_inputs(head):
    return dict(Cnet=head['Cnet'],G=output_ref.decode_unorm8(head['GATES_code'])[...,:3],
                P=head['PREC'][...,:3],W4=output_ref.decode_unorm8(head['CANDW_code']),
                PM=np.zeros((19,29,4),np.float16),Pe=np.zeros((19,29,2),np.float16),
                Pn=np.zeros((19,29,2),np.float16),Hprev=np.zeros((30,45,4),np.float16),
                output_size=(29,19),jitter=(.125,-.125),E=np.float32(1.25),Ep=np.float32(.75),
                invE=np.float32(.8),reset=False,constants=symbols())


def append_output(arena,kernels,dispatches,a,base,head_arena,mutation=False):
    expected={name:base+head_arena.ptr[key] for name,key in HEAD_LINKS.items()}
    bound=expected.copy()
    if mutation:
        bound['Cnet']=arena.reserve(a['Cnet'].size*2)
    args,offs,sizes=output.prepare(arena,a,packed=True,input_bindings=bound)
    for name in ('main','pool'):
        kernels['output-'+name]=(build.compile_kernel('rr_output.hlsl','rr_output_'+name,
                                                      {'WG_BLOCK':output.BLOCK}),output.NARGS)
    wh,hh=args[17:19]
    dispatches.extend([('output-main',(wh*hh+output.BLOCK-1)//output.BLOCK,args),
                       ('output-pool',(((wh+1)//2)*((hh+1)//2)+output.BLOCK-1)//output.BLOCK,args)])
    return args,offs,sizes,expected,bound


def head_from_gpu(img,base,head_arena):
    def read(name,shape,dtype):
        off=base+head_arena.ptr[name]
        n=int(np.prod(shape))*np.dtype(dtype).itemsize
        return img[off:off+n].copy().view(dtype).reshape(shape)
    return dict(Cnet=read('cnet',(32,32,4),'<f2'),GATES_code=read('gates',(19,29,4),np.uint8),
                PREC=read('prec',(19,29,4),'<f2'),CANDW_code=read('candw',(19,29,4),np.uint8))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json',required=True)
    ap.add_argument('--mutation',action='store_true',help='disconnect the Cnet producer; must fail numerically')
    args=ap.parse_args()
    if 'WG_RR_BUILD' not in os.environ: ap.error('set WG_RR_BUILD to private clean-root scratch')
    ar,kernels,dispatches,calls,reference,base,ha,body_bindings,regroup=network.prepare()
    a=output_inputs(reference['head'])
    output_args,offs,sizes,expected_bindings,_=append_output(ar,kernels,dispatches,a,base,ha,args.mutation)
    bound=dict(zip(HEAD_LINKS,output_args[:4]))
    print('Computing full prepared-graph CPU reference',flush=True)
    expected,trace=output.oracle_with_trace(a)
    print(f'Running {len(dispatches)} connected GPU dispatches',flush=True)
    img,log=build.run(ar,kernels,dispatches,require_double=True,
                      workdir=str(Path(build.BUILD)/('pipeline-mutant' if args.mutation else 'pipeline')))
    body_report=network.analyze(img,calls,reference,base,ha,regroup)
    full_report,_=output.qualify(img,[('full-prepared-graph',a,offs,sizes,expected,trace)],reset_pairs=())
    # Head tolerances can legally change a downstream discrete decision. Check
    # output's hard rules against the GPU head values as well as the whole CPU graph.
    actual_a=output_inputs(head_from_gpu(img,base,ha))
    own_expected,own_trace=output.oracle_with_trace(actual_a)
    own_report,_=output.qualify(img,[('from-actual-GPU-head',actual_a,offs,sizes,own_expected,own_trace)],reset_pairs=())
    connected=bound==expected_bindings and output_args[-1]==1
    # This fixture deliberately uses a stricter whole-graph gate as well as the
    # local output-stage contract; it is not a universal model acceptance rule.
    passed=body_report['passed'] and full_report[0]['passed'] and own_report[0]['passed'] and connected
    numerical_change=body_report['passed'] and any(
        not comparison['passed'] for row in (full_report[0],own_report[0])
        for comparison in row['comparison'].values())
    report=dict(passed=passed,mutation=args.mutation,body=body_report,full_graph=full_report[0],
                output_from_gpu_head=own_report[0],connected=connected,expected_head_bindings=expected_bindings,
                actual_head_bindings=bound,packed_head=output_args[-1],mutation_detected_numerically=bool(args.mutation and numerical_change),
                body_bindings=body_bindings,dispatches=[d[0] for d in dispatches],
                host_log=log,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__),Path(network.__file__),Path(output.__file__),Path(output_ref.__file__),
                     Path(build.KDIR)/'rr_output.hlsl')},
                shaders={n:hashlib.sha256(Path(k[0]).read_bytes()).hexdigest() for n,k in kernels.items()},
                scope='one prepared synthetic frame; live GPU body/head->CH/Q/PH1; no input/resolve, temporal runtime, player model or game')
    Path(args.json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.json).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS' if passed else 'FAIL',flush=True)
    print(json.dumps(dict(body=body_report['passed'],full_graph=full_report[0],from_gpu_head=own_report[0]),indent=2),flush=True)
    if args.mutation and not numerical_change:
        print('Invalid negative control: upstream body failed or no numeric output difference',flush=True)
        return 2
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
