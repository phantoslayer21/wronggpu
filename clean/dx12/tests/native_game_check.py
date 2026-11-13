# SPDX-License-Identifier: MIT
"""Run the private synthetic candidate and retain raw validation evidence.

Run this script only under a GPU lock (WG_GPU_LOCK, BUILDING.md) when other GPU jobs may run. It performs no
performance measurement: readbacks and debug checks deliberately synchronize.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',required=True)
    p.add_argument('--build',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True,help='New evidence directory; existing directory is refused')
    p.add_argument('--queue',choices=('direct','compute'),default='direct')
    p.add_argument('--oracle',choices=('native','hip'),default='native')
    p.add_argument('--shape',nargs=4,type=int,default=(853,481,1279,721),metavar=('RW','RH','OW','OH'))
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--cycles',type=int,default=2)
    a=p.parse_args()
    if a.frames<3 or a.cycles<1 or min(a.shape)<1:
        p.error('at least 3 frames, 1 lifecycle cycle and positive dimensions are required')
    out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for name in ('HIP_PATH','ROCM_PATH','LLVM_PATH','HIP_DEVICE_LIB_PATH','HIP_CLANG_PATH','WG_DX12_KERNELS'):
        env.pop(name,None)
    trace=out/'dispatch.trace'
    env['WG_LAUNCH_TRACE']=str(trace)
    env['TMP']=env['TEMP']=str(out)
    cmd=[a.exe,a.build,a.model,a.queue,a.oracle,*map(str,a.shape),str(a.frames),str(a.cycles)]
    start=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,errors='replace',timeout=900)
    (out/'stdout.log').write_text(result.stdout)
    (out/'stderr.log').write_text(result.stderr)
    lines=trace.read_text().splitlines() if trace.exists() else []
    native=Counter(x.split()[0] for x in lines if x.endswith(' d3d12'))
    other=Counter(x.split()[0] for x in lines if not x.endswith(' d3d12'))
    # Every selected neural stage must execute once for each native frame. The
    # candidate also dispatches one extra outstanding frame per lifecycle cycle.
    frame_count=a.cycles*(a.frames+1+(a.frames if a.oracle=='native' else 0))
    stages=('input_encoder','swin_block_c64_nc','patch_merge_c64','swin_block_c96_nc','patch_merge_c96',
            'bneck/swin_bneck6_nc','swin_block_c96_add_ex_nc','expand_c96',
            'd1p12/swin_block_c64_add_dec0_nc','w64/post_tf_f32_p12_diet_vsit')
    gates={'exit_zero':result.returncode==0,'final_pass':'PASS native game candidate' in result.stdout,
           'all_required_stages':all(native[s]==frame_count for s in stages),
           'borrowed_driver_probe':native['native_driver_probe']==a.cycles,
           'supplied_exposure':native['exposure_pair_diet']>0,'auto_exposure':native['exposure_auto_diet']>0,
           'pre_exposure_helpers':native['color_divide']>0 and native['scale_rgb']>0,
           'mutation_detected':'mutation=1' in result.stdout,
           'ready_fence_order':result.stdout.count('PASS delayed input-ready fence held final completion')==a.cycles,
           'debug_available_clean':'DEBUG enabled' in result.stdout and 'DEBUG clean' in result.stdout and
              'DEBUG unavailable' not in result.stdout and 'DEBUG ERROR' not in result.stdout}
    if a.oracle=='native':
        gates['native_only_trace']=not other
        gates['fresh_module_checks']=result.stdout.count('MODULES after-destroy')==a.cycles and 'MODULES fresh' in result.stdout
        gates['no_hip_modules']='amdhip' not in result.stdout.lower() and 'hiprtc' not in result.stdout.lower()
    manifest=Path(a.build)/'dx12/manifest.json'
    report={'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,
            'returncode':result.returncode,'gates':gates,'pass':all(gates.values()),
            'native_trace_counts':dict(native),'other_trace_counts':dict(other),'expected_neural_frames':frame_count,
            'sha256':{str(Path(a.exe).resolve()):sha(a.exe),str(manifest.resolve()):sha(manifest)},
            'domain':'original synthetic weights/frames; shared original Converter arithmetic; Model M selected route',
            'exclusions':'unwritten b0-b4/dec1/display/B; unused C allocation tail; previous state on reset',
            'performance':'not a timing run; synchronized diagnostic validation only',
            'telemetry':'must be supplied by outer owned GPU-lock/activity monitor'}
    for entry in json.loads(manifest.read_text())['kernels']:
        path=manifest.parent/entry['file']
        report['sha256'][str(path.resolve())]=sha(path)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pass':report['pass'],'gates':gates,'report':str(out/'report.json')},indent=2))
    if result.returncode:
        print(result.stderr[-4000:],file=sys.stderr)
    return 0 if report['pass'] else 1


if __name__=='__main__':
    sys.exit(main())
