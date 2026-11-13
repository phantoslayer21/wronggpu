# SPDX-License-Identifier: MIT
"""Original synthetic HDR output-storage regression; no model data or game captures.

Run under the shared GPU lock. --baseline additionally checks unchanged FP32,
history and in-range FP16 bytes against accepted code objects. This is a post
stage test, not proof that a particular game's visual symptom is fixed.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from clean.reference import post as ref
from clean.runner import post as ps
from clean.runner.stage_runner import Runner


def tf_plan(dev, ow, oh, brightness, exposure, pi, half):
    dw, dh = ps.recon_size_for(ow), ps.recon_size_for(oh)
    rw, rh = max(1, ow * 2 // 3), max(1, oh * 2 // 3)
    x = ref.random_inputs(np.random.default_rng(1729), rw, rh, dw=dw, dh=dh)
    x.update(color=np.full((rh, rw, 3), brightness, np.float32),
             exposure=exposure, exposure_prev=exposure, reset=True,
             jitter=np.zeros(2, np.float32))
    data = ps.pack12_inputs(ps.post_device_inputs(x))
    pl = ps._Plan(dev)
    try:
        for name, payload in data.items():
            pl.alloc(name, len(payload), payload)
        pl.alloc('w', 16, ps.pack({'s': np.array(1., np.float32)}))
        pl.alloc('history', dw * dh * 8)
        pl.alloc('reduced', max((dw // 2) * (dh // 2), 1) * 4)
        # Exercise a padded row even for aligned output widths.
        pitch = ((ow * 8 + 255) // 256 + 1) * 256
        pl.alloc('out', pitch * oh if half else ow * oh * 12)
        expo = np.array([exposure, exposure, 0, 1, .01 / exposure], np.float32)
        pl.alloc('expo', expo.nbytes, expo.tobytes())
        b, P, I, F = pl.plan['bufs'], C.c_void_p, C.c_int, C.c_float
        args = [P(b[k]) for k in ('color', 'A', 'B', 'C', 'hist_prev',
                'm_near', 'm_center', 'm_far', 'w', 'history', 'reduced', 'out')]
        args += [ps.post_params(x), P(b['expo']), I(ow), I(oh), F(.01 / exposure),
                 F(pi), I(pitch), I(68), I(0x7fffffff), I(68)]
        kernel = 'post_tf_' + ('p12' if half else 'f32_p12') + '_diet_vsit'
        pl.plan['launches'] = [(kernel, ps.tf_grid(ow, oh, 68), 512, args)]
        pl.plan['outputs'] = [('history', (dh, dw, 3)),
                              ('reduced', (dh // 2, dw // 2, 3)), ('out', (oh, ow, 3))]
        def decode_half(raw, shape):
            rows = np.frombuffer(raw, np.uint8).reshape(oh, pitch)
            return rows[:, :ow * 8].copy().view(np.float16).reshape(oh, ow, 4)[..., :3].astype(np.float32)
        pl.plan['decoders'] = {'history': ps.decode_history,
                              'reduced': ps.decode_reduced,
                              'out': decode_half if half else ps.decode_f32}
        return pl.plan
    except Exception:
        pl.release()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', required=True)
    ap.add_argument('--baseline')
    ap.add_argument('--rocm', required=True)
    ap.add_argument('--json', required=True)
    a = ap.parse_args()
    for key in ('LLVM_PATH', 'ROCM_PATH', 'HIP_PATH', 'HIP_DEVICE_LIB_PATH'):
        os.environ.pop(key, None)
    runners = {'candidate': Runner(a.build, a.rocm, backend='hip')}
    if a.baseline:
        runners['baseline'] = Runner(a.baseline, a.rocm, backend='hip')
    rows, errors = [], []
    for ow, oh in ((16, 16), (61, 33)):
        for brightness, exposure in ((1., 1.), (60000., .001), (65504., .001)):
            for pi in (1., 2.):
                outputs = {}
                for side, runner in runners.items():
                    outputs[side] = {half: ps.run_plan(runner,
                        tf_plan(runner.dev, ow, oh, brightness, exposure, pi, half),
                        ps.CODE_OBJECT_W64) for half in (False, True)}
                got = outputs['candidate']
                f32 = got[False]['out'] * np.float32(pi)
                want = np.clip(f32, -65504., 65504.).astype(np.float16).astype(np.float32)
                out = got[True]['out']
                bad = int(np.count_nonzero(out.view(np.uint32) != want.view(np.uint32)))
                row = {'output': [ow, oh], 'brightness': brightness, 'exposure': exposure,
                       'pre_exposure': pi, 'half_nonfinite': int((~np.isfinite(out)).sum()),
                       'half_vs_saturated_fp32_changed_values': bad}
                if bad or row['half_nonfinite']:
                    errors.append(row)
                if 'baseline' in outputs:
                    old = outputs['baseline']
                    row['baseline_half_nonfinite'] = int((~np.isfinite(old[True]['out'])).sum())
                    changed = {f'{half}:{k}': int(np.count_nonzero(
                        got[half][k].view(np.uint32) != old[half][k].view(np.uint32)))
                        for half in (False, True) for k in ('history', 'reduced', 'out')
                        if not (half and k == 'out')}
                    row['unchanged_path_differences'] = changed
                    mask = np.isfinite(old[True]['out'])
                    row['in_range_half_differences'] = int(np.count_nonzero(
                        got[True]['out'][mask].view(np.uint32) != old[True]['out'][mask].view(np.uint32)))
                    if any(changed.values()) or row['in_range_half_differences']:
                        errors.append(row)
                rows.append(row)
    report = {'scope': 'synthetic production wave64 TF output; no network or game quality claim',
              'cases': rows, 'failed_checks': len(errors)}
    Path(a.json).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(rows), 'failed_checks': len(errors),
                     'candidate_nonfinite': sum(r['half_nonfinite'] for r in rows),
                     'baseline_nonfinite': sum(r.get('baseline_half_nonfinite', 0) for r in rows)}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
