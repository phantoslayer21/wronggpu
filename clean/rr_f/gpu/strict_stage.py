# SPDX-License-Identifier: MIT
"""Opt-in correctness-only fragment/common-stage/D0R qualification; no default changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import build, exact_dot, layout, stage
from ..reference import fmt
from ..reference.stages import run_stage
from ..tests import synth
from ..head.dx12 import run_dx12 as head


def acc_pack(m):
    return m.reshape(2, 8, 16).transpose(0, 2, 1).reshape(32, 8)


def acc_unpack(a):
    return a.reshape(2, 16, 8).transpose(0, 2, 1).reshape(16, 16)


def fragment_probe(out):
    out = Path(out)
    rng = np.random.default_rng(5102)
    finite = np.array([x for x in range(256) if x & 127 != 127], np.uint8)
    count = 16
    inp = np.zeros((count, 32, 64), np.uint8)
    expected = []
    for w in range(count):
        a = rng.choice(finite, (16, 32))
        b = rng.choice(finite, (32, 16))
        c = rng.integers(0, 65536, (16, 16), dtype=np.uint16)
        if w == 0:
            a[:] = 0; b[:] = 0
        elif w == 1:
            a[:, 16:] = a[:, :16]
            b[16:] = b[:16] ^ 128
        elif w == 2:
            a[7, 19] = 127; b[5, 9] = 255
            c[:, 2] = 0x7c00; c[:, 3] = 0xfc00
        elif w == 3:
            a[:] = 0; b[:] = 0
            a[:, :2] = [0x38, 0x10]; b[0] = 0x38; b[1] = 0x08
            c[:] = 1
        af, bf = layout.frags(a)[0], layout.frags(b.T)[0]
        inp[w, :, 0:8], inp[w, :, 8:16] = af[0], bf[0]
        inp[w, :, 16:24], inp[w, :, 24:32] = af[1], bf[1]
        with np.errstate(invalid='ignore'):
            cp = acc_pack(fmt.from_f16_bits(c)).astype(np.float32)
        inp[w, :, 32:] = cp.view(np.uint8).reshape(32, 32)
        expected.append(fmt.to_f16_bits(fmt.mm(fmt.d8(a), fmt.d8(b), fmt.from_f16_bits(c))))
    ar = build.Arena(); pin = ar.add(inp); pout = ar.reserve(count * 32 * 32)
    dx = build.compile_kernel('rr_exact_dot.hlsl', 'rr_exact_frag', {})
    img, log = build.run(ar, {0: (dx, 3)}, [(0, count // 4, [pin, pout, count * 32])],
                         require_double=True, lock=True, workdir=str(out / 'run'))
    raw = img[pout:pout + count * 32 * 32].view(np.float32).reshape(count, 32, 8)
    actual = np.stack([fmt.to_f16_bits(acc_unpack(x).astype(np.float64)) for x in raw])
    expected = np.stack(expected)
    mismatches = int(np.count_nonzero(~exact_dot.same_bits(actual, expected)))
    (out / 'host.log').write_text(log, encoding='utf-8')
    actual.tofile(out / 'actual.u16'); expected.tofile(out / 'expected.u16')
    return dict(count=int(actual.size), mismatches=mismatches, passed=mismatches == 0)


def common_probe(out, seed=5103, hot=False, zmode='supplied', step='strict', special=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    name = 'IN0'
    wc, _ = synth.window_consts(rng, 'A')
    consts = synth.stage_consts(rng, hot=hot)
    weights = synth.stage_weights(rng, name, hot=hot)
    inputs = synth.stage_inputs(rng, name, 8, 8, hot=hot)
    if special in ('x_nan_pos', 'x_nan_neg'):
        inputs['x'][0, 0, 0] = 0x7f if special == 'x_nan_pos' else 0xff
    elif special in ('bias_nan_pos', 'bias_nan_neg'):
        weights['bo'][0] = fmt.from_f16_bits(np.uint16(0x7e00 if special == 'bias_nan_pos' else 0xfe00))
    elif special is not None:
        raise ValueError('unknown special fixture')
    expected, aux = run_stage(name, inputs, weights, consts, wc, (0, 0))
    call = stage.StageCall(name, inputs, weights, consts, wc, (0, 0),
                           Z=aux['Z'] if zmode == 'supplied' else None)
    defines = stage.stage_defines(name, zmode, step='E' if step == 'strict' else step)
    if step == 'strict':
        defines['RR_STEP'] = 2
    dx = build.compile_kernel('rr_block.hlsl', 'rr_block', defines, tag='rr_IN0_'+step+'_'+zmode)
    img, log = build.run(call.arena, {0: (dx, len(call.args))}, [(0, call.groups(), call.args)],
                         require_double=True, lock=True, workdir=str(out / 'run'))
    got = call.outputs(img)
    comparisons = {k: stage.compare(k, v, expected[k]) for k, v in got.items()}
    for k, v in got.items():
        v.tofile(out / (k+'-actual.bin'))
        (np.asarray(expected[k], np.uint8) if v.dtype == np.uint8 else stage.f16b(expected[k])).tofile(out / (k+'-expected.bin'))
    (out/'host.log').write_text(log, encoding='utf-8')
    return dict(stage=name, seed=seed, hot=hot, zmode=zmode, step=step, special=special, outputs=comparisons,
                passed=all(x['diff'] == 0 for x in comparisons.values()))


def head_probe(out, case_name='stage_edge', strict=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = head.V.load_manifest(head.VDIR)
    c = next(c for c in manifest['cases'] if c['name'] == case_name)
    ld = lambda x: head.V.load_array(head.VDIR, x)
    p = head.case_params(c)
    inputs = c['inputs']
    weights = {n: ld(inputs[n]) for n in ('Wq', 'Wv', 'Wo', 'W1', 'W2', 'Tb', 'bo', 'b1', 'b2')}
    ha, job = head.make_job(p, 'stage', 'dbgz', W=weights, x=ld(inputs['x']), hold=ld(inputs['hold']),
                            hc=ld(inputs['Hc']), Zref=ld(c['diagnostics']['Z']))
    words = np.frombuffer(job, '<u4')
    nargs = int(words[2]); args = words[3:3+nargs].tolist(); groups = int(words[3+nargs])
    ar = build.Arena(); ar.size = ha.size
    ar.parts = [(off, np.frombuffer(data, np.uint8)) for off, data in ha.init]
    source = Path(head.HERE)/'rr_d0r.hlsl'
    defines = dict(WG_BLOCK=128, WG_WAVE=32, RR_DEBUG=1, RR_ZSUP=1, RR_STRICT_STEP=int(strict))
    key = hashlib.sha256(source.read_bytes() + build._key(str(source), 'rr_d0r_head', defines).encode()).hexdigest()[:16]
    dx = str(Path(build.BUILD)/'dxil'/('rr_head_step_'+key+'.dxil'))
    Path(dx).parent.mkdir(parents=True, exist_ok=True)
    bd = build.build_dx12
    asm = bd.build_asm_tool(build.BUILD)
    if not Path(dx).exists():
        bd.compile_hlsl(str(source), 'rr_d0r_head', dx, defines,
                        includes=[str(Path(build.REPO)/'clean/dx12/kernels')], asm_tool=asm)
    img, log = build.run(ar, {0: (dx, nargs)}, [(0, groups, args)],
                         require_double=True, lock=True, workdir=str(out/'run'))
    output_path = out/'head.out'
    output_path.write_bytes(b''.join(img[off:off+size].tobytes() for _, off, size in ha.reads))
    j = dict(variant='dbgz', out=str(output_path), reads=ha.reads)
    report = head.compare_case(c, [j])
    report['strict'] = strict
    report['shader_sha256'] = hashlib.sha256(Path(dx).read_bytes()).hexdigest()
    # Existing verdict expects an own-Z 'dbg' record. Here the identical record
    # is supplied-Z; retain every hard rule, including exact y2h, without changing
    # the existing comparator or widening any tolerance.
    record = report['variants']['dbgz']
    report['verdict'] = head.verdict(dict(report, variants={'dbg': record, 'dbgz': record}))
    report['passed'] = report['verdict'] != 'failing'
    (out/'host.log').write_text(log, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--only', choices=('fragment', 'common', 'head', 'all', 'default-controls', 'nan-controls'), default='all')
    args = parser.parse_args()
    if 'WG_RR_BUILD' not in os.environ:
        parser.error('set private WG_RR_BUILD first')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.only == 'nan-controls':
        for special in ('x_nan_pos', 'x_nan_neg', 'bias_nan_pos', 'bias_nan_neg'):
            report[special] = common_probe(out/special, special=special)
            print(json.dumps(report[special]), flush=True)
    if args.only == 'default-controls':
        for step in ('E', 'X'):
            report['common_'+step] = common_probe(out/('common_'+step), step=step)
            print(json.dumps(report['common_'+step]), flush=True)
        report['head_default'] = head_probe(out/'head_default', case_name='stage_basic', strict=False)
        print(json.dumps(report['head_default']), flush=True)
    if args.only in ('all', 'fragment'):
        report['fragment'] = fragment_probe(out/'fragment')
        print(json.dumps(report['fragment']), flush=True)
        if not report['fragment']['passed']:
            (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return 1
    if args.only in ('all', 'common'):
        report['common'] = common_probe(out/'common')
        print(json.dumps(report['common']), flush=True)
        (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.only in ('all', 'head'):
        report['head'] = head_probe(out/'head')
        print(json.dumps(report['head']), flush=True)
    report['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__), Path(build.KDIR)/'rr_exact_math.hlsli', Path(build.KDIR)/'rr_exact_dot.hlsl',
         Path(build.KDIR)/'rr_block.hlsl', Path(head.HERE)/'rr_d0r.hlsl')}
    report['passed'] = all(v['passed'] for k, v in report.items() if k != 'sources')
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
