# SPDX-License-Identifier: MIT
"""Retain direct-resource correctness or matched benchmark evidence.

Run only under the machine-wide the GPU lock (WG_GPU_LOCK, BUILDING.md). This tool does not establish
GPU isolation: qualified timing also needs root-owned pre/during/post telemetry.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

COMMON = ('swin_block_c64_nc','patch_merge_c64','swin_block_c96_nc','patch_merge_c96',
          'bneck/swin_bneck6_nc','swin_block_c96_add_ex_nc','expand_c96',
          'd1p12/swin_block_c64_add_dec0_nc')
MARKERS = {'native_direct_begin':('direct','begin'),'native_direct_end':('direct','end'),
           'native_staged_begin':('staged','begin'),'native_staged_end':('staged','end')}
CONVERSION = ('native_staged_input_conversion','native_staged_output_conversion')
# The post kernel the runtime selects (runtime.cpp: the cap-352 kernel where tf_supported takes the shape, else
# post_tf4_dl_* for native AA and ratios between 1.0 and 1.5, Brief 15; WG_RUNTIME_OPTIONS post_tf4=force: always the
# latter); set by main from --shape
POST='w64/post_tf_f32_p12_diet_vsit'


# The neural launches with the native D3D12 fusions of runtime.cpp kDefaultD3d12Fusion (pm,xe,mb,rv,xo,xd; its lc is
# Model L's): enc2's block
# with enc1's merge in reverse order (pm, rv), the bottleneck with enc2's merge and the overlap counters (mb, xo), dec2's
# block with dec1's expand (xe) launched as xd on frames of at most kXdMaxPixels render pixels (WG_D3D12_XD_MAX_PIXELS)
# else xo, and the fused dec1 + dec0 in its xd form on those frames. Set by main from --shape (COMMON above: the
# unfused list, kept for WG_D3D12_FUSION=none and the CPU fixtures).
XD_MAX_PIXELS=1200000
DIRECT_HELPERS=('f16lut_build_direct',)   # the direct post's output table (wg_f16lut.hlsli), at most once per frame


def fused_common(rw,rh):
    xd=rw*rh<=int(os.environ.get('WG_D3D12_XD_MAX_PIXELS',XD_MAX_PIXELS) or 1<<62)
    return ('swin_block_c64_nc','swin_block_c96_pm_rev_nc','bneck/swin_bneck6_mbo_nc',
            'swin_block_c96_add_xe_xd_nc' if xd else 'swin_block_c96_add_xe_xo_nc',
            'd1p12/swin_block_c64_add_dec0_xd_nc' if xd else 'd1p12/swin_block_c64_add_dec0_nc')


def post_kernel(rw,rh,ow,oh,options=''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from clean.runner import post as PS
    dw,dh=(3*ow+1)//2,(3*oh+1)//2
    if PS.tf_supported(rw,rh,dw,dh) and 'post_tf4=force' not in options:
        return 'w64/post_tf_f32_p12_diet_vsit'
    return 'w64/post_tf4_dl_f32_p12_diet_vsit'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trace_scopes(lines):
    scopes=[]
    errors=[]
    current=None
    outside=Counter()
    for index,line in enumerate(lines,1):
        fields=line.split()
        if len(fields)!=4 or fields[-1]!='d3d12':
            errors.append({'line':index,'problem':'not a native D3D12 trace record','text':line})
            continue
        name=fields[0]
        if name in MARKERS:
            route,kind=MARKERS[name]
            if kind=='begin':
                if current is not None:
                    errors.append({'line':index,'problem':'nested frame begin'})
                current={'route':route,'begin_line':index,'kernels':Counter()}
            elif current is None or current['route']!=route:
                errors.append({'line':index,'problem':'unmatched frame end'})
            else:
                current['end_line']=index
                scopes.append(current)
                current=None
        elif current is None:
            outside[name]+=1
        else:
            current['kernels'][name]+=1
    if current is not None:
        errors.append({'line':current['begin_line'],'problem':'unfinished frame'})
    for scope in scopes:
        k=scope['kernels'];direct=scope['route']=='direct'
        suffix='_direct' if direct else ''
        expected=(*COMMON,'input_encoder'+suffix,POST+suffix)
        exposure=('exposure_auto_diet'+suffix,'exposure_pair_diet'+suffix)
        if any(k[s]!=1 for s in expected) or sum(k[s] for s in exposure)!=1:
            errors.append({'line':scope['begin_line'],'problem':'missing/repeated selected neural stage','kernels':dict(k)})
        if direct:
            # An exact allow-list catches old helpers, standalone conversion
            # markers, unexpected fallback/copy/helper dispatches or variants.
            if any(k[s]>1 for s in DIRECT_HELPERS):
                errors.append({'line':scope['begin_line'],'problem':'repeated direct helper','kernels':dict(k)})
            extra=set(k)-set(expected)-set(exposure)-set(DIRECT_HELPERS)
            if extra:
                errors.append({'line':scope['begin_line'],'problem':'unexpected direct frame work','extra':sorted(extra)})
        elif any(k[s]!=1 for s in CONVERSION):
            errors.append({'line':scope['begin_line'],'problem':'staged converter oracle missing'})
    return scopes,errors,dict(outside)


def records(text,prefix):
    result=[]
    for line in text.splitlines():
        if not line.startswith(prefix+' '):
            continue
        r={}
        for token in line[len(prefix)+1:].split():
            if '=' not in token:
                continue
            name,value=token.split('=',1)
            try:
                parsed=float(value) if '.' in value else int(value)
                # Preserve a nonfinite numeric token as text, as in time_frames,
                # so raw evidence remains representable in standards-valid JSON.
                r[name]=value if isinstance(parsed,float) and not math.isfinite(parsed) else parsed
            except ValueError:r[name]=value
        result.append(r)
    return result


def performance(stdout,rounds,sample_count,minimum_seconds=1.0):
    samples=records(stdout,'SAMPLE')
    warming=[r for r in records(stdout,'PHASE') if 'elapsed_ms' in r]
    groups=defaultdict(list)
    for row in samples:groups[(row.get('route'),row.get('round'))].append(row)
    columns=('gpu_span_ms','wall_ms','dispatch_api_ms','submission_ms','final_wait_ms')
    invalid=[]
    for index,row in enumerate(samples):
        for field in columns:
            raw=row.get(field)
            try:value=float(raw)
            except (ValueError,TypeError):value=float('nan')
            positive=field in ('gpu_span_ms','wall_ms')
            if not math.isfinite(value) or (value<=0 if positive else value<0):
                invalid.append({'sample_index':index,'route':row.get('route'),'round':row.get('round'),
                                'sample':row.get('sample'),'field':field,'value':raw,
                                'required':'finite and >0' if positive else 'finite and >=0'})
    def warm_valid(row):
        try:return math.isfinite(float(row['elapsed_ms'])) and float(row['elapsed_ms'])>=3000
        except (ValueError,TypeError,KeyError):return False
    actual_ids=[(s.get('route'),s.get('round'),s.get('sample')) for s in samples]
    fences=[(s.get('route'),s.get('native_completion_value')) for s in samples]
    expected_ids={(route,r,i) for route in ('direct','staged') for r in range(rounds) for i in range(sample_count)}
    windows=records(stdout,'WINDOW')
    def duration_valid(row):
        try:return (math.isfinite(float(row['actual_seconds'])) and float(row['actual_seconds'])>=minimum_seconds and
                    row['actual_samples']==sample_count and float(row['minimum_seconds'])==minimum_seconds)
        except (ValueError,TypeError,KeyError):return False
    expected_windows={(route,r) for route in ('direct','staged') for r in range(rounds)}
    gates={'all_samples':len(samples)==2*rounds*sample_count,
           'matched_round_counts':all(len(groups[(route,r)])==sample_count for route in ('direct','staged') for r in range(rounds)),
           'unique_sample_ids_and_routes':len(actual_ids)==len(set(actual_ids)) and set(actual_ids)==expected_ids,
           'unique_native_completion_values':len(fences)==len(set(fences)) and all(isinstance(v,int) and v>0 for _,v in fences),
           'actual_measurement_durations':len(windows)==2*rounds and {(w.get('route'),w.get('round')) for w in windows}==expected_windows and all(duration_valid(w) for w in windows),
           'warm_at_least_3_seconds':len(warming)==2*rounds and all(warm_valid(r) for r in warming),
           'timing_samples_valid':bool(samples) and not invalid}
    # An invalid clock sample invalidates every aggregate. Keep the entire raw
    # sequence, including bad samples; never rescue a run by filtering them out.
    summary_valid=all(gates.values())
    medians=None
    paired=None
    if summary_valid:
        medians={}
        for route in ('direct','staged'):
            rows=[s for s in samples if s['route']==route]
            medians[route]={c:statistics.median(float(s[c]) for s in rows) for c in columns}
        paired=[]
        for r in range(rounds):
            direct={c:statistics.median(float(s[c]) for s in groups[('direct',r)]) for c in columns}
            staged={c:statistics.median(float(s[c]) for s in groups[('staged',r)]) for c in columns}
            paired.append({'round':r,'direct':direct,'staged':staged,
                           'direct_vs_staged_percent':{c:(direct[c]/staged[c]-1)*100 if staged[c] else None for c in columns}})
    begins={(r.get('route'),r.get('round')):r.get('utc') for r in records(stdout,'PHASE measure_begin')}
    marker_windows=[]
    for row in records(stdout,'PHASE measure_end'):
        key=(row.get('route'),row.get('round'));start=begins.get(key);stop=row.get('utc')
        try:duration=(datetime.fromisoformat(stop.replace('Z','+00:00'))-datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()
        except (TypeError,ValueError,AttributeError):duration=None
        marker_windows.append({'route':key[0],'round':key[1],'begin_utc':start,'end_utc':stop,'actual_seconds':duration})
    return {'samples':samples,'warm':warming,'measured_windows':windows,'medians_ms':medians,'paired_rounds':paired,'gates':gates,
            'summary_valid':summary_valid,'invalid_timing_samples':invalid,'stdout_phase_windows':marker_windows,
            'qualification':'Exploratory until external pre/during/post process and GPU telemetry qualifies isolation.',
            'definitions':{'gpu_span_ms':'Supplied-queue timestamps before dispatch through final output/state restoration; END timestamp precedes ResolveQueryData, so GPU span excludes query resolve and includes CPU submission gaps.',
                           'wall_ms':'Start timestamp submission through query resolve and completion of final queue fence; excludes subsequent query map and all texture readback.',
                           'dispatch_api_ms':'CPU wall within Candidate::dispatch, including its synchronization and prior-frame resource-retirement work; no pre-sample sync moves that work outside wall time.',
                           'submission_ms':'CPU wall before final wait, including both timestamp submissions and Candidate::dispatch.',
                           'final_wait_ms':'CPU wait for the final query-resolve queue fence.',
                           'uploads_readbacks':'Synthetic texture uploads occur before warming/measurement; texture/stage readback absent in performance mode. Query map is outside the timed wall span.',
                           'state_matching':'After >=3 second warm-up each leg resets temporal state outside measurement, then measures identical fixed inputs/settings and frame counts.'}}


def performance_parser_selftest():
    """CPU fixture regression: one corrupt sample invalidates, never shrinks, a run."""
    lines=[]
    for route in ('direct','staged'):
        lines.append(f'PHASE warm_end route={route} round=0 elapsed_ms=3000.0 frames=1000')
        lines.append(f'WINDOW route={route} round=0 actual_samples=2 actual_seconds=1.25 minimum_seconds=1.0 valid_duration=1')
        for sample in range(2):
            lines.append(f'SAMPLE route={route} round=0 sample={sample} native_completion_value={sample+10} gpu_span_ms=2.0 wall_ms=3.0 '
                         'dispatch_api_ms=0.0 submission_ms=0.0 final_wait_ms=0.0')
    text='\n'.join(lines)
    valid=performance(text,1,2)
    assert all(valid['gates'].values()) and valid['summary_valid']
    assert len(valid['samples'])==4 and valid['medians_ms']['direct']['gpu_span_ms']==2.0
    fixtures={'negative_gpu':('gpu_span_ms=2.0','gpu_span_ms=-1.0'),
              'nan_wall':('wall_ms=3.0','wall_ms=nan'),
              'zero_gpu':('gpu_span_ms=2.0','gpu_span_ms=0.0'),
              'infinite_cpu':('final_wait_ms=0.0','final_wait_ms=inf'),
              'overflow_gpu':('gpu_span_ms=2.0','gpu_span_ms=1.0e999'),
              'negative_cpu':('dispatch_api_ms=0.0','dispatch_api_ms=-0.1')}
    results={}
    for name,(before,after) in fixtures.items():
        broken=text.replace(before,after,1)
        report=performance(broken,1,2)
        assert not report['gates']['timing_samples_valid'] and not report['summary_valid'],name
        assert report['medians_ms'] is None and report['paired_rounds'] is None,name
        assert len(report['samples'])==4 and report['samples']==records(broken,'SAMPLE'),name
        assert len(report['invalid_timing_samples'])==1,name
        results[name]={'passed':True,'raw_samples_retained':len(report['samples']),
                       'invalid_timing_samples':report['invalid_timing_samples']}
    for name,before,after,gate in (
            ('duplicate_sample','sample=1','sample=0','unique_sample_ids_and_routes'),
            ('unknown_route','SAMPLE route=direct','SAMPLE route=unknown','unique_sample_ids_and_routes'),
            ('duplicate_fence','native_completion_value=11','native_completion_value=10','unique_native_completion_values'),
            ('short_window','actual_seconds=1.25','actual_seconds=0.25','actual_measurement_durations')):
        broken=text.replace(before,after,1);report=performance(broken,1,2)
        assert not report['gates'][gate] and not report['summary_valid'],name
        assert report['medians_ms'] is None and report['paired_rounds'] is None and len(report['samples'])==4,name
        results[name]={'passed':True,'raw_samples_retained':len(report['samples']),'failed_gate':gate}
    return {'pass':True,'valid_fixture_passed':True,'fixtures':results,'gpu_run':False}


def benchmark_workload(stdout):
    """Actual fixed workload emitted by the executable, never inferred from timing."""
    options=records(stdout,'OPTIONS');configs=records(stdout,'BENCH_CONFIG');errors=[]
    if len(options)!=1 or len(configs)!=1:
        return {'valid':False,'errors':['exactly one OPTIONS and BENCH_CONFIG record required'],'options':options,'configs':configs}
    policy=options[0].get('pre_exposure');hdr=options[0].get('hdr');config=configs[0]
    try:pi=float(config['pi'])
    except (ValueError,TypeError,KeyError):pi=None
    if policy not in ('scale','original') or hdr not in (0,1):errors.append('unknown pre-exposure policy or HDR flag')
    if pi is None or not math.isfinite(pi) or pi<=0:errors.append('invalid fixed benchmark pre-exposure')
    if config.get('exposure')!=2:errors.append('this benchmark profile contract requires computed exposure')
    return {'valid':not errors,'errors':errors,'pre_exposure':policy,'hdr':bool(hdr),'pi':pi,
            'exposure_mode':config.get('exposure'),'emitted_options':options[0],'emitted_benchmark_config':config,
            'basis':'Unmodified OPTIONS and BENCH_CONFIG records in the measured executable stdout.'}


def measured_profile(rows,samples,workload=None):
    """Join only the returned native fence, with no adjacent-fence inference."""
    if not samples:
        return {'requested':False,'valid':None,'reason':'no measured SAMPLE records; whole-process diagnostics only'}
    if not workload or workload.get('valid') is not True:
        return {'requested':True,'valid':False,'errors':[{'reason':'missing/invalid emitted benchmark workload metadata'}],
                'frames':[],'aggregate':None,'workload':workload}
    by_session=defaultdict(list)
    by_fence=defaultdict(list)
    for row in rows:
        by_session[row['session']].append(row)
        if row.get('type')=='gpu':by_fence[(row['session'],row.get('fence'))].append(row)
    routes=defaultdict(list);errors=[]
    if {s.get('route') for s in samples}!={'direct','staged'}:
        errors.append({'reason':'measured samples must include both known routes'})
    for session,items in by_session.items():
        entries={r.get('name') for r in items if r.get('type')=='gpu' and r.get('kind')=='dispatch'}
        if 'input_encoder_direct' in entries and 'input_encoder' in entries:
            errors.append({'session':session,'reason':'ambiguous mixed-route session'})
        elif 'input_encoder_direct' in entries:routes['direct'].append(session)
        elif 'input_encoder' in entries:routes['staged'].append(session)
    for route in ('direct','staged'):
        if len(routes[route])!=1:errors.append({'route':route,'reason':'measured route must map to exactly one encoder-identified session','sessions':routes[route]})
    frames=[];joined=set()
    # Ten nanoseconds expressed in milliseconds: a small allowance for printed
    # SAMPLE/profile decimal rounding, never permission to hide a negative gap.
    residual_tolerance_ms=1e-5
    for sample in samples:
        route=sample.get('route');fence=sample.get('native_completion_value')
        identity={'route':route,'round':sample.get('round'),'sample':sample.get('sample'),'native_completion_value':fence}
        if route not in ('direct','staged') or len(routes[route])!=1 or not isinstance(fence,int) or fence<=0:
            errors.append({**identity,'reason':'missing/ambiguous route or native completion fence'});continue
        session=routes[route][0];key=(session,fence)
        if key in joined:errors.append({**identity,'reason':'native fence reused by measured samples'});continue
        joined.add(key);items=by_session[session]
        operations=by_fence[(session,fence)]
        dispatches=Counter(r['name'] for r in operations if r.get('kind')=='dispatch')
        suffix='_direct' if route=='direct' else ''
        expected_neural=Counter((*COMMON,'input_encoder'+suffix,'exposure_auto_diet'+suffix,POST+suffix))
        expected_helpers=Counter(('color_divide','scale_rgb')) if route=='staged' and workload['pre_exposure']=='scale' and workload['pi']!=1.0 else Counter()
        expected=expected_neural+expected_helpers
        passes=Counter(r['name'] for r in operations if r.get('kind')=='external_pass')
        expected_passes=Counter(('direct_input_states','direct_output_states') if route=='direct' else
                                ('staged_input_conversion','staged_output_conversion'))
        reasons=[]
        if dispatches!=expected:reasons.append('returned fence does not contain exactly 11 neural dispatches plus the emitted-workload pre-exposure helpers; split/missing/extra work is not inferred')
        if passes!=expected_passes:reasons.append('returned fence does not contain exactly the two expected external passes')
        sequence=[r.get('sequence') for r in operations]
        if not operations or None in sequence or len(sequence)!=len(set(sequence)):reasons.append('missing or duplicated GPU operation sequence')
        if any(r.get('kind') not in ('dispatch','external_pass','barrier','copy') for r in operations):reasons.append('unexpected transfer/fill/unknown work in measured native fence')
        try:
            values=[float(r['ms']) for r in operations]
            if any(r.get('valid') is not True or not math.isfinite(v) or v<0 for r,v in zip(operations,values)):
                reasons.append('invalid native GPU operation timestamp')
            complete=float(sample['gpu_span_ms'])
            if not math.isfinite(complete) or complete<=0:reasons.append('invalid complete GPU span')
        except (TypeError,ValueError,KeyError):reasons.append('missing/malformed timestamp fields');complete=0
        endings=[r for r in items if r.get('type')=='end']
        if len(endings)!=1 or endings[0].get('valid') is not True:reasons.append('profile session did not finish validly')
        if reasons:
            errors.append({**identity,'session':session,'reason':reasons,'observed_dispatches':dict(dispatches),
                           'expected_pre_exposure_helpers':dict(expected_helpers),'observed_external_passes':dict(passes)});continue
        groups=defaultdict(list)
        for r in operations:groups[(r['kind'],r['name'])].append(r)
        breakdown={}
        for (kind,name),group in groups.items():
            breakdown[kind+'/'+name]={'kind':kind,'name':name,'count':len(group),'ms':sum(float(r['ms']) for r in group),
                                     'bytes':sum(r.get('bytes',0) for r in group)}
        category={'neural_dispatch_ms':0.0,'input_divide_ms':0.0,'output_scale_ms':0.0,'input_conversion_ms':0.0,'output_conversion_ms':0.0,
                  'direct_input_states_ms':0.0,'direct_output_states_ms':0.0,'arena_barrier_ms':0.0,'gpu_copy_ms':0.0}
        for r in operations:
            field={'dispatch':'neural_dispatch_ms','barrier':'arena_barrier_ms','copy':'gpu_copy_ms'}.get(r['kind'])
            if r['kind']=='dispatch' and r['name'] in ('color_divide','scale_rgb'):
                field={'color_divide':'input_divide_ms','scale_rgb':'output_scale_ms'}[r['name']]
            if r['kind']=='external_pass':
                field={'staged_input_conversion':'input_conversion_ms','staged_output_conversion':'output_conversion_ms',
                       'direct_input_states':'direct_input_states_ms','direct_output_states':'direct_output_states_ms'}[r['name']]
            category[field]+=float(r['ms'])
        attributed=sum(category.values())
        residual=complete-attributed
        residual_valid=residual>=-residual_tolerance_ms
        if not residual_valid:
            errors.append({**identity,'session':session,'reason':'sum of nonoverlapping measured GPU intervals exceeds complete GPU span',
                           'attributed_gpu_ms':attributed,'complete_gpu_span_ms':complete,
                           'unattributed_complete_span_ms':residual,'printed_rounding_tolerance_ms':residual_tolerance_ms})
        frames.append({**identity,'session':session,'gpu_operation_count':len(operations),'operation_sequences':sequence,
                       'neural_dispatch_count':sum(expected_neural.values()),'pre_exposure_helper_counts':dict(expected_helpers),
                       'operations':list(breakdown.values()),'category_ms':category,
                       'category_share_percent_of_complete_gpu_span':{name:value/complete*100 for name,value in category.items()},
                       'complete_gpu_span_ms':complete,'attributed_gpu_ms':attributed,
                       'unattributed_complete_span_ms':residual,'interval_sum_valid':residual_valid})
    valid=not errors and len(frames)==len(samples)
    aggregate=None
    if valid:
        aggregate={}
        for route in ('direct','staged'):
            selected=[f for f in frames if f['route']==route]
            aggregate[route]={'frames':len(selected),'median_category_ms':{name:statistics.median(f['category_ms'][name] for f in selected) for name in selected[0]['category_ms']},
                              'median_category_share_percent':{name:statistics.median(f['category_share_percent_of_complete_gpu_span'][name] for f in selected) for name in selected[0]['category_ms']}}
    return {'requested':True,'valid':valid,'errors':errors,'frames':frames,'aggregate':aggregate,'workload':workload,
            'printed_rounding_tolerance_ms':residual_tolerance_ms,
            'definition':'Measured SAMPLE rows joined to the exact encoder-identified native Device session and returned completion fence. Initialization/warm-up fences are excluded, never estimated. Every joined fence contains exactly 11 neural dispatches and both expected external passes. Only staged scale-policy frames with emitted pi !=1 additionally require exactly one color_divide and one scale_rgb; these are reported separately from neural work. Direct and original-policy frames permit neither helper. Interval shares are instrumented bottom-of-pipe intervals, not isolated hardware costs; gaps/CPU submission remain unattributed. An attributed sum exceeding complete span by >0.00001 ms invalidates attribution; raw operations and negative residual remain unchanged. Whole-process CPU profiler aggregates remain diagnostic.'}


def measured_profile_selftest():
    import copy
    rows=[];samples=[]
    workload={'valid':True,'pre_exposure':'scale','pi':1.375,'hdr':False,'exposure_mode':2,'basis':'explicit CPU fixture'}
    for route in ('direct','staged'):
        session=route+'-fixture';suffix='_direct' if route=='direct' else ''
        rows.append({'type':'meta','session':session,'frequency_hz':1_000_000})
        rows.append({'type':'gpu','session':session,'sequence':1,'fence':7,'kind':'dispatch','name':'input_encoder'+suffix,'ms':1000.0,'valid':True})
        operations=[('dispatch',name) for name in (*COMMON,'input_encoder'+suffix,'exposure_auto_diet'+suffix,POST+suffix)]
        if route=='staged':operations += [('dispatch','color_divide'),('dispatch','scale_rgb')]
        operations += [('external_pass',name) for name in (('direct_input_states','direct_output_states') if route=='direct' else ('staged_input_conversion','staged_output_conversion'))]
        operations += [('barrier','uav')]
        for sequence,(kind,name) in enumerate(operations,2):
            rows.append({'type':'gpu','session':session,'sequence':sequence,'fence':19,'kind':kind,'name':name,'ms':0.01,'valid':True,'bytes':0})
        rows.append({'type':'end','session':session,'valid':True})
        samples.append({'route':route,'round':0,'sample':0,'native_completion_value':19,'gpu_span_ms':1.0})
    good=measured_profile(rows,samples,workload)
    assert good['valid'] and len(good['frames'])==2
    assert all(abs(f['attributed_gpu_ms']-(0.14 if f['route']=='direct' else 0.16))<1e-9 and 1 not in f['operation_sequences'] for f in good['frames'])
    staged=next(f for f in good['frames'] if f['route']=='staged')
    assert staged['neural_dispatch_count']==11 and staged['category_ms']['input_divide_ms']==0.01 and staged['category_ms']['output_scale_ms']==0.01
    outcomes={}
    for label in ('missing_dispatch','split_fence','extra_pass','duplicate_operation','wrong_completion','interval_oversum',
                  'missing_expected_helper','extra_expected_helper','direct_helper','original_forbids_helpers'):
        bad=copy.deepcopy(rows);bad_samples=copy.deepcopy(samples)
        index=next(i for i,r in enumerate(bad) if r.get('session')=='direct-fixture' and r.get('fence')==19 and r.get('kind')=='dispatch')
        if label=='missing_dispatch':bad.pop(index)
        elif label=='split_fence':bad[index]['fence']=18
        elif label=='extra_pass':
            item=copy.deepcopy(next(r for r in bad if r.get('session')=='direct-fixture' and r.get('kind')=='external_pass'))
            item['sequence']=99;bad.append(item)
        elif label=='duplicate_operation':bad.append(copy.deepcopy(bad[index]))
        elif label=='wrong_completion':bad_samples[0]['native_completion_value']=999
        elif label=='interval_oversum':bad[index]['ms']=2.0
        elif label=='missing_expected_helper':bad=[r for r in bad if not(r.get('session')=='staged-fixture' and r.get('name')=='color_divide')]
        elif label=='extra_expected_helper':
            item=copy.deepcopy(next(r for r in bad if r.get('name')=='scale_rgb'));item['sequence']=99;bad.append(item)
        elif label=='direct_helper':
            item=copy.deepcopy(next(r for r in bad if r.get('name')=='scale_rgb'));item['session']='direct-fixture';item['sequence']=99;bad.append(item)
        tested_workload={**workload,'pre_exposure':'original'} if label=='original_forbids_helpers' else workload
        result=measured_profile(bad,bad_samples,tested_workload)
        assert not result['valid'] and result['aggregate'] is None,label
        if label=='interval_oversum':
            frame=next(f for f in result['frames'] if f['route']=='direct')
            assert frame['unattributed_complete_span_ms']<0 and frame['interval_sum_valid'] is False
            assert any(op['ms']==2.0 for op in frame['operations']),'oversum operation was clamped/discarded'
        outcomes[label]={'passed':True,'errors':result['errors']}
    original_rows=[r for r in rows if r.get('name') not in ('color_divide','scale_rgb')]
    assert measured_profile(original_rows,samples,{**workload,'pre_exposure':'original'})['valid']
    assert measured_profile(original_rows,samples,{**workload,'pi':1.0})['valid']
    return {'pass':True,'gpu_run':False,'warm_fence_1000ms_excluded':True,'scale_staged_13_direct_11_dispatches_accepted':True,
            'original_and_unit_pi_11_dispatches_accepted':True,'negative_fixtures':outcomes}


def profile_summary(path,samples=None,workload=None):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_session=defaultdict(list)
    for row in rows:by_session[row['session']].append(row)
    summary=[]
    for session,items in by_session.items():
        gpu=[x for x in items if x['type']=='gpu']
        direct=any(x['name']=='input_encoder_direct' for x in gpu)
        groups=defaultdict(list)
        for x in gpu:groups[(x['kind'],x['name'])].append(x['ms'])
        cpu=defaultdict(list)
        for x in items:
            if x['type']=='cpu':cpu[x['name']].append(x['ms'])
        summary.append({'session':session,'route':'direct' if direct else 'staged','meta':next((x for x in items if x['type']=='meta'),None),
            'gpu_operations':[{'kind':k,'name':n,'count':len(v),'total_ms':sum(v),'median_ms':statistics.median(v)} for (k,n),v in groups.items()],
            'cpu_operations':[{'name':n,'count':len(v),'total_ms':sum(v),'median_ms':statistics.median(v)} for n,v in cpu.items()],
            'ends':[x for x in items if x['type']=='end'],
            'direct_copy_count':sum(x['kind']=='copy' for x in gpu) if direct else None})
    measured=measured_profile(rows,samples or [],workload)
    return {'sessions':summary,'scope':'whole_process_diagnostic','measured_gpu':measured,
            'measured_phase_attribution_available':measured.get('valid') is True,
            'definition':'The sessions totals are whole-process diagnostic and include initialization/warm-up/diagnostics; CPU scopes overlap. Only measured_gpu contains exact sample-fence attribution. GPU barrier intervals are not isolated hardware costs.'}


def marker_windows(markers,rounds,minimum_seconds=1.0):
    """Validate the real child marker stream independently of activity telemetry."""
    grouped=defaultdict(list)
    for m in markers:grouped[(m.get('config'),m.get('round'))].append(m)
    expected={(route,r) for route in ('direct','staged') for r in range(rounds)}
    errors=[];windows=[];phases=set();pids=set()
    if set(grouped)!=expected:errors.append('route/round windows missing or unexpected')
    for key,rows in grouped.items():
        begin=[m for m in rows if m.get('kind')=='steady_begin'];end=[m for m in rows if m.get('kind')=='steady_end']
        if len(begin)!=1 or len(end)!=1 or len(rows)!=2:
            errors.append(f'nonunique begin/end for {key}');continue
        a,b=begin[0],end[0]
        try:
            duration=(b['monotonic_ns']-a['monotonic_ns'])/1e9
            if not isinstance(a['monotonic_ns'],int) or not isinstance(b['monotonic_ns'],int):raise ValueError('clock type')
            if not math.isfinite(duration) or duration<minimum_seconds:errors.append(f'short/invalid marker duration for {key}')
            if a['pid']!=b['pid'] or not isinstance(a['pid'],int):errors.append(f'PID mismatch for {key}')
            if a.get('phase')!=b.get('phase') or a.get('phase') in phases:errors.append(f'nonunique phase for {key}')
            phases.add(a.get('phase'));pids.add(a['pid'])
            warm=float(a['warm_seconds'])
            if not math.isfinite(warm) or warm<3 or warm!=float(b['warm_seconds']):errors.append(f'warmup invalid for {key}')
            for record in (a,b):
                if not record['utc'].endswith('Z'):raise ValueError('non-UTC marker')
                datetime.fromisoformat(record['utc'].replace('Z','+00:00'))
            windows.append({'config':key[0],'round':key[1],'actual_seconds':duration,'begin':a,'end':b})
        except (ValueError,TypeError,KeyError) as error:errors.append(f'malformed marker {key}: {error}')
    if len(pids)!=1:errors.append('expected one measured child PID')
    return {'valid':not errors and bool(windows),'errors':errors,'windows':windows,'minimum_seconds':minimum_seconds}


def marker_contract_selftest(exe,out):
    """Actual C++ QPC/file contract plus explicitly synthetic qualifier fixtures."""
    import copy
    import gpu_evidence as G
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);directory=out/'markers';directory.mkdir()
    env=os.environ.copy();env['WG_GPU_EVIDENCE_MARKER_DIR']=str(directory)
    before=time.perf_counter_ns();result=subprocess.run([exe,'--marker-contract'],cwd=out,env=env,capture_output=True,text=True,timeout=20);after=time.perf_counter_ns()
    (out/'stdout.log').write_text(result.stdout);(out/'stderr.log').write_text(result.stderr)
    actual,errors=G.read_markers(directory)
    assert result.returncode==0 and not errors and len(actual)==4
    assert all(before<=m['monotonic_ns']<=after for m in actual),'C++ QPC does not match Python perf_counter_ns epoch'
    assert all(m['warm_seconds']==0 and m['cpu_contract_fixture'] is True for m in actual),'CPU fixture must not claim GPU warm-up'
    step=250_000_000;lo=actual[0]['monotonic_ns'];hi=actual[-1]['monotonic_ns']
    origin=(lo//step-2)*step;finish=((hi+step-1)//step+2)*step;rows=[]
    for begin in range(origin,finish,step):
        end=begin+step;phase='pre' if end<=lo else 'post' if begin>=hi else 'during'
        rows.append({'interval_begin_ns':begin,'interval_end_ns':end,'phase':phase,'phase_begin':phase,'valid':True,
                     'attribution':{'unknown_active_pids':[],'unrelated_max_pid_percent':0,'unrelated_total_percent':0,
                                    'active_declared_competitors':[],'workload_total_percent':50 if phase=='during' else 0,'strict_zero_unrelated':True}})
    args=SimpleNamespace(min_samples=2,max_interval=1,max_unrelated_pid=5,max_unrelated_total=10)
    cold=G.qualify(rows,actual,args)
    assert not cold['activity_qualified'] and all('insufficient_reported_warmup' in w['reasons'] for w in cold['windows'])
    synthetic=copy.deepcopy(actual)
    for m in synthetic:m['warm_seconds']=3.25
    good=G.qualify(rows,synthetic,args)
    assert good['activity_qualified'] and marker_windows(synthetic,1)['valid']
    missing=G.qualify(rows,synthetic[:-1],args);assert not missing['activity_qualified']
    duplicate=synthetic[:1]+synthetic;assert not G.qualify(rows,duplicate,args)['activity_qualified']
    short=copy.deepcopy(synthetic);short[1]['monotonic_ns']=short[0]['monotonic_ns']+100_000_000
    assert not marker_windows(short,1)['valid']
    report={'pass':True,'gpu_run':False,'actual_cpp_markers':actual,'qpc_epoch_matches_python':True,
            'actual_zero_warmup_rejected':cold,'synthetic_warm_fixture_qualified':good,
            'missing_end_rejected':True,'duplicate_begin_rejected':True,'short_window_rejected':True,
            'qualifier_path':str(Path(G.__file__).resolve()),'qualifier_sha256':sha(G.__file__),
            'note':'CPU contract only. Telemetry rows and positive warm metadata are explicitly synthetic in-memory fixtures; original C++ marker files report warm_seconds=0 and are preserved unchanged.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pass':True,'report':str(out/'report.json'),'gpu_run':False}))
    return 0


def reparse_profile(parent_report,out):
    """CPU-only correction of helper attribution; original evidence is immutable."""
    import copy
    parent_report=Path(parent_report).resolve();out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    old=json.loads(parent_report.read_text());report=copy.deepcopy(old)
    assert old['mode']=='bench' and old['instrumented_profile'] is True,'requires an existing instrumented benchmark report'
    stdout_path=parent_report.parent/'stdout.log';profile_path=parent_report.parent/'profile.jsonl'
    stdout=stdout_path.read_text();workload=benchmark_workload(stdout)
    assert workload['valid'] and workload['pre_exposure']==old['pre_exposure'] and workload['hdr']==old['hdr'],'emitted workload/options mismatch'
    config=workload['emitted_benchmark_config']
    timings=performance(stdout,config['rounds'],config['samples'],old['minimum_measurement_seconds'])
    assert timings['samples']==old['timings']['samples'],'offline reparse must retain the identical sample sequence'
    timings['gates']['actual_marker_contract']=old['timings']['gates']['actual_marker_contract']
    profile=profile_summary(profile_path,timings['samples'],workload);measured=profile['measured_gpu']
    report['gates'].update(timings['gates'])
    report['gates']['benchmark_workload_metadata']=True
    report['gates']['profile_complete']=bool(profile['sessions']) and all(s['ends'] and all(x['valid'] for x in s['ends']) for s in profile['sessions'])
    report['gates']['measured_profile_complete']=measured.get('valid') is True
    report['gates']['no_direct_gpu_copies']=measured.get('valid') is True and all(f['category_ms']['gpu_copy_ms']==0 for f in measured['frames'] if f['route']=='direct')
    report['pass']=all(report['gates'].values())
    if not report['pass']:timings.update(summary_valid=False,medians_ms=None,paired_rounds=None)
    report.update(timings=timings,profile=profile,fixed_benchmark_workload=workload)
    report['offline_reparse']={'utc':datetime.now(timezone.utc).isoformat(),'gpu_run':False,
        'parent_report':str(parent_report),'parent_report_sha256':sha(parent_report),'parent_report_pass':old['pass'],
        'raw_stdout_sha256':sha(stdout_path),'raw_profile_sha256':sha(profile_path),'parser_sha256':sha(__file__),
        'cause':'The original parser required exactly 11 dispatches for staged scale-policy pi=1.375, incorrectly rejecting the legitimate color_divide and scale_rgb helpers. The corrected parser still requires exactly 11 neural stages and separately exactly these two helpers for this emitted workload; direct/original-policy helpers are forbidden.',
        'unchanged_gates':'All numeric, sample-identity, native-fence, duration, marker, interval-sum tolerance and unrelated original report gates remain in force. No samples/operations discarded or clamped. External activity qualification is unchanged and remains the integration owner responsibility.',
        'prior_attribution_error_count':len(old['profile']['measured_gpu']['errors']),'new_attribution_error_count':len(measured['errors'])}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pass':report['pass'],'profile_attribution_valid':measured['valid'],'report':str(out/'report.json'),
                      'aggregate':measured['aggregate'],'gpu_run':False},indent=2))
    return 0 if report['pass'] else 1


def main():
    if sys.argv[1:]==['--self-test']:
        print(json.dumps(performance_parser_selftest(),indent=2))
        return 0
    if sys.argv[1:]==['--profile-self-test']:
        print(json.dumps(measured_profile_selftest(),indent=2))
        return 0
    if sys.argv[1:2]==['--reparse-profile']:
        p=argparse.ArgumentParser(description='CPU-only strict profile attribution correction; preserve original report')
        p.add_argument('--input',required=True);p.add_argument('--out',required=True)
        a=p.parse_args(sys.argv[2:]);return reparse_profile(a.input,a.out)
    if sys.argv[1:2]==['--marker-self-test']:
        p=argparse.ArgumentParser(description='CPU-only C++ marker and unchanged GPU-evidence qualifier contract')
        p.add_argument('--exe',required=True);p.add_argument('--out',required=True)
        a=p.parse_args(sys.argv[2:]);return marker_contract_selftest(a.exe,a.out)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',required=True);p.add_argument('--build',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',required=True,help='New evidence directory; existing directory refused')
    p.add_argument('--queue',choices=('direct','compute'),default='direct')
    p.add_argument('--mode',choices=('check','matrix','bench'),default='check')
    p.add_argument('--shape',nargs=4,type=int,default=(853,481,1279,721),metavar=('RW','RH','OW','OH'))
    p.add_argument('--frames',type=int,default=8);p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--warm',type=float,default=3);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--samples',type=int,default=600,help='Samples per route/round (default 600). Use actual marker durations and external activity sampling to qualify coverage; count alone does not guarantee it.')
    p.add_argument('--min-seconds',type=float,default=1.0,help='Reject shorter actual measured windows (>=1); preserve and rerun BOTH routes with a larger shared sample count')
    p.add_argument('--pre-exposure',choices=('scale','original'),default='scale')
    p.add_argument('--hdr',action='store_true',help='Set the Model M input HDR flag on both routes')
    p.add_argument('--profile',action='store_true',help='Enable diagnostic native per-operation timestamps; use separate runs for uninstrumented timing')
    p.add_argument('--diagnostic-trace',action='store_true',help='Enable launch trace during benchmark (off by default for performance)')
    a=p.parse_args()
    if (a.frames<3 or a.cycles<1 or min(a.shape)<1 or not math.isfinite(a.warm) or a.warm<3 or
            a.rounds<1 or a.samples<1 or not math.isfinite(a.min_seconds) or a.min_seconds<1):
        p.error('invalid counts/dimensions, warm <3 seconds or minimum duration <1 second')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    global POST,COMMON
    POST=post_kernel(*a.shape,env.get('WG_RUNTIME_OPTIONS',''))
    if env.get('WG_D3D12_FUSION','') != 'none':
        COMMON=fused_common(a.shape[0],a.shape[1])
    for name in ('HIP_PATH','ROCM_PATH','LLVM_PATH','HIP_DEVICE_LIB_PATH','HIP_CLANG_PATH','WG_DX12_KERNELS','WG_D3D12_PROFILE','WG_LAUNCH_TRACE'):
        env.pop(name,None)
    trace=out/'dispatch.trace';profile=out/'profile.jsonl'
    trace_enabled=a.mode!='bench' or a.diagnostic_trace or a.profile
    if trace_enabled:env['WG_LAUNCH_TRACE']=str(trace)
    env['TMP']=env['TEMP']=str(out)
    if a.profile:env['WG_D3D12_PROFILE']=str(profile)
    marker_directory=Path(env.get('WG_GPU_EVIDENCE_MARKER_DIR',str(out/'markers'))).resolve()
    marker_directory.mkdir(parents=True,exist_ok=True);env['WG_GPU_EVIDENCE_MARKER_DIR']=str(marker_directory)
    cmd=[a.exe,a.build,a.model,a.queue,a.mode,*map(str,a.shape),str(a.frames),str(a.cycles),str(a.warm),str(a.rounds),str(a.samples),
         str(a.min_seconds),a.pre_exposure,str(int(a.hdr))]
    exe_hash_before=sha(a.exe)
    start=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,errors='replace',timeout=1800)
    (out/'stdout.log').write_text(result.stdout);(out/'stderr.log').write_text(result.stderr)
    scopes,trace_errors,outside=trace_scopes(trace.read_text().splitlines() if trace.exists() else [])
    counts=Counter(s['route'] for s in scopes)
    gates={'exit_zero':result.returncode==0,'fresh_modules':'MODULES fresh' in result.stdout and 'MODULES after-destroy' in result.stdout,
           'no_hip_modules':'amdhip' not in result.stdout.lower() and 'hiprtc' not in result.stdout.lower()}
    if trace_enabled:
        gates.update({'complete_native_trace':bool(scopes) and not trace_errors,
                      'both_routes':counts['direct']>0 and counts['staged']>0,
                      'driver_probes':outside.get('native_driver_probe',0)==(2 if a.mode=='bench' else 2*a.cycles)})
    else:gates['launch_trace_disabled_for_timing']=not trace.exists()
    timings=None
    marker_data=None
    workload=benchmark_workload(result.stdout) if a.mode=='bench' else None
    if a.mode=='bench':
        gates['benchmark_workload_metadata']=workload['valid'] and workload.get('pre_exposure')==a.pre_exposure and workload.get('hdr')==a.hdr
        import gpu_evidence as G
        emitted=[json.loads(line[len('EVIDENCE_MARKER '):]) for line in result.stdout.splitlines() if line.startswith('EVIDENCE_MARKER ')]
        all_markers,marker_errors=G.read_markers(marker_directory)
        child_pids={m.get('pid') for m in emitted}
        actual=[m for m in all_markers if m.get('pid') in child_pids]
        marker_data=marker_windows(actual,a.rounds,a.min_seconds)
        marker_data.update(directory=str(marker_directory),read_errors=marker_errors,stdout_matches_file=emitted==actual)
        timings=performance(result.stdout,a.rounds,a.samples,a.min_seconds)
        timings['gates']['actual_marker_contract']=marker_data['valid'] and not marker_errors and emitted==actual
        if not all(timings['gates'].values()):
            timings.update(summary_valid=False,medians_ms=None,paired_rounds=None)
        gates.update(timings['gates'])
        gates['final_pass']='PASS native direct benchmark' in result.stdout
    else:
        combinations=a.cycles*(24 if a.mode=='matrix' else 1)
        expected=combinations*a.frames
        gates.update({'exact_frame_counts':counts['direct']==expected+(a.cycles if a.mode=='check' else 0) and counts['staged']==expected,
                      'format_count':len(records(result.stdout,'FORMAT'))==combinations,
                      'inventory_count':len(records(result.stdout,'INVENTORY'))==expected,
                      'mutation_detected':'mutation=1' in result.stdout,'ready_fence_order':'PASS delayed input-ready fence held final completion' in result.stdout,
                      'debug_clean':'DEBUG enabled' in result.stdout and 'DEBUG clean' in result.stdout and 'DEBUG unavailable' not in result.stdout and 'DEBUG ERROR' not in result.stdout,
                      'final_pass':'PASS native direct check' in result.stdout})
    profile_data=profile_summary(profile,records(result.stdout,'SAMPLE'),workload) if a.profile and profile.exists() else None
    if a.profile:
        gates['profile_complete']=bool(profile_data) and all(s['ends'] and all(x['valid'] for x in s['ends']) for s in profile_data['sessions'])
        if a.mode=='bench':
            measured=profile_data.get('measured_gpu',{}) if profile_data else {}
            gates['measured_profile_complete']=measured.get('valid') is True
            gates['no_direct_gpu_copies']=measured.get('valid') is True and all(f['category_ms']['gpu_copy_ms']==0 for f in measured['frames'] if f['route']=='direct')
        else:
            gates['no_direct_gpu_copies']=bool(profile_data) and all(s['direct_copy_count']==0 for s in profile_data['sessions'] if s['route']=='direct')
    manifest=Path(a.build)/'dx12/manifest.json'
    hashes={str(Path(a.exe).resolve()):sha(a.exe),str(manifest.resolve()):sha(manifest)}
    for k in json.loads(manifest.read_text())['kernels']:
        path=manifest.parent/k['file'];hashes[str(path.resolve())]=sha(path)
    build_json=Path(a.exe).resolve().parent/'native-direct-build.json'
    build_link=json.loads(build_json.read_text()) if build_json.exists() else None
    gates['exe_build_link']=bool(build_link) and build_link.get('exe_sha256')==exe_hash_before==sha(a.exe) and not build_link.get('changed_during_build')
    if build_link:hashes[str(build_json)]=sha(build_json)
    adapter_rows=[json.loads(line[len('ADAPTER_JSON '):]) for line in result.stdout.splitlines() if line.startswith('ADAPTER_JSON ')]
    gates['actual_adapter_metadata']=len(adapter_rows)==1 and adapter_rows[0].get('timestamp_frequency_hz',0)>0 and bool(adapter_rows[0].get('description'))
    model_files={str(path.relative_to(Path(a.model))):sha(path) for path in sorted(Path(a.model).rglob('*')) if path.is_file()}
    gates['model_payload_hashed']=bool(model_files)
    if timings and not all(gates.values()):timings.update(summary_valid=False,medians_ms=None,paired_rounds=None)
    report={'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'mode':a.mode,'shape':a.shape,
            'pass':all(gates.values()),'gates':gates,'returncode':result.returncode,'sha256':hashes,
            'frame_scope_counts':dict(counts),'trace_errors':trace_errors,'outside_frame_trace':outside,
            'scopes':scopes,'inventory':records(result.stdout,'INVENTORY'),'formats':records(result.stdout,'FORMAT'),
            'timings':timings,'profile':profile_data,'diagnostic_launch_trace':trace_enabled,'instrumented_profile':a.profile,
            'actual_adapter':adapter_rows,'exe_build_json':{'path':str(build_json),'record':build_link},
            'model_payload':{'path':str(Path(a.model).resolve()),'sha256':model_files},'markers':marker_data,
            'pre_exposure':a.pre_exposure,'hdr':a.hdr,'minimum_measurement_seconds':a.min_seconds,
            'fixed_benchmark_workload':workload,
            'domain':'Original synthetic weights and scene; direct typed-texture Model M versus retained staged Converter/native route on same device/queue.',
            'excluded_unwritten':'b0-b4, dec1, post.display, dec0.B, unused C allocation tail and previous state on reset; direct removed IO buffers explicitly unavailable.',
            'memory_definition':'removed_io_bytes is avoided logical suballocation capacity, not physical VRAM freed.',
            'telemetry':'GPU lock and external pre/during/post activity evidence are owned by root; this script cannot qualify idle hardware.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pass':report['pass'],'gates':gates,'report':str(out/'report.json')},indent=2))
    if result.returncode:print(result.stderr[-5000:],file=sys.stderr)
    return 0 if report['pass'] else 1


if __name__=='__main__':sys.exit(main())
