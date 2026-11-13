# SPDX-License-Identifier: MIT
"""Offline, metadata-only triage of WrongGPU support ZIPs. Never extracts or runs files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024

def analyze(path):
    if Path(path).stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Support archive exceeds file-size limit')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 256 or len({i.filename for i in infos}) != len(infos):
            raise ValueError('Too many or duplicate ZIP members')
        total = 0
        def read(name):
            nonlocal total
            item = archive.getinfo(name)
            if item.file_size > MAX_FILE or total + item.file_size > MAX_TOTAL:
                raise ValueError('Support archive exceeds bounded read limits')
            total += item.file_size
            return archive.read(item)
        names = {i.filename for i in infos}
        coverage = []
        manifest = json.loads(read('support-manifest.json')) if 'support-manifest.json' in names else {}
        if not isinstance(manifest, dict) or (manifest and manifest.get('schema') != 1):
            raise ValueError('Unsupported support manifest schema')
        integrity = []
        for artifact in manifest.get('artifacts', []):
            name = artifact.get('name', '')
            if artifact.get('status') != 'included':
                coverage.append({'artifact': name, 'status': artifact.get('status')})
                continue
            # Do not inspect memory dumps, executables, models or unknown binary payloads.
            if not name.lower().endswith(('.txt', '.ini', '.log', '.log.old', '.jsonl', '.jsonl.old', '.record', '.json')):
                coverage.append({'artifact': name, 'status': 'unsupported_artifact_not_read'})
                continue
            if name not in names:
                integrity.append({'name': name, 'matches': False, 'reason': 'missing_member'})
                continue
            digest = hashlib.sha256(read(name)).hexdigest()
            integrity.append({'name': name, 'matches': digest == artifact.get('sha256')})
            if artifact.get('truncated'):
                coverage.append({'artifact': name, 'status': 'truncated'})
        failures = manifest.get('failures', [])
        frames = manifest.get('frame_observations', [])
        # Legacy files can still be classified, but cannot invent absent frame/session evidence.
        if not manifest:
            coverage.append({'status': 'legacy_bundle_no_structured_manifest'})
            for name in sorted(names):
                if Path(name).name.lower().startswith('wronggpu') and name.lower().endswith(('.log', '.log.old')):
                    text = read(name).decode('utf-8', 'replace')
                    for line in text.splitlines():
                        if 'HIP error 719' in line:
                            failures.append({'session': None, 'module': Path(name).name,
                                'data': {'code': 'WG-HIP-719', 'observed_at': 'input_wait' if 'hipStreamWaitValue32' in line else 'hip_api',
                                         'message': line, 'root_cause_proven': False}})
                        elif 'more than 8' in line and 'vtables' in line:
                            failures.append({'session': None, 'module': Path(name).name,
                                'data': {'code': 'WG-HOOK-TABLE', 'observed_at': 'hook_registration',
                                         'message': line, 'root_cause_proven': False}})
        codes = sorted({f.get('data', {}).get('code', 'WG-ERROR') for f in failures})
        observations = []
        if 'WG-HIP-719' in codes:
            observations.append('HIP 719 was observed; the reporting API may detect an earlier asynchronous GPU failure. A driver cause is unproven.')
        if 'WG-HOOK-TABLE' in codes:
            observations.append('A hook-registration failure was recorded. Compare the exact frontend build with the table-growth regression fix.')
        if 'WG-NO-PROVIDER' in codes:
            observations.append('No usable provider was recorded at this boundary. Check loaded-provider paths, file hashes and fallback configuration.')
        if not failures:
            observations.append('No explicit failure record was captured. This does not prove a healthy session or rule out a hang/crash.')
        if manifest.get('malformed_diagnostic_lines', 0):
            coverage.append({'status': 'malformed_diagnostic_lines', 'count': manifest['malformed_diagnostic_lines']})
        runtime = manifest.get('runtime_observations', [])
        for event in runtime:
            if event.get('event') in ('records_dropped', 'module_snapshot_unavailable'):
                coverage.append({'status': event['event'], 'session': event.get('session'), 'data': event.get('data')})
        if manifest and not runtime:
            coverage.append({'status': 'no_runtime_identity_observations'})
        resolved_stacks = []
        for stack in runtime:
            if stack.get('event') != 'cpu_detection_stack':
                continue
            maps = [event for event in runtime if event.get('event') == 'loaded_modules' and
                    event.get('session') == stack.get('session') and event.get('module') == stack.get('module')]
            # Prefer the most recent map at/before detection. Later maps are usable
            # only with an explicit lifetime/order uncertainty, never as proof.
            before = [event for event in maps if event.get('qpc', 0) <= stack.get('qpc', 0)]
            mapping = max(before or maps, key=lambda event: event.get('qpc', 0), default={})
            resolved = []
            for item in stack.get('data', {}).get('frames', []):
                pc = int(item.get('pc', 0))
                match = next((m for m in mapping.get('data', {}).get('modules', []) if
                              int(m.get('base_address', 0)) <= pc < int(m.get('base_address', 0)) + int(m.get('image_size', 0))), None)
                resolved.append({'pc': str(pc), 'module': match.get('path') if match else None,
                    'offset': pc - int(match['base_address']) if match else None,
                    'module_sha256': match.get('sha256') if match else None})
            resolved_stacks.append({'incident': stack.get('data', {}).get('incident'),
                'faulting_stack_proven': False, 'mapping_captured_before_detection': bool(before), 'frames': resolved})
        return {'schema': 1, 'bundle_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'root_cause_proven': False, 'artifact_integrity': integrity, 'coverage_gaps': coverage,
            'error_codes': codes, 'observations': observations, 'failures': failures, 'detection_stacks': resolved_stacks,
            'reproduction_context': {'capture': 'metadata_only', 'gpu_replay_available': False,
                'resource_bytes_available': False, 'object_ids_are_session_local': True,
                'frames': frames, 'installed_files': manifest.get('installed_files', []),
                'runtime_observations': manifest.get('runtime_observations', []),
                'adapters': manifest.get('adapters', []), 'driver_registry': manifest.get('driver_registry', []),
                'hip_candidates': manifest.get('hip_candidates', [])}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = analyze(args.bundle)
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    return 1 if any(not item['matches'] for item in result['artifact_integrity']) else 0

if __name__ == '__main__':
    raise SystemExit(main())
