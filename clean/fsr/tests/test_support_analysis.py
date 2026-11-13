# SPDX-License-Identifier: MIT
"""Original bounded/offline support analysis regressions; synthetic logs only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
spec = importlib.util.spec_from_file_location('support', Path(__file__).parent.parent/'tools/analyze_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

class Reports(unittest.TestCase):
    def make(self, folder, manifest=None, data=b'original', legacy=False):
        target = Path(folder)/'report.zip'
        with zipfile.ZipFile(target, 'w') as z:
            if legacy:
                z.writestr('wronggpu-fsr.log', 'ERROR: hipStreamWaitValue32 failed: HIP error 719 (unspecified launch failure)\n')
            else:
                z.writestr('wronggpu-fsr.log', data)
                z.writestr('support-manifest.json', json.dumps(manifest or {'schema':1, 'failures':[], 'artifacts':[
                    {'name':'wronggpu-fsr.log','status':'included','sha256':hashlib.sha256(b'original').hexdigest()}]}))
        return target
    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            report = support.analyze(self.make(folder, data=b'changed'))
            self.assertFalse(report['artifact_integrity'][0]['matches'])
            self.assertFalse(report['root_cause_proven'])
    def test_correct_hash_no_false_health_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            report = support.analyze(self.make(folder))
            self.assertTrue(report['artifact_integrity'][0]['matches'])
            self.assertIn('does not prove', report['observations'][0])
    def test_legacy719_keeps_attribution_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            report = support.analyze(self.make(folder, legacy=True))
            self.assertEqual(report['error_codes'], ['WG-HIP-719'])
            self.assertFalse(report['reproduction_context']['gpu_replay_available'])
            self.assertEqual(report['failures'][0]['data']['observed_at'], 'input_wait')
    def test_unavailable_or_truncated_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'schema':1, 'artifacts':[{'name':'locked.log','status':'unreadable'}],
                        'malformed_diagnostic_lines':2,
                        'runtime_observations':[{'event':'records_dropped','session':'test','data':{'count':3}}]}
            report = support.analyze(self.make(folder, manifest=manifest))
            self.assertTrue(any(g['status']=='malformed_diagnostic_lines' for g in report['coverage_gaps']))
            self.assertTrue(any(g['status']=='records_dropped' for g in report['coverage_gaps']))
    def test_unsupported_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError): support.analyze(self.make(folder, manifest={'schema':99}))
    def test_stack_offsets_require_matching_session_map(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'schema':1, 'runtime_observations':[
                {'session':'s','module':'fe','event':'loaded_modules','qpc':1,
                 'data':{'modules':[{'path':'own.dll','base_address':'1000','image_size':100,'sha256':'exact'}]}},
                {'session':'s','module':'fe','event':'cpu_detection_stack','qpc':2,
                 'data':{'incident':'i','frames':[{'pc':'1017'},{'pc':'9999'}]}}]}
            stack = support.analyze(self.make(folder, manifest=manifest))['detection_stacks'][0]
            self.assertEqual(stack['frames'][0]['offset'],17)
            self.assertIsNone(stack['frames'][1]['module'])
            self.assertFalse(stack['faulting_stack_proven'])
    def test_oversized_member_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.make(folder)
            with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z: z.writestr('wronggpu-fsr.log.old', b'a'*(support.MAX_FILE+1))
            # Legacy selection exercises member-size checks before decompression.
            report = Path(folder)/'legacy.zip'
            with zipfile.ZipFile(report,'w',zipfile.ZIP_DEFLATED) as z: z.writestr('wronggpu-fsr.log', b'a'*(support.MAX_FILE+1))
            with self.assertRaises(ValueError): support.analyze(report)

if __name__ == '__main__': unittest.main()
