import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom


class SlicerValidationTests(unittest.TestCase):
    def test_pixel_values_cannot_pass_by_uint16_truncation_or_wrap(self):
        import hashlib
        from medic.slicer_validation import check_pixels, check_mask
        pixels = np.arange(4 * 16 * 16).reshape(4, 16, 16).astype('<u2')
        expected = hashlib.sha256(pixels.tobytes()).hexdigest()
        check_pixels(pixels, expected)
        for wrong in (pixels.astype(float) + .5, pixels.astype(float) + 65536):
            with self.assertRaises(ValueError):
                check_pixels(wrong, expected)
        mask = np.zeros((4, 16, 16), dtype=np.uint8)
        mask[1:3, 4:7, 5:9] = 1
        exported = mask.transpose(2, 1, 0)
        check_mask(exported, mask)
        with self.assertRaises(ValueError):
            check_mask(exported.transpose(1, 0, 2), mask)

    def test_result_collision_and_write_failure_always_exit_slicer(self):
        import runpy
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        script = Path(__file__).resolve().parent.parent / 'scripts' / 'slicer_roundtrip_test.py'
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / 'config.json'
            result_path = Path(temp) / 'result.json'
            config_path.write_text(json.dumps({'result_path': str(result_path)}))
            scope = runpy.run_path(str(script))
            entry = scope['main']
            fake_run = Mock(return_value={'status': 'PASSED'})
            entry.__globals__['run'] = fake_run
            app = SimpleNamespace(exit=Mock())
            result_path.write_text('{"status":"PASSED"}')
            with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=app)}):
                self.assertEqual(entry(str(config_path)), 1)
            fake_run.assert_not_called()
            app.exit.assert_called_once_with(1)
            result_path.unlink()
            app.exit.reset_mock()
            entry.__globals__['_write_result'] = Mock(side_effect=OSError('cannot save'))
            with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=app)}):
                self.assertEqual(entry(str(config_path)), 1)
            app.exit.assert_called_once_with(1)

    def test_bundle_contains_known_oblique_anisotropic_synthetic_ct(self):
        from medic.slicer_validation import prepare_validation
        with tempfile.TemporaryDirectory() as temp:
            bundle = prepare_validation(Path(temp) / '.medic')
            config = json.loads(Path(bundle['config_path']).read_text())
            manifest = json.loads(Path(config['manifest_path']).read_text())
            self.assertEqual(bundle['status'], 'PREPARED_NOT_RUN')
            self.assertEqual(manifest['series'][0]['geometry']['shape'], [16, 16, 4])
            self.assertTrue(manifest['series'][0]['measurement_eligible'])
            self.assertEqual(len(manifest['instances']), 4)
            instance = pydicom.dcmread(manifest['instances'][0]['stored_path'])
            self.assertEqual(str(instance.PatientID), 'MEDIC_SYNTHETIC')
            self.assertEqual(instance.pixel_array.shape, (16, 16))
            affine = np.array(manifest['series'][0]['geometry']['affine_lps'])
            self.assertTrue(np.allclose(np.linalg.norm(affine[:3, :3], axis=0), [3, 2, 4]))
            self.assertNotEqual(affine[0, 1], 0)
            self.assertAlmostEqual(config['expected']['distance_mm'], math.sqrt(145), places=9)
            self.assertEqual(config['expected']['angle_degrees'], 90.0)
            self.assertEqual(config['expected']['volume_mm3'], 576.0)
            self.assertFalse(Path(config['result_path']).exists())

    def test_wrong_measurement_or_missing_metric_cannot_pass(self):
        from medic.slicer_validation import check_metrics
        expected = {'distance_mm': math.sqrt(145), 'angle_degrees': 90.0, 'volume_mm3': 576.0,
                    'evaluation_roi_volume_mm3': 6912.0, 'fraction_of_evaluation_roi': 1 / 12}
        observed = dict(expected)
        check_metrics(observed, expected)
        for key in expected:
            with self.subTest(metric=key):
                wrong = {**observed, key: observed[key] + .01}
                with self.assertRaises(ValueError):
                    check_metrics(wrong, expected)
                with self.assertRaises(ValueError):
                    check_metrics({k: v for k, v in observed.items() if k != key}, expected)
        with self.assertRaises(ValueError):
            check_metrics({**observed, 'angle_degrees': float('nan')}, expected)

    def test_bundle_requires_local_nonlinked_medic_directory(self):
        from medic.slicer_validation import prepare_validation
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            prepare_validation(Path(temp) / 'outside-store')
