import unittest


class ComparisonTests(unittest.TestCase):
    def test_changed_acquisition_is_not_called_progression(self):
        from medic.comparison import compare_results
        from types import SimpleNamespace
        class Synthesis:
            def model_dump(self, **kwargs):
                return {'agreements': ['different']}
        before = SimpleNamespace(run_id='old', case_id='same', patient_context={'fact_details': [],
            'imaging_reviews': [{'review_id':'old', 'conditions': 'supine', 'angle': 10}]},
            evidence_cards=(), calls=(), synthesis=Synthesis(), created_at=None, data_cutoff=None)
        after = SimpleNamespace(**{**before.__dict__, 'run_id':'new', 'patient_context': {'fact_details': [],
            'imaging_reviews': [{'review_id':'new', 'conditions': 'standing', 'angle': 12}]}})
        result = compare_results(before, after)
        self.assertEqual(result['clinical_change'], 'NOT_INFERRED')
        self.assertTrue(result['imaging_changed'])
        after.case_id = 'different'
        with self.assertRaises(ValueError):
            compare_results(before, after)
