import tempfile
import unittest
from pathlib import Path

from medic.ledger import AppendOnlyJsonlLedger


class ImagingAmendmentTests(unittest.TestCase):
    def test_amendment_links_latest_and_rejects_stale_or_wrong_series(self):
        from medic.imaging_history import record_import
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            review = {'study_uid': 'study', 'series_uid': 'series'}
            def save(identifier, value=review, **kwargs):
                return record_import(ledger, 'case', identifier, value, 'sha256:source', **kwargs)
            save('old')
            save('new', replaces='old', reason='corrected point')
            before = len(ledger.records())
            save('new', replaces='old', reason='corrected point')
            self.assertEqual(len(ledger.records()), before)
            event = ledger.records()[-1]['payload']
            self.assertEqual(event['replaces_review_id'], 'old')
            self.assertEqual(event['amendment_reason'], 'corrected point')
            for identifier, value, kwargs in (
                ('third', review, {'replaces': 'old', 'reason': 'stale'}),
                ('other-series', {**review, 'series_uid': 'different'}, {'replaces': 'new', 'reason': 'wrong'}),
                ('self', review, {'replaces': 'self', 'reason': 'self'}),
                ('missing-reason', review, {'replaces': 'new'}),
                ('missing-target', review, {'reason': 'missing'}),
            ):
                with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                    save(identifier, value, **kwargs)
            with self.assertRaises(ValueError):
                record_import(ledger, 'other-case', 'third', review, 'sha256:source', replaces='new', reason='wrong case')
            self.assertEqual(len(ledger.records()), before)

    def test_latest_withdrawn_review_can_have_explicit_corrected_successor(self):
        from medic.imaging_history import record_import
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            review = {'study_uid': 'study', 'series_uid': 'series'}
            record_import(ledger, 'case', 'old', review, 'sha256:source')
            ledger.append_once('IMAGING_REVIEW_WITHDRAWN', 'case:old', {'case_id': 'case', 'review_id': 'old'})
            record_import(ledger, 'case', 'new', review, 'sha256:source', replaces='old', reason='corrected')
            self.assertEqual(ledger.records()[-1]['payload']['replaces_review_id'], 'old')
