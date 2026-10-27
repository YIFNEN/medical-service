import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from medic.cli import main, _latest_context
from medic.ledger import AppendOnlyJsonlLedger


class ImagingWithdrawalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = AppendOnlyJsonlLedger(Path(self.temp.name) / 'ledger.jsonl')
        self.ledger.append_once('HIGH_IMPACT_FACTS_CONFIRMED', 'context',
            {'case_id': 'case', 'context': {'fact_details': [], 'confirmed_facts': [], 'uncertainties': []}})

    def add_review(self, review_id):
        self.ledger.append_once('IMAGING_REVIEW_IMPORTED', review_id, {'case_id': 'case',
            'review_id': review_id, 'review': {'study_uid': 'study', 'series_uid': 'series',
                'review_status': 'USER_REVIEWED', 'review_status_provenance': 'USER_DECLARED',
                'measurements': [{'value': 5.0, 'unit': 'mm'}], 'segmentations': [],
                'interpretation': 'manual', 'unmarked_tissue': 'NOT_EVALUATED'}})

    def withdraw(self, review_id, case='case', reason='incorrect point'):
        with redirect_stdout(StringIO()):
            self.assertEqual(main(['dicom', 'withdraw-review', '--case', case, '--review-id', review_id,
                '--reviewer', 'user', '--reason', reason, '--store', self.temp.name]), 0)

    def test_latest_withdrawal_does_not_restore_old_measurement(self):
        self.add_review('old')
        self.add_review('latest')
        self.withdraw('latest')
        self.assertFalse(_latest_context(self.ledger, 'case')['imaging_reviews'])
        before = len(self.ledger.records())
        self.withdraw('latest')
        self.assertEqual(len(self.ledger.records()), before)
        self.add_review('new')
        self.assertEqual(_latest_context(self.ledger, 'case')['imaging_reviews'][0]['review_id'], 'new')
        self.assertEqual(sum(r['kind'] == 'IMAGING_REVIEW_IMPORTED' for r in self.ledger.records()), 3)

    def test_withdrawing_older_review_does_not_remove_latest(self):
        self.add_review('old')
        self.add_review('latest')
        self.withdraw('old')
        self.assertEqual(_latest_context(self.ledger, 'case')['imaging_reviews'][0]['review_id'], 'latest')

    def test_unknown_case_or_blank_reason_cannot_write_event(self):
        self.add_review('review')
        before = len(self.ledger.records())
        with self.assertRaises(ValueError):
            self.withdraw('review', case='other')
        with self.assertRaises(ValueError):
            self.withdraw('review', reason='  ')
        self.assertEqual(len(self.ledger.records()), before)
