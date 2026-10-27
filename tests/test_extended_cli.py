import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from medic.cli import main


class ExtendedCliTests(unittest.TestCase):
    def test_fetch_retry_option_reaches_resolver_without_paid_gateway(self):
        from unittest.mock import patch
        from tests.test_decision_literature import card
        item = card().model_copy(update={'fulltext_required': True})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cards.json'
            path.write_text(json.dumps([item.model_dump(mode='json')]))
            with patch('medic.literature.FullTextResolver.fetch', return_value=item) as fetch:
                code, _ = self.invoke(['literature', 'fetch', '--cards', str(path), '--store', temp,
                                       '--retry-unavailable'])
            self.assertEqual(code, 0)
            fetch.assert_called_once()
            self.assertTrue(fetch.call_args.kwargs['retry_unavailable'])

    def test_dicom_review_reaches_case_context(self):
        from tests.test_imaging import dicom
        from medic.cli import _latest_context
        from medic.ledger import AppendOnlyJsonlLedger
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input'
            source.mkdir()
            first = dicom(source / 'one')
            dicom(source / 'two', study=first.StudyInstanceUID, series=first.SeriesInstanceUID, z=4)
            store = root / '.medic'
            _, inv = self.invoke(['inventory', str(source), '--store', str(store)])
            _, inspection = self.invoke(['dicom', 'inspect', '--inventory', inv['path'], '--store', str(store)])
            _, export = self.invoke(['dicom', 'export', '--manifest', inspection['path'],
                                     '--series', first.SeriesInstanceUID, '--store', str(store)])
            review = json.loads(Path(export['path']).read_text())
            review['markups'] = [{'kind':'line', 'coordinate_system':'LPS', 'points':[[0,0,0],[3,4,0]]}]
            path = store / 'manual-review.json'
            path.write_text(json.dumps(review))
            _, result = self.invoke(['dicom', 'import-review', '--manifest', inspection['path'], '--review', str(path),
                                     '--case', 'case', '--reviewer', 'local-user', '--store', str(store)])
            self.assertEqual(result['measurement_count'], 1)
            context = _latest_context(AppendOnlyJsonlLedger(store / 'ledger.jsonl'), 'case')
            self.assertEqual(context['imaging_reviews'][0]['measurements'][0]['value'], 5.0)
            measurement = context['imaging_reviews'][0]['measurements'][0]
            self.assertTrue(measurement['fact_id'].startswith('imaging-observation-'))
            self.assertEqual(measurement['confirmation_scope'], 'MANUAL_IMAGING_OBSERVATION_NOT_DIAGNOSIS')
            self.assertFalse(context['fact_details'])
            review['markups'][0]['points'] = [[0, 0, 0], [0, 4, 0]]
            path.write_text(json.dumps(review))
            args = ['dicom', 'import-review', '--manifest', inspection['path'], '--review', str(path),
                    '--case', 'case', '--reviewer', 'local-user', '--store', str(store),
                    '--replaces-review-id', result['review_id'], '--amendment-reason', 'corrected point']
            _, updated = self.invoke(args)
            current = _latest_context(AppendOnlyJsonlLedger(store / 'ledger.jsonl'), 'case')['imaging_reviews'][0]
            self.assertEqual(current['replaces_review_id'], result['review_id'])
            self.assertEqual(current['measurements'][0]['value'], 4.0)
            self.assertNotEqual(updated['review_id'], result['review_id'])

    def invoke(self, args):
        output = StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_inventory_and_inspect_are_local_and_report_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input'
            source.mkdir()
            (source / 'unknown.txt').write_text('unclassified')
            store = root / '.medic'
            code, output = self.invoke(['inventory', str(source), '--store', str(store)])
            self.assertEqual(code, 0)
            inventory = Path(output['path'])
            self.assertTrue(inventory.exists())
            code, result = self.invoke(['dicom', 'inspect', '--inventory', str(inventory), '--store', str(store)])
            self.assertEqual(code, 0)
            self.assertEqual(result['series_count'], 0)
            self.assertEqual((source / 'unknown.txt').read_text(), 'unclassified')

    def test_approval_without_budget_is_rejected_before_live_call(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'budget'):
                self.invoke(['deliberate', 'case', '--approve-cost', '--store', temp])

    def test_literature_appraisal_without_approval_is_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cards.json'
            path.write_text('[]')
            code, output = self.invoke(['literature', 'appraise', '--cards', str(path), '--store', temp])
            self.assertEqual(code, 2)
            self.assertEqual(output['paid_calls_made'], 0)

    def test_run_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'run'):
                self.invoke(['resume', '../escape', '--store', temp])
