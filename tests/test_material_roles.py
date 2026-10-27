import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from medic.cli import main


class MaterialRoleTests(unittest.TestCase):
    def test_edited_inventory_cannot_override_executable_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'viewer.exe').write_bytes(b'MZblocked')
            store = root / '.medic'
            path = self.invoke(['inventory', str(source), '--store', str(store)])['path']
            inventory = json.loads(Path(path).read_text())
            inventory['records'][0].update(status='ACCEPTED', reason=None)
            edited = root / 'edited.json'
            edited.write_text(json.dumps(inventory))
            view = self.invoke(['inventory-review', 'status', '--inventory', str(edited), '--store', str(store)])
            record = json.loads(Path(view['path']).read_text())['records'][0]
            with self.assertRaises(ValueError):
                self.invoke(['inventory-review', 'assign', '--inventory', str(edited), '--record-id', record['record_id'],
                    '--role', 'PATIENT_RECORD', '--case', 'case', '--reviewer', 'user', '--reason', 'review', '--store', str(store)])
            self.assertFalse((store / 'ledger.jsonl').exists())

    def test_retained_inventory_allows_withdrawal_after_source_is_removed(self):
        from medic.inventory import scan
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            original = source / 'record.txt'
            original.write_text('record')
            inventory = root / 'external-inventory.json'
            inventory.write_text(json.dumps(scan(source)))
            store = root / '.medic'
            view = self.invoke(['inventory-review', 'status', '--inventory', str(inventory), '--store', str(store)])
            record = json.loads(Path(view['path']).read_text())['records'][0]
            result = self.invoke(['inventory-review', 'assign', '--inventory', str(inventory), '--record-id', record['record_id'],
                '--role', 'PATIENT_RECORD', '--case', 'case', '--reviewer', 'user', '--reason', 'review', '--store', str(store)])
            saved = result['inventory_path']
            self.assertNotEqual(Path(saved), inventory)
            inventory.unlink()
            original.unlink()
            self.invoke(['inventory-review', 'withdraw', '--inventory', saved, '--record-id', record['record_id'],
                         '--reviewer', 'user', '--reason', 'withdraw', '--store', str(store)])

    def invoke(self, args):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 0)
        return json.loads(output.getvalue())

    def test_explicit_roles_corrections_withdrawal_and_snapshot_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            path = source / 'patient-study.pdf'
            path.write_bytes(b'unknown document')
            store = root / '.medic'
            inventory = self.invoke(['inventory', str(source), '--store', str(store)])['path']
            def status(inv=inventory):
                output = self.invoke(['inventory-review', 'status', '--inventory', inv, '--store', str(store)])
                return json.loads(Path(output['path']).read_text(encoding='utf-8'))['records'][0]
            first = status()
            self.assertEqual(first['role'], 'UNKNOWN')
            args = ['inventory-review', 'assign', '--inventory', inventory, '--record-id', first['record_id'],
                    '--reviewer', 'user', '--reason', 'read document', '--store', str(store)]
            with self.assertRaises(ValueError):
                self.invoke(args + ['--role', 'PATIENT_RECORD'])
            assigned = self.invoke(args + ['--role', 'PATIENT_RECORD', '--case', 'case'])
            again = self.invoke(args + ['--role', 'PATIENT_RECORD', '--case', 'case'])
            self.assertEqual(assigned['event_id'], again['event_id'])
            self.assertEqual(status()['case_id'], 'case')
            with self.assertRaises(ValueError):
                self.invoke(args + ['--role', 'LITERATURE'])
            self.invoke(args + ['--role', 'LITERATURE', '--citation', 'Reviewed title, authors, year'])
            self.assertIsNone(status()['case_id'])
            self.assertEqual(status()['role'], 'LITERATURE')
            self.invoke(['inventory-review', 'withdraw', '--inventory', inventory, '--record-id', first['record_id'],
                         '--reviewer', 'user', '--reason', 'uncertain identity', '--store', str(store)])
            self.assertEqual(status()['role'], 'UNKNOWN')
            path.write_bytes(b'changed document')
            changed = self.invoke(['inventory', str(source), '--store', str(store)])['path']
            self.assertEqual(status(changed)['role'], 'UNKNOWN')
            self.assertNotEqual(status(changed)['record_id'], first['record_id'])
            self.assertEqual(len((store / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()), 3)

    def test_rejected_material_cannot_be_assigned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'viewer.exe').write_bytes(b'MZblocked')
            store = root / '.medic'
            inventory = self.invoke(['inventory', str(source), '--store', str(store)])['path']
            view = self.invoke(['inventory-review', 'status', '--inventory', inventory, '--store', str(store)])
            record = json.loads(Path(view['path']).read_text())['records'][0]
            with self.assertRaises(ValueError):
                self.invoke(['inventory-review', 'assign', '--inventory', inventory, '--record-id', record['record_id'],
                    '--role', 'OTHER', '--reviewer', 'user', '--reason', 'check', '--store', str(store)])
            self.assertFalse((store / 'ledger.jsonl').exists())
