import zipfile

import pytest

from medic.inventory import scan
from medic.ledger import AppendOnlyJsonlLedger
from medic.material_roles import role_view, review_role


def test_ingest_selected_archive_record_requires_role_and_preserves_provenance(tmp_path):
    from medic.inventory_ingest import ingest_selected
    source = tmp_path / 'source'
    source.mkdir()
    archive = source / 'records.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('record.txt', 'synthetic right ankle observation')
    inventory = scan(source)
    store = tmp_path / '.medic'
    ledger = AppendOnlyJsonlLedger(store / 'ledger.jsonl')
    leaf = next(r for r in role_view(inventory, ledger)['records'] if r['classification'] == 'UNKNOWN')
    with pytest.raises(ValueError, match='reviewed'):
        ingest_selected(store, inventory, 'case', [leaf['record_id']])
    review_role(inventory, ledger, leaf['record_id'], role='PATIENT_RECORD', case_id='case', reviewer='operator', reason='synthetic record')
    result = ingest_selected(store, inventory, 'case', [leaf['record_id']])
    assert result['file_count'] == 1
    provenance = [r['payload'] for r in ledger.records() if r['kind'] == 'INVENTORY_MATERIAL_INGESTED'][0]
    assert provenance['record']['archive_path'] == ['record.txt']
    assert result == ingest_selected(store, inventory, 'case', [leaf['record_id']])
    with pytest.raises(ValueError):
        ingest_selected(store, inventory, 'different-case', [leaf['record_id']])
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('record.txt', 'changed synthetic record')
    with pytest.raises(ValueError):
        ingest_selected(store, inventory, 'case', [leaf['record_id']])
