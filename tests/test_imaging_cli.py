import json

from medic.cli import main
from medic.imaging import inspect_inventory
from medic.inventory import scan
from tests.test_imaging import dicom


def test_prepare_slicer_session_uses_selected_series_without_launch(tmp_path, capsys):
    source = tmp_path / 'source'
    source.mkdir()
    a = dicom(source / 'a')
    dicom(source / 'b', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4)
    store = tmp_path / '.medic'
    manifest = inspect_inventory(scan(source), store / 'imaging' / 'originals')
    manifest_path = store / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    assert main(['dicom', 'launch-review', '--manifest', str(manifest_path), '--series', str(a.SeriesInstanceUID),
                 '--store', str(store)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['launched'] is False
    session = json.loads(open(result['session'], encoding='utf-8').read())
    assert session['series_uid'] == str(a.SeriesInstanceUID)
    assert session['template']['original_hashes'] == manifest['series'][0]['original_hashes']
    assert result['clinical_validation'] == 'NOT_VALIDATED'
