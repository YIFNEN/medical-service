import io
import json
import stat
import zipfile

import pytest


def zip_bytes(items):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in items:
            if isinstance(name, str):
                original = name
                name = zipfile.ZipInfo(name)
                name.filename = original
                name.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(name, data)
    return out.getvalue()


def test_nested_archive_provenance_and_unknown(tmp_path):
    from medic.inventory import scan, read_record
    blob = zip_bytes([('nested.zip', zip_bytes([('unknown', b'hello')]))])
    (tmp_path / 'outer.zip').write_bytes(blob)
    result = scan(tmp_path)
    json.dumps(result, allow_nan=False)
    leaf = next(r for r in result['records'] if r['archive_path'] == ['nested.zip', 'unknown'])
    assert leaf['classification'] == 'UNKNOWN'
    assert leaf['sha256'].startswith('sha256:')
    assert len(leaf['container_hashes']) == 2
    assert read_record(result, leaf) == b'hello'
    assert (tmp_path / 'outer.zip').read_bytes() == blob
    (tmp_path / 'outer.zip').write_bytes(b'changed')
    with pytest.raises(ValueError):
        read_record(result, leaf)


def test_reject_unsafe_zip_and_limits(tmp_path):
    from medic.inventory import scan, Limits
    link = zipfile.ZipInfo('link')
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    (tmp_path / 'bad.zip').write_bytes(zip_bytes([
        ('../escape', b'x'), ('C:/escape', b'x'), ('a\\b', b'x'),
        ('run.exe', b'MZ123'), (link, b'target'), ('bomb', b'0' * 10000)]))
    result = scan(tmp_path, limits=Limits(max_member_bytes=2000, max_ratio=20))
    leaves = [r for r in result['records'] if r['archive_path']]
    assert len(leaves) == 6
    assert all(r['status'] == 'REJECTED' for r in leaves)
    assert not (tmp_path.parent / 'escape').exists()


def test_iso_and_nested_zip(tmp_path):
    import pycdlib
    from medic.inventory import scan, read_record
    blob = zip_bytes([('leaf', b'unknown')])
    iso = pycdlib.PyCdlib()
    iso.new()
    iso.add_fp(io.BytesIO(blob), len(blob), iso_path='/NEST.ZIP;1')
    iso.write(str(tmp_path / 'disc.iso'))
    iso.close()
    result = scan(tmp_path)
    leaf = next(r for r in result['records'] if r['archive_path'][-1:] == ['leaf'])
    assert read_record(result, leaf) == b'unknown'


def test_duplicates_and_depth_rejected(tmp_path):
    from medic.inventory import scan, Limits
    with pytest.warns(UserWarning):
        data = zip_bytes([('same', b'a'), ('same', b'b')])
    (tmp_path / 'dupe.zip').write_bytes(data)
    result = scan(tmp_path)
    assert all(r['status'] == 'REJECTED' for r in result['records'] if r['archive_path'])
    result = scan(tmp_path, limits=Limits(max_depth=0))
    assert result['records'][0]['reason'] == 'archive_depth'


def test_limits_finite_and_total_budget(tmp_path):
    from medic.inventory import scan, Limits
    with pytest.raises(ValueError):
        Limits(max_ratio=float('nan'))
    with pytest.raises(ValueError):
        Limits(max_entries=1.5)
    (tmp_path / 'a').write_bytes(b'a'*20)
    (tmp_path / 'b').write_bytes(b'b'*20)
    result = scan(tmp_path, limits=Limits(max_total_bytes=30))
    assert sum(r['status'] == 'ACCEPTED' for r in result['records']) == 1
    assert result['bytes_read'] <= 30
    assert scan(tmp_path,limits=Limits(max_entries=1))['truncated']


def test_symlink_not_followed(tmp_path):
    from medic.inventory import scan
    target = tmp_path / 'target'
    target.write_bytes(b'original')
    try:
        (tmp_path/'link').symlink_to(target)
    except OSError:
        pytest.skip('OS does not permit creating a synthetic symlink')
    link = next(r for r in scan(tmp_path)['records'] if r['source_path'] == 'link')
    assert link['status'] == 'REJECTED'
    assert link['sha256'] is None


def test_verified_batch_visit_and_changed_sources(tmp_path):
    from medic.inventory import scan, visit_records
    (tmp_path/'archive.zip').write_bytes(zip_bytes([('a',b'a'),('b',b'b')]))
    inventory = scan(tmp_path)
    collected = []
    missing = visit_records(inventory, lambda r,data: collected.append(data))
    assert sorted(collected) == [b'a',b'b']
    assert missing == []
    (tmp_path/'archive.zip').write_bytes(zip_bytes([('a',b'changed'),('b',b'b')]))
    collected.clear()
    missing = visit_records(inventory, lambda r,data: collected.append(data))
    assert collected == []
    assert len(missing) == 2
