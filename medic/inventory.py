"""Read-only, bounded local inventory. Paths and hashes are sensitive local data."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import io
import math
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


@dataclass(frozen=True)
class Limits:
    max_member_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    max_entries: int = 100000
    max_depth: int = 4
    max_ratio: float = 200

    def __post_init__(self):
        if (any(type(v) is not int for k,v in asdict(self).items() if k != 'max_ratio')
                or not math.isfinite(self.max_ratio)
                or any(v <= 0 for k, v in asdict(self).items() if k != 'max_depth') or self.max_depth < 0):
            raise ValueError('invalid inventory limits')


def digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def safe_name(name):
    return (bool(name) and '\\' not in name and ':' not in name and '\x00' not in name
            and not PurePosixPath(name).is_absolute()
            and all(p not in ('', '.', '..') for p in name.split('/')))


def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


def _executable(name, data=b''):
    return (Path(name.split(';')[0]).suffix.lower() in {
        '.exe', '.dll', '.com', '.bat', '.cmd', '.ps1', '.sh', '.msi', '.scr',
        '.js', '.vbs', '.py', '.jar', '.lnk', '.app', '.so', '.dylib'}
        or data.startswith((b'MZ', b'\x7fELF', b'#!', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf')))


def _archive(data):
    if zipfile.is_zipfile(io.BytesIO(data)):
        return 'ZIP'
    if len(data) > 32774 and data[32769:32774] == b'CD001':
        return 'ISO'
    return None


class _BudgetExceeded(ValueError):
    pass


def _scan(root, limits, visitor=None, only=None):
    root = Path(root).absolute()
    if not root.is_dir() or any(_linked(p) for p in (root, *root.parents)):
        raise ValueError('root must be a real local directory')
    result = {'version': 1, 'root': str(root), 'limits': asdict(limits), 'records': [], 'truncated': False}
    total = 0

    def record(source, chain, parents, size):
        if len(result['records']) >= limits.max_entries:
            raise _BudgetExceeded()
        r = dict(source_path=source, archive_path=chain, container_hashes=parents,
                 size_bytes=size, sha256=None, classification='UNKNOWN', status='ACCEPTED', reason=None)
        result['records'].append(r)
        return r

    def reject(r, reason):
        r.update(status='REJECTED', reason=reason)

    def consume(stream, size):
        nonlocal total
        if size > limits.max_member_bytes or total + size > limits.max_total_bytes:
            raise ValueError('size_limit')
        data = stream.read(min(limits.max_member_bytes, limits.max_total_bytes - total) + 1)
        total += len(data)
        if len(data) != size or len(data) > limits.max_member_bytes or total > limits.max_total_bytes:
            raise ValueError('size_limit')
        return data

    def process(r, data, depth):
        r['sha256'] = digest(data)
        if _executable((r['archive_path'] or [r['source_path']])[-1], data):
            reject(r, 'executable')
            return
        kind = _archive(data)
        if not kind:
            if visitor:
                visitor(r, data)
            return
        r['classification'] = kind
        if depth >= limits.max_depth:
            reject(r, 'archive_depth')
            return
        parents = r['container_hashes'] + [r['sha256']]

        def child(name, size, stream_factory, reason=None):
            c = record(r['source_path'], r['archive_path'] + [name], parents, size)
            if not safe_name(name):
                reason = 'unsafe_path'
            if _executable(name):
                reason = 'executable'
            if reason:
                reject(c, reason)
                return
            try:
                with stream_factory() as stream:
                    process(c, consume(stream, size), depth + 1)
            except _BudgetExceeded:
                raise
            except Exception:
                reject(c, 'unreadable_or_size_limit')

        try:
            if kind == 'ZIP':
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    infos = z.infolist()
                    counts = Counter(i.filename for i in infos)
                    for i in infos:
                        if i.is_dir():
                            continue
                        mode = i.external_attr >> 16
                        reason = None
                        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                            reason = 'symlink_or_special'
                        elif counts[i.filename] != 1:
                            reason = 'duplicate_member'
                        elif i.flag_bits & 1:
                            reason = 'encrypted'
                        elif i.file_size / max(i.compress_size, 1) > limits.max_ratio:
                            reason = 'compression_ratio'
                        child(i.orig_filename, i.file_size, lambda i=i: z.open(i), reason)
            else:
                import pycdlib
                iso = pycdlib.PyCdlib()
                try:
                    iso.open_fp(io.BytesIO(data))
                    namespace = 'rr_path' if iso.has_rock_ridge() else ('joliet_path' if iso.has_joliet() else 'iso_path')
                    for directory, dirs, files in iso.walk(**{namespace: '/'}):
                        for name in files:
                            path = directory.rstrip('/') + '/' + name
                            entry = iso.get_record(**{namespace: path})
                            reason = 'symlink_or_special' if entry.is_symlink() else None
                            size = entry.data_length
                            def extract(path=path, size=size):
                                if size > limits.max_member_bytes or total + size > limits.max_total_bytes:
                                    raise ValueError('size_limit')
                                out = io.BytesIO()
                                iso.get_file_from_iso_fp(out, **{namespace: path})
                                out.seek(0)
                                return out
                            child(path.lstrip('/'), size, extract, reason)
                finally:
                    iso.close()
        except _BudgetExceeded:
            raise
        except Exception:
            reject(r, 'invalid_or_unsupported_archive')

    def walk(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                source = path.relative_to(root).as_posix()
                if _linked(path):
                    reject(record(source, [], [], 0), 'symlink_or_reparse_point')
                elif entry.is_dir(follow_symlinks=False):
                    yield from walk(path)
                else:
                    yield path

    try:
        for path in ([root / only] if only else walk(root)):
            source = path.relative_to(root).as_posix()
            r = record(source, [], [], path.lstat().st_size)
            try:
                if any(_linked(p) for p in (path, *path.parents)) or not stat.S_ISREG(path.stat().st_mode):
                    reject(r, 'symlink_or_special')
                    continue
                with path.open('rb') as stream:
                    process(r, consume(stream, r['size_bytes']), 0)
            except _BudgetExceeded:
                raise
            except Exception:
                reject(r, 'unreadable_or_size_limit')
    except _BudgetExceeded:
        result['truncated'] = True
    result['bytes_read'] = total
    return result


def scan(root, *, limits=None):
    """Inventory without extraction, content role inference, or changes to originals."""
    return _scan(root, limits or Limits())


def read_record(inventory, record):
    """Re-read bounded content and verify every container and leaf hash."""
    if record not in inventory['records'] or record['status'] != 'ACCEPTED' or not safe_name(record['source_path']):
        raise ValueError('unaccepted source')
    matches = []
    def visit(current, data):
        if current == record:
            matches.append(data)
    _scan(inventory['root'], Limits(**inventory['limits']), visit, record['source_path'])
    if len(matches) != 1:
        raise ValueError('source changed or provenance mismatch')
    return matches[0]


def visit_records(inventory, visitor):
    """Reverify leaves in one bounded scan; call visitor(record, bytes), return missing records."""
    expected = {(r['source_path'], tuple(r['archive_path'])): r for r in inventory['records']
                if r['status'] == 'ACCEPTED' and r['classification'] == 'UNKNOWN'}
    visited = set()
    def visit(current, data):
        key = (current['source_path'], tuple(current['archive_path']))
        if expected.get(key) == current:
            visitor(current, data)
            visited.add(key)
    _scan(inventory['root'], Limits(**inventory['limits']), visit)
    return [r for key,r in expected.items() if key not in visited]
