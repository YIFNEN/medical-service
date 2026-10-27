"""Local DICOM inspection, provenance and conservative technical geometry gates."""
from collections import defaultdict
import io
import math
from pathlib import Path
import warnings

import pydicom
from pydicom.uid import UID

from .inventory import visit_records, digest, _linked
from .imaging_geometry import validate_geometry, validate_plane


def _numbers(value, length):
    try:
        out = [float(x) for x in value]
        return out if len(out) == length and all(math.isfinite(x) for x in out) else None
    except (ValueError, TypeError):
        return None


def _uid(ds, key):
    value = str(getattr(ds, key, ''))
    return value if value and UID(value).is_valid else None


def _inspect(data):
    # pydicom inflates deflated datasets before pixel handling; reject before that allocation.
    from pydicom.filereader import read_preamble, _read_file_meta_info
    header = io.BytesIO(data)
    read_preamble(header, force=True)
    meta = _read_file_meta_info(header)
    if str(getattr(meta, 'TransferSyntaxUID', '')) == '1.2.840.10008.1.2.1.99':
        raise ValueError('deflated dataset requires bounded decoding; inspect only')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=False)
        except pydicom.errors.InvalidDicomError:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
        sop_class = _uid(ds, 'SOPClassUID') or _uid(ds.file_meta, 'MediaStorageSOPClassUID')
        if sop_class == '1.2.840.10008.1.3.10':
            return {'kind':'DICOMDIR', 'measurement_eligible':False}
        if not sop_class or not all(_uid(ds, k) for k in ('StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID')):
            return None
        transfer = getattr(ds.file_meta, 'TransferSyntaxUID', None)
        rows, columns = int(getattr(ds,'Rows',0)), int(getattr(ds,'Columns',0))
        frames = int(getattr(ds,'NumberOfFrames',1))
        bits = int(getattr(ds,'BitsAllocated',0))
        samples = int(getattr(ds,'SamplesPerPixel',0))
        pixel_length = len(getattr(ds,'PixelData',b''))
        expected = rows*columns*frames*samples*bits//8
        pixel_valid = (transfer is not None and not UID(transfer).is_compressed and bits in (8,16)
            and samples == 1 and rows > 0 and columns > 0 and frames > 0
            and pixel_length in (expected, expected + (expected % 2))
            and _uid(ds.file_meta,'MediaStorageSOPClassUID') in (None, sop_class)
            and _uid(ds.file_meta,'MediaStorageSOPInstanceUID') in (None, _uid(ds,'SOPInstanceUID')))
        return dict(kind='DICOM', study_uid=_uid(ds,'StudyInstanceUID'), series_uid=_uid(ds,'SeriesInstanceUID'),
            sop_uid=_uid(ds,'SOPInstanceUID'), sop_class_uid=sop_class,
            frame_of_reference_uid=_uid(ds,'FrameOfReferenceUID'), modality=str(getattr(ds,'Modality','UNKNOWN')),
            date=str(getattr(ds,'AcquisitionDate',getattr(ds,'StudyDate',''))) or None,
            laterality=str(getattr(ds,'ImageLaterality',getattr(ds,'Laterality','UNKNOWN'))),
            orientation=_numbers(getattr(ds,'ImageOrientationPatient',None),6),
            position=_numbers(getattr(ds,'ImagePositionPatient',None),3),
            spacing=_numbers(getattr(ds,'PixelSpacing',None),2),
            imager_spacing=_numbers(getattr(ds,'ImagerPixelSpacing',None),2),
            calibration=str(getattr(ds,'PixelSpacingCalibrationType','')) or None,
            rows=int(getattr(ds,'Rows',0)), columns=int(getattr(ds,'Columns',0)),
            frames=frames, pixel_payload_valid=bool(pixel_valid), weight_bearing='UNKNOWN')


def inspect_inventory(inventory, store):
    """Copy recognized, hash-verified DICOM only. Store must be under local .medic."""
    store = Path(store).absolute()
    if '.medic' not in store.parts or str(store).startswith('\\\\'):
        raise ValueError('DICOM store must be local and under .medic')
    if any(p.exists() and _linked(p) for p in (store, *store.parents)):
        raise ValueError('linked store is not permitted')
    manifest = dict(version=1, instances=[], directories=[], series=[], conflicts=[], rejected=[])
    def inspect_record(record, data):
        if '.medic' in Path(record['source_path']).parts:
            return
        try:
            item = _inspect(data)
            if item is None:
                return
            store.mkdir(parents=True, exist_ok=True)
            target = store / (record['sha256'].split(':')[1] + '.dcm')
            if target.exists():
                if _linked(target) or digest(target.read_bytes()) != record['sha256']:
                    raise ValueError('store hash mismatch')
            else:
                with target.open('xb') as f:
                    f.write(data)
            item.update(sha256=record['sha256'], source=record, stored_path=str(target))
            manifest['directories' if item['kind'] == 'DICOMDIR' else 'instances'].append(item)
        except Exception:
            manifest['rejected'].append(dict(sha256=record['sha256'], reason='unreadable_changed_or_invalid'))
    missing = visit_records(inventory, inspect_record)
    manifest['rejected'].extend(dict(sha256=r['sha256'],reason='source_changed_or_unavailable') for r in missing)
    by_sop = defaultdict(set)
    groups = defaultdict(list)
    for item in manifest['instances']:
        by_sop[item['sop_uid']].add(item['sha256'])
        groups[(item['study_uid'], item['series_uid'])].append(item)
    bad = {sop for sop, hashes in by_sop.items() if len(hashes) > 1}
    manifest['conflicts'] = [dict(sop_uid=sop, hashes=sorted(by_sop[sop])) for sop in sorted(bad)]
    for (study, series), instances in groups.items():
        unique = list({i['sha256']:i for i in instances}.values())
        s = dict(study_uid=study, series_uid=series, original_hashes=sorted({i['sha256'] for i in unique}),
                 measurement_eligible=False, geometry=None, weight_bearing='UNKNOWN', reasons=[])
        try:
            if len({key[0] for key in groups if key[1] == series}) != 1:
                raise ValueError('Series UID reused across studies')
            if any(i['sop_uid'] in bad for i in unique):
                raise ValueError('SOP hash conflict')
            if any(not i['pixel_payload_valid'] for i in unique):
                raise ValueError('missing, compressed, or unsupported pixel payload / file meta mismatch')
            modalities = {i['modality'] for i in unique}
            if len(modalities) != 1 or any(i['frames'] != 1 for i in unique):
                raise ValueError('mixed modality or multiframe inspect only')
            modality = unique[0]['modality']
            if modality in ('CT','MR'):
                allowed = {'CT':'1.2.840.10008.5.1.4.1.1.2', 'MR':'1.2.840.10008.5.1.4.1.1.4'}
                if any(i['sop_class_uid'] != allowed[modality] for i in unique):
                    raise ValueError('unsupported SOP class')
                if len({i['frame_of_reference_uid'] for i in unique}) != 1 or not unique[0]['frame_of_reference_uid']:
                    raise ValueError('missing or inconsistent frame of reference')
                s['geometry'] = validate_geometry(unique)
                s['ordered_hashes'] = [unique[i]['sha256'] for i in s['geometry']['slice_order']]
            elif modality in ('DX','CR'):
                if any(not i['spacing'] or min(i['spacing']) <= 0 or i['calibration'] not in ('GEOMETRY','FIDUCIAL') for i in unique):
                    raise ValueError('explicit calibration required')
                if len(unique) != 1:
                    raise ValueError('projection series must contain one image')
                allowed = {'CR': {'1.2.840.10008.5.1.4.1.1.1'},
                           'DX': {'1.2.840.10008.5.1.4.1.1.1.1', '1.2.840.10008.5.1.4.1.1.1.1.1'}}
                if unique[0]['sop_class_uid'] not in allowed[modality]:
                    raise ValueError('unsupported projection SOP class')
                s['geometry'] = validate_plane(unique[0])
                s['ordered_hashes'] = [unique[0]['sha256']]
            else:
                raise ValueError('modality inspect only')
            s['measurement_eligible'] = True
        except (ValueError, TypeError, KeyError) as exc:
            s['reasons'].append(str(exc))
        manifest['series'].append(s)
    return manifest
