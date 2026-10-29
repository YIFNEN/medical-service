"""Prepare local Slicer sessions and pair existing, active manual measurements."""
import json
import os
from pathlib import Path
import subprocess
import uuid

from .extended_cli import read_json, save_json
from .inventory import _linked
from .ledger import AppendOnlyJsonlLedger


def register(dicom):
    launch = dicom.add_parser('launch-review')
    launch.add_argument('--manifest', type=Path, required=True)
    launch.add_argument('--series', required=True)
    launch.add_argument('--slicer', type=Path)
    launch.add_argument('--launch', action='store_true')
    launch.add_argument('--store', type=Path, default=Path('.medic'))
    compare = dicom.add_parser('compare-measurements')
    compare.add_argument('--case', required=True)
    for field in ('before-review', 'before-measurement', 'after-review', 'after-measurement'):
        compare.add_argument('--' + field, required=True)
    compare.add_argument('--repeatability-bound', type=float)
    compare.add_argument('--repeatability-source')
    compare.add_argument('--store', type=Path, default=Path('.medic'))


def execute(args):
    if args.command != 'dicom' or args.action not in ('launch-review', 'compare-measurements'):
        return None
    root = Path(args.store).resolve()
    if args.action == 'compare-measurements':
        from .measurement_comparison import active_measurement, compare_measurements
        ledger = AppendOnlyJsonlLedger(root / 'ledger.jsonl')
        before = active_measurement(ledger, args.case, args.before_review, args.before_measurement)
        after = active_measurement(ledger, args.case, args.after_review, args.after_measurement)
        result = compare_measurements(before, after, repeatability_bound=args.repeatability_bound,
                                      repeatability_source=args.repeatability_source)
        result.update(case_id=args.case, before_review=args.before_review, after_review=args.after_review)
        return 0, {**result, 'path': str(save_json(root, 'imaging/comparisons', result))}
    if '.medic' not in root.parts or str(root).startswith('\\\\') or any(_linked(p) for p in (root, *root.parents)):
        raise ValueError('local unlinked .medic workspace required')
    from .slicer_review import review_template
    manifest = read_json(args.manifest)
    template = review_template(manifest, args.series)
    session = {'version': 'medic-slicer-session/1', 'series_uid': args.series,
               'manifest': manifest, 'template': template,
               'workspace': str(root / 'imaging' / 'slicer-sessions' / uuid.uuid4().hex)}
    path = save_json(root, 'imaging/sessions', session)
    executable = args.slicer or os.environ.get('MEDIC_SLICER_EXECUTABLE')
    script = Path(__file__).resolve().parent.parent / 'scripts' / 'slicer_session.py'
    result = {'session': str(path), 'launched': False, 'clinical_validation': 'NOT_VALIDATED',
              'instruction': 'Use standard Slicer Markups; medic_bind(node), then medic_export([node]). Review metadata before comparison.'}
    if args.launch:
        if not executable or not Path(executable).is_file() or Path(executable).name.lower() not in ('slicer.exe', 'slicer'):
            raise ValueError('explicit installed Slicer executable required')
        environment = {**os.environ, 'MEDIC_SLICER_SESSION': str(path)}
        process = subprocess.Popen([str(Path(executable).resolve()), '--no-splash', '--python-script', str(script)],
                                   env=environment, shell=False)
        result.update(launched=True, pid=process.pid, runtime_status='STARTED_NOT_VERIFIED')
    return 0, result
