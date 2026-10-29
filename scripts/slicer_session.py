"""Slicer entry point for a local, selected-series review session."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.slicer_review import load_series, export_review, bind_markup

session = json.loads(Path(os.environ['MEDIC_SLICER_SESSION']).read_text(encoding='utf-8'))
medic_manifest = session['manifest']
medic_volume = load_series(medic_manifest, session['series_uid'], session['workspace'])

import slicer
slicer.util.setSliceViewerLayers(background=medic_volume)
slicer.util.resetSliceViews()


def medic_bind(*nodes):
    for node in nodes:
        bind_markup(medic_volume, node)


def medic_export(nodes):
    """Export only explicitly selected, source-bound standard Slicer Markups."""
    return export_review(medic_manifest, medic_volume, nodes, session['workspace'])


print('MEDIC selected series loaded. Select Markups explicitly: medic_bind(node); medic_export([node]).')
