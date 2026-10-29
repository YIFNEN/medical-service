"""Read-only local readiness inspection; never loads credentials into output."""
import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path

from .config import ModelRegistry


def inspect_environment(store, *, slicer=None):
    packages = {}
    for package in ('pydantic', 'openai', 'anthropic', 'google-genai', 'pydicom', 'pynrrd', 'PyMuPDF'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(store).resolve()
    ancestor = root
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    configured = slicer or os.environ.get('MEDIC_SLICER_EXECUTABLE') or shutil.which('Slicer')
    slicer_path = Path(configured).resolve() if configured else None
    registry = ModelRegistry.default()
    result = {
        'schema_version': 'medic-doctor/1', 'network_calls_made': 0, 'paid_calls_made': 0,
        'python': {'version': platform.python_version(), 'supported': sys.version_info >= (3, 12)},
        'packages': packages,
        'store': {'path': str(root), 'exists': root.exists(),
                  'parent_writable': ancestor.is_dir() and os.access(ancestor, os.W_OK)},
        'slicer': {'path': str(slicer_path) if slicer_path else None,
                   'available': bool(slicer_path and slicer_path.is_file()), 'runtime_verified': False},
        'pricing_as_of': registry.pricing_as_of.isoformat(), 'pricing_status': 'CONFIGURED_NOT_LIVE_VERIFIED',
        'providers': [{'provider': p.provider, 'model': p.model, 'key_available': p.key_available,
                       'input_usd_per_million': p.input_usd_per_million,
                       'output_usd_per_million': p.output_usd_per_million} for p in registry.providers],
    }
    result['offline_ready'] = result['python']['supported'] and all(packages.values()) and result['store']['parent_writable']
    result['live_ready'] = False  # keys alone do not verify models, prices, or approval.
    return result
