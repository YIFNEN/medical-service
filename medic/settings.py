"""Versioned processing limits shared by execution, estimates and reports."""
from pathlib import Path
from typing import Literal

from pydantic import Field

from .research_contracts import StrictFrozenModel


class ProcessingSettings(StrictFrozenModel):
    schema_version: Literal['medic-processing/1'] = 'medic-processing/1'
    max_results: int = Field(default=10, ge=1)
    screening_limit: int = Field(default=200, ge=1)
    evidence_limit: int = Field(default=20, ge=1)
    fulltext_limit: int = Field(default=5, ge=0)
    screening_batch_size: int = Field(default=10, ge=1)
    abstract_excerpt_chars: int = Field(default=12000, ge=1)
    fulltext_chunk_chars: int = Field(default=20000, ge=1)
    fulltext_max_chunks: int = Field(default=12, ge=1)


LIMIT_FIELDS = tuple(name for name in ProcessingSettings.model_fields if name != 'schema_version')


def resolve_settings(preset='standard', config: Path | None = None, overrides=None):
    if preset not in ('standard', 'pilot'):
        raise ValueError('unknown processing preset')
    values = ProcessingSettings().model_dump(mode='json')
    if preset == 'pilot':
        values.update(max_results=5, screening_limit=20, evidence_limit=3, fulltext_limit=1)
    if config is not None:
        import json
        supplied = json.loads(Path(config).read_text(encoding='utf-8'))
        if not isinstance(supplied, dict):
            raise ValueError('settings file must be a JSON object')
        values.update(supplied)
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return ProcessingSettings.model_validate(values)


def add_settings_arguments(parser):
    parser.add_argument('--preset', choices=('standard', 'pilot'))
    parser.add_argument('--config', type=Path)
    for name in LIMIT_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), type=int)


def settings_from_args(args):
    return resolve_settings(args.preset or 'standard', args.config,
                            {key: getattr(args, key, None) for key in LIMIT_FIELDS})


def has_settings_override(args):
    return bool(args.preset or args.config or any(getattr(args, key, None) is not None for key in LIMIT_FIELDS))
