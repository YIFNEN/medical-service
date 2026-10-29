"""Shared strict schema base; separate to keep runtime contracts acyclic."""
from pydantic import BaseModel, ConfigDict


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, allow_inf_nan=False)
