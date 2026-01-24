from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def resolve_config_path() -> Path:
    filename = Path.cwd() / "util\config.yaml"
    print(filename)
    return Path(filename)


@lru_cache(maxsize=1)
def load_raw_config() -> dict:
    path = resolve_config_path()
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def load_config(model: Type[T], *, section: str | None = None) -> T:
    raw = load_raw_config()

    data = raw[section] if section else raw
    if data is None:
        raise RuntimeError(f"Missing config section: {section}")

    return model.model_validate(data)