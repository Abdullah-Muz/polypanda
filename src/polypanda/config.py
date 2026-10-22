"""Strict construction of (nested) frozen dataclass configs from YAML."""

import dataclasses
import typing
from pathlib import Path

import yaml


def from_dict[T](cls: type[T], data: dict) -> T:
    """Build `cls` from a dict, recursing into dataclass fields; unknown keys are an error."""
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            value = from_dict(hint, value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_yaml[T](cls: type[T], path: str | Path) -> T:
    with open(path) as f:
        return from_dict(cls, yaml.safe_load(f) or {})
