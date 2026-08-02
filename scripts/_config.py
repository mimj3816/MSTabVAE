"""Tiny YAML config loader shared by the CLI scripts."""

from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
