"""Load externalized prompt text from ``prompts.yaml`` and fill placeholders.

Prompt strings live in ``prompts.yaml`` next to this module so they can be
edited without touching Python. Values are plain strings that may contain
``{{placeholder}}`` tokens; :func:`render` substitutes them. Any literal ``{``
or ``}`` (e.g. JSON examples) is left untouched.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")


@lru_cache(maxsize=1)
def _prompts() -> dict:
    with open(_PROMPTS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def P(key: str) -> str:
    """Return the prompt string registered under *key* in ``prompts.yaml``."""
    return _prompts()[key]


def render(template: str, **values: object) -> str:
    """Replace every ``{{name}}`` token in *template* with ``values[name]``."""
    out = template
    for name, value in values.items():
        out = out.replace("{{" + name + "}}", str(value))
    return out
