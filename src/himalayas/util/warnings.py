"""Minimal warning helpers with no side effects."""

from __future__ import annotations

import warnings
from typing import Type


def warn(message: str, category: Type[Warning] = RuntimeWarning, stacklevel: int = 2) -> None:
    warnings.warn(message, category=category, stacklevel=stacklevel)
