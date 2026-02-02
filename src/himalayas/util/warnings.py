"""
himalayas/util/warnings
"""

from __future__ import annotations

import warnings
from typing import Type


def warn(message: str, category: Type[Warning] = RuntimeWarning, stacklevel: int = 2) -> None:
    """
    Emits a warning with a default stacklevel.

    Args:
        message (str): Warning message text.
        category (Type[Warning]): Warning category class. Defaults to RuntimeWarning.
        stacklevel (int): Stacklevel to report. Defaults to 2.
    """
    warnings.warn(message, category=category, stacklevel=stacklevel)
