"""
himalayas
~~~~~~~~~

HiMaLAYAS: Hierarchical Matrix Layout and Annotation Software
"""

from .core.matrix import Matrix
from .core.annotations import Annotations
from .core.clustering import cluster
from .core.enrichment import run_first_pass
from .core.results import Results

__all__ = [
    "Matrix",
    "Annotations",
    "cluster",
    "run_first_pass",
    "Results",
]

__version__ = "0.0.2-beta.0"
