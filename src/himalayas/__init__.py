"""
himalayas
~~~~~~~~~

Hierarchical Matrix Layout and Annotation Software (HiMaLAYAS)
Enrichment-based annotation of hierarchically clustered matrices
"""

from .core.matrix import Matrix
from .core.annotations import Annotations
from .core.analysis import Analysis
from .core.clustering import cluster
from .core.results import Results

__all__ = [
    "Matrix",
    "Annotations",
    "Analysis",
    "cluster",
    "Results",
]

__version__ = "0.0.7"
