"""Lineup-chemistry vertical slice: stints, synthetic data, splits, models.

The public surface is intentionally small and import-light so that
``courtgraph doctor`` (and the dependency-free path) never pulls in NumPy.
Import the submodules directly for the modeling code.
"""

from __future__ import annotations

CHEMISTRY_STINT_SCHEMA_VERSION = 1
CHEMISTRY_ARTIFACT_SCHEMA_VERSION = 2

__all__ = [
    "CHEMISTRY_ARTIFACT_SCHEMA_VERSION",
    "CHEMISTRY_STINT_SCHEMA_VERSION",
]
