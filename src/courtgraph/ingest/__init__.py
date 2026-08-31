"""Offline NBA snapshot -> validated stint ingestion.

This package converts a stored, immutable snapshot of stats.nba.com responses
into records accepted by :mod:`courtgraph.chemistry.stints`, using ``pbpstats``
in **file-only mode** purely as a possession / lineup reconstruction tool
(``DATA_SOURCES.md`` SRC-PBPSTATS). It never contacts the network, never
mutates the snapshot, and quarantines rather than fabricates when an input is
ambiguous or incomplete.

The public surface is import-light: ``courtgraph doctor`` and the chemistry
path never import this package, and this module never imports ``pbpstats`` or
``numpy`` at module load. Import the submodules directly.
"""

from __future__ import annotations

# One documented snapshot layout for cycle 1 (see snapshot.py for the contract).
SNAPSHOT_FORMAT = "stats_nba_pbpstats/v1"

# Bump when the parser/policy contract changes in a way that alters emitted rows.
INGEST_PIPELINE_VERSION = 1

__all__ = ["SNAPSHOT_FORMAT", "INGEST_PIPELINE_VERSION"]
