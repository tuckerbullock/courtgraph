"""Versioned, deterministic serialization for a fitted rung-3 model.

Mirrors :mod:`courtgraph.chemistry.artifact` (the ``ChemistryModel`` artifact)
but uses its own schema key and top-level shape, so a rung-3 file can never be
mistaken for -- or accidentally loaded as -- a synthetic ``ChemistryModel``
artifact or vice versa. Rung 3 (``HierarchicalRidge``, in ``hierarchical.py``)
has no interaction term: this is the additive-talent-plus-context model whose
calibrated uncertainty is the one validated result of research cycle 1
(see ``docs/RESEARCH_REPORT.md``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from courtgraph import __version__
from courtgraph.chemistry.hierarchical import (
    HIERARCHICAL_SCHEMA_VERSION,
    HierarchicalRidge,
)

RUNG3_ARTIFACT_SCHEMA_VERSION = 1
_TOP_KEYS = ("rung3_artifact_schema_version", "courtgraph_version", "model", "metadata")


def save_model(
    model: HierarchicalRidge,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "rung3_artifact_schema_version": RUNG3_ARTIFACT_SCHEMA_VERSION,
        "courtgraph_version": __version__,
        "model": model.to_dict(),
        "metadata": dict(metadata or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model(path: str | Path) -> tuple[HierarchicalRidge, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rung-3 model artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _TOP_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"{path}: not a CourtGraph rung-3 model artifact (missing {missing})"
        )
    version = payload["rung3_artifact_schema_version"]
    if version != RUNG3_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: rung3_artifact_schema_version {version} != "
            f"{RUNG3_ARTIFACT_SCHEMA_VERSION}"
        )
    model_data = payload["model"]
    if model_data.get("hierarchical_schema_version") != HIERARCHICAL_SCHEMA_VERSION:
        raise ValueError(f"{path}: unrecognized hierarchical model schema version")
    model = HierarchicalRidge.from_dict(model_data)
    metadata = dict(payload.get("metadata", {}))
    return model, metadata
