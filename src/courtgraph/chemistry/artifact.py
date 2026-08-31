"""Versioned, deterministic serialization for a fitted chemistry model.

The artifact is plain JSON: model weights, the feature space, training-support
tables, and a small metadata block. No timestamps are written unless the caller
supplies them, so ``fit`` output is byte-reproducible for a given seed and input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from courtgraph import __version__
from courtgraph.chemistry.chemistry_model import ARTIFACT_SCHEMA_VERSION, ChemistryModel

_TOP_KEYS = ("artifact_schema_version", "courtgraph_version", "model", "metadata")


def save_model(
    model: ChemistryModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "courtgraph_version": __version__,
        "model": model.to_dict(),
        "metadata": dict(metadata or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model(path: str | Path) -> tuple[ChemistryModel, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _TOP_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path}: not a CourtGraph model artifact (missing {missing})")
    version = payload["artifact_schema_version"]
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: artifact_schema_version {version} != {ARTIFACT_SCHEMA_VERSION}"
        )
    model = ChemistryModel.from_dict(payload["model"])
    metadata = dict(payload.get("metadata", {}))
    return model, metadata
