"""Explicit, versioned ingestion policy.

Every research choice the adapter makes -- the garbage-time rule, the
reconciliation tolerance, what to do with an ambiguous possession -- lives here
as a named field with a documented default, not as a magic constant buried in
the parser (``AGENTS.md``: "Store configuration explicitly"). ``policy_version``
is written into every audit manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

POLICY_VERSION = "cg-ingest-policy/1"


@dataclass(frozen=True)
class IngestPolicy:
    """Deterministic knobs for one ingestion run."""

    policy_version: str = POLICY_VERSION

    # --- reconciliation ---------------------------------------------------
    # Final score is reconciled exactly against the independent box-score total
    # in the snapshot metadata. A game that fails is quarantined whole unless
    # ``allow_score_mismatch`` is set (then it is emitted with a manifest flag).
    require_exact_final_score: bool = True
    allow_score_mismatch: bool = False
    period_score_tolerance: int = 0  # informational only; deltas always recorded

    # --- possession acceptance -----------------------------------------
    # A reconstructed possession is dropped (not fabricated) when:
    max_possession_points: int = 6  # > this after tech-FT adjustment => ambiguous
    drop_empty_possessions: bool = True  # possession with no value/'ending' event
    # Split-lineup possessions (a substitution between live events) are never
    # attributed to a stint; "downweight" is a later task (master plan 7.4).
    quarantine_split_lineup_possessions: bool = True

    # --- stint shaping -------------------------------------------------
    min_offensive_possessions_per_stint: int = 1

    # --- garbage-time weight (master plan 7.6 "explicit deterministic rule") --
    # Baseline: full weight, except a blowout in the final minutes of the game.
    garbage_time_weight_full: float = 1.0
    garbage_time_weight_blowout: float = 0.2
    garbage_time_period_at_or_after: int = 4
    garbage_time_seconds_remaining_below: float = 300.0
    garbage_time_margin_above: int = 25

    def garbage_time_weight(
        self, period: int, seconds_remaining: float, score_margin_abs: int
    ) -> float:
        """The deterministic baseline weight for a stint (always in ``(0, 1]``)."""

        if (
            period >= self.garbage_time_period_at_or_after
            and seconds_remaining < self.garbage_time_seconds_remaining_below
            and score_margin_abs > self.garbage_time_margin_above
        ):
            return self.garbage_time_weight_blowout
        return self.garbage_time_weight_full

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
