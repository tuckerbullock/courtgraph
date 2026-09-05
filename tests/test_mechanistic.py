"""Mechanistic-outcome evaluation (candidate idea #2)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402
from test_role_interaction import _role_dataset  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.features.stint_shots import ShotAttribution


def _planted_attribution(
    table: Any, clustering: Any, delta: Any, *, sigma: float = 0.06, seed: int = 1
) -> ShotAttribution:
    """Fabricate per-stint shot aggregates so that points-per-shot carries a
    planted role-pair effect: pps = 1.05 + sum(delta over offense role pairs)
    + noise, at a fixed 12 FGA per stint."""

    import numpy as np

    from courtgraph.features.stint_shots import ShotAttribution, StintShots

    rng = np.random.default_rng(seed)
    per_stint = {}
    for stint in table:
        roles = [clustering.role_of(p) for p in stint.offense_player_ids]
        surplus = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                ra, rb = roles[a], roles[b]
                surplus += delta[(min(ra, rb), max(ra, rb))]
        pps = 1.05 + 0.1 * surplus + float(rng.normal(0, sigma))
        fga = 12
        fg_points = int(round(pps * fga))
        per_stint[stint.stint_id] = StintShots(
            fga=fga,
            fgm=fg_points // 2,
            fg3a=4,
            fg3m=1,
            rim_fga=4,
            mid_fga=4,
            corner3_fga=1,
            fg_points=fg_points,
        )
    return ShotAttribution(
        per_stint=per_stint,
        shots_total=len(per_stint) * 12,
        shots_matched=len(per_stint) * 12,
        shots_unmatched=0,
    )


def _planted_event_attribution(
    table: Any,
    clustering: Any,
    delta: Any,
    *,
    outcome: str,
    sigma: float = 0.03,
    seed: int = 2,
) -> Any:
    """Fabricate per-stint turnover/assist aggregates with a planted role-pair
    effect on ``outcome`` (``turnover_rate`` or ``assist_rate``), at a fixed
    10 possessions / 8 FGM per stint."""

    import numpy as np

    from courtgraph.features.stint_events import EventAttribution, StintPlayEvents

    rng = np.random.default_rng(seed)
    per_stint = {}
    possessions, fgm = 10, 8
    for stint in table:
        roles = [clustering.role_of(p) for p in stint.offense_player_ids]
        surplus = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                ra, rb = roles[a], roles[b]
                surplus += delta[(min(ra, rb), max(ra, rb))]
        if outcome == "turnover_rate":
            rate = max(0.0, 0.14 + 0.02 * surplus + float(rng.normal(0, sigma)))
            turnovers = int(round(rate * possessions))
            per_stint[stint.stint_id] = StintPlayEvents(
                turnovers=turnovers,
                offensive_possessions=possessions,
                fgm=fgm,
                assisted_fgm=fgm // 2,
            )
        else:
            rate = min(
                1.0, max(0.0, 0.55 + 0.05 * surplus + float(rng.normal(0, sigma)))
            )
            assisted = int(round(rate * fgm))
            per_stint[stint.stint_id] = StintPlayEvents(
                turnovers=1,
                offensive_possessions=possessions,
                fgm=fgm,
                assisted_fgm=assisted,
            )
    n = len(per_stint)
    return EventAttribution(
        per_stint=per_stint,
        events_total=n * 2,
        events_matched=n * 2,
        events_unmatched=0,
    )


@unittest.skipUnless(HAS_NUMPY, "mechanistic eval requires numpy")
class MechanisticEvalTests(unittest.TestCase):
    def test_detects_a_planted_role_effect_on_shot_quality(self) -> None:
        from courtgraph.chemistry.mechanistic import evaluate_mechanistic

        table, clustering, delta = _role_dataset(n_stints=8000, tau_role=2.0, seed=6)
        att = _planted_attribution(table, clustering, delta)
        comp = evaluate_mechanistic(
            table, att, clustering, outcome="pts_per_shot", min_fga=3, n_boot=0
        )
        self.assertEqual(comp.outcome, "pts_per_shot")
        self.assertEqual(len(comp.holdouts), 3)
        self.assertGreater(comp.n_stints_kept, 5000)
        wins = sum(
            1 for h in comp.holdouts if h.role_macro_rmse < h.role_placebo_macro_rmse
        )
        self.assertGreaterEqual(wins, 2)

    def test_rejects_an_unknown_outcome(self) -> None:
        from courtgraph.chemistry.mechanistic import evaluate_mechanistic

        table, clustering, delta = _role_dataset(n_stints=3000)
        att = _planted_attribution(table, clustering, delta)
        with self.assertRaises(ValueError):
            evaluate_mechanistic(table, att, clustering, outcome="nonsense")

    def test_detects_a_planted_role_effect_on_turnover_rate(self) -> None:
        from courtgraph.chemistry.mechanistic import evaluate_mechanistic

        table, clustering, delta = _role_dataset(n_stints=8000, tau_role=2.0, seed=7)
        att = _planted_event_attribution(
            table, clustering, delta, outcome="turnover_rate"
        )
        comp = evaluate_mechanistic(
            table, att, clustering, outcome="turnover_rate", min_fga=3, n_boot=0
        )
        self.assertEqual(comp.outcome, "turnover_rate")
        self.assertGreater(comp.n_stints_kept, 5000)
        wins = sum(
            1 for h in comp.holdouts if h.role_macro_rmse < h.role_placebo_macro_rmse
        )
        self.assertGreaterEqual(wins, 2)

    def test_detects_a_planted_role_effect_on_assist_rate(self) -> None:
        from courtgraph.chemistry.mechanistic import evaluate_mechanistic

        table, clustering, delta = _role_dataset(n_stints=8000, tau_role=2.0, seed=8)
        att = _planted_event_attribution(
            table, clustering, delta, outcome="assist_rate"
        )
        comp = evaluate_mechanistic(
            table, att, clustering, outcome="assist_rate", min_fga=3, n_boot=0
        )
        self.assertEqual(comp.outcome, "assist_rate")
        wins = sum(
            1 for h in comp.holdouts if h.role_macro_rmse < h.role_placebo_macro_rmse
        )
        self.assertGreaterEqual(wins, 2)

    def test_min_fga_filter_drops_thin_stints(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.mechanistic import mechanistic_table_and_design
        from courtgraph.features.stint_shots import ShotAttribution, StintShots

        table, clustering, delta = _role_dataset(n_stints=1200)
        att = _planted_attribution(table, clustering, delta)
        # zero out FGA for the first 200 stints
        thin = dict(att.per_stint)
        for stint in list(table)[:200]:
            thin[stint.stint_id] = StintShots(1, 0, 0, 0, 0, 0, 0, 0)
        att2 = ShotAttribution(thin, att.shots_total, att.shots_matched, 0)
        space = FeatureSpace.from_training(table)
        kt, design = mechanistic_table_and_design(
            space, table, att2, "pts_per_shot", min_fga=3
        )
        self.assertEqual(len(kt), len(table) - 200)
        self.assertTrue(np.all(design.weight >= 3))


if __name__ == "__main__":
    unittest.main()
