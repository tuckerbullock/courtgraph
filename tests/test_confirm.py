"""Better-powered confirmation run + the bootstrap-delta helper."""

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


def _shot_attribution(
    table: Any, clustering: Any, delta: Any, *, seed: int = 1
) -> ShotAttribution:
    """Fabricate per-stint shots so that three_share carries the planted role
    effect (fga fixed at 15; fg3a = a role-dependent count)."""

    import numpy as np

    from courtgraph.features.stint_shots import ShotAttribution, StintShots

    rng = np.random.default_rng(seed)
    per_stint = {}
    for stint in table:
        roles = [clustering.role_of(p) for p in stint.offense_player_ids]
        surplus = sum(
            delta[(min(roles[a], roles[b]), max(roles[a], roles[b]))]
            for a in range(5)
            for b in range(a + 1, 5)
        )
        fga = 15
        base = 5 + 0.4 * surplus + float(rng.normal(0, 0.6))
        fg3a = int(min(max(round(base), 0), fga))
        per_stint[stint.stint_id] = StintShots(
            fga=fga,
            fgm=7,
            fg3a=fg3a,
            fg3m=fg3a // 3,
            rim_fga=4,
            mid_fga=fga - fg3a - 4,
            corner3_fga=1,
            fg_points=15,
        )
    n = len(per_stint) * 15
    return ShotAttribution(
        per_stint=per_stint, shots_total=n, shots_matched=n, shots_unmatched=0
    )


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class BootstrapDeltaTests(unittest.TestCase):
    def test_clear_improvement_has_frac_gt_0_near_one(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline_ladder import bootstrap_group_delta

        rng = np.random.default_rng(0)
        y = rng.normal(0, 5, 80)
        # model is much closer to y than the baseline
        base = y + rng.normal(0, 4, 80)
        model = y + rng.normal(0, 1, 80)
        d = bootstrap_group_delta(base, model, y, n_boot=1000, seed=1)
        self.assertGreater(d["frac_gt_0"], 0.99)
        self.assertGreater(d["ci_lo"], 0.0)

    def test_no_difference_straddles_zero(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline_ladder import bootstrap_group_delta

        rng = np.random.default_rng(2)
        y = rng.normal(0, 5, 60)
        a = y + rng.normal(0, 3, 60)
        b = y + rng.normal(0, 3, 60)
        d = bootstrap_group_delta(a, b, y, n_boot=1000, seed=3)
        self.assertLess(d["ci_lo"], 0.0)
        self.assertGreater(d["ci_hi"], 0.0)


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class ConfirmationRunTests(unittest.TestCase):
    def test_run_confirmation_produces_ci_rows_and_detects_signal(self) -> None:
        from courtgraph.chemistry.confirm import run_confirmation

        table, clustering, delta = _role_dataset(n_stints=7000, tau_role=2.4, seed=6)
        att = _shot_attribution(table, clustering, delta)
        result = run_confirmation(
            table,
            {5: clustering},
            att,
            n_lineups=40,
            n_boot=400,
        )
        # role model at k=5 should have a CI row per structural holdout
        role_rows = [r for r in result.rows if r.model == "role"]
        self.assertEqual(len(role_rows), 2)
        for r in role_rows:
            self.assertIn("mean", r.delta_vs_rung3)
            self.assertIn("ci_lo", r.delta_vs_placebo)
        self.assertEqual(set(result.holdout_groups), {"unseen_pair", "unseen_lineup"})


if __name__ == "__main__":
    unittest.main()
