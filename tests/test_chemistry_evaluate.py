"""Evaluation: leakage-safety, recovery of a real signal, no spurious signal."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import (  # noqa: E402
    HAS_NUMPY,
    fast_chemistry,
    recovery_synthetic,
)

if TYPE_CHECKING:
    from courtgraph.chemistry.evaluate import HoldoutResult


@unittest.skipUnless(HAS_NUMPY, "evaluation requires numpy")
class RecoveryTests(unittest.TestCase):
    """Master plan 33.3: the model recovers a real interaction signal and does
    not manufacture one when none exists."""

    signal: HoldoutResult
    null: HoldoutResult

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.evaluate import evaluate_holdout
        from courtgraph.chemistry.splits import make_all_splits
        from courtgraph.chemistry.synthetic import generate

        cfg = recovery_synthetic()
        chem = fast_chemistry()

        signal_table, signal_truth = generate(cfg)
        cls.signal = evaluate_holdout(
            signal_table,
            make_all_splits(signal_table)["unseen_lineup"],
            config=chem,
            truth=signal_truth,
        )

        null_table, null_truth = generate(cfg.with_no_interaction())
        cls.null = evaluate_holdout(
            null_table,
            make_all_splits(null_table)["unseen_lineup"],
            config=chem,
            truth=null_truth,
        )

    def test_holdout_has_no_leakage(self) -> None:
        self.assertEqual(self.signal.leakage_violations, ())
        self.assertEqual(self.null.leakage_violations, ())

    def test_full_model_beats_additive_on_unseen_lineups_with_a_real_signal(
        self,
    ) -> None:
        m = self.signal.metrics
        self.assertLess(
            m["full_rmse_truth_macro"],
            m["additive_rmse_truth_macro"],
            "full model should predict unseen lineup value better than additive",
        )
        self.assertGreater(self.signal.headline_improvement_pct, 8.0)
        self.assertGreater(m["interaction_recovery_corr"], 0.25)
        self.assertGreater(
            self.signal.approximate_delta_interval["prob_full_better"], 0.75
        )

    def test_no_spurious_chemistry_when_the_true_signal_is_zero(self) -> None:
        self.assertLess(abs(self.null.headline_improvement_pct), 5.0)
        self.assertLess(self.null.metrics["predicted_interaction_sd"], 1.5)

    def test_group_rows_carry_support_and_novelty(self) -> None:
        for row in self.signal.groups:
            self.assertIn(row.novelty, {"seen", "partially-seen", "unseen"})
            self.assertGreater(row.test_possessions, 0)
            self.assertGreaterEqual(row.prob_interaction_positive, 0.0)
            self.assertLessEqual(row.prob_interaction_positive, 1.0)

    def test_group_uncertainty_is_derived_from_group_level_bootstrap_samples(
        self,
    ) -> None:
        # each group's P(C>0) is a fraction of the B ensemble members whose
        # possession-weighted group prediction is positive -> a multiple of 1/B.
        # (Averaging row-level probabilities, the old behaviour, would not be.)
        boot = fast_chemistry().n_bootstrap
        for row in self.signal.groups:
            scaled = row.prob_interaction_positive * boot
            self.assertAlmostEqual(scaled, round(scaled), places=6, msg=row.group_id)


@unittest.skipUnless(HAS_NUMPY, "evaluation requires numpy")
class SuiteShapeTests(unittest.TestCase):
    def test_suite_runs_all_three_holdouts_and_is_json_serializable(self) -> None:
        from courtgraph.chemistry.evaluate import evaluate_suite
        from courtgraph.chemistry.splits import make_all_splits
        from courtgraph.chemistry.synthetic import generate

        table, truth = generate(recovery_synthetic())
        summary = evaluate_suite(
            table,
            make_all_splits(table),
            config=fast_chemistry(),
            truth=truth,
        )
        self.assertEqual(
            [h.kind for h in summary.holdouts],
            ["chronological", "unseen_pair", "unseen_lineup"],
        )
        self.assertIn("offensive_talent_corr", summary.recovery)
        self.assertGreater(summary.recovery["offensive_talent_corr"], 0.45)
        json.dumps(summary.as_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
