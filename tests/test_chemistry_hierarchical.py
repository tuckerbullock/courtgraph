"""Rung 3 -- the empirical-Bayes hierarchical player model."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import (  # noqa: E402
    HAS_NUMPY,
    hierarchical_config,
    recovery_synthetic,
    wellspec_synthetic,
)

if TYPE_CHECKING:
    from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
    from courtgraph.chemistry.hierarchical import HierarchicalRidge
    from courtgraph.chemistry.stints import StintTable
    from courtgraph.chemistry.synthetic import GroundTruth


@unittest.skipUnless(HAS_NUMPY, "hierarchical model requires numpy")
class HierarchicalRidgeTests(unittest.TestCase):
    table: StintTable
    truth: GroundTruth
    space: FeatureSpace
    design: DesignMatrices
    model: HierarchicalRidge

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.hierarchical import HierarchicalRidge
        from courtgraph.chemistry.synthetic import generate

        cls.table, cls.truth = generate(wellspec_synthetic())
        cls.space = FeatureSpace.from_training(cls.table)
        cls.design = cls.space.build(cls.table)
        # a non-monotone EM step warns; make that fatal for the whole suite.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.model = HierarchicalRidge.fit(
                cls.design, cls.space, config=hierarchical_config()
            )

    def test_recovers_the_realised_player_effect_sd(self) -> None:
        import numpy as np

        vc = self.model.variance_components()
        realised_off = float(np.std(self.truth.off_talent))
        realised_def = float(np.std(self.truth.def_talent))
        self.assertLess(abs(vc["tau_off"] / realised_off - 1.0), 0.15)
        self.assertLess(abs(vc["tau_def"] / realised_def - 1.0), 0.15)
        self.assertTrue(self.model.converged)
        self.assertGreater(vc["sigma"], 0.0)

    def test_em_does_not_decrease_the_log_likelihood(self) -> None:
        from courtgraph.chemistry.hierarchical import HierarchicalRidge

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = HierarchicalRidge.fit(
                self.design, self.space, config=hierarchical_config()
            )
        self.assertTrue(model.converged)

    def test_shrinkage_is_between_no_pool_and_full_pool(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline import AdditiveRidge

        no_pool = AdditiveRidge.fit(
            self.design, self.space, l2_player=1e-6, l2_context=1e-3
        )
        full_pool = AdditiveRidge.fit(
            self.design, self.space, l2_player=1e9, l2_context=1e-3
        )
        eb_norm = float(np.linalg.norm(self.model.offense_coef))
        self.assertLess(eb_norm, float(np.linalg.norm(no_pool.offense_coef)))
        self.assertGreater(eb_norm, float(np.linalg.norm(full_pool.offense_coef)))

    def test_is_deterministic(self) -> None:
        import numpy as np

        from courtgraph.chemistry.hierarchical import HierarchicalRidge

        again = HierarchicalRidge.fit(
            self.design, self.space, config=hierarchical_config()
        )
        self.assertTrue(np.array_equal(again.offense_coef, self.model.offense_coef))
        self.assertTrue(np.array_equal(again.defense_coef, self.model.defense_coef))
        self.assertEqual(again.tau_off2, self.model.tau_off2)
        self.assertEqual(again.sigma2, self.model.sigma2)

    def test_serialization_round_trip(self) -> None:
        import numpy as np

        from courtgraph.chemistry.hierarchical import HierarchicalRidge

        restored = HierarchicalRidge.from_dict(self.model.to_dict())
        self.assertTrue(
            np.allclose(restored.predict(self.design), self.model.predict(self.design))
        )
        self.assertEqual(restored.tau_off2, self.model.tau_off2)
        self.assertEqual(restored.to_dict(), self.model.to_dict())

    def test_decomposition_identity(self) -> None:
        d = self.model.decompose_row(self.design, 0)
        self.assertAlmostEqual(d.talent + d.context, d.total, places=9)

    def test_group_interval_coverage_is_near_nominal(self) -> None:
        import numpy as np

        from courtgraph.chemistry.calibration import coverage

        rng = np.random.default_rng(0)
        rows = rng.choice(len(self.table), size=800, replace=False)
        groups = {str(r): np.array([int(r)], dtype=np.int64) for r in rows}
        pred = self.model.group_predictive(self.design, groups)

        point = np.array([pred[k][0] for k in groups])
        sd = np.array([pred[k][1] for k in groups])
        realized = np.array([self.design.y[int(k)] for k in groups])
        cov = coverage(point, sd, realized)
        self.assertGreaterEqual(cov[0.5], 0.40)
        self.assertLessEqual(cov[0.5], 0.60)
        self.assertGreaterEqual(cov[0.8], 0.70)
        self.assertLessEqual(cov[0.8], 0.90)
        self.assertGreaterEqual(cov[0.95], 0.88)


@unittest.skipUnless(HAS_NUMPY, "hierarchical model requires numpy")
class HierarchicalPointAccuracyTests(unittest.TestCase):
    def test_talent_recovery_tracks_the_additive_baseline(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline import AdditiveRidge
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.hierarchical import HierarchicalRidge
        from courtgraph.chemistry.synthetic import generate

        table, truth = generate(recovery_synthetic())
        space = FeatureSpace.from_training(table)
        design = space.build(table)
        rung2 = AdditiveRidge.fit(design, space)
        rung3 = HierarchicalRidge.fit(design, space, config=hierarchical_config())

        pids = list(space.player_ids)
        truth_off = np.array([truth.off_talent[pids.index(p)] for p in pids])
        corr2 = np.corrcoef([rung2.talent_of(p)[0] for p in pids], truth_off)[0, 1]
        corr3 = np.corrcoef([rung3.talent_of(p)[0] for p in pids], truth_off)[0, 1]
        self.assertGreater(corr3, corr2 - 0.05)


if __name__ == "__main__":
    unittest.main()
