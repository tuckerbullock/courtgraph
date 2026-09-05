"""Model-ladder rungs 0 and 1 (context mean, EB-shrunk lineup ratings)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Any  # noqa: E402

from _chemistry_support import HAS_NUMPY, wellspec_synthetic  # noqa: E402


@unittest.skipUnless(HAS_NUMPY, "rung 0/1 require numpy")
class Rung01Tests(unittest.TestCase):
    def _fit(self) -> tuple[Any, Any, Any, Any]:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.rung01 import ContextMeanModel, LineupMeanModel
        from courtgraph.chemistry.synthetic import generate

        table, _ = generate(wellspec_synthetic())
        space = FeatureSpace.from_training(table)
        design = space.build(table)
        rung0 = ContextMeanModel.fit(design, space)
        rung1 = LineupMeanModel.fit(table, design, space)
        return table, design, rung0, rung1

    def test_rung0_predicts_one_value_per_context_not_per_lineup(self) -> None:
        import numpy as np

        table, design, rung0, _ = self._fit()
        preds = rung0.predict(design)
        # rows sharing a context vector must share a prediction
        _, inverse = np.unique(design.context, axis=0, return_inverse=True)
        for g in np.unique(inverse):
            block = preds[inverse == g]
            self.assertLess(float(block.max() - block.min()), 1e-9)

    def test_rung1_shrinks_toward_rung0_and_helps_in_sample(self) -> None:
        import numpy as np

        table, design, rung0, rung1 = self._fit()
        p0 = rung0.predict(design)
        p1 = rung1.predict(table, design)
        # every adjustment is a shrunk (|B|<=1) fraction of the raw residual
        self.assertGreater(rung1.tau2, 0.0)
        self.assertGreater(rung1.n_lineups, 50)
        # in-sample, adding the (shrunk, real) lineup means cannot hurt much and
        # should help: possession-weighted RMSE of rung 1 <= rung 0.
        w = design.weight
        rmse0 = float(np.sqrt(np.average((p0 - design.y) ** 2, weights=w)))
        rmse1 = float(np.sqrt(np.average((p1 - design.y) ** 2, weights=w)))
        self.assertLessEqual(rmse1, rmse0 + 1e-9)

    def test_rung1_equals_rung0_for_an_unseen_lineup(self) -> None:
        from dataclasses import replace

        import numpy as np

        from courtgraph.chemistry.stints import StintTable

        table, _design, rung0, rung1 = self._fit()
        space = rung0.feature_space
        seen = {s.offense_lineup_id for s in table}
        players = sorted(space.player_ids)
        combo = tuple(sorted(players[:5]))
        probe = StintTable.from_stints(
            [replace(table.stints[0], offense_player_ids=combo)]
        )
        if probe.stints[0].offense_lineup_id in seen:
            self.skipTest("probe lineup happened to be seen")
        pd = space.build(probe)
        self.assertTrue(np.allclose(rung0.predict(pd), rung1.predict(probe, pd)))


if __name__ == "__main__":
    unittest.main()
