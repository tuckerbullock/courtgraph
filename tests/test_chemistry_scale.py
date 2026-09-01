"""The chemistry model must fit a full-season-shaped player pool without a
per-stint dense (n, n_players) allocation. Guards the sparse-Gram rework
against a regression to the old dense design.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import (  # noqa: E402
    HAS_NUMPY,
    hierarchical_config,
    scale_chemistry,
    scale_synthetic,
)


@unittest.skipUnless(HAS_NUMPY, "model requires numpy")
class ScaleTests(unittest.TestCase):
    def test_fit_scales_to_a_full_season_pool(self) -> None:
        from courtgraph.chemistry.chemistry_model import ChemistryModel
        from courtgraph.chemistry.synthetic import generate

        table, _truth = generate(scale_synthetic())
        n_players = len(table.player_ids())
        self.assertGreater(len(table), 15_000)
        self.assertGreater(n_players, 300)

        tracemalloc.start()
        start = time.perf_counter()
        model = ChemistryModel.fit(table, scale_chemistry())
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(model.interaction.provision.shape, (n_players + 1, 3))
        # A dense (n, n_players) one-hot here is ~50 MB; the old (n, n_players,
        # rank) ALS buffer ~150 MB. 1 GB leaves generous headroom for the
        # legitimate (n_players*rank)**2 Gram (~10 MB) and working arrays.
        self.assertLess(peak, 1_000_000_000, f"peak tracemalloc {peak / 1e6:.0f} MB")
        # Loose: a dense fit at this size took >1 h. Anything under 5 min means
        # the sparse path is engaged.
        self.assertLess(elapsed, 300.0, f"fit took {elapsed:.0f}s")

    def test_hierarchical_fit_scales_to_a_full_season_pool(self) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.hierarchical import HierarchicalRidge
        from courtgraph.chemistry.synthetic import generate

        table, _truth = generate(scale_synthetic())
        space = FeatureSpace.from_training(table)
        design = space.build(table)
        n_players = len(table.player_ids())

        tracemalloc.start()
        start = time.perf_counter()
        model = HierarchicalRidge.fit(design, space, config=hierarchical_config())
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(model.converged)
        self.assertEqual(model.offense_coef.shape, (n_players,))
        self.assertLess(peak, 1_500_000_000, f"peak tracemalloc {peak / 1e6:.0f} MB")
        self.assertLess(elapsed, 60.0, f"EM fit took {elapsed:.0f}s")

        # group predictive over a few hundred synthetic groups must be quick
        import numpy as np

        rng = np.random.default_rng(0)
        rows = rng.choice(len(table), size=400, replace=False)
        groups = {str(r): np.array([int(r)], dtype=np.int64) for r in rows}
        start = time.perf_counter()
        pred = model.group_predictive(design, groups)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(len(pred), 400)


if __name__ == "__main__":
    unittest.main()
