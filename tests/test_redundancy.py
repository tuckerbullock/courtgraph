"""Skill-redundancy / anti-synergy model (candidate idea #3)."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.stints import StintTable
    from courtgraph.features.role_clusters import RoleClustering


def _redundancy_dataset(
    *,
    n_players: int = 60,
    n_stints: int = 9000,
    d: int = 3,
    rho: tuple[float, ...] = (-0.6, 0.5, 0.2),
    tau_off: float = 3.0,
    sigma: float = 9.0,
    seed: int = 4,
) -> tuple[StintTable, RoleClustering, tuple[float, ...]]:
    """Additive talent + a planted per-dimension concentration effect
    ``sum_d rho_d * ((sum_i z_id)^2 - sum_i z_id^2)`` on standardized role
    vectors."""

    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable
    from courtgraph.features.role_clusters import RoleClustering

    rng = np.random.default_rng(seed)
    players = list(range(700, 700 + n_players))
    zmat = rng.normal(0, 1, (n_players, d))
    zmat = (zmat - zmat.mean(axis=0)) / zmat.std(axis=0)
    z = {players[i]: zmat[i] for i in range(n_players)}
    off_talent = dict(
        zip(players, rng.normal(0, tau_off, n_players).tolist(), strict=True)
    )
    def_talent = dict(zip(players, rng.normal(0, 2.4, n_players).tolist(), strict=True))
    rotations = [
        tuple(sorted(int(x) for x in rng.choice(players, 5, replace=False)))
        for _ in range(40)
    ]

    stints: list[Any] = []
    for i in range(n_stints):
        if rng.random() < 0.8:
            off = rotations[int(rng.integers(len(rotations)))]
        else:
            off = tuple(sorted(int(x) for x in rng.choice(players, 5, replace=False)))
        deff = tuple(sorted(int(x) for x in rng.choice(players, 5, replace=False)))
        if set(off) & set(deff):
            continue
        zsum = np.sum([z[p] for p in off], axis=0)
        zsq = np.sum([z[p] ** 2 for p in off], axis=0)
        conc = zsum**2 - zsq
        w = int(rng.integers(4, 16))
        value = (
            110.0
            + sum(off_talent[p] for p in off)
            - sum(def_talent[p] for p in deff)
            + float(np.dot(rho, conc))
        )
        y = value + float(rng.normal(0, sigma / np.sqrt(w)))
        stints.append(
            Stint(
                stint_id=f"s{i}",
                game_id=f"g{i // 10}",
                game_date=f"2022-{1 + (i // 400) % 12:02d}-{1 + i % 27:02d}",
                season="2021-22",
                season_index=0,
                period=1 + i % 4,
                start_time_seconds=float(i % 600),
                offense_team_id=1,
                defense_team_id=2,
                offense_player_ids=off,  # type: ignore[arg-type]
                defense_player_ids=deff,  # type: ignore[arg-type]
                offensive_possessions=w,
                points_scored=int(round(y * w / 100.0)),
                home_offense=bool(i % 2),
                score_margin_offense=0,
                playoff=False,
                days_rest_offense=1,
                garbage_time_weight=1.0,
            )
        )

    clustering = RoleClustering(
        features=tuple(f"f{k}" for k in range(d)),
        n_clusters=3,
        mean=np.zeros(d),
        std=np.ones(d),
        centers=np.zeros((3, d)),
        assignment={},
        player_cluster={p: i % 3 for i, p in enumerate(players)},
        player_vector={p: tuple(float(x) for x in z[p]) for p in players},
        seed=0,
    )
    return StintTable.from_stints(stints), clustering, rho


@unittest.skipUnless(HAS_NUMPY, "redundancy model requires numpy")
class RedundancyInteractionTests(unittest.TestCase):
    table: StintTable
    clustering: RoleClustering
    rho: tuple[float, ...]
    model: Any

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.redundancy import (
            RedundancyConfig,
            RedundancyInteraction,
        )

        cls.table, cls.clustering, cls.rho = _redundancy_dataset()
        space = FeatureSpace.from_training(cls.table)
        design = space.build(cls.table)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.model = RedundancyInteraction.fit(
                design,
                space,
                cls.clustering,
                config=RedundancyConfig(tol=1e-6, max_iters=80),
            )

    def test_recovers_the_rho_vector(self) -> None:
        import numpy as np

        # rho is fitted on standardized concentration features, so compare
        # sign and rank rather than raw magnitude
        est = np.array(list(self.model.rho_by_feature().values()))
        tru = np.array(self.rho)
        self.assertGreater(float(np.corrcoef(est, tru)[0, 1]), 0.9)
        # the strongest planted effect (rho_0 < 0) stays clearly negative
        self.assertLess(est[0], 0.0)
        self.assertTrue(self.model.converged)

    def test_placebo_role_vectors_shrink_tau_rho_and_fit_worse(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.redundancy import (
            RedundancyConfig,
            RedundancyInteraction,
        )
        from courtgraph.features.role_clusters import permuted_clustering

        space = FeatureSpace.from_training(self.table)
        design = space.build(self.table)
        placebo = RedundancyInteraction.fit(
            design,
            space,
            permuted_clustering(self.clustering, seed=2),
            config=RedundancyConfig(tol=1e-6, max_iters=80),
        )
        real_rss = float(np.sum(self.model.residuals(design) ** 2))
        placebo_rss = float(np.sum(placebo.residuals(design) ** 2))
        self.assertGreater(placebo_rss, real_rss * 1.02)
        self.assertLess(
            placebo.variance_components()["tau_rho"],
            0.7 * self.model.variance_components()["tau_rho"],
        )

    def test_decomposition_identity_and_determinism(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.redundancy import (
            RedundancyConfig,
            RedundancyInteraction,
        )

        space = FeatureSpace.from_training(self.table)
        design = space.build(self.table)
        again = RedundancyInteraction.fit(
            design,
            space,
            self.clustering,
            config=RedundancyConfig(tol=1e-6, max_iters=80),
        )
        self.assertTrue(np.array_equal(again.rho, self.model.rho))
        d = self.model.decompose_row(design, 0)
        self.assertAlmostEqual(d.talent + d.context, d.total, places=6)


@unittest.skipUnless(HAS_NUMPY, "redundancy eval requires numpy")
class RedundancyEvalTests(unittest.TestCase):
    def test_evaluate_runs_and_beats_placebo_with_signal(self) -> None:
        from courtgraph.chemistry.redundancy_eval import evaluate_redundancy
        from courtgraph.chemistry.splits import make_all_splits

        table, clustering, _ = _redundancy_dataset(n_stints=7000, seed=8)
        comp = evaluate_redundancy(table, make_all_splits(table), clustering, n_boot=0)
        self.assertEqual(len(comp.holdouts), 3)
        self.assertIn("f0", comp.rho)
        wins = sum(
            1
            for h in comp.holdouts
            if h.redundancy_macro_rmse < h.redundancy_placebo_macro_rmse
        )
        self.assertGreaterEqual(wins, 2)

    def test_raw_concentration_hand_case(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.redundancy import raw_concentration
        from courtgraph.chemistry.stints import Stint, StintTable
        from courtgraph.features.role_clusters import RoleClustering

        stint = Stint(
            stint_id="s0",
            game_id="g0",
            game_date="2022-01-01",
            season="2021-22",
            season_index=0,
            period=1,
            start_time_seconds=0.0,
            offense_team_id=1,
            defense_team_id=2,
            offense_player_ids=(1, 2, 3, 4, 5),
            defense_player_ids=(6, 7, 8, 9, 10),
            offensive_possessions=10,
            points_scored=10,
            home_offense=True,
            score_margin_offense=0,
            playoff=False,
            days_rest_offense=1,
            garbage_time_weight=1.0,
        )
        table = StintTable.from_stints([stint])
        space = FeatureSpace.from_training(table)
        # players 1-5 have 1-D role vectors [2, -1, 0, 1, -2]; player 3 has none
        vecs: dict[int, tuple[float, ...]] = {
            1: (2.0,),
            2: (-1.0,),
            4: (1.0,),
            5: (-2.0,),
        }
        clustering = RoleClustering(
            features=("f0",),
            n_clusters=2,
            mean=np.zeros(1),
            std=np.ones(1),
            centers=np.zeros((2, 1)),
            assignment={},
            player_cluster={p: 0 for p in (1, 2, 4, 5)},
            player_vector=vecs,
            seed=0,
        )
        conc = raw_concentration(space.build(table).offense_index, space, clustering)
        # sum = 2 - 1 + 1 - 2 = 0 ; sum of squares = 4 + 1 + 1 + 4 = 10
        self.assertAlmostEqual(conc[0, 0], 0.0**2 - 10.0)


if __name__ == "__main__":
    unittest.main()
