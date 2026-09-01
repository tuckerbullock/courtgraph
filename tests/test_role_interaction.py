"""Role-conditioned interaction RAPM (candidate idea #1)."""

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


def _role_dataset(
    *,
    n_roles: int = 5,
    per_role: int = 12,
    n_stints: int = 9000,
    tau_off: float = 3.0,
    tau_role: float = 1.6,
    sigma: float = 9.0,
    seed: int = 3,
) -> tuple[StintTable, RoleClustering, dict[tuple[int, int], float]]:
    """Additive talent + a planted symmetric role-pair effect ``delta_{a,b}``.

    Players are split into ``n_roles`` equal role groups; a lineup's interaction
    contribution is the sum of ``delta`` over its 10 offensive role pairs.
    """

    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable, pair_id  # noqa: F401
    from courtgraph.features.role_clusters import RoleClustering

    rng = np.random.default_rng(seed)
    n_players = n_roles * per_role
    players = list(range(500, 500 + n_players))
    role_of = {p: (i // per_role) for i, p in enumerate(players)}
    off_talent = dict(
        zip(players, rng.normal(0, tau_off, n_players).tolist(), strict=True)
    )
    def_talent = dict(zip(players, rng.normal(0, 2.4, n_players).tolist(), strict=True))

    delta: dict[tuple[int, int], float] = {}
    for a in range(n_roles):
        for b in range(a, n_roles):
            delta[(a, b)] = float(rng.normal(0, tau_role))

    def d(ra: int, rb: int) -> float:
        return delta[(ra, rb) if ra <= rb else (rb, ra)]

    # a fixed pool of recurring lineups so unseen_lineup / chronological splits
    # get real macro groups (random 5-of-N never repeats)
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
        w = int(rng.integers(4, 16))
        value = (
            110.0
            + sum(off_talent[p] for p in off)
            - sum(def_talent[p] for p in deff)
            + sum(
                d(role_of[off[a]], role_of[off[b]])
                for a in range(5)
                for b in range(a + 1, 5)
            )
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
        features=("usage",),
        n_clusters=n_roles,
        mean=np.zeros(1),
        std=np.ones(1),
        centers=np.arange(n_roles, dtype=np.float64).reshape(-1, 1),
        assignment={(p, "2021-22"): role_of[p] for p in players},
        player_cluster=dict(role_of),
        seed=0,
    )
    return StintTable.from_stints(stints), clustering, delta


@unittest.skipUnless(HAS_NUMPY, "role interaction requires numpy")
class RoleClusterInteractionTests(unittest.TestCase):
    table: StintTable
    clustering: RoleClustering
    delta: dict[tuple[int, int], float]
    model: Any

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.role_interaction import (
            RoleClusterInteraction,
            RoleInteractionConfig,
        )

        cls.table, cls.clustering, cls.delta = _role_dataset()
        space = FeatureSpace.from_training(cls.table)
        design = space.build(cls.table)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.model = RoleClusterInteraction.fit(
                design,
                space,
                cls.clustering,
                config=RoleInteractionConfig(tol=1e-6, max_iters=80),
            )

    def test_recovers_the_role_pair_matrix(self) -> None:
        import numpy as np

        k = self.clustering.n_clusters
        est = self.model.role_pair_matrix()
        true = np.array(
            [[self.delta[(min(a, b), max(a, b))] for b in range(k)] for a in range(k)]
        )
        corr = float(np.corrcoef(est.ravel(), true.ravel())[0, 1])
        self.assertGreater(corr, 0.85)
        self.assertTrue(self.model.converged)

    def test_recovers_tau_role(self) -> None:
        import numpy as np

        realised = float(np.std([v for v in self.delta.values()]))
        vc = self.model.variance_components()
        self.assertLess(abs(vc["tau_role"] / realised - 1.0), 0.30)
        self.assertEqual(vc["n_role_pairs"], 15)

    def test_placebo_clustering_destroys_the_signal(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.role_interaction import (
            RoleClusterInteraction,
            RoleInteractionConfig,
        )
        from courtgraph.features.role_clusters import permuted_clustering

        space = FeatureSpace.from_training(self.table)
        design = space.build(self.table)
        placebo = RoleClusterInteraction.fit(
            design,
            space,
            permuted_clustering(self.clustering, seed=1),
            config=RoleInteractionConfig(tol=1e-6, max_iters=80),
        )
        real_rss = float(np.sum(self.model.residuals(design) ** 2))
        placebo_rss = float(np.sum(placebo.residuals(design) ** 2))
        self.assertGreater(placebo_rss, real_rss * 1.03)
        # the placebo's role variance component shrinks toward zero
        self.assertLess(
            placebo.variance_components()["tau_role"],
            0.6 * self.model.variance_components()["tau_role"],
        )

    def test_is_deterministic_and_decomposes(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.role_interaction import (
            RoleClusterInteraction,
            RoleInteractionConfig,
        )

        space = FeatureSpace.from_training(self.table)
        design = space.build(self.table)
        again = RoleClusterInteraction.fit(
            design,
            space,
            self.clustering,
            config=RoleInteractionConfig(tol=1e-6, max_iters=80),
        )
        self.assertTrue(np.array_equal(again.role_coef, self.model.role_coef))
        d = self.model.decompose_row(design, 0)
        self.assertAlmostEqual(d.talent + d.context, d.total, places=6)


@unittest.skipUnless(HAS_NUMPY, "role clustering requires numpy")
class RoleClusteringTests(unittest.TestCase):
    def test_kmeans_recovers_well_separated_role_groups(self) -> None:
        from courtgraph.features.player_season import PlayerSeasonProfile
        from courtgraph.features.role_clusters import fit_role_clusters

        profiles = []
        # three well-separated archetypes, 8 players each
        archetypes = {
            "guard": dict(usage=0.30, three_rate=0.35, rim_rate=0.20),
            "wing": dict(usage=0.20, three_rate=0.55, rim_rate=0.15),
            "big": dict(usage=0.22, three_rate=0.05, rim_rate=0.55),
        }
        pid = 100
        for _label, base in archetypes.items():
            for _ in range(8):
                profiles.append(
                    PlayerSeasonProfile(
                        player_id=pid,
                        season="2023-24",
                        games=60,
                        off_possessions=3000,
                        def_possessions=3000,
                        fga=800,
                        fgm=380,
                        fg3a=200,
                        fg3m=80,
                        fta=200,
                        ftm=160,
                        assists=200,
                        turnovers=120,
                        off_rebounds=60,
                        def_rebounds=200,
                        steals=40,
                        blocks=20,
                        personal_fouls=120,
                        rim_fga=200,
                        mid_fga=200,
                        corner3_fga=40,
                        usage=base["usage"],
                        assist_per100=6.0,
                        turnover_per100=4.0,
                        oreb_per100=2.0,
                        dreb_per100=6.0,
                        steal_per100=1.3,
                        block_per100=0.7,
                        three_rate=base["three_rate"],
                        rim_rate=base["rim_rate"],
                        corner3_rate=0.05,
                        ft_rate=0.25,
                    )
                )
                pid += 1

        clustering = fit_role_clusters(
            profiles, n_clusters=3, features=("usage", "three_rate", "rim_rate"), seed=0
        )
        # every player of one archetype lands in the same cluster
        for start in (100, 108, 116):
            labels = {clustering.role_of(p) for p in range(start, start + 8)}
            self.assertEqual(len(labels), 1)
        self.assertEqual(len({clustering.role_of(p) for p in range(100, 124)}), 3)


@unittest.skipUnless(HAS_NUMPY, "role eval requires numpy")
class RoleEvalTests(unittest.TestCase):
    def test_evaluate_role_interaction_runs_and_beats_placebo_with_signal(
        self,
    ) -> None:
        import numpy as np

        from courtgraph.chemistry.role_eval import evaluate_role_interaction
        from courtgraph.chemistry.splits import make_all_splits

        table, clustering, _ = _role_dataset(n_stints=7000, tau_role=1.8, seed=5)
        splits = make_all_splits(table)
        comp = evaluate_role_interaction(table, splits, clustering, n_boot=0)
        self.assertEqual(len(comp.holdouts), 3)
        self.assertEqual(np.array(comp.role_pair_matrix).shape, (5, 5))
        # with a strong planted role effect, role beats its permuted placebo on
        # at least the structural holdouts
        wins = sum(
            1 for h in comp.holdouts if h.role_macro_rmse < h.role_placebo_macro_rmse
        )
        self.assertGreaterEqual(wins, 2)

    def test_cli_roles_smoke(self) -> None:
        import json as _json
        from io import StringIO
        from tempfile import TemporaryDirectory

        from courtgraph.chemistry.stints import write_stints
        from courtgraph.cli import main
        from courtgraph.features.player_season import write_player_profiles

        table, clustering, _ = _role_dataset(n_stints=4000, seed=9)
        # synthesize minimal profiles so the CLI's clustering step has input
        from courtgraph.features.player_season import PlayerSeasonProfile

        profiles = []
        for (pid, _season), _lbl in clustering.assignment.items():
            profiles.append(
                PlayerSeasonProfile(
                    player_id=pid,
                    season="2021-22",
                    games=60,
                    off_possessions=3000,
                    def_possessions=3000,
                    fga=800,
                    fgm=380,
                    fg3a=200,
                    fg3m=80,
                    fta=200,
                    ftm=160,
                    assists=200,
                    turnovers=120,
                    off_rebounds=60,
                    def_rebounds=200,
                    steals=40,
                    blocks=20,
                    personal_fouls=120,
                    rim_fga=200,
                    mid_fga=200,
                    corner3_fga=40,
                    usage=0.2 + 0.03 * (pid % 5),
                    assist_per100=5.0,
                    turnover_per100=4.0,
                    oreb_per100=2.0,
                    dreb_per100=6.0,
                    steal_per100=1.3,
                    block_per100=0.7,
                    three_rate=0.1 + 0.08 * (pid % 5),
                    rim_rate=0.5 - 0.05 * (pid % 5),
                    corner3_rate=0.05,
                    ft_rate=0.25,
                )
            )

        with TemporaryDirectory() as tmp:
            sp = Path(tmp) / "stints.jsonl"
            pp = Path(tmp) / "profiles.jsonl"
            write_stints(table, sp)
            write_player_profiles(profiles, pp)
            buf = StringIO()
            code = main(
                [
                    "roles",
                    "--input",
                    str(sp),
                    "--profiles",
                    str(pp),
                    "--clusters",
                    "4",
                    "--bootstrap",
                    "0",
                    "--json",
                ],
                output=buf,
            )
            self.assertEqual(code, 0)
            payload = _json.loads(buf.getvalue())
            self.assertEqual(payload["n_clusters"], 4)
            self.assertEqual(len(payload["holdouts"]), 3)
            self.assertIn("role_macro_rmse", payload["holdouts"][0])


if __name__ == "__main__":
    unittest.main()
