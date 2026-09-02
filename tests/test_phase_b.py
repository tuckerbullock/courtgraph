"""Master plan §45 Phase B -- the per-player-production lift model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.phase_b import PhaseBDesign


def _planted_design(
    *,
    n_players: int = 50,
    n_rows: int = 30000,
    tau_base: float = 8.0,
    lift_scale: float = 0.0,
    sigma: float = 6.0,
    seed: int = 3,
) -> tuple[PhaseBDesign, dict[int, float]]:
    import numpy as np

    from courtgraph.chemistry.phase_b import PhaseBDesign

    rng = np.random.default_rng(seed)
    players = list(range(600, 600 + n_players))
    base = rng.normal(38.0, tau_base, n_players)
    lift = np.zeros(n_players)
    if lift_scale > 0:
        givers = rng.choice(n_players, n_players // 3, replace=False)
        lift[givers] = rng.normal(0, lift_scale, len(givers))

    rec = rng.integers(0, n_players, n_rows)
    team = np.empty((n_rows, 4), dtype=np.int64)
    for i in range(n_rows):
        others = rng.choice(
            [p for p in range(n_players) if p != rec[i]], 4, replace=False
        )
        team[i] = others
    w = rng.integers(3, 14, n_rows).astype(np.float64)
    ctx = np.column_stack(
        [rng.integers(0, 2, n_rows), np.zeros(n_rows), np.ones(n_rows)]
    ).astype(np.float64)
    y = (
        base[rec]
        + lift[team].sum(axis=1)
        + 1.5 * ctx[:, 0]
        + rng.normal(0, sigma / np.sqrt(w))
    )
    design = PhaseBDesign(
        y=y.astype(np.float64),
        w=w,
        receiver=rec.astype(np.int64),
        teammates=team,
        context=ctx,
        player_ids=tuple(players),
        context_names=("home", "playoff", "garbage"),
    )
    return design, {players[i]: float(lift[i]) for i in range(n_players)}


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class PhaseBModelTests(unittest.TestCase):
    def test_recovers_a_planted_production_lift(self) -> None:
        import numpy as np

        from courtgraph.chemistry.phase_b import PhaseBModel

        design, lift = _planted_design(lift_scale=3.0, n_rows=40000, seed=7)
        model = PhaseBModel.fit(design)
        self.assertGreater(model.tau_lift2, 1e-3)
        planted = np.array([lift[p] for p in design.player_ids])
        fitted = model.lift
        self.assertGreater(float(np.corrcoef(planted, fitted)[0, 1]), 0.4)

    def test_no_planted_lift_gives_tiny_tau(self) -> None:
        from courtgraph.chemistry.phase_b import PhaseBModel

        design, _ = _planted_design(lift_scale=0.0, n_rows=30000, seed=2)
        model = PhaseBModel.fit(design)
        # the residual carries no teammate structure -> lift near zero
        self.assertLess(model.variance_components()["lift_abs_max"], 1.5)

    def test_placebo_is_weaker_than_the_real_fit(self) -> None:
        import numpy as np

        from courtgraph.chemistry.phase_b import PhaseBModel

        design, _ = _planted_design(lift_scale=3.0, n_rows=40000, seed=9)
        real = PhaseBModel.fit(design, seed=0)
        placebo = PhaseBModel.fit(design, seed=0, permuted=True)
        real_rmse = float(np.sqrt(np.mean((design.y - real.predict(design)) ** 2)))
        plc_rmse = float(np.sqrt(np.mean((design.y - placebo.predict(design)) ** 2)))
        self.assertLess(real_rmse, plc_rmse)

    def test_deterministic(self) -> None:
        import numpy as np

        from courtgraph.chemistry.phase_b import PhaseBModel

        design, _ = _planted_design(lift_scale=2.0, n_rows=20000, seed=5)
        a = PhaseBModel.fit(design, seed=0)
        b = PhaseBModel.fit(design, seed=0)
        self.assertTrue(np.array_equal(a.lift, b.lift))
        self.assertEqual(a.tau_lift2, b.tau_lift2)


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class PhaseBDesignTests(unittest.TestCase):
    def test_build_design_from_stints_and_production(self) -> None:
        from courtgraph.chemistry.phase_b import build_phase_b_design
        from courtgraph.chemistry.stints import Stint, StintTable
        from courtgraph.features.player_production import PlayerStintProduction

        stints = []
        prod = []
        for i in range(400):
            off = tuple(range(1 + (i % 3), 6 + (i % 3)))
            stints.append(
                Stint(
                    stint_id=f"s{i}",
                    game_id=f"g{i // 5}",
                    game_date=f"2022-01-{1 + i % 27:02d}",
                    season="2021-22",
                    season_index=0,
                    period=1,
                    start_time_seconds=0.0,
                    offense_team_id=1,
                    defense_team_id=2,
                    offense_player_ids=off,  # type: ignore[arg-type]
                    defense_player_ids=(20, 21, 22, 23, 24),
                    offensive_possessions=10,
                    points_scored=11,
                    home_offense=True,
                    score_margin_offense=0,
                    playoff=False,
                    days_rest_offense=1,
                    garbage_time_weight=1.0,
                )
            )
            for pid in off:
                prod.append(
                    PlayerStintProduction(
                        stint_id=f"s{i}",
                        game_id=f"g{i // 5}",
                        season="2021-22",
                        season_index=0,
                        player_id=pid,
                        team_id=1,
                        offensive_possessions=10,
                        fg_points=2 if pid % 2 else 0,
                        ft_points=0,
                        assisted_points=2,
                    )
                )
        design = build_phase_b_design(StintTable.from_stints(stints), prod)
        self.assertGreater(len(design.y), 100)
        self.assertEqual(design.teammates.shape[1], 4)
        # credited production per 100: (fg+ft) + 0.5*assisted, /poss*100
        self.assertTrue((design.y >= 0).all())


if __name__ == "__main__":
    unittest.main()
