"""Master plan §45 Phase A -- pooled player-lift on lineup value."""

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


def _lift_dataset(
    *,
    n_players: int = 60,
    n_stints: int = 9000,
    tau_off: float = 3.0,
    lift_frac: float = 0.4,
    lift_scale: float = 0.5,
    sigma: float = 3.5,
    seed: int = 5,
) -> tuple[StintTable, dict[int, float]]:
    """Additive talent + a planted **lift**: a stint's non-additive contribution
    is ``sum_i lift_i * (A_off - alpha_i)`` -- high-lift players make lineups
    with strong teammates outperform the additive sum."""

    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable

    rng = np.random.default_rng(seed)
    players = list(range(700, 700 + n_players))
    off_talent = dict(
        zip(players, rng.normal(0, tau_off, n_players).tolist(), strict=True)
    )
    def_talent = dict(zip(players, rng.normal(0, 2.2, n_players).tolist(), strict=True))
    lift = {p: 0.0 for p in players}
    givers = rng.choice(players, int(lift_frac * n_players), replace=False)
    for p in givers:
        lift[int(p)] = float(rng.normal(0, lift_scale))

    rotations = [
        tuple(sorted(int(x) for x in rng.choice(players, 5, replace=False)))
        for _ in range(45)
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
        a_off = sum(off_talent[p] for p in off)
        value = (
            110.0
            + a_off
            - sum(def_talent[p] for p in deff)
            + sum(lift[p] * (a_off - off_talent[p]) for p in off)
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
    return StintTable.from_stints(stints), lift


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class PlayerLiftTests(unittest.TestCase):
    def test_recovers_a_planted_lift_and_beats_rung3(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.hierarchical import HierarchicalRidge
        from courtgraph.chemistry.player_lift import PlayerLift

        table, lift = _lift_dataset(n_stints=14000, lift_scale=0.5, seed=7)
        space = FeatureSpace.from_training(table)
        design = space.build(table)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = PlayerLift.fit(design, space)

        # marginal likelihood finds an interior optimum (not the grid floor)
        self.assertGreater(model.tau_lambda2, 1e-4)
        # the fitted lambda correlates with the planted lift
        pidx = space.player_index()
        planted = np.array([lift[pid] for pid in pidx])
        fitted = np.array([model.lambda_[pidx[pid]] for pid in pidx])
        self.assertGreater(float(np.corrcoef(planted, fitted)[0, 1]), 0.4)

        # in-sample: lift model fits the residual rung 3 leaves
        rung3 = HierarchicalRidge.fit(design, space)
        r3_rmse = float(np.sqrt(np.mean(rung3.residuals(design) ** 2)))
        lift_rmse = float(np.sqrt(np.mean(model.residuals(design) ** 2)))
        self.assertLess(lift_rmse, r3_rmse)

    def test_no_out_of_sample_gain_with_no_planted_lift(self) -> None:
        from courtgraph.chemistry.player_lift_eval import evaluate_player_lift
        from courtgraph.chemistry.splits import make_all_splits

        table, _ = _lift_dataset(n_stints=12000, lift_frac=0.0, lift_scale=0.0, seed=3)
        splits = make_all_splits(table, n_pairs=30, n_lineups=40)
        result = evaluate_player_lift(table, splits, n_boot=0)
        for h in result.holdouts:
            # no planted signal: the lift model must not beat rung 3 out of
            # sample by more than noise, and must not beat its own placebo
            self.assertLess(
                h.rung3_macro_rmse - h.lift_macro_rmse, 0.15 * h.rung3_macro_rmse
            )
            self.assertLess(
                h.lift_placebo_macro_rmse - h.lift_macro_rmse,
                0.15 * h.rung3_macro_rmse,
            )

    def test_deterministic(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.player_lift import PlayerLift

        table, _ = _lift_dataset(n_stints=6000, seed=11)
        space = FeatureSpace.from_training(table)
        design = space.build(table)
        a = PlayerLift.fit(design, space, seed=0)
        b = PlayerLift.fit(design, space, seed=0)
        self.assertTrue(np.array_equal(a.lambda_, b.lambda_))
        self.assertEqual(a.tau_lambda2, b.tau_lambda2)

    def test_placebo_is_a_real_control(self) -> None:
        import numpy as np

        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.player_lift import PlayerLift

        table, _ = _lift_dataset(n_stints=12000, lift_scale=0.09, seed=9)
        space = FeatureSpace.from_training(table)
        design = space.build(table)
        real = PlayerLift.fit(design, space, seed=0)
        placebo = PlayerLift.fit(design, space, seed=0, permuted=True)
        real_rmse = float(np.sqrt(np.mean(real.residuals(design) ** 2)))
        placebo_rmse = float(np.sqrt(np.mean(placebo.residuals(design) ** 2)))
        # the real fit captures the planted structure; the scrambled one cannot
        self.assertLess(real_rmse, placebo_rmse)

    def test_evaluate_player_lift_runs(self) -> None:
        from courtgraph.chemistry.player_lift_eval import evaluate_player_lift
        from courtgraph.chemistry.splits import make_all_splits

        table, _ = _lift_dataset(n_stints=10000, seed=4)
        splits = make_all_splits(table, n_pairs=30, n_lineups=30)
        result = evaluate_player_lift(table, splits, n_boot=0)
        self.assertEqual(len(result.holdouts), 3)
        for h in result.holdouts:
            self.assertIn("z_sd", h.lift_calibration)
        self.assertTrue(result.top_lifts)


if __name__ == "__main__":
    unittest.main()
