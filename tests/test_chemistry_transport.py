"""Phase-transport evaluation: train on one table, test on a disjoint second one."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.stints import StintTable
    from courtgraph.chemistry.transport import TransportResult


def _phase_dataset(
    *,
    n_players: int = 44,
    n_train: int = 10000,
    n_test: int = 2600,
    tau_pair: float = 0.0,
    sigma: float = 9.0,
    seed: int = 11,
) -> tuple[StintTable, StintTable, dict[str, float]]:
    """A regular-season table and a disjoint 'playoff' table over the **same**
    player pool, with a shared additive-plus-per-pair generative model.

    When ``tau_pair > 0`` the planted ``gamma_ij`` is identical in both phases --
    a *transferable* pair effect a rung-4 model fit on the regular season should
    be able to carry into the playoffs. Game ids are phase-prefixed so the two
    tables never share a game. Returns ``(train, test, {pair_key: gamma})``.
    """

    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable, pair_id

    rng = np.random.default_rng(seed)
    players = list(range(2000, 2000 + n_players))
    off_talent = dict(zip(players, rng.normal(0, 3.0, n_players).tolist(), strict=True))
    def_talent = dict(zip(players, rng.normal(0, 2.5, n_players).tolist(), strict=True))
    gsize = n_players // 4
    groups = [players[i * gsize : (i + 1) * gsize] for i in range(4)]
    # a fixed pool of recurring rotations per group -- gives lineup recurrence
    # (macro groups) while keeping teammate pairs varied enough that the pair
    # effect is not collinear with "which player sat".
    n_rot = 40
    rotations = [
        [
            tuple(sorted(int(x) for x in rng.choice(g, 5, replace=False)))
            for _ in range(n_rot)
        ]
        for g in groups
    ]

    all_pairs = [
        pair_id(players[i], players[j])
        for i in range(n_players)
        for j in range(i + 1, n_players)
    ]
    gamma: dict[str, float] = (
        {k: float(rng.normal(0, tau_pair)) for k in all_pairs} if tau_pair > 0 else {}
    )

    def build(n: int, tag: str, year: int) -> StintTable:
        rows: list[Any] = []
        for i in range(n):
            gi, gj, gk = (int(x) for x in rng.choice(4, 3, replace=False))
            draw = rng.random()
            if draw < 0.1:
                pool = groups[gi] + groups[gk]
                off = tuple(sorted(int(x) for x in rng.choice(pool, 5, replace=False)))
            elif draw < 0.25:
                off = tuple(
                    sorted(int(x) for x in rng.choice(groups[gi], 5, replace=False))
                )
            else:
                off = rotations[gi][int(rng.integers(n_rot))]
            deff = tuple(
                sorted(int(x) for x in rng.choice(groups[gj], 5, replace=False))
            )
            w = int(rng.integers(4, 16))
            value = (
                110.0
                + sum(off_talent[p] for p in off)
                - sum(def_talent[p] for p in deff)
                + sum(
                    gamma.get(pair_id(off[a], off[b]), 0.0)
                    for a in range(5)
                    for b in range(a + 1, 5)
                )
            )
            y = value + float(rng.normal(0, sigma / np.sqrt(w)))
            rows.append(
                Stint(
                    stint_id=f"{tag}-s{i}",
                    game_id=f"{tag}-g{i // 8}",
                    game_date=f"{year}-{1 + (i // 200) % 12:02d}-{1 + i % 27:02d}",
                    season="2022-23",
                    season_index=0,
                    period=1 + i % 4,
                    start_time_seconds=float(i % 600),
                    offense_team_id=1 + gi,
                    defense_team_id=1 + gj,
                    offense_player_ids=off,  # type: ignore[arg-type]
                    defense_player_ids=deff,  # type: ignore[arg-type]
                    offensive_possessions=w,
                    points_scored=int(round(y * w / 100.0)),
                    home_offense=bool(i % 2),
                    score_margin_offense=int(rng.integers(-12, 13)),
                    playoff=(tag == "po"),
                    days_rest_offense=1 + i % 3,
                    garbage_time_weight=1.0,
                )
            )
        return StintTable.from_stints(rows)

    return build(n_train, "rs", 2022), build(n_test, "po", 2023), gamma


@unittest.skipUnless(HAS_NUMPY, "transport evaluation requires numpy")
class TransportLeakageTests(unittest.TestCase):
    def test_clean_two_table_split_has_no_violations_and_full_coverage(self) -> None:
        from courtgraph.chemistry.transport import evaluate_transport

        train, test, _ = _phase_dataset(tau_pair=0.0, n_train=4000, n_test=800)
        result = evaluate_transport(train, test, n_boot=0)
        self.assertEqual(result.leakage_violations, ())
        self.assertEqual(result.coverage["test_players_unseen_in_train"], 0.0)
        self.assertGreater(result.coverage["test_offensive_pairs"], 0.0)
        self.assertEqual(
            result.coverage["test_stints_seen_lineup"]
            + result.coverage["test_stints_partially_seen_lineup"]
            + result.coverage["test_stints_unseen_lineup"],
            result.coverage["n_test_stints"],
        )

    def test_overlapping_games_are_reported_as_leakage(self) -> None:
        from courtgraph.chemistry.transport import evaluate_transport

        train, _test, _ = _phase_dataset(n_train=4000, n_test=400)
        result = evaluate_transport(train, train, n_boot=0)
        self.assertTrue(any("game" in v for v in result.leakage_violations))


@unittest.skipUnless(HAS_NUMPY, "transport evaluation requires numpy")
class TransportPairSignalTests(unittest.TestCase):
    """The pair-level playoff test must find a *transferable* pair effect when
    one is planted, and must not manufacture one from noise when none is."""

    signal: TransportResult
    null: TransportResult

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.pair_interaction import PairHierarchicalConfig
        from courtgraph.chemistry.transport import evaluate_transport

        cfg = PairHierarchicalConfig(tol=1e-6, max_iters=40, min_co_stints=40)
        tr_s, te_s, _ = _phase_dataset(tau_pair=2.6, seed=7)
        tr_n, te_n, _ = _phase_dataset(tau_pair=0.0, seed=7)
        cls.signal = evaluate_transport(tr_s, te_s, n_boot=0, rung4_config=cfg)
        cls.null = evaluate_transport(tr_n, te_n, n_boot=0, rung4_config=cfg)

    def test_transferable_pair_signal_beats_the_placebo(self) -> None:
        pl = self.signal.rung4_pair_level
        assert pl is not None
        self.assertGreater(pl["n_pair_groups"], 50)
        # a real, transferable pair effect: rung 4 clearly below its placebo
        self.assertLess(pl["rung4_macro_rmse"], 0.95 * pl["rung4_placebo_macro_rmse"])
        # ... and below the additive model on the same pair groups
        self.assertLess(pl["rung4_macro_rmse"], pl["rung2_macro_rmse"])

    def test_no_signal_means_rung4_matches_its_placebo(self) -> None:
        pl = self.null.rung4_pair_level
        assert pl is not None
        ratio = pl["rung4_macro_rmse"] / pl["rung4_placebo_macro_rmse"]
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)

    def test_reports_novelty_and_clutch_breakdowns(self) -> None:
        by_nov = self.null.by_novelty
        self.assertIn("seen", by_nov)
        self.assertIn("partially-seen", by_nov)
        covered = sum(int(v.get("n_groups", 0)) for v in by_nov.values())
        self.assertEqual(covered, self.null.n_lineup_groups)
        micro = self.null.micro_rmse
        self.assertEqual(
            int(micro["clutch"]["n_stints"] + micro["non_clutch"]["n_stints"]),
            int(micro["all"]["n_stints"]),
        )
        self.assertIn("rung2_micro_rmse", micro["clutch"])

    def test_playoff_indicator_is_zeroed_so_intervals_stay_finite(self) -> None:
        # 'playoff' never varies in the (regular-season) train table, so it is
        # zeroed in the test design -- otherwise every playoff row would inherit
        # the ~tau_c2 context prior and the rung-3 predictive SD would explode.
        self.assertIn("playoff", self.null.zeroed_context_columns)
        self.assertLess(self.null.rung3_calibration["mean_predictive_sd"], 100.0)
        self.assertGreater(self.null.rung3_calibration["z_sd"], 0.3)

    def test_as_dict_round_trips_the_shape(self) -> None:
        d = self.signal.as_dict()
        self.assertEqual(d["leakage_violations"], [])
        self.assertIn("playoff", d["zeroed_context_columns"])
        self.assertIn("rung4_pair_level", d)
        self.assertIn("by_novelty", d)
        self.assertIn("n_test_stints", d["coverage"])


if __name__ == "__main__":
    unittest.main()
