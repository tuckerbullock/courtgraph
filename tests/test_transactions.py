"""Transaction backtest -- roster changes as natural experiments (contract T4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.stints import StintTable


def _multiseason_dataset(
    *,
    n_seasons: int = 4,
    n_teams: int = 8,
    per_team: int = 9,
    stints_per_season: int = 4500,
    n_movers: int = 16,
    fit_scale: float = 0.0,  # planted roster-fit bump for a mover on their new team
    sigma: float = 8.0,
    seed: int = 4,
) -> tuple[StintTable, dict[int, float]]:
    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable

    rng = np.random.default_rng(seed)
    seasons = [f"20{16 + i}-{17 + i}" for i in range(n_seasons)]
    n_players = n_teams * per_team
    players = list(range(400, 400 + n_players))
    off_talent = dict(zip(players, rng.normal(0, 3.0, n_players).tolist(), strict=True))
    def_talent = dict(zip(players, rng.normal(0, 2.2, n_players).tolist(), strict=True))
    team_ids = list(range(1610612737, 1610612737 + n_teams))

    # base roster: each team gets `per_team` players
    roster: dict[str, dict[int, list[int]]] = {}
    base = {
        t: players[i * per_team : (i + 1) * per_team] for i, t in enumerate(team_ids)
    }
    for s in seasons:
        roster[s] = {t: list(base[t]) for t in team_ids}

    # movers: pick n_movers players, move each from its team to another between
    # two consecutive seasons; plant a roster-fit bump on the new team
    fit: dict[int, float] = {}
    movers = rng.choice(players, n_movers, replace=False)
    for pid in movers:
        pid = int(pid)
        cur_team = next(t for t in team_ids if pid in base[t])
        new_team = int(rng.choice([t for t in team_ids if t != cur_team]))
        cut = int(rng.integers(1, n_seasons))  # first new-team season index
        for si in range(cut, n_seasons):
            roster[seasons[si]][cur_team] = [
                x for x in roster[seasons[si]][cur_team] if x != pid
            ]
            roster[seasons[si]][new_team].append(pid)
        fit[pid] = float(rng.normal(0, fit_scale)) if fit_scale > 0 else 0.0

    def team_of(season: str, pid: int) -> int | None:
        for t in team_ids:
            if pid in roster[season][t]:
                return t
        return None

    stints: list[Any] = []
    sid = 0
    for si, season in enumerate(seasons):
        for _ in range(stints_per_season):
            ot = int(rng.choice(team_ids))
            dt = int(rng.choice([t for t in team_ids if t != ot]))
            opool = roster[season][ot]
            dpool = roster[season][dt]
            if len(opool) < 5 or len(dpool) < 5:
                continue
            off = tuple(sorted(int(x) for x in rng.choice(opool, 5, replace=False)))
            deff = tuple(sorted(int(x) for x in rng.choice(dpool, 5, replace=False)))
            w = int(rng.integers(4, 16))
            fit_bump = sum(
                fit.get(p, 0.0)
                for p in off
                if p in fit and team_of(season, p) == ot and season != seasons[0]
            )
            value = (
                110.0
                + sum(off_talent[p] for p in off)
                - sum(def_talent[p] for p in deff)
                + fit_bump
            )
            y = value + float(rng.normal(0, sigma / np.sqrt(w)))
            stints.append(
                Stint(
                    stint_id=f"s{sid}",
                    game_id=f"g{si}_{sid // 10}",
                    game_date=f"{2016 + si + (1 if False else 0)}-"
                    f"{1 + (sid // 200) % 12:02d}-{1 + sid % 27:02d}",
                    season=season,
                    season_index=si,
                    period=1 + sid % 4,
                    start_time_seconds=float(sid % 600),
                    offense_team_id=ot,
                    defense_team_id=dt,
                    offense_player_ids=off,  # type: ignore[arg-type]
                    defense_player_ids=deff,  # type: ignore[arg-type]
                    offensive_possessions=w,
                    points_scored=int(round(y * w / 100.0)),
                    home_offense=bool(sid % 2),
                    score_margin_offense=0,
                    playoff=False,
                    days_rest_offense=1,
                    garbage_time_weight=1.0,
                )
            )
            sid += 1
    return StintTable.from_stints(stints), fit


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class TransactionCohortTests(unittest.TestCase):
    def test_finds_clean_cross_season_switches(self) -> None:
        from courtgraph.chemistry.transactions import (
            find_transactions,
            leakage_safe_train,
        )

        table, _ = _multiseason_dataset(n_movers=16, seed=2)
        txns = find_transactions(table, min_poss_each_side=300)
        self.assertGreater(len(txns), 5)
        for t in txns:
            self.assertNotEqual(t.from_team_id, t.to_team_id)
            self.assertTrue(t.post_stint_ids)
            # the leakage-safe train has no stint in or after the cutover season
            train = leakage_safe_train(table, t.cutover_season)
            seasons = {s.season for s in train}
            self.assertNotIn(t.cutover_season, seasons)

    def test_phantom_cohort_excludes_real_movers(self) -> None:
        from courtgraph.chemistry.transactions import (
            find_transactions,
            phantom_transactions,
        )

        table, _ = _multiseason_dataset(n_movers=16, seed=2)
        real = find_transactions(table, min_poss_each_side=300)
        phantom = phantom_transactions(table, real, min_poss_each_side=300, seed=1)
        real_keys = {(t.player_id, t.cutover_season) for t in real}
        for ph in phantom:
            self.assertNotIn((ph.player_id, ph.cutover_season), real_keys)
            self.assertEqual(ph.from_team_id, ph.to_team_id)


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class TransactionBacktestTests(unittest.TestCase):
    def test_additive_world_matches_phantom(self) -> None:
        from courtgraph.chemistry.transaction_backtest import run_backtest

        table, _ = _multiseason_dataset(fit_scale=0.0, n_movers=20, seed=6)
        result = run_backtest(
            table, min_poss_each_side=300, n_phantom=40, n_boot=800, seed=0
        )
        self.assertGreater(result.n_transactions, 5)
        g = result.real_vs_phantom_abs
        # no planted roster fit: real |Delta| should not exceed phantom's
        self.assertLess(g["ci_lo"], 0.0)

    def test_planted_roster_fit_is_detected(self) -> None:
        from courtgraph.chemistry.transaction_backtest import run_backtest

        table, fit = _multiseason_dataset(
            fit_scale=6.0, n_movers=24, stints_per_season=6000, seed=8
        )
        result = run_backtest(
            table, min_poss_each_side=300, n_phantom=40, n_boot=1000, seed=0
        )
        g = result.real_vs_phantom_abs
        # movers carry a real roster-specific bump -> real |Delta| > phantom
        self.assertGreater(g["mean"], 0.0)
        self.assertGreater(g["frac_gt_0"], 0.9)
        # and per-transaction Delta tracks the planted fit
        by_pid = {d.player_id: d.delta for d in result.deltas}
        import numpy as np

        common = [p for p in fit if p in by_pid and fit[p] != 0.0]
        if len(common) >= 5:
            a = np.array([fit[p] for p in common])
            b = np.array([by_pid[p] for p in common])
            self.assertGreater(float(np.corrcoef(a, b)[0, 1]), 0.3)

    def test_deterministic(self) -> None:
        from courtgraph.chemistry.transaction_backtest import run_backtest

        table, _ = _multiseason_dataset(n_movers=16, seed=3)
        a = run_backtest(table, min_poss_each_side=300, n_phantom=20, n_boot=200)
        b = run_backtest(table, min_poss_each_side=300, n_phantom=20, n_boot=200)
        self.assertEqual(a.real, b.real)
        self.assertEqual(a.real_vs_phantom_abs, b.real_vs_phantom_abs)


if __name__ == "__main__":
    unittest.main()
