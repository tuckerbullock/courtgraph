"""Leakage-safe holdout construction and the verification gate."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY, tiny_synthetic  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.splits import SplitManifest
    from courtgraph.chemistry.stints import StintTable


@unittest.skipUnless(HAS_NUMPY, "split fixtures use the synthetic generator")
class SplitConstructionTests(unittest.TestCase):
    table: StintTable

    def setUp(self) -> None:
        from courtgraph.chemistry.synthetic import generate

        self.table, _ = generate(tiny_synthetic())

    def test_all_three_splits_are_leakage_free(self) -> None:
        from courtgraph.chemistry.splits import make_all_splits, verify_split

        for kind, manifest in make_all_splits(self.table).items():
            self.assertEqual(verify_split(self.table, manifest), [], kind)
            self.assertGreater(manifest.n_train, 0, kind)
            self.assertGreater(manifest.n_test, 0, kind)
            self.assertEqual(manifest.n_train + manifest.n_test, len(self.table), kind)

    def test_chronological_split_cuts_on_a_game_boundary(self) -> None:
        from courtgraph.chemistry.splits import make_chronological_split

        manifest = make_chronological_split(self.table, train_fraction=0.55)
        train_games = {s.game_id for s in manifest.train_table(self.table)}
        test_games = {s.game_id for s in manifest.test_table(self.table)}
        self.assertEqual(train_games & test_games, set())

    def test_chronological_ordering_uses_game_date_not_game_id(self) -> None:
        """game_id sorts opposite to game_date; the split must follow game_date."""

        from courtgraph.chemistry.splits import make_chronological_split, verify_split
        from courtgraph.chemistry.stints import Stint, StintTable

        stints = []
        for g in range(6):  # game_date ascending, game_id lexically descending
            for p in range(4):
                stints.append(
                    Stint(
                        stint_id=f"z{9 - g}-{p}",
                        game_id=f"z{9 - g}",
                        game_date=f"2021-03-{g + 1:02d}",
                        season="S1",
                        season_index=0,
                        period=1 + p // 2,
                        start_time_seconds=float(p * 100),
                        offense_team_id=1,
                        defense_team_id=2,
                        offense_player_ids=(1, 2, 3, 4, 5),
                        defense_player_ids=(6, 7, 8, 9, 10),
                        offensive_possessions=5,
                        points_scored=5,
                        home_offense=True,
                        score_margin_offense=0,
                        playoff=False,
                        days_rest_offense=1,
                        garbage_time_weight=1.0,
                    )
                )
        table = StintTable.from_stints(stints)
        manifest = make_chronological_split(table, train_fraction=0.5)
        self.assertEqual(verify_split(table, manifest), [])
        train_dates = {s.game_date for s in manifest.train_table(table)}
        test_dates = {s.game_date for s in manifest.test_table(table)}
        self.assertLess(max(train_dates), min(test_dates))
        self.assertIn("2021-03-01", train_dates)
        self.assertIn("2021-03-06", test_dates)

    def test_unseen_pair_keeps_both_players_individually_observed(self) -> None:
        from courtgraph.chemistry.splits import make_unseen_pair_split

        manifest = make_unseen_pair_split(self.table, n_pairs=6)
        self.assertGreaterEqual(len(manifest.held_out_pairs), 1)
        train_offense: set[int] = set()
        for stint in manifest.train_table(self.table):
            train_offense.update(stint.offense_player_ids)
        for key in manifest.held_out_pairs:
            a, b = (int(x) for x in key.split("-"))
            self.assertIn(a, train_offense, key)
            self.assertIn(b, train_offense, key)

    def test_unseen_pair_removes_every_shared_training_stint(self) -> None:
        from courtgraph.chemistry.splits import make_unseen_pair_split

        manifest = make_unseen_pair_split(self.table, n_pairs=6)
        self.assertGreaterEqual(len(manifest.held_out_pairs), 1)
        train = list(manifest.train_table(self.table))
        for key in manifest.held_out_pairs:
            a, b = (int(x) for x in key.split("-"))
            leaked = [
                s
                for s in train
                if a in s.offense_player_ids and b in s.offense_player_ids
            ]
            self.assertEqual(leaked, [], key)

    def test_unseen_pair_budget_is_a_hard_upper_bound_including_the_first_pair(
        self,
    ) -> None:
        """An oversized top candidate must be skipped, not admitted for free."""

        from courtgraph.chemistry.splits import make_unseen_pair_split, verify_split
        from courtgraph.chemistry.stints import pair_id

        counts: dict[str, int] = {}
        for stint in self.table.sorted_chronologically():
            ids = stint.offense_player_ids
            for i in range(5):
                for j in range(i + 1, 5):
                    key = pair_id(ids[i], ids[j])
                    counts[key] = counts.get(key, 0) + 1
        top_size = max(counts.values())

        # budget below the single largest pair but above many smaller ones
        frac = (top_size - 1) / len(self.table)
        budget = int(frac * len(self.table))
        self.assertGreater(budget, 0)
        self.assertTrue(
            any(c > budget for c in counts.values()),
            "test needs at least one over-budget candidate",
        )

        manifest = make_unseen_pair_split(self.table, n_pairs=8, max_test_fraction=frac)
        self.assertGreaterEqual(len(manifest.held_out_pairs), 1)
        self.assertLessEqual(manifest.n_test, budget)  # true upper bound
        self.assertLess(manifest.n_test, top_size)  # oversized pair did not slip in
        for key in manifest.held_out_pairs:
            self.assertLessEqual(counts[key], budget, key)
        self.assertEqual(verify_split(self.table, manifest), [])

    def test_unseen_pair_raises_when_no_pair_fits_the_budget(self) -> None:
        from courtgraph.chemistry.splits import (
            UnsatisfiableSplitError,
            make_unseen_pair_split,
        )

        with self.assertRaises(UnsatisfiableSplitError) as ctx:
            make_unseen_pair_split(self.table, max_test_fraction=0.0)
        message = str(ctx.exception)
        self.assertIn("max_test_fraction", message)
        self.assertIn("budget", message)

    def test_unseen_lineup_removes_every_exact_training_stint(self) -> None:
        from courtgraph.chemistry.splits import make_unseen_lineup_split

        manifest = make_unseen_lineup_split(self.table, n_lineups=8)
        train_lineups = {s.offense_lineup_id for s in manifest.train_table(self.table)}
        for lid in manifest.held_out_lineups:
            self.assertNotIn(lid, train_lineups)

    def test_default_splits_hold_many_macro_groups_on_a_full_pool(self) -> None:
        from _chemistry_support import scale_synthetic
        from courtgraph.chemistry.evaluate import _group_index
        from courtgraph.chemistry.splits import make_all_splits, verify_split
        from courtgraph.chemistry.synthetic import generate

        table, _ = generate(scale_synthetic())  # ~330 players, ~19k stints, 2 seasons
        splits = make_all_splits(table)
        # was 2 / 8 / 12 before the widening; a macro comparison needs more.
        # (chronological is limited by the fixture's 2 compressed seasons; on the
        # real 5-season data it buckets into ~13 months.)
        minimum = {"chronological": 4, "unseen_pair": 20, "unseen_lineup": 20}
        for kind, manifest in splits.items():
            self.assertEqual(verify_split(table, manifest), [], kind)
            groups = _group_index(manifest.test_table(table), manifest)
            self.assertGreaterEqual(
                len(groups), minimum[kind], f"{kind}: {len(groups)} groups"
            )

    def test_unseen_pair_caps_any_single_pair(self) -> None:
        from courtgraph.chemistry.splits import make_unseen_pair_split
        from courtgraph.chemistry.stints import pair_id

        manifest = make_unseen_pair_split(self.table, n_pairs=20)
        cap = manifest.parameters["per_pair_cap"]
        by_pair: dict[str, int] = {}
        for stint in manifest.test_table(self.table):
            ids = stint.offense_player_ids
            for a in range(5):
                for b in range(a + 1, 5):
                    key = pair_id(ids[a], ids[b])
                    if key in manifest.held_out_pairs:
                        by_pair[key] = by_pair.get(key, 0) + 1
        for key, count in by_pair.items():
            self.assertLessEqual(count, cap, key)

    def test_manifest_round_trips_through_json(self) -> None:
        from courtgraph.chemistry.splits import SplitManifest, make_unseen_pair_split

        manifest = make_unseen_pair_split(self.table, n_pairs=4)
        with self.subTest("dict"):
            self.assertEqual(manifest, SplitManifest.from_dict(manifest.to_dict()))
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            manifest.write(path)
            import json

            loaded = SplitManifest.from_dict(json.loads(path.read_text()))
            self.assertEqual(manifest, loaded)


@unittest.skipUnless(HAS_NUMPY, "split fixtures use the synthetic generator")
class LeakageGuardTests(unittest.TestCase):
    table: StintTable
    splits: dict[str, SplitManifest]

    def setUp(self) -> None:
        from courtgraph.chemistry.splits import make_all_splits
        from courtgraph.chemistry.synthetic import generate

        self.table, _ = generate(tiny_synthetic())
        self.splits = make_all_splits(self.table)

    def test_catches_a_test_game_leaked_into_training(self) -> None:
        from courtgraph.chemistry.splits import verify_split

        manifest = self.splits["chronological"]
        leaked_id = manifest.test_stint_ids[0]
        broken = replace(
            manifest,
            train_stint_ids=manifest.train_stint_ids + (leaked_id,),
        )
        violations = verify_split(self.table, broken)
        self.assertTrue(violations)
        self.assertTrue(any("both train and test" in v for v in violations))

    def test_catches_a_held_out_pair_co_play_in_training(self) -> None:
        from courtgraph.chemistry.splits import verify_split

        manifest = self.splits["unseen_pair"]
        # move a co-play test stint into training
        leaked_id = manifest.test_stint_ids[0]
        broken = replace(
            manifest,
            train_stint_ids=manifest.train_stint_ids + (leaked_id,),
            test_stint_ids=manifest.test_stint_ids[1:],
        )
        violations = verify_split(self.table, broken)
        self.assertTrue(any("on offense" in v for v in violations))

    def test_catches_a_held_out_player_missing_from_training(self) -> None:
        from courtgraph.chemistry.splits import verify_split

        manifest = self.splits["unseen_pair"]
        by_id = {s.stint_id: s for s in self.table}
        player_a = int(manifest.held_out_pairs[0].split("-")[0])
        moved = tuple(
            sid
            for sid in manifest.train_stint_ids
            if player_a in by_id[sid].offense_player_ids
        )
        moved_set = set(moved)
        broken = replace(
            manifest,
            train_stint_ids=tuple(
                sid for sid in manifest.train_stint_ids if sid not in moved_set
            ),
            test_stint_ids=manifest.test_stint_ids + moved,
        )
        violations = verify_split(self.table, broken)
        self.assertTrue(any("individually observed" in v for v in violations))

    def test_catches_an_exact_held_out_lineup_in_training(self) -> None:
        from courtgraph.chemistry.splits import verify_split

        manifest = self.splits["unseen_lineup"]
        leaked_id = manifest.test_stint_ids[0]
        broken = replace(
            manifest,
            train_stint_ids=manifest.train_stint_ids + (leaked_id,),
            test_stint_ids=manifest.test_stint_ids[1:],
        )
        violations = verify_split(self.table, broken)
        self.assertTrue(
            any("training stint(s) use it exactly" in v for v in violations)
        )

    def test_catches_a_stint_dropped_from_both_sides(self) -> None:
        from courtgraph.chemistry.splits import verify_split

        manifest = self.splits["chronological"]
        broken = replace(manifest, test_stint_ids=manifest.test_stint_ids[:-3])
        violations = verify_split(self.table, broken)
        self.assertTrue(any("neither train nor test" in v for v in violations))


if __name__ == "__main__":
    unittest.main()
