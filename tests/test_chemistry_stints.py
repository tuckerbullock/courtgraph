"""Versioned stint format: validation, canonicalisation, and round-trip IO."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from courtgraph.chemistry.stints import (
    SCHEMA_VERSION,
    Stint,
    StintSchemaError,
    StintTable,
    lineup_id,
    pair_id,
    read_stints,
    write_stints,
)


def make_stint(**overrides: object) -> Stint:
    base = dict(
        stint_id="g1-p0",
        game_id="g1",
        game_date="2021-11-03",
        season="S1",
        season_index=0,
        period=1,
        start_time_seconds=0.0,
        offense_team_id=1,
        defense_team_id=2,
        offense_player_ids=(5, 4, 3, 2, 1),
        defense_player_ids=(10, 9, 8, 7, 6),
        offensive_possessions=8,
        points_scored=9,
        home_offense=True,
        score_margin_offense=-2,
        playoff=False,
        days_rest_offense=1,
        garbage_time_weight=1.0,
    )
    base.update(overrides)
    return Stint(**base)  # type: ignore[arg-type]


class StintValidationTests(unittest.TestCase):
    def test_lineups_are_canonicalised_to_sorted_sets(self) -> None:
        stint = make_stint(offense_player_ids=(5, 4, 3, 2, 1))
        self.assertEqual(stint.offense_player_ids, (1, 2, 3, 4, 5))
        self.assertEqual(stint.offense_lineup_id, "1-2-3-4-5")

    def test_offensive_rating_is_points_per_100(self) -> None:
        stint = make_stint(points_scored=12, offensive_possessions=10)
        self.assertAlmostEqual(stint.offensive_rating, 120.0)

    def test_rejects_overlapping_offense_and_defense(self) -> None:
        with self.assertRaises(StintSchemaError):
            make_stint(defense_player_ids=(1, 6, 7, 8, 9))

    def test_rejects_wrong_lineup_size(self) -> None:
        with self.assertRaises(StintSchemaError):
            make_stint(offense_player_ids=(1, 2, 3, 4))

    def test_rejects_nonpositive_possessions_and_negative_points(self) -> None:
        with self.assertRaises(StintSchemaError):
            make_stint(offensive_possessions=0)
        with self.assertRaises(StintSchemaError):
            make_stint(points_scored=-1)

    def test_rejects_garbage_weight_out_of_range(self) -> None:
        with self.assertRaises(StintSchemaError):
            make_stint(garbage_time_weight=0.0)
        with self.assertRaises(StintSchemaError):
            make_stint(garbage_time_weight=1.5)

    def test_unknown_field_is_rejected_on_load(self) -> None:
        record = make_stint().to_record()
        record["mystery"] = 1
        with self.assertRaises(StintSchemaError):
            Stint.from_record(record)

    def test_schema_version_is_2_and_requires_game_date(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 2)
        record = make_stint().to_record()
        self.assertEqual(record["game_date"], "2021-11-03")

    def test_future_schema_version_is_rejected(self) -> None:
        record = make_stint().to_record()
        record["schema_version"] = SCHEMA_VERSION + 1
        with self.assertRaises(StintSchemaError):
            Stint.from_record(record)

    def test_schema_version_1_is_rejected_on_load(self) -> None:
        record = make_stint().to_record()
        record["schema_version"] = 1
        with self.assertRaises(StintSchemaError):
            Stint.from_record(record)

    def test_rejects_a_non_iso_game_date(self) -> None:
        with self.assertRaises(StintSchemaError):
            make_stint(game_date="11/03/2021")
        with self.assertRaises(StintSchemaError):
            make_stint(game_date="not-a-date")


class IdHelperTests(unittest.TestCase):
    def test_lineup_and_pair_ids_are_order_independent(self) -> None:
        self.assertEqual(lineup_id([3, 1, 2]), lineup_id([2, 3, 1]))
        self.assertEqual(pair_id(9, 4), pair_id(4, 9))
        self.assertEqual(pair_id(4, 9), "4-9")


class StintTableTests(unittest.TestCase):
    def _table(self) -> StintTable:
        stints = [
            make_stint(stint_id="g1-p0", game_id="g1", season="S1", season_index=0),
            make_stint(
                stint_id="g2-p0",
                game_id="g2",
                season="S2",
                season_index=1,
                offense_player_ids=(2, 3, 4, 5, 11),
            ),
        ]
        return StintTable.from_stints(stints)

    def test_duplicate_stint_ids_are_rejected(self) -> None:
        with self.assertRaises(StintSchemaError):
            StintTable.from_stints([make_stint(), make_stint()])

    def test_player_ids_and_season_order(self) -> None:
        table = self._table()
        self.assertEqual(table.season_order(), ("S1", "S2"))
        self.assertIn(6, table.player_ids())
        self.assertEqual(table.total_possessions(), 16)

    def test_subset_preserves_order(self) -> None:
        table = self._table()
        subset = table.subset({"g2-p0"})
        self.assertEqual([s.stint_id for s in subset], ["g2-p0"])

    def test_jsonl_and_json_round_trip_is_exact(self) -> None:
        table = self._table()
        with TemporaryDirectory() as directory:
            for name in ("stints.jsonl", "stints.json"):
                path = Path(directory) / name
                write_stints(table, path)
                reloaded = read_stints(path)
                self.assertEqual(
                    [s.to_record() for s in table],
                    [s.to_record() for s in reloaded],
                )

    def test_jsonl_rows_carry_the_schema_version(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "s.jsonl"
            write_stints(self._table(), path)
            first = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(first["schema_version"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
