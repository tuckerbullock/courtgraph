"""Deterministic synthetic generator and its known ground truth."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY, tiny_synthetic  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.stints import StintTable
    from courtgraph.chemistry.synthetic import GroundTruth, SyntheticConfig


@unittest.skipUnless(HAS_NUMPY, "synthetic generator requires numpy")
class SyntheticGeneratorTests(unittest.TestCase):
    config: SyntheticConfig
    table: StintTable
    truth: GroundTruth

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.synthetic import generate

        cls.config = tiny_synthetic()
        cls.table, cls.truth = generate(cls.config)

    def test_is_deterministic_for_a_seed(self) -> None:
        from courtgraph.chemistry.synthetic import generate

        again, _ = generate(self.config)
        self.assertEqual(
            [s.to_record() for s in self.table],
            [s.to_record() for s in again],
        )

    def test_a_different_seed_changes_the_data(self) -> None:
        from courtgraph.chemistry.synthetic import generate

        other, _ = generate(replace(self.config, seed=self.config.seed + 1))
        self.assertNotEqual(
            [s.to_record() for s in self.table][:20],
            [s.to_record() for s in other][:20],
        )

    def test_every_stint_is_schema_valid_and_five_on_five(self) -> None:
        for stint in self.table:
            self.assertEqual(len(set(stint.offense_player_ids)), 5)
            self.assertEqual(len(set(stint.defense_player_ids)), 5)
            self.assertFalse(
                set(stint.offense_player_ids) & set(stint.defense_player_ids)
            )
            self.assertGreater(stint.offensive_possessions, 0)

    def test_ground_truth_value_matches_decomposition_parts(self) -> None:
        stint = self.table[0]
        context = stint.context_vector()
        context["season_index"] = float(stint.season_index)
        total = self.truth.lineup_value(
            stint.offense_player_ids, stint.defense_player_ids, context
        )
        parts = (
            self.truth.additive_talent(
                stint.offense_player_ids, stint.defense_player_ids
            )
            + self.truth.lineup_interaction(stint.offense_player_ids)
            + self.truth.context_value(context)
        )
        self.assertAlmostEqual(total, parts, places=9)

    def test_no_interaction_variant_has_zero_true_chemistry(self) -> None:
        from courtgraph.chemistry.synthetic import generate

        _table, truth = generate(self.config.with_no_interaction())
        for stint in list(_table)[:50]:
            self.assertAlmostEqual(
                truth.lineup_interaction(stint.offense_player_ids), 0.0, places=9
            )
            self.assertAlmostEqual(
                truth.pair_surplus(*stint.offense_player_ids[:2]), 0.0, places=9
            )

    def test_lineup_interaction_is_permutation_invariant(self) -> None:
        stint = self.table[3]
        forward = self.truth.lineup_interaction(stint.offense_player_ids)
        reversed_ = self.truth.lineup_interaction(stint.offense_player_ids[::-1])
        self.assertAlmostEqual(forward, reversed_, places=10)

    def test_rejects_league_that_cannot_field_rotations(self) -> None:
        from courtgraph.chemistry.synthetic import generate

        with self.assertRaises(ValueError):
            generate(replace(self.config, n_players=10))


if __name__ == "__main__":
    unittest.main()
