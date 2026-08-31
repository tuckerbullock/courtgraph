"""The additive baseline and the low-rank chemistry model.

Covers the decomposition identity, permutation invariance, deterministic
serialization round-trips, and additive-talent recovery.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import (  # noqa: E402
    HAS_NUMPY,
    fast_chemistry,
    tiny_synthetic,
)

if TYPE_CHECKING:
    from courtgraph.chemistry.chemistry_model import ChemistryModel
    from courtgraph.chemistry.stints import StintTable
    from courtgraph.chemistry.synthetic import GroundTruth

CONTEXT: dict[str, Any] = {
    "home_offense": True,
    "score_margin_offense": 3,
    "period": 3,
    "playoff": True,
    "days_rest_offense": 2,
    "garbage_time_weight": 1.0,
    "season_index": 1,
}


@unittest.skipUnless(HAS_NUMPY, "model requires numpy")
class ChemistryModelTests(unittest.TestCase):
    table: StintTable
    truth: GroundTruth
    model: ChemistryModel
    players: list[int]
    offense: tuple[int, ...]
    defense: tuple[int, ...]

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.chemistry_model import ChemistryModel
        from courtgraph.chemistry.synthetic import generate

        cls.table, cls.truth = generate(tiny_synthetic())
        cls.model = ChemistryModel.fit(cls.table, fast_chemistry())
        cls.players = list(cls.model.feature_space.player_ids)
        cls.offense = tuple(sorted(cls.players[:5]))
        cls.defense = tuple(sorted(cls.players[5:10]))

    def test_decomposition_is_exactly_additive(self) -> None:
        d = self.model.decompose(self.offense, self.defense, CONTEXT)
        self.assertAlmostEqual(d.talent + d.interaction + d.context, d.total, places=9)

    def test_decomposition_matches_full_prediction(self) -> None:
        from courtgraph.chemistry.stints import Stint, StintTable

        stint = Stint(
            stint_id="q",
            game_id="q",
            game_date="2022-01-15",
            season="S2",
            season_index=1,
            period=3,
            start_time_seconds=0.0,
            offense_team_id=1,
            defense_team_id=2,
            offense_player_ids=self.offense,  # type: ignore[arg-type]
            defense_player_ids=self.defense,  # type: ignore[arg-type]
            offensive_possessions=100,
            points_scored=100,
            home_offense=True,
            score_margin_offense=3,
            playoff=True,
            days_rest_offense=2,
            garbage_time_weight=1.0,
        )
        design = self.model.feature_space.build(StintTable.from_stints([stint]))
        full = float(self.model.predict_total(design)[0])
        d = self.model.decompose(self.offense, self.defense, CONTEXT)
        self.assertAlmostEqual(full, d.total, places=6)

    def test_interaction_is_permutation_invariant_over_offense(self) -> None:
        base = self.model.decompose(self.offense, self.defense, CONTEXT)
        for perm in list(permutations(self.offense))[::17]:
            d = self.model.decompose(tuple(perm), self.defense, CONTEXT)
            self.assertAlmostEqual(d.interaction, base.interaction, places=9)
            self.assertAlmostEqual(d.total, base.total, places=9)

    def test_defense_order_does_not_change_the_prediction(self) -> None:
        base = self.model.decompose(self.offense, self.defense, CONTEXT)
        for perm in list(permutations(self.defense))[::23]:
            d = self.model.decompose(self.offense, tuple(perm), CONTEXT)
            self.assertAlmostEqual(d.total, base.total, places=9)

    def test_low_rank_pathway_sum_pooling_equals_pairwise_sum(self) -> None:
        import numpy as np

        interaction = self.model.interaction
        index = self.model.feature_space.player_index()
        pos = [index[p] for p in self.offense]
        idx = np.array([pos], dtype=np.int64)
        pooled = float(interaction.interaction(idx)[0])
        pairwise = sum(
            interaction.pair_surplus(pos[a], pos[b])
            for a in range(5)
            for b in range(a + 1, 5)
        )
        self.assertAlmostEqual(pooled, pairwise, places=8)

    def test_model_fit_is_deterministic(self) -> None:
        from courtgraph.chemistry.chemistry_model import ChemistryModel

        again = ChemistryModel.fit(self.table, fast_chemistry())
        self.assertEqual(again.to_dict(), self.model.to_dict())

    def test_bootstrap_count_controls_exact_ensemble_size(self) -> None:
        from dataclasses import replace

        from courtgraph.chemistry.chemistry_model import ChemistryModel

        base = fast_chemistry()
        for n in (0, 1, 4):
            model = ChemistryModel.fit(self.table, replace(base, n_bootstrap=n))
            self.assertEqual(len(model.interaction_ensemble), n)
            self.assertEqual(len(model.ensemble_references), n)
            # round-trips exactly
            self.assertEqual(
                len(ChemistryModel.from_dict(model.to_dict()).interaction_ensemble), n
            )
        zero = ChemistryModel.fit(self.table, replace(base, n_bootstrap=0))
        # with no ensemble the group interval collapses to a point
        interval = zero.interaction_interval(self.offense)
        self.assertEqual(interval["lower"], interval["upper"])

    def test_serialization_round_trip_is_exact(self) -> None:
        from courtgraph.chemistry.artifact import load_model, save_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(self.model, path, metadata={"source": "unit-test"})
            reloaded, metadata = load_model(path)
        self.assertEqual(reloaded.to_dict(), self.model.to_dict())
        self.assertEqual(metadata["source"], "unit-test")
        before = self.model.decompose(self.offense, self.defense, CONTEXT)
        after = reloaded.decompose(self.offense, self.defense, CONTEXT)
        self.assertEqual(before.as_dict(), after.as_dict())

    def test_rejects_a_foreign_artifact(self) -> None:
        from courtgraph.chemistry.artifact import load_model

        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"hello": "world"}))
            with self.assertRaises(ValueError):
                load_model(bad)

    def test_additive_ridge_recovers_individual_talent(self) -> None:
        import numpy as np

        truth_index = {p: i for i, p in enumerate(self.truth.player_ids)}
        model_off = np.array(
            [self.model.additive.talent_of(p)[0] for p in self.players]
        )
        true_off = np.array(
            [self.truth.off_talent[truth_index[p]] for p in self.players]
        )
        corr = float(np.corrcoef(model_off, true_off)[0, 1])
        self.assertGreater(corr, 0.45)

    def test_unseen_offensive_player_makes_prediction_additive_only(self) -> None:
        import numpy as np

        from courtgraph.chemistry.stints import Stint, StintTable

        # one known lineup, one lineup with a single unseen player
        known = self.offense
        mixed = (self.offense[0], self.offense[1], 900001, 900002, 900003)
        rows = []
        for off in (known, mixed):
            rows.append(
                Stint(
                    stint_id=f"q-{off[2]}",
                    game_id="q",
                    game_date="2022-01-15",
                    season="S2",
                    season_index=1,
                    period=2,
                    start_time_seconds=0.0,
                    offense_team_id=1,
                    defense_team_id=2,
                    offense_player_ids=tuple(sorted(off)),  # type: ignore[arg-type]
                    defense_player_ids=self.defense,  # type: ignore[arg-type]
                    offensive_possessions=100,
                    points_scored=100,
                    home_offense=True,
                    score_margin_offense=0,
                    playoff=False,
                    days_rest_offense=1,
                    garbage_time_weight=1.0,
                )
            )
        design = self.model.feature_space.build(StintTable.from_stints(rows))
        additive = self.model.predict_additive(design)
        full = self.model.predict_total(design)
        inter = self.model.interaction_component(design)
        samples = self.model.interaction_samples(design.offense_index)
        # row 0 (fully known): interaction is generally nonzero
        # row 1 (has an unseen player): interaction is exactly zero everywhere
        self.assertAlmostEqual(full[1], additive[1], places=12)
        self.assertEqual(inter[1], 0.0)
        self.assertTrue(np.all(samples[:, 1] == 0.0))

        d = self.model.decompose(mixed, self.defense, CONTEXT)
        self.assertEqual(d.offense_novelty, "unseen")
        self.assertEqual(d.interaction, 0.0)
        self.assertAlmostEqual(d.talent + d.context, d.total, places=12)
        interval = self.model.interaction_interval(mixed)
        self.assertEqual(interval["method"], "unseen-player-no-estimate")
        self.assertEqual(interval["point"], 0.0)
        self.assertEqual(interval["lower"], 0.0)
        self.assertEqual(interval["upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
