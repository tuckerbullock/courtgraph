"""Rung-3 model artifact round-trip and the real-lineup predictor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import (  # noqa: E402
    HAS_NUMPY,
    fast_chemistry,
    hierarchical_config,
    wellspec_synthetic,
)

if TYPE_CHECKING:
    from courtgraph.chemistry.hierarchical import HierarchicalRidge
    from courtgraph.chemistry.stints import StintTable


@unittest.skipUnless(HAS_NUMPY, "rung-3 artifact requires numpy")
class Rung3ArtifactTests(unittest.TestCase):
    table: StintTable
    model: HierarchicalRidge
    _dir: TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.hierarchical import HierarchicalRidge
        from courtgraph.chemistry.synthetic import generate

        cls._dir = TemporaryDirectory()
        cls.table, _truth = generate(wellspec_synthetic())
        space = FeatureSpace.from_training(cls.table)
        design = space.build(cls.table)
        cls.model = HierarchicalRidge.fit(design, space, config=hierarchical_config())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_round_trip_save_load(self) -> None:
        from courtgraph.chemistry import rung3_artifact

        path = Path(self._dir.name) / "rung3.json"
        rung3_artifact.save_model(self.model, path, metadata={"source": "test"})
        loaded, meta = rung3_artifact.load_model(path)
        self.assertEqual(meta["source"], "test")
        self.assertEqual(loaded.to_dict(), self.model.to_dict())

    def test_load_rejects_a_chemistry_model_artifact(self) -> None:
        from courtgraph.chemistry import rung3_artifact
        from courtgraph.chemistry.artifact import save_model as save_chemistry_model
        from courtgraph.chemistry.chemistry_model import ChemistryModel

        chem = ChemistryModel.fit(self.table, fast_chemistry())
        path = Path(self._dir.name) / "chem.json"
        save_chemistry_model(chem, path)
        with self.assertRaises(ValueError):
            rung3_artifact.load_model(path)

    def test_predict_lineup_rung3_matches_group_predictive(self) -> None:
        import numpy as np

        from courtgraph.chemistry import rung3_artifact
        from courtgraph.chemistry.chemistry_model import _reference_stint
        from courtgraph.chemistry.pipeline import predict_lineup_rung3
        from courtgraph.chemistry.stints import StintTable

        path = Path(self._dir.name) / "rung3_predict.json"
        possessions: dict[int, int] = {}
        for stint in self.table:
            for pid in stint.offense_player_ids:
                possessions[pid] = possessions.get(pid, 0) + stint.offensive_possessions
        poss_str = {str(k): v for k, v in possessions.items()}
        rung3_artifact.save_model(
            self.model, path, metadata={"training_player_possessions": poss_str}
        )

        offense = (1001, 1002, 1003, 1004, 1005)
        defense = (1006, 1007, 1008, 1009, 1010)
        result = predict_lineup_rung3(path, list(offense), list(defense))

        stint = _reference_stint(offense, defense, result.context)
        design = self.model.feature_space.build(StintTable.from_stints([stint]))
        point, sd, _w = self.model.group_predictive(
            design, {"lineup": np.array([0], dtype=np.int64)}
        )["lineup"]
        self.assertAlmostEqual(result.total, point, places=8)
        self.assertAlmostEqual(result.predictive_sd, sd, places=8)
        self.assertAlmostEqual(result.talent + result.context_value, point, places=8)

    def test_unseen_player_is_flagged_not_guessed(self) -> None:
        from courtgraph.chemistry import rung3_artifact
        from courtgraph.chemistry.pipeline import predict_lineup_rung3

        path = Path(self._dir.name) / "rung3_unseen.json"
        rung3_artifact.save_model(self.model, path)
        result = predict_lineup_rung3(
            path,
            [1001, 1002, 1003, 1004, 999999],
            [1006, 1007, 1008, 1009, 1010],
        )
        self.assertEqual(result.support["unseen_offense_players"], [999999])

    def test_rejects_a_bad_lineup(self) -> None:
        from courtgraph.chemistry import rung3_artifact
        from courtgraph.chemistry.pipeline import predict_lineup_rung3

        path = Path(self._dir.name) / "rung3_bad.json"
        rung3_artifact.save_model(self.model, path)
        off4 = [1001, 1002, 1003, 1004]
        def5 = [1006, 1007, 1008, 1009, 1010]
        with self.assertRaises(ValueError):
            predict_lineup_rung3(path, off4, def5)
        with self.assertRaises(ValueError):
            predict_lineup_rung3(
                path,
                [1001, 1002, 1003, 1004, 1004],
                [1006, 1007, 1008, 1009, 1010],
            )

    def test_result_has_no_interaction_field(self) -> None:
        """Structural guarantee: rung 3 has no chemistry/interaction term, so
        the prediction result type must not expose one, even accidentally."""
        from courtgraph.chemistry import rung3_artifact
        from courtgraph.chemistry.pipeline import predict_lineup_rung3

        path = Path(self._dir.name) / "rung3_struct.json"
        rung3_artifact.save_model(self.model, path)
        result = predict_lineup_rung3(
            path, [1001, 1002, 1003, 1004, 1005], [1006, 1007, 1008, 1009, 1010]
        )
        payload = result.as_dict()
        self.assertNotIn("interaction", payload)
        self.assertNotIn("interaction_interval", payload)
        self.assertIn("note", payload)


if __name__ == "__main__":
    unittest.main()
