"""End-to-end CLI behaviour for ``demo``, ``fit``, and ``predict``."""

from __future__ import annotations

import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY, cli_synthetic, fast_chemistry  # noqa: E402
from courtgraph.cli import main  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.pipeline import DemoResult


class ArgParsingTests(unittest.TestCase):
    def test_unknown_command_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["frobnicate"])
        self.assertEqual(ctx.exception.code, 2)

    def test_predict_requires_offense_and_defense(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["predict", "--model", "x.json"])
        self.assertEqual(ctx.exception.code, 2)


@unittest.skipUnless(HAS_NUMPY, "demo/fit/predict require numpy")
class ChemistryCommandTests(unittest.TestCase):
    _dir: TemporaryDirectory[str]
    demo: DemoResult

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.pipeline import run_demo

        cls._dir = TemporaryDirectory()
        out = Path(cls._dir.name)
        cls.demo = run_demo(
            out_dir=out / "demo",
            report_path=out / "report.html",
            seed=5,
            n_boot=3,
            synthetic_config=cli_synthetic(),
            chemistry_config=fast_chemistry(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_demo_wrote_stints_model_splits_and_report(self) -> None:
        self.assertTrue(self.demo.stints_path.is_file())
        self.assertTrue(self.demo.model_path.is_file())
        self.assertTrue(self.demo.report_path is not None)
        assert self.demo.report_path is not None
        self.assertTrue(self.demo.report_path.is_file())
        for holdout in self.demo.summary.holdouts:
            self.assertEqual(holdout.leakage_violations, ())

    def test_bootstrap_option_sets_exact_ensemble_size_in_the_saved_artifact(
        self,
    ) -> None:
        from courtgraph.chemistry.artifact import load_model
        from courtgraph.chemistry.pipeline import run_demo

        for n in (0, 2):
            sub = Path(self._dir.name) / f"boot{n}"
            result = run_demo(
                out_dir=sub,
                seed=5,
                n_boot=n,
                synthetic_config=cli_synthetic(),
                chemistry_config=fast_chemistry(),
            )
            model, _meta = load_model(result.model_path)
            self.assertEqual(len(model.interaction_ensemble), n)
            self.assertEqual(len(model.ensemble_references), n)

    def test_predict_json_for_an_unseen_offensive_player_is_additive_only(self) -> None:
        out = StringIO()
        code = main(
            [
                "predict",
                "--model",
                str(self.demo.model_path),
                "--offense",
                "1001,1002,777001,777002,777003",
                "--defense",
                "1006,1007,1008,1009,1010",
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        d = payload["decomposition"]
        self.assertEqual(d["interaction"], 0.0)
        self.assertEqual(d["interaction_lower"], 0.0)
        self.assertEqual(d["interaction_upper"], 0.0)
        self.assertAlmostEqual(d["talent"] + d["context"], d["total"], places=9)
        self.assertEqual(
            payload["interaction_interval"]["method"], "unseen-player-no-estimate"
        )

    def test_report_is_self_contained_and_labeled_synthetic(self) -> None:
        assert self.demo.report_path is not None
        html = self.demo.report_path.read_text()
        self.assertIn("SYNTHETIC DEMONSTRATION DATA", html)
        self.assertIn("talent", html)
        self.assertIn("interaction", html)
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("http://", html)
        self.assertNotIn("cdn", html.lower())

    def test_predict_decomposition_is_additive_and_labeled(self) -> None:
        from courtgraph.chemistry.stints import read_stints

        stint = next(iter(read_stints(self.demo.stints_path)))
        out = StringIO()
        code = main(
            [
                "predict",
                "--model",
                str(self.demo.model_path),
                "--offense",
                ",".join(str(p) for p in stint.offense_player_ids),
                "--defense",
                ",".join(str(p) for p in stint.defense_player_ids),
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        d = payload["decomposition"]
        self.assertAlmostEqual(
            d["talent"] + d["interaction"] + d["context"], d["total"], places=6
        )
        self.assertEqual(payload["model_metadata"]["source"], "synthetic-demo")

    def test_fit_then_predict_round_trip(self) -> None:
        out_model = Path(self._dir.name) / "fitted.json"
        out = StringIO()
        code = main(
            [
                "fit",
                "--input",
                str(self.demo.stints_path),
                "--model-out",
                str(out_model),
                "--rank",
                "3",
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        self.assertTrue(out_model.is_file())
        self.assertGreater(json.loads(out.getvalue())["training_stints"], 100)

        pred = StringIO()
        code = main(
            [
                "predict",
                "--model",
                str(out_model),
                "--offense",
                "1001,1002,1003,1004,1005",
                "--defense",
                "1006,1007,1008,1009,1010",
                "--context",
                "playoff=1",
            ],
            output=pred,
        )
        self.assertEqual(code, 0)
        self.assertIn("total value V", pred.getvalue())

    def test_fit_bootstrap_zero_and_evaluate(self) -> None:
        from courtgraph.chemistry.artifact import load_model

        out_model = Path(self._dir.name) / "fit_boot0.json"
        out = StringIO()
        code = main(
            [
                "fit",
                "--input",
                str(self.demo.stints_path),
                "--model-out",
                str(out_model),
                "--bootstrap",
                "0",
                "--evaluate",
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(len(payload["holdouts"]), 3)
        model, _meta = load_model(out_model)
        self.assertEqual(len(model.interaction_ensemble), 0)

    def test_fit_rejects_negative_bootstrap(self) -> None:
        code = main(
            [
                "fit",
                "--input",
                str(self.demo.stints_path),
                "--model-out",
                str(Path(self._dir.name) / "never.json"),
                "--bootstrap",
                "-1",
            ],
            output=StringIO(),
        )
        self.assertEqual(code, 2)

    def test_predict_rejects_a_bad_lineup(self) -> None:
        from courtgraph.chemistry.pipeline import predict_lineup

        with self.assertRaises(ValueError):
            predict_lineup(self.demo.model_path, [1, 2, 3, 4], [5, 6, 7, 8, 9])

    def test_baselines_compares_rung2_and_rung3_calibration(self) -> None:
        out = StringIO()
        code = main(
            [
                "baselines",
                "--input",
                str(self.demo.stints_path),
                "--bootstrap",
                "10",
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(len(payload["holdouts"]), 3)
        vc = payload["variance_components"]
        self.assertGreater(vc["tau_off"], 0.0)
        self.assertGreater(vc["sigma"], 0.0)
        first = payload["holdouts"][0]
        self.assertIn("coverage_95", first["rung3_calibration"])
        self.assertIn("rung2_macro_rmse", first)

    def test_baselines_rung4_adds_the_pair_interaction_columns(self) -> None:
        out = StringIO()
        code = main(
            [
                "baselines",
                "--input",
                str(self.demo.stints_path),
                "--bootstrap",
                "5",
                "--rung4",
                "--json",
            ],
            output=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        for h in payload["holdouts"]:
            self.assertIn("rung4_macro_rmse", h)
            self.assertIn("coverage_95", h["rung4_calibration"])
            self.assertIn("rung4_n_admitted_pairs", h)
        chron = next(h for h in payload["holdouts"] if h["kind"] == "chronological")
        self.assertIn("rung4_pair_covered", chron)
        self.assertIn("rung4_pair_degraded", chron)
        self.assertIn("rung4_pair_level", chron)
        self.assertIn("n_pair_groups", chron["rung4_pair_level"])

    def test_baselines_rejects_negative_bootstrap(self) -> None:
        code = main(
            [
                "baselines",
                "--input",
                str(self.demo.stints_path),
                "--bootstrap",
                "-1",
            ],
            output=StringIO(),
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
