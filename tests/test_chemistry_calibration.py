"""Calibration diagnostics -- coverage, calibration line, width vs error."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402


@unittest.skipUnless(HAS_NUMPY, "calibration needs numpy")
class CalibrationTests(unittest.TestCase):
    def test_well_specified_intervals_cover_at_nominal_rates(self) -> None:
        import numpy as np

        from courtgraph.chemistry.calibration import (
            calibration_line,
            coverage,
            z_moments,
        )

        rng = np.random.default_rng(0)
        n = 20_000
        point = rng.normal(0.0, 5.0, size=n)
        sd = rng.uniform(0.5, 3.0, size=n)
        realized = point + sd * rng.normal(size=n)

        cov = coverage(point, sd, realized)
        self.assertAlmostEqual(cov[0.5], 0.50, delta=0.02)
        self.assertAlmostEqual(cov[0.8], 0.80, delta=0.02)
        self.assertAlmostEqual(cov[0.95], 0.95, delta=0.02)

        line = calibration_line(point, sd, realized)
        self.assertAlmostEqual(line["slope"], 1.0, delta=0.03)
        self.assertAlmostEqual(line["intercept"], 0.0, delta=0.15)

        zm = z_moments(point, sd, realized)
        self.assertAlmostEqual(zm["z_sd"], 1.0, delta=0.03)
        self.assertAlmostEqual(zm["z_mean"], 0.0, delta=0.03)

    def test_slope_detects_a_scaled_signal(self) -> None:
        import numpy as np

        from courtgraph.chemistry.calibration import calibration_line

        rng = np.random.default_rng(1)
        n = 20_000
        point = rng.normal(0.0, 4.0, size=n)
        sd = np.full(n, 1.0)
        realized = 2.0 * point + sd * rng.normal(size=n)
        self.assertAlmostEqual(
            calibration_line(point, sd, realized)["slope"], 2.0, delta=0.05
        )

    def test_underdispersed_intervals_undercover(self) -> None:
        import numpy as np

        from courtgraph.chemistry.calibration import coverage, z_moments

        rng = np.random.default_rng(2)
        n = 20_000
        point = rng.normal(0.0, 5.0, size=n)
        sd = np.full(n, 2.0)
        realized = point + 3.0 * sd * rng.normal(size=n)  # true noise 3x claimed

        self.assertLess(coverage(point, sd, realized)[0.95], 0.75)
        self.assertAlmostEqual(z_moments(point, sd, realized)["z_sd"], 3.0, delta=0.1)

    def test_width_tracks_error_when_noise_scales_with_1_over_sqrt_m(self) -> None:
        import numpy as np

        from courtgraph.chemistry.calibration import width_vs_error

        rng = np.random.default_rng(3)
        n = 5_000
        m = rng.integers(10, 400, size=n).astype(float)
        point = rng.normal(0.0, 3.0, size=n)
        sd = 1.0 / np.sqrt(m)
        realized = point + sd * rng.normal(size=n)
        self.assertGreater(width_vs_error(point, sd, realized)["corr_sd_abserr"], 0.2)


if __name__ == "__main__":
    unittest.main()
