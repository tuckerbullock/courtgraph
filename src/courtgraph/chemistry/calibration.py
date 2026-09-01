"""Calibration diagnostics for interval-valued predictions.

The research contract's currency for lineup / pair value calibration
(sections 14-16) is: **interval coverage at 50 / 80 / 95%**, the calibration
line (slope -> 1, intercept -> 0), the standardized-residual moments, and the
relationship between predicted interval width and realized error -- reported
per holdout, including under chronological and unseen-lineup shift.

Pure NumPy. Inputs are per-group arrays: ``point`` (predicted value),
``sd`` (predictive standard deviation, already including outcome noise),
``realized`` (observed value), all same length.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Phi^-1(0.5 + level/2) for the only three levels the contract names.
_HALF_WIDTH_Z: dict[float, float] = {
    0.5: 0.6744897501960817,
    0.8: 1.2815515655446004,
    0.95: 1.959963984540054,
}
COVERAGE_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.95)


def _z(point: FloatArray, sd: FloatArray, realized: FloatArray) -> FloatArray:
    return np.asarray((realized - point) / sd, dtype=np.float64)


def coverage(
    point: FloatArray,
    sd: FloatArray,
    realized: FloatArray,
    levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> dict[float, float]:
    """Fraction of groups whose central ``level`` interval contains ``realized``."""

    z = np.abs(_z(point, sd, realized))
    return {lvl: float(np.mean(z <= _HALF_WIDTH_Z[lvl])) for lvl in levels}


def calibration_line(
    point: FloatArray, sd: FloatArray, realized: FloatArray
) -> dict[str, float]:
    """WLS fit ``realized ~ intercept + slope * point`` with weights ``1/sd**2``.

    A well-calibrated point model has slope ~ 1 and intercept ~ 0.
    """

    weight = 1.0 / sd**2
    design = np.column_stack([np.ones_like(point), point])
    dw = design * weight[:, None]
    beta = np.linalg.solve(dw.T @ design, dw.T @ realized)
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def z_moments(
    point: FloatArray, sd: FloatArray, realized: FloatArray
) -> dict[str, float]:
    """Mean and SD of the standardized residuals; ideal is 0 and 1."""

    z = _z(point, sd, realized)
    ddof = 1 if len(z) > 1 else 0
    return {"z_mean": float(z.mean()), "z_sd": float(z.std(ddof=ddof))}


def width_vs_error(
    point: FloatArray, sd: FloatArray, realized: FloatArray
) -> dict[str, float]:
    """Does a wider predicted interval go with a larger realized error?"""

    abs_err = np.abs(realized - point)
    if np.std(sd) == 0.0 or np.std(abs_err) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(sd, abs_err)[0, 1])
    return {"corr_sd_abserr": corr, "mean_predictive_sd": float(sd.mean())}


def calibration_report(
    point: FloatArray, sd: FloatArray, realized: FloatArray
) -> dict[str, float]:
    """All the diagnostics above flattened into one dict (JSON-friendly keys)."""

    out: dict[str, float] = {}
    for lvl, cov in coverage(point, sd, realized).items():
        out[f"coverage_{int(round(lvl * 100))}"] = cov
    out.update(calibration_line(point, sd, realized))
    out.update(z_moments(point, sd, realized))
    out.update(width_vs_error(point, sd, realized))
    out["n_groups"] = float(len(point))
    return out
