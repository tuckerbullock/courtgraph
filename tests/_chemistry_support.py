"""Shared fixtures for the chemistry test suite (not a test module).

Kept import-light so the NumPy-free test modules can import it: the config
factories lazily import the modeling package only when called.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from courtgraph.chemistry.chemistry_model import ChemistryConfig
    from courtgraph.chemistry.hierarchical import HierarchicalConfig
    from courtgraph.chemistry.synthetic import SyntheticConfig

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def tiny_synthetic() -> SyntheticConfig:
    """A tiny league: fast to generate and fit; enough for schema / split /
    serialization / invariance tests."""

    from courtgraph.chemistry.synthetic import SyntheticConfig

    return SyntheticConfig(
        seed=101,
        n_players=64,
        n_teams=8,
        rotation_size=8,
        n_seasons=3,
        games_per_matchup=1,
        stints_per_game=10,
    )


def recovery_synthetic() -> SyntheticConfig:
    """A medium league with a deliberately strong (still residual) chemistry
    signal, for the recovery tests. ~6k stints; a few seconds to fit."""

    from courtgraph.chemistry.synthetic import SyntheticConfig

    return SyntheticConfig(
        seed=202,
        n_players=90,
        n_teams=9,
        rotation_size=9,
        n_seasons=3,
        games_per_matchup=2,
        stints_per_game=14,
        embedding_rank=2,
        embedding_sd=1.1,
        interaction_scale=2.4,
        per_possession_sd=0.85,
    )


def cli_synthetic() -> SyntheticConfig:
    from courtgraph.chemistry.synthetic import SyntheticConfig

    return SyntheticConfig(
        seed=5,
        n_players=64,
        n_teams=8,
        rotation_size=8,
        n_seasons=3,
        games_per_matchup=1,
        stints_per_game=10,
    )


def fast_chemistry() -> ChemistryConfig:
    from courtgraph.chemistry.chemistry_model import ChemistryConfig

    return ChemistryConfig(
        seed=0,
        rank=3,
        cross_fit_folds=3,
        als_sweeps=14,
        selection_folds=2,
        n_bootstrap=3,
        reference_sample=800,
    )


def scale_synthetic() -> SyntheticConfig:
    """A full-season-shaped player pool (~330 players, ~19k stints) -- large
    enough that a re-introduced (n, n_players) allocation would blow memory /
    time, small enough to fit in ~20-30 s."""

    from courtgraph.chemistry.synthetic import SyntheticConfig

    return SyntheticConfig(
        seed=303,
        n_players=330,
        n_teams=30,
        rotation_size=10,
        n_seasons=2,
        games_per_matchup=2,
        stints_per_game=11,
    )


def scale_chemistry() -> ChemistryConfig:
    from courtgraph.chemistry.chemistry_model import ChemistryConfig

    return ChemistryConfig(
        seed=0,
        rank=3,
        cross_fit_folds=2,
        als_sweeps=6,
        selection_folds=2,
        n_bootstrap=0,
        reference_sample=500,
    )


def wellspec_synthetic() -> SyntheticConfig:
    """``recovery_synthetic`` with the chemistry signal switched off -- a
    well-specified additive world, for exact variance-component and
    interval-coverage checks of the hierarchical (rung-3) model."""

    return recovery_synthetic().with_no_interaction()


def hierarchical_config() -> HierarchicalConfig:
    """Faster EM settings for tests (looser tol, capped iters)."""

    from courtgraph.chemistry.hierarchical import HierarchicalConfig

    return HierarchicalConfig(tol=1e-6, max_iters=100)
