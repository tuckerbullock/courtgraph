"""Derived player features (role / skill profiles) for the interaction models.

Reads a validated ``stats_nba_pbpstats/v1`` snapshot plus a stint file and
emits one profile per (player, season): counting stats from the raw
play-by-play and shot-chart payloads, on-court possession exposure from the
stints, and the per-possession rates the role-conditioned interaction model
consumes. No network, no new downloads -- the same local snapshot the ingest
already produced.
"""

from courtgraph.features.player_season import (
    PLAYER_PROFILE_SCHEMA_VERSION,
    PlayerSeasonProfile,
    build_from_paths,
    build_player_profiles,
    read_player_profiles,
    write_player_profiles,
)

__all__ = [
    "PLAYER_PROFILE_SCHEMA_VERSION",
    "PlayerSeasonProfile",
    "build_from_paths",
    "build_player_profiles",
    "read_player_profiles",
    "write_player_profiles",
]
