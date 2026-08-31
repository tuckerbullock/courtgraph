"""Deterministic synthetic stint generator with a known ground truth.

The generator is the backbone of the model-recovery suite (master plan 33.3):
it produces stints whose lineup value has a *known* decomposition

    V(L_o, L_d, c) = alpha
                   + sum_{i in L_o} off_talent[i]
                   - sum_{j in L_d} def_talent[j]        (additive talent, T)
                   + interaction_scale * C_true(L_o)     (interaction surplus, C)
                   + K_true(c)                           (context, K)

where the interaction surplus is a genuine low-rank provision/need structure

    C_true(L_o) = sum_{i<j in L_o} ( p_i . n_j + p_j . n_i ),

which a permutation-invariant low-rank embedding model can recover but an
additive ridge baseline cannot. Per-possession points are drawn around V/100
with realistic possession-level variance, so the recoverable signal is small
relative to noise -- exactly the regime the research question lives in.

Everything flows from a single ``numpy.random.Generator`` seeded once, so a
given :class:`SyntheticConfig` always produces byte-identical output.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.stints import LINEUP_SIZE, Stint, StintTable

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticConfig:
    """Every knob of the generator. Defaults give a small, fast demo dataset."""

    seed: int = 20260830
    n_players: int = 120
    n_teams: int = 12
    n_seasons: int = 3
    games_per_matchup: int = 2
    stints_per_game: int = 22
    rotation_size: int = 9
    roster_change_prob: float = 0.18
    playoff_fraction: float = 0.06
    embedding_rank: int = 2
    talent_sd: float = 6.0
    defensive_talent_sd: float = 5.0
    embedding_sd: float = 1.0
    interaction_scale: float = 1.4
    base_offensive_rating: float = 110.0
    per_possession_sd: float = 0.85
    min_possessions: int = 4
    mean_extra_possessions: float = 6.0

    def with_no_interaction(self) -> SyntheticConfig:
        """A matched config with the chemistry signal switched off."""

        return replace(self, interaction_scale=0.0)


@dataclass(frozen=True)
class GroundTruth:
    """The latent parameters the generator used, for recovery evaluation."""

    config: SyntheticConfig
    player_ids: tuple[int, ...]
    alpha: float
    off_talent: Vector  # indexed by player position in ``player_ids``
    def_talent: Vector
    provision: Vector  # (n_players, rank)
    need: Vector  # (n_players, rank)
    context_weights: dict[str, float]
    interaction_scale: float

    def _index(self, player_id: int) -> int:
        return self._index_map[player_id]

    @property
    def _index_map(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.player_ids)}

    def pair_surplus(self, player_a: int, player_b: int) -> float:
        """True same-team offensive interaction surplus for one pair."""

        ia, ib = self._index(player_a), self._index(player_b)
        pair = float(
            self.provision[ia] @ self.need[ib] + self.provision[ib] @ self.need[ia]
        )
        return self.interaction_scale * pair

    def lineup_interaction(self, offense_ids: tuple[int, ...]) -> float:
        """True five-player chemistry surplus C(L_o), before centering."""

        idx = [self._index(pid) for pid in offense_ids]
        p_sum = self.provision[idx].sum(axis=0)
        n_sum = self.need[idx].sum(axis=0)
        diag = float((self.provision[idx] * self.need[idx]).sum())
        return self.interaction_scale * (float(p_sum @ n_sum) - diag)

    def additive_talent(
        self, offense_ids: tuple[int, ...], defense_ids: tuple[int, ...]
    ) -> float:
        off = float(self.off_talent[[self._index(p) for p in offense_ids]].sum())
        deff = float(self.def_talent[[self._index(p) for p in defense_ids]].sum())
        return self.alpha + off - deff

    def context_value(self, context: dict[str, float]) -> float:
        w = self.context_weights
        return (
            w["home_offense"] * context["home_offense"]
            + w["score_margin_offense"] * context["score_margin_offense"]
            + w["period"] * (context["period"] - 2.5)
            + w["playoff"] * context["playoff"]
            + w["days_rest_offense"] * context["days_rest_offense"]
            + w["garbage_time_weight"] * (context["garbage_time_weight"] - 1.0)
            + w["season_pace_drift"] * context["season_index"]
        )

    def lineup_value(
        self,
        offense_ids: tuple[int, ...],
        defense_ids: tuple[int, ...],
        context: dict[str, float],
    ) -> float:
        return (
            self.additive_talent(offense_ids, defense_ids)
            + self.lineup_interaction(offense_ids)
            + self.context_value(context)
        )

    def decomposition(
        self,
        offense_ids: tuple[int, ...],
        defense_ids: tuple[int, ...],
        context: dict[str, float],
        interaction_reference: float,
    ) -> dict[str, float]:
        """(talent, interaction, context, total), interaction centered by a ref."""

        talent = self.additive_talent(offense_ids, defense_ids)
        interaction = self.lineup_interaction(offense_ids) - interaction_reference
        ctx = self.context_value(context)
        return {
            "talent": talent + interaction_reference,
            "interaction": interaction,
            "context": ctx,
            "total": talent + self.lineup_interaction(offense_ids) + ctx,
        }

    def mean_lineup_interaction(self, sample: int = 4000, seed: int = 0) -> float:
        """Reference interaction level: mean C_true over random offensive fives."""

        rng = np.random.default_rng(seed)
        n = len(self.player_ids)
        total = 0.0
        for _ in range(sample):
            idx = rng.choice(n, size=5, replace=False)
            ids = tuple(int(self.player_ids[k]) for k in idx)
            total += self.lineup_interaction(ids)
        return total / sample

    def as_dict(self) -> dict[str, Any]:
        """Compact JSON-friendly summary for reports (not the full vectors)."""

        return {
            "alpha": self.alpha,
            "interaction_scale": self.interaction_scale,
            "embedding_rank": self.config.embedding_rank,
            "context_weights": dict(self.context_weights),
            "off_talent_sd": float(np.std(self.off_talent)),
            "def_talent_sd": float(np.std(self.def_talent)),
        }


@dataclass
class _Season:
    """One season's team assignment, rotations, and minute weights."""

    team_of: dict[int, int]
    rotation_of: dict[int, list[int]]  # team id -> 9 player ids
    weight_of: dict[int, Vector]  # team id -> minute weights over its rotation


def generate(config: SyntheticConfig | None = None) -> tuple[StintTable, GroundTruth]:
    """Generate a deterministic stint table and its ground truth."""

    cfg = config or SyntheticConfig()
    if cfg.rotation_size < LINEUP_SIZE:
        raise ValueError("rotation_size must be at least 5")
    if cfg.n_players < cfg.n_teams * cfg.rotation_size:
        raise ValueError(
            f"n_players ({cfg.n_players}) must be >= n_teams * rotation_size "
            f"({cfg.n_teams * cfg.rotation_size}); grow the pool or shrink the league"
        )
    rng = np.random.default_rng(cfg.seed)
    player_ids = tuple(1001 + i for i in range(cfg.n_players))

    off_talent = rng.normal(0.0, cfg.talent_sd, size=cfg.n_players)
    def_talent = rng.normal(0.0, cfg.defensive_talent_sd, size=cfg.n_players)
    off_talent -= off_talent.mean()
    def_talent -= def_talent.mean()

    provision = rng.normal(
        0.0, cfg.embedding_sd, size=(cfg.n_players, cfg.embedding_rank)
    )
    need = rng.normal(0.0, cfg.embedding_sd, size=(cfg.n_players, cfg.embedding_rank))
    provision -= provision.mean(axis=0, keepdims=True)
    need -= need.mean(axis=0, keepdims=True)

    context_weights = {
        "home_offense": 2.2,
        "score_margin_offense": -0.18,
        "period": -0.6,
        "playoff": -1.4,
        "days_rest_offense": 0.5,
        "garbage_time_weight": -3.0,
        "season_pace_drift": 0.8,
    }

    truth = GroundTruth(
        config=cfg,
        player_ids=player_ids,
        alpha=cfg.base_offensive_rating,
        off_talent=off_talent,
        def_talent=def_talent,
        provision=provision,
        need=need,
        context_weights=context_weights,
        interaction_scale=cfg.interaction_scale,
    )

    seasons = _build_seasons(cfg, rng, list(player_ids))
    stints = _simulate(cfg, rng, truth, seasons)
    table = StintTable.from_stints(stints).sorted_chronologically()
    return table, truth


def _build_seasons(
    cfg: SyntheticConfig, rng: np.random.Generator, player_ids: list[int]
) -> list[_Season]:
    seasons: list[_Season] = []
    team_of = {pid: i % cfg.n_teams for i, pid in enumerate(player_ids)}
    for season in range(cfg.n_seasons):
        if season > 0:
            for pid in player_ids:
                if rng.random() < cfg.roster_change_prob:
                    team_of[pid] = int(rng.integers(cfg.n_teams))
        team_of = _balance_rosters(cfg, rng, player_ids, dict(team_of))
        rotation_of: dict[int, list[int]] = {}
        weight_of: dict[int, Vector] = {}
        for team in range(cfg.n_teams):
            roster = sorted(p for p, t in team_of.items() if t == team)
            rotation = sorted(
                int(p)
                for p in rng.choice(roster, size=cfg.rotation_size, replace=False)
            )
            weights = rng.uniform(0.6, 2.4, size=cfg.rotation_size)
            rotation_of[team] = rotation
            weight_of[team] = weights / weights.sum()
        seasons.append(_Season(dict(team_of), rotation_of, weight_of))
    return seasons


def _balance_rosters(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    player_ids: list[int],
    team_of: dict[int, int],
) -> dict[int, int]:
    """Ensure every team has at least ``rotation_size`` players."""

    min_size = cfg.rotation_size
    changed = True
    while changed:
        changed = False
        sizes = {t: 0 for t in range(cfg.n_teams)}
        for t in team_of.values():
            sizes[t] += 1
        short = [t for t, n in sizes.items() if n < min_size]
        if not short:
            break
        surplus_players = [p for p in player_ids if sizes[team_of[p]] > min_size]
        rng.shuffle(surplus_players)
        for team in short:
            need = min_size - sum(1 for v in team_of.values() if v == team)
            for p in surplus_players[:need]:
                team_of[p] = team
            surplus_players = surplus_players[need:]
            changed = True
    return team_of


def _draw_lineup(
    rng: np.random.Generator, rotation: list[int], weights: Vector
) -> tuple[int, int, int, int, int]:
    picked = rng.choice(len(rotation), size=5, replace=False, p=weights)
    return tuple(sorted(int(rotation[k]) for k in picked))  # type: ignore[return-value]


def _simulate(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    truth: GroundTruth,
    seasons: list[_Season],
) -> list[Stint]:
    stints: list[Stint] = []
    season_labels = [f"S{i + 1}" for i in range(cfg.n_seasons)]
    game_counter = 0
    for season in range(cfg.n_seasons):
        state = seasons[season]
        season_start = _dt.date(2020, 10, 20) + _dt.timedelta(days=365 * season)
        matchups = [
            (a, b)
            for a in range(cfg.n_teams)
            for b in range(cfg.n_teams)
            if a != b
            for _ in range(cfg.games_per_matchup)
        ]
        rng.shuffle(matchups)
        n_playoff = int(round(cfg.playoff_fraction * len(matchups)))
        n_games = len(matchups)
        for game_number, matchup in enumerate(matchups):
            home, away = int(matchup[0]), int(matchup[1])
            game_counter += 1
            playoff = game_number >= n_games - n_playoff
            game_id = f"{season_labels[season]}-G{game_counter:04d}"
            game_date = (
                season_start
                + _dt.timedelta(days=int(game_number * 175 / max(1, n_games - 1)))
            ).isoformat()
            rest_home = int(rng.integers(0, 4))
            rest_away = int(rng.integers(0, 4))
            margin = 0
            for stint_index in range(cfg.stints_per_game):
                period = 1 + (stint_index * 4) // cfg.stints_per_game
                offense_is_home = stint_index % 2 == 0
                off_team, def_team = (home, away) if offense_is_home else (away, home)
                o5 = _draw_lineup(
                    rng, state.rotation_of[off_team], state.weight_of[off_team]
                )
                d5 = _draw_lineup(
                    rng, state.rotation_of[def_team], state.weight_of[def_team]
                )
                possessions = cfg.min_possessions + int(
                    rng.poisson(cfg.mean_extra_possessions)
                )
                margin += int(rng.normal(0.0, 6.0))
                garbage = 1.0 if abs(margin) < 22 or period < 4 else 0.55
                context = {
                    "home_offense": float(offense_is_home),
                    "score_margin_offense": float(margin),
                    "period": float(period),
                    "playoff": float(playoff),
                    "days_rest_offense": float(
                        rest_home if offense_is_home else rest_away
                    ),
                    "garbage_time_weight": garbage,
                    "season_index": float(season),
                }
                value = truth.lineup_value(o5, d5, context)
                per_poss = rng.normal(
                    value / 100.0, cfg.per_possession_sd, size=possessions
                )
                points = max(0, int(round(float(per_poss.sum()))))
                stints.append(
                    Stint(
                        stint_id=f"{game_id}-P{stint_index:02d}",
                        game_id=game_id,
                        game_date=game_date,
                        season=season_labels[season],
                        season_index=season,
                        period=period,
                        start_time_seconds=float(stint_index * 144),
                        offense_team_id=10 + off_team,
                        defense_team_id=10 + def_team,
                        offense_player_ids=o5,
                        defense_player_ids=d5,
                        offensive_possessions=possessions,
                        points_scored=points,
                        home_offense=offense_is_home,
                        score_margin_offense=int(margin),
                        playoff=playoff,
                        days_rest_offense=int(
                            rest_home if offense_is_home else rest_away
                        ),
                        garbage_time_weight=garbage,
                        source="synthetic",
                    )
                )
    return stints
