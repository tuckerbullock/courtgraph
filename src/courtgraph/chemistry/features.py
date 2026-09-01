"""Design-matrix construction shared by the additive baseline and the model.

The :class:`FeatureSpace` fixes the player vocabulary, the context columns, and
the standardization constants **from the training stints only** (research
contract 13: scalers are fit within the training cutoff). Serialized models
carry the same information so ``courtgraph predict`` reproduces training-time
features exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from courtgraph.chemistry.stints import Stint, StintTable

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Context columns produced from a stint. period is one-hot with period 1 as the
# held-out baseline; season is one-hot with the first season as baseline.
_STANDARDIZE = ("score_margin_offense", "days_rest_offense")


@dataclass(frozen=True)
class DesignMatrices:
    """Everything a linear or embedding model needs for one stint table."""

    context: FloatArray  # (n, n_context)
    offense_index: IntArray  # (n, 5) player positions on offense, -1 = unseen
    defense_index: IntArray  # (n, 5) player positions on defense, -1 = unseen
    y: FloatArray  # (n,) offensive rating (points / 100 possessions)
    weight: FloatArray  # (n,) offensive possessions (exposure)
    game_ids: tuple[str, ...]
    stint_ids: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.context.shape[0])


@dataclass(frozen=True)
class FeatureSpace:
    """Fixed feature vocabulary and standardization, fit on training stints."""

    player_ids: tuple[int, ...]
    context_columns: tuple[str, ...]
    season_labels: tuple[str, ...]
    standardize_mean: dict[str, float]
    standardize_std: dict[str, float]

    @classmethod
    def from_training(cls, table: StintTable) -> FeatureSpace:
        player_ids = table.player_ids()
        season_labels = table.season_order()
        margins = np.array([s.score_margin_offense for s in table], dtype=np.float64)
        rests = np.array([s.days_rest_offense for s in table], dtype=np.float64)
        raw = {"score_margin_offense": margins, "days_rest_offense": rests}
        mean = {k: float(v.mean()) for k, v in raw.items()}
        std = {k: float(v.std()) or 1.0 for k, v in raw.items()}
        context_columns = cls._context_columns(season_labels)
        return cls(
            player_ids=player_ids,
            context_columns=context_columns,
            season_labels=season_labels,
            standardize_mean=mean,
            standardize_std=std,
        )

    @staticmethod
    def _context_columns(season_labels: tuple[str, ...]) -> tuple[str, ...]:
        cols = [
            "intercept",
            "home_offense",
            "score_margin_offense_z",
            "score_margin_offense_z_sq",
            "period_2",
            "period_3",
            "period_4",
            "playoff",
            "days_rest_offense_z",
            "garbage_time_deficit",
        ]
        cols += [f"season_{label}" for label in season_labels[1:]]
        return tuple(cols)

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def n_context(self) -> int:
        return len(self.context_columns)

    def player_index(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.player_ids)}

    def context_row(self, stint: Stint) -> dict[str, float]:
        margin_z = (
            stint.score_margin_offense - self.standardize_mean["score_margin_offense"]
        ) / self.standardize_std["score_margin_offense"]
        rest_z = (
            stint.days_rest_offense - self.standardize_mean["days_rest_offense"]
        ) / self.standardize_std["days_rest_offense"]
        row = {
            "intercept": 1.0,
            "home_offense": float(stint.home_offense),
            "score_margin_offense_z": margin_z,
            "score_margin_offense_z_sq": margin_z * margin_z,
            "period_2": float(stint.period == 2),
            "period_3": float(stint.period == 3),
            "period_4": float(stint.period >= 4),
            "playoff": float(stint.playoff),
            "days_rest_offense_z": rest_z,
            "garbage_time_deficit": stint.garbage_time_weight - 1.0,
        }
        for label in self.season_labels[1:]:
            row[f"season_{label}"] = float(stint.season == label)
        return row

    def build(self, table: StintTable) -> DesignMatrices:
        n = len(table)
        index = self.player_index()
        context = np.zeros((n, self.n_context), dtype=np.float64)
        offense_index = np.full((n, 5), -1, dtype=np.int64)
        defense_index = np.full((n, 5), -1, dtype=np.int64)
        y = np.zeros(n, dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        for r, stint in enumerate(table):
            row = self.context_row(stint)
            for c, name in enumerate(self.context_columns):
                context[r, c] = row[name]
            for k, pid in enumerate(stint.offense_player_ids):
                offense_index[r, k] = index.get(pid, -1)
            for k, pid in enumerate(stint.defense_player_ids):
                defense_index[r, k] = index.get(pid, -1)
            y[r] = stint.offensive_rating
            weight[r] = float(stint.offensive_possessions)
        return DesignMatrices(
            context=context,
            offense_index=offense_index,
            defense_index=defense_index,
            y=y,
            weight=weight,
            game_ids=tuple(s.game_id for s in table),
            stint_ids=tuple(s.stint_id for s in table),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_ids": list(self.player_ids),
            "context_columns": list(self.context_columns),
            "season_labels": list(self.season_labels),
            "standardize_mean": dict(self.standardize_mean),
            "standardize_std": dict(self.standardize_std),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpace:
        return cls(
            player_ids=tuple(int(p) for p in data["player_ids"]),
            context_columns=tuple(data["context_columns"]),
            season_labels=tuple(data["season_labels"]),
            standardize_mean={k: float(v) for k, v in data["standardize_mean"].items()},
            standardize_std={k: float(v) for k, v in data["standardize_std"].items()},
        )
