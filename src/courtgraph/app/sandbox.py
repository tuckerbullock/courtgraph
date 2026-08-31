"""A small, in-memory synthetic model; never accepts NBA artifacts or names."""

from __future__ import annotations

from typing import Any

from courtgraph.chemistry.chemistry_model import ChemistryConfig, ChemistryModel
from courtgraph.chemistry.pipeline import DEFAULT_CONTEXT
from courtgraph.chemistry.synthetic import SyntheticConfig, generate


class Sandbox:
    def __init__(self) -> None:
        self.seed = 20260831
        # Large enough that the cross-fitted interaction fit clears its
        # out-of-fold selection gate and recovers a non-zero chemistry surplus;
        # a smaller pool leaves the sandbox's interaction component identically
        # zero. ~12.7k stints, deterministic, fits in a few seconds at startup.
        table, _ = generate(
            SyntheticConfig(
                seed=self.seed,
                n_players=96,
                n_teams=12,
                rotation_size=8,
                n_seasons=3,
                games_per_matchup=2,
                stints_per_game=16,
            )
        )
        self.model = ChemistryModel.fit(
            table,
            ChemistryConfig(
                seed=0,
                cross_fit_folds=3,
                als_sweeps=14,
                selection_folds=2,
                n_bootstrap=8,
                reference_sample=800,
            ),
        )
        self.players = sorted(self.model.feature_space.player_ids)
        last = table.stints[-1]
        self.offense = list(last.offense_player_ids)
        self.defense = list(last.defense_player_ids)
        self.alternative = self.offense[:-1] + [
            next(p for p in self.players if p not in self.offense + self.defense)
        ]
        self.cutoff = max(s.game_date for s in table)

    def catalog(self) -> dict[str, Any]:
        return {
            "mode": "synthetic",
            "seed": self.seed,
            "model": "Synthetic additive ridge + low-rank interactions",
            "players": [
                {"id": p, "name": f"Synthetic Player {i + 1:02d}"}
                for i, p in enumerate(self.players)
            ],
            "offense": self.offense,
            "alternative": self.alternative,
            "defense": self.defense,
            "training_stints": self.model.training_stints,
            "training_possessions": self.model.training_possessions,
            "cutoff": self.cutoff,
            "bootstrap_members": len(self.model.interaction_ensemble),
            "warning": (
                "Fictional players, synthetic training data. These are not "
                "NBA predictions. This app does not establish held-out "
                "accuracy."
            ),
        }

    def _lineup(self, raw: Any) -> tuple[int, ...]:
        if (
            not isinstance(raw, list)
            or len(raw) != 5
            or any(type(p) is not int for p in raw)
        ):
            raise ValueError("Select exactly five synthetic player IDs per lineup")
        if len(set(raw)) != 5:
            raise ValueError("Each lineup must contain five distinct players")
        if not set(raw).issubset(self.players):
            raise ValueError("Only players from this synthetic catalog are allowed")
        return tuple(sorted(raw))

    def compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) - {
            "offense",
            "alternative",
            "defense",
            "home",
            "playoff",
            "rest",
        }:
            raise ValueError("Unknown comparison field")
        offense = self._lineup(payload.get("offense"))
        alternative = self._lineup(payload.get("alternative"))
        defense = self._lineup(payload.get("defense"))
        if set(defense) & (set(offense) | set(alternative)):
            raise ValueError(
                "An offensive player cannot also be on the opposing lineup"
            )
        home, playoff, rest = (
            payload.get("home", True),
            payload.get("playoff", False),
            payload.get("rest", 1),
        )
        if (
            type(home) is not bool
            or type(playoff) is not bool
            or type(rest) is not int
            or not 0 <= rest <= 7
        ):
            raise ValueError(
                "Invalid context: home/playoff must be booleans and rest 0–7 days"
            )
        context = {
            **DEFAULT_CONTEXT,
            "home_offense": home,
            "playoff": playoff,
            "days_rest_offense": rest,
            "season_index": 2,
        }
        results: list[dict[str, Any]] = []
        for lineup in (offense, alternative):
            d = self.model.decompose(lineup, defense, context)
            results.append(
                {
                    "offense": list(lineup),
                    "decomposition": d.as_dict(),
                    "support": self.model.lineup_support(lineup),
                    "interval": self.model.interaction_interval(lineup),
                }
            )
        return {
            "mode": "synthetic",
            "units": "Offensive points per 100 possessions; not net rating",
            "context": context,
            "defense": list(defense),
            "results": results,
            "delta": {
                key: results[1]["decomposition"][key] - results[0]["decomposition"][key]
                for key in ("talent", "interaction", "context", "total")
            },
            "uncertainty_note": (
                "80% approximate bootstrap spread for the interaction "
                "component only; not a calibrated interval for total "
                "performance or the A/B difference."
            ),
        }
