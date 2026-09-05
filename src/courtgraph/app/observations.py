"""Read an explicit ingest output without changing or retraining on its contents."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from courtgraph.chemistry.stints import Stint, StintTable, read_stints


def _object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


class Observations:
    def __init__(self, directory: Path, names_path: Path | None = None) -> None:
        self.manifest = _object(directory / "manifest.json")
        path = directory / "stints.jsonl"
        expected = self.manifest.get("outputs", {}).get("stints_sha256")
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("stints.jsonl does not match its manifest checksum")
        self.table = read_stints(path)
        self.games = self.manifest.get("games", [])
        if not isinstance(self.games, list):
            raise ValueError("manifest games must be a list")
        game_ids = [str(g["game_id"]) for g in self.games]
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("manifest contains duplicate games")
        accepted = {
            str(g["game_id"]): g for g in self.games if g["status"] == "accepted"
        }
        counts = Counter(s.game_id for s in self.table)
        possessions: Counter[str] = Counter()
        for stint in self.table:
            game = accepted.get(stint.game_id)
            if game is None or stint.game_date != game["game_date"]:
                raise ValueError("stint does not belong to a dated accepted game")
            possessions[stint.game_id] += stint.offensive_possessions
        if len(self.table) != self.manifest["outputs"]["stints_written"]:
            raise ValueError("manifest stint count does not match the data")
        for gid, game in accepted.items():
            if (
                counts[gid] != game["stints_emitted"]
                or possessions[gid] != game["accepted_possessions"]
            ):
                raise ValueError("game exposure does not match the manifest")
        names = _object(names_path) if names_path else {}
        self.names: dict[str, dict[str, str]] = {}
        for kind in ("players", "teams"):
            values = names.get(kind, {})
            if not isinstance(values, dict) or any(
                not isinstance(v, str) for v in values.values()
            ):
                raise ValueError(f"display names for {kind} must be strings")
            self.names[kind] = {str(k): v for k, v in values.items()}
        self._rung3_model: Any = None
        self._rung3_metadata: dict[str, Any] = {}

    def team_name(self, team: int | str) -> str:
        return self.names["teams"].get(str(team), f"Team {team}")

    def player_name(self, player: int) -> str:
        return self.names["players"].get(str(player), f"Player {player}")

    def overview(self) -> dict[str, Any]:
        provenance = self.manifest.get("source_provenance") or {}
        coverage = provenance.get("archive_coverage") or {}
        games = []
        for game in self.games:
            rec = game.get("reconciliation", {})
            scores = rec.get("final_score_official", {})
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")
            team_ids = [team for team in (away_id, home_id) if team is not None]
            if not team_ids:
                team_ids = list(scores)
            games.append(
                {
                    "id": game["game_id"],
                    "date": game["game_date"],
                    "status": game["status"],
                    "home_team": self.team_name(home_id) if home_id is not None else "",
                    "away_team": self.team_name(away_id) if away_id is not None else "",
                    "stints": game["stints_emitted"],
                    "possessions": game["accepted_possessions"],
                    "scores": [
                        {
                            "team": self.team_name(team),
                            "points": scores.get(str(team), scores.get(team)),
                        }
                        for team in team_ids
                        if str(team) in scores or team in scores
                    ],
                    "score_matched": rec.get("final_score_matched"),
                    "score_source": rec.get("official_score_source", "Not recorded"),
                    "quarantine_reason": game.get("quarantine_reason", ""),
                    "flags": game.get("flags", []),
                    "exclusions": dict(
                        Counter(
                            item.get("reason", "unknown")
                            for item in game.get("excluded_possessions", [])
                        )
                    ),
                }
            )
        for excluded in coverage.get("excluded_games", []):
            team_ids = excluded.get("team_ids", [])
            games.append(
                {
                    "id": excluded.get("game_id", "unknown"),
                    "date": excluded.get("game_date", ""),
                    "status": "source_incomplete",
                    "home_team": "",
                    "away_team": "",
                    "stints": 0,
                    "possessions": 0,
                    "scores": [
                        {"team": self.team_name(team), "points": None}
                        for team in team_ids
                    ],
                    "score_matched": None,
                    "score_source": "Not attempted: required archive input missing",
                    "quarantine_reason": "missing archive input: "
                    + ", ".join(excluded.get("missing_inputs", [])),
                    "flags": [],
                    "exclusions": {},
                }
            )
        games.sort(key=lambda game: (game["date"], game["id"]))
        teams = sorted(
            {s.offense_team_id for s in self.table}
            | {s.defense_team_id for s in self.table}
        )
        players = sorted(
            {
                p
                for stint in self.table
                for p in (*stint.offense_player_ids, *stint.defense_player_ids)
            }
        )
        totals = self.manifest.get("totals", {})
        coverage_summary = {
            "archive_games": int(
                coverage.get("archive_games", totals.get("games_in", len(self.games)))
            ),
            "complete_games": int(
                coverage.get("complete_games", totals.get("games_in", len(self.games)))
            ),
            "attempted_games": int(totals.get("games_in", len(self.games))),
            "accepted_games": int(
                totals.get(
                    "games_accepted",
                    sum(game["status"] == "accepted" for game in self.games),
                )
            ),
            "quarantined_games": int(
                totals.get(
                    "games_quarantined",
                    sum(game["status"] == "quarantined" for game in self.games),
                )
            ),
            "source_incomplete_games": len(coverage.get("excluded_games", [])),
        }
        return {
            "loaded": True,
            "mode": "observations",
            "games": games,
            "coverage": coverage_summary,
            "teams": [{"id": team, "name": self.team_name(team)} for team in teams],
            "players": [
                {"id": player, "name": self.player_name(player)} for player in players
            ],
            "source": provenance.get(
                "source", "Source not recorded; do not assume real NBA data"
            ),
            "converter": provenance.get("converter_version", "Not recorded"),
            "source_commit": provenance.get("pinned_commit", "Not recorded"),
            "checksum": self.manifest["outputs"]["stints_sha256"],
            "created_utc": self.manifest.get("created_utc", "Not recorded"),
            "parser": self.manifest.get("parser", {}),
            "cutoff": max((stint.game_date for stint in self.table), default=None),
            "warning": (
                "Observed accepted possessions only. Not adjusted impact, "
                "chemistry, or a prediction. Participants are not a "
                "complete roster."
            ),
        }

    def query(
        self, *, game: str = "", team: str = "", player: str = "", minimum: int = 1
    ) -> dict[str, Any]:
        if minimum < 1 or minimum > 1_000_000:
            raise ValueError("minimum possessions must be between 1 and 1000000")
        rows = [
            s
            for s in self.table
            if (not game or s.game_id == game)
            and (not team or str(s.offense_team_id) == team)
            and (not player or player in {str(p) for p in s.offense_player_ids})
        ]
        groups: dict[tuple[int, tuple[int, ...]], list[Stint]] = {}
        for stint in rows:
            groups.setdefault(
                (stint.offense_team_id, stint.offense_player_ids), []
            ).append(stint)
        lineups: list[dict[str, Any]] = []
        for (tid, ids), stints in groups.items():
            possessions = sum(s.offensive_possessions for s in stints)
            if possessions < minimum:
                continue
            points = sum(s.points_scored for s in stints)
            lineups.append(
                {
                    "id": f"{tid}:" + "-".join(map(str, ids)),
                    "team": self.team_name(tid),
                    "players": [self.player_name(p) for p in ids],
                    "player_ids": list(ids),
                    "possessions": possessions,
                    "points": points,
                    "rating": 100 * points / possessions,
                    "stints": len(stints),
                    "games": len({s.game_id for s in stints}),
                    "opponents": sorted(
                        {self.team_name(s.defense_team_id) for s in stints}
                    ),
                    "dates": sorted({s.game_date for s in stints}),
                    "downweighted_stints": sum(
                        s.garbage_time_weight < 1 for s in stints
                    ),
                }
            )
        lineups.sort(key=lambda row: (-row["possessions"], row["id"]))
        table = StintTable.from_stints(rows)
        return {
            "lineups": lineups,
            "stints": len(rows),
            "possessions": table.total_possessions(),
            "points": sum(s.points_scored for s in rows),
            "games": len({s.game_id for s in rows}),
            "minimum": minimum,
            "weighting": (
                "Raw possession totals; no garbage-time weighting. Minimum "
                "filters lineup rows only."
            ),
        }

    def player_pool(self, team: str) -> dict[str, Any]:
        """Players observed on this team's offense or defense anywhere in the
        loaded ingest window -- an *inferred exposure set*, not an official
        roster. A player who barely played, or who joined/left mid-window,
        may be over- or under-represented; there is no dated roster source."""

        if not team:
            raise ValueError("team is required")
        possessions: Counter[int] = Counter()
        for stint in self.table:
            if str(stint.offense_team_id) == team:
                for pid in stint.offense_player_ids:
                    possessions[pid] += stint.offensive_possessions
            elif str(stint.defense_team_id) == team:
                for pid in stint.defense_player_ids:
                    possessions[pid] += stint.offensive_possessions
        if not possessions:
            raise ValueError(f"no players observed for team {team!r}")
        players = [
            {"id": pid, "name": self.player_name(pid), "possessions": poss}
            for pid, poss in sorted(possessions.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return {
            "team": self.team_name(team),
            "players": players,
            "source": (
                "Observed in stints for this team in the loaded ingest "
                "window, not an official roster."
            ),
        }

    def _ensure_rung3(self) -> tuple[Any, dict[str, Any]]:
        if self._rung3_model is None:
            from courtgraph.chemistry.features import FeatureSpace
            from courtgraph.chemistry.hierarchical import HierarchicalRidge

            space = FeatureSpace.from_training(self.table)
            design = space.build(self.table)
            self._rung3_model = HierarchicalRidge.fit(design, space)
            possessions: Counter[int] = Counter()
            for stint in self.table:
                for pid in stint.offense_player_ids:
                    possessions[pid] += stint.offensive_possessions
            self._rung3_metadata = {
                "training_player_possessions": {
                    str(k): v for k, v in possessions.items()
                }
            }
        return self._rung3_model, self._rung3_metadata

    @staticmethod
    def _lineup_ids(raw: Any, label: str) -> list[int]:
        if (
            not isinstance(raw, list)
            or len(raw) != 5
            or any(type(p) is not int for p in raw)
        ):
            raise ValueError(f"{label} must be exactly five integer player ids")
        return list(raw)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score a real 5-vs-5 lineup with rung 3 -- additive talent + context
        and a calibrated interval, fit once (lazily, cached) on this
        ingest's stints. No chemistry/interaction claim; see
        :data:`courtgraph.chemistry.pipeline.RUNG3_NOTE`."""

        if set(payload) - {"offense", "defense", "home", "playoff", "rest"}:
            raise ValueError("Unknown prediction field")
        offense = self._lineup_ids(payload.get("offense"), "offense")
        defense = self._lineup_ids(payload.get("defense"), "defense")
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
                "Invalid context: home/playoff must be booleans and rest 0-7 days"
            )
        context = {"home_offense": home, "playoff": playoff, "days_rest_offense": rest}

        from courtgraph.chemistry.pipeline import _predict_lineup_rung3_with_model

        model, meta = self._ensure_rung3()
        result = _predict_lineup_rung3_with_model(
            model, meta, offense, defense, context
        )
        return result.as_dict()
