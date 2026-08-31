"""CourtGraph's own checks on reconstructed possessions.

``pbpstats`` output is treated as a proposal. This module re-derives what it can
from that proposal and the *independent* box-score totals in the snapshot
metadata, and decides -- per possession and per game -- what is trustworthy
enough to become a stint. Nothing here fabricates a value to satisfy the schema:
the outcomes are "accept", "exclude this possession", or "quarantine this game".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from courtgraph.ingest.policy import IngestPolicy
from courtgraph.ingest.possessions import PossessionView
from courtgraph.ingest.snapshot import GameMetadata


@dataclass(frozen=True)
class Exclusion:
    """One possession (or whole game) that did not become a stint."""

    game_id: str
    level: str  # "possession" | "game"
    reason: str
    detail: str
    period: int | None = None
    possession_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "game_id": self.game_id,
            "level": self.level,
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.period is not None:
            out["period"] = self.period
        if self.possession_number is not None:
            out["possession_number"] = self.possession_number
        return out


@dataclass(frozen=True)
class AcceptedPossession:
    """A possession that passed validation, with its settled offensive points."""

    view: PossessionView
    points: int


@dataclass
class GameValidation:
    game_id: str
    game_quarantined: bool = False
    quarantine_reason: str = ""
    accepted: list[AcceptedPossession] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    reconciliation: dict[str, object] = field(default_factory=dict)
    source_event_counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    reconstructed_possessions: int = 0


def _settle_points(view: PossessionView) -> int:
    """Offensive points on the possession, technical free throws removed."""

    return view.offense_points_raw - view.offense_technical_ft_points


def validate_game(
    views: list[PossessionView],
    metadata: GameMetadata,
    policy: IngestPolicy,
) -> GameValidation:
    result = GameValidation(
        game_id=metadata.game_id, reconstructed_possessions=len(views)
    )

    gaps = metadata.missing_context()
    if gaps:
        result.game_quarantined = True
        result.quarantine_reason = "missing_context"
        result.exclusions.append(
            Exclusion(
                game_id=metadata.game_id,
                level="game",
                reason="missing_context",
                detail=f"required metadata absent: {', '.join(gaps)}",
            )
        )
        return result

    valid_teams = {metadata.home_team_id, metadata.away_team_id}
    event_counts: dict[str, int] = {}
    reconcilable: list[tuple[PossessionView, int]] = []
    game_technical_ft: dict[int, int] = {}

    for view in views:
        for name, count in view.event_type_counts.items():
            event_counts[name] = event_counts.get(name, 0) + count
        for team, points in view.technical_ft_points_by_team.items():
            game_technical_ft[team] = game_technical_ft.get(team, 0) + points

        if {view.offense_team_id, view.defense_team_id} != valid_teams:
            result.exclusions.append(
                _exc(
                    view,
                    "unknown_team",
                    f"offense/defense {view.offense_team_id}/{view.defense_team_id} "
                    f"not in {sorted(valid_teams)}",
                )
            )
            continue

        if policy.drop_empty_possessions and not view.has_real_ending_event:
            result.exclusions.append(
                _exc(view, "empty_possession", "no shot / rebound / turnover event")
            )
            continue

        off_n = len(view.lineup_5(view.offense_team_id))
        def_n = len(view.lineup_5(view.defense_team_id))
        if off_n != 5 or def_n != 5:
            result.exclusions.append(
                _exc(
                    view,
                    "lineup_not_five",
                    f"offense has {off_n} players, defense has {def_n}",
                )
            )
            continue

        settled = _settle_points(view)
        if settled < 0 or settled > policy.max_possession_points:
            result.exclusions.append(
                _exc(
                    view,
                    "ambiguous_scoring",
                    f"raw={view.offense_points_raw} "
                    f"tech_ft={view.offense_technical_ft_points} settled={settled}",
                )
            )
            continue

        # points are now settled -> this possession counts toward reconciliation
        reconcilable.append((view, settled))
        if view.offense_technical_ft_points:
            _flag(result, "technical_ft_in_possession")

        if policy.quarantine_split_lineup_possessions and view.is_split_lineup:
            result.exclusions.append(
                _exc(
                    view,
                    "split_lineup_possession",
                    "substitution between live events; not attributed to a stint",
                )
            )
            continue

        result.accepted.append(AcceptedPossession(view=view, points=settled))

    result.source_event_counts = event_counts
    if any(game_technical_ft.values()):
        _flag(result, "technical_free_throws_in_game")

    _check_alternation(views, result)
    _reconcile_scores(reconcilable, game_technical_ft, metadata, policy, result)

    return result


def _exc(view: PossessionView, reason: str, detail: str) -> Exclusion:
    return Exclusion(
        game_id=view.game_id,
        level="possession",
        reason=reason,
        detail=detail,
        period=view.period,
        possession_number=view.number,
    )


def _flag(result: GameValidation, flag: str) -> None:
    if flag not in result.flags:
        result.flags.append(flag)


def _check_alternation(views: list[PossessionView], result: GameValidation) -> None:
    """Independent re-derivation of possession alternation (pbpstats also checks)."""

    by_period: dict[int, list[PossessionView]] = {}
    for view in views:
        by_period.setdefault(view.period, []).append(view)
    for period, period_views in by_period.items():
        ordered = sorted(period_views, key=lambda v: v.number)
        for prev, curr in zip(ordered, ordered[1:], strict=False):
            if prev.offense_team_id == curr.offense_team_id:
                result.game_quarantined = True
                result.quarantine_reason = "possession_alternation_failed"
                result.exclusions.append(
                    Exclusion(
                        game_id=result.game_id,
                        level="game",
                        reason="possession_alternation_failed",
                        detail=f"period {period}: possessions "
                        f"{prev.number} and {curr.number} share offense "
                        f"{curr.offense_team_id}",
                    )
                )
                return
        numbers = [v.number for v in ordered]
        if numbers and numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            _flag(result, f"noncontiguous_possession_numbers_period_{period}")


def _reconcile_scores(
    reconcilable: list[tuple[PossessionView, int]],
    technical_ft_by_team: dict[int, int],
    metadata: GameMetadata,
    policy: IngestPolicy,
    result: GameValidation,
) -> None:
    # Every point is accounted for: possession points, plus technical free
    # throws (real points that belong to no possession's offense).
    derived_final: dict[int, int] = {
        metadata.home_team_id: 0,
        metadata.away_team_id: 0,
    }
    derived_periods: dict[int, dict[int, int]] = {}
    for view, points in reconcilable:
        derived_final[view.offense_team_id] = (
            derived_final.get(view.offense_team_id, 0) + points
        )
        derived_periods.setdefault(view.period, {}).setdefault(view.offense_team_id, 0)
        derived_periods[view.period][view.offense_team_id] += points
    for team, points in technical_ft_by_team.items():
        derived_final[team] = derived_final.get(team, 0) + points

    official = metadata.final_score
    game_teams = (metadata.home_team_id, metadata.away_team_id)
    # Reconcile over BOTH teams: a final score present for only one of them
    # must not be treated as a match (see GameMetadata.missing_context, which
    # already quarantines this case before we get here).
    final_deltas = {
        team: derived_final.get(team, 0) - official.get(team, 0) for team in game_teams
    }
    matched = all(team in official for team in game_teams) and all(
        delta == 0 for delta in final_deltas.values()
    )

    period_deltas: dict[int, dict[int, int]] = {}
    for period, official_list in _official_period_scores(metadata).items():
        for team, official_pts in official_list.items():
            got = derived_periods.get(period, {}).get(team, 0)
            period_deltas.setdefault(period, {})[team] = got - official_pts

    result.reconciliation = {
        "final_score_official": {str(k): v for k, v in official.items()},
        "final_score_derived": {str(k): v for k, v in derived_final.items()},
        "final_score_delta": {str(k): v for k, v in final_deltas.items()},
        "final_score_matched": matched,
        "period_score_delta": {
            str(p): {str(t): d for t, d in deltas.items()}
            for p, deltas in period_deltas.items()
        },
    }

    if not matched and policy.require_exact_final_score:
        if policy.allow_score_mismatch:
            _flag(result, "score_reconciliation_mismatch_allowed")
        else:
            result.game_quarantined = True
            result.quarantine_reason = "score_reconciliation_failed"
            result.exclusions.append(
                Exclusion(
                    game_id=result.game_id,
                    level="game",
                    reason="score_reconciliation_failed",
                    detail=f"derived {derived_final} vs official {official}",
                )
            )
            result.accepted = []


def _official_period_scores(metadata: GameMetadata) -> dict[int, dict[int, int]]:
    out: dict[int, dict[int, int]] = {}
    for team, points_by_period in metadata.period_scores.items():
        for index, points in enumerate(points_by_period, start=1):
            out.setdefault(index, {})[team] = points
    return out
