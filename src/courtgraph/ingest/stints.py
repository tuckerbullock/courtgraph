"""Turn validated possessions into ``courtgraph.chemistry.stints`` records.

A stint here is a *maximal run of consecutive validated possessions with the
same ten players on the floor in the same period* (master plan 7.5). Each run
emits up to two one-sided :class:`~courtgraph.chemistry.stints.Stint` rows -- one
per team that had the ball -- matching the shape the chemistry code expects.

Two things break a run, both required for correct attribution:

* the ten-player set changes -- non-contiguous appearances of the same five are
  never merged; and
* a **gap** in the original possession sequence -- if any possession between
  two accepted ones was excluded (lineup change, ambiguous scoring, ...), the
  accepted possessions on either side belong to different stints even when
  their lineups match.
"""

from __future__ import annotations

from courtgraph.chemistry.stints import Stint
from courtgraph.ingest.policy import IngestPolicy
from courtgraph.ingest.snapshot import GameMetadata
from courtgraph.ingest.validate import AcceptedPossession


def _period_length_seconds(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def _lineup_key(
    poss: AcceptedPossession,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    view = poss.view
    teams = sorted(view.lineups)
    return (
        view.period,
        tuple(sorted(view.lineups[teams[0]])),
        tuple(sorted(view.lineups[teams[1]])),
    )


def possessions_to_stints(
    accepted: list[AcceptedPossession],
    metadata: GameMetadata,
    policy: IngestPolicy,
    season_index: int,
) -> list[Stint]:
    ordered = sorted(accepted, key=lambda p: p.view.sequence_index)
    stints: list[Stint] = []
    run: list[AcceptedPossession] = []
    run_index = 0

    def flush() -> None:
        nonlocal run_index
        if not run:
            return
        run_index += 1
        stints.extend(_run_to_stints(run, metadata, policy, season_index, run_index))
        run.clear()

    current_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
    prev_seq: int | None = None
    for poss in ordered:
        key = _lineup_key(poss)
        contiguous = prev_seq is not None and poss.view.sequence_index == prev_seq + 1
        if run and (key != current_key or not contiguous):
            flush()
        run.append(poss)
        current_key = key
        prev_seq = poss.view.sequence_index
    flush()
    return stints


def _run_to_stints(
    run: list[AcceptedPossession],
    metadata: GameMetadata,
    policy: IngestPolicy,
    season_index: int,
    run_index: int,
) -> list[Stint]:
    first = run[0].view
    period = first.period
    period_length = _period_length_seconds(period)
    run_start_remaining = first.start_seconds_remaining
    start_time_seconds = min(
        max(period_length - run_start_remaining, 0.0), period_length
    )
    run_start_score = dict(first.start_score)

    by_offense: dict[int, list[AcceptedPossession]] = {}
    for poss in run:
        by_offense.setdefault(poss.view.offense_team_id, []).append(poss)

    out: list[Stint] = []
    for offense_team_id, group in sorted(by_offense.items()):
        if len(group) < policy.min_offensive_possessions_per_stint:
            continue
        teams = sorted(first.lineups)
        defense_team_id = teams[0] if teams[1] == offense_team_id else teams[1]
        offense_ids = tuple(sorted(first.lineups[offense_team_id]))
        defense_ids = tuple(sorted(first.lineups[defense_team_id]))

        margin = int(
            run_start_score.get(offense_team_id, 0)
            - run_start_score.get(defense_team_id, 0)
        )
        weight = policy.garbage_time_weight(period, run_start_remaining, abs(margin))

        out.append(
            Stint(
                stint_id=(
                    f"{metadata.game_id}-P{period}-R{run_index:03d}-O{offense_team_id}"
                ),
                game_id=metadata.game_id,
                game_date=metadata.game_date,
                season=metadata.season,
                season_index=season_index,
                period=period,
                start_time_seconds=float(start_time_seconds),
                offense_team_id=offense_team_id,
                defense_team_id=defense_team_id,
                offense_player_ids=offense_ids,  # type: ignore[arg-type]
                defense_player_ids=defense_ids,  # type: ignore[arg-type]
                offensive_possessions=len(group),
                points_scored=sum(p.points for p in group),
                home_offense=offense_team_id == metadata.home_team_id,
                score_margin_offense=margin,
                playoff=metadata.playoff,
                days_rest_offense=int(metadata.days_rest[offense_team_id]),
                garbage_time_weight=weight,
                source="nba-stats-pbpstats",
            )
        )
    return out
