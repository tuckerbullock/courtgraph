"""Hand-authored, NBA-shaped snapshot fixtures for the ingestion tests.

These build **real ``stats.nba.com`` ``playbyplayv2``-shaped JSON** (a
``resultSets`` payload with the real column headers and integer ``EVENTMSGTYPE``
codes) plus the sibling files ``pbpstats`` needs, then write them into the
documented ``stats_nba_pbpstats/v1`` snapshot layout. Nothing here mocks the
parser: the tests point the real pipeline (and therefore real ``pbpstats``) at
what this module writes.

Not a test module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Real playbyplayv2 column order (a representative subset that pbpstats reads).
PBP_COLUMNS = [
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "WCTIMESTRING",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
    "PERSON1TYPE",
    "PLAYER1_ID",
    "PLAYER1_NAME",
    "PLAYER1_TEAM_ID",
    "PERSON2TYPE",
    "PLAYER2_ID",
    "PLAYER2_NAME",
    "PLAYER2_TEAM_ID",
    "PERSON3TYPE",
    "PLAYER3_ID",
    "PLAYER3_NAME",
    "PLAYER3_TEAM_ID",
    "VIDEO_AVAILABLE_FLAG",
]

# EVENTMSGTYPE
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER = 1, 2, 3, 4, 5
FOUL, SUBSTITUTION, TIMEOUT, JUMP_BALL = 6, 8, 9, 10
START_PERIOD, END_PERIOD = 12, 13

REGULATION_SECONDS = 720
OVERTIME_SECONDS = 300


def _clock(seconds_remaining: int) -> str:
    seconds_remaining = max(seconds_remaining, 0)
    return f"{seconds_remaining // 60}:{seconds_remaining % 60:02d}"


@dataclass
class GameBuilder:
    """Accumulate playbyplayv2 rows for one game while tracking the true score."""

    game_id: str
    home_team_id: int
    away_team_id: int
    home_players: list[int]
    away_players: list[int]
    _rows: list[list[object]] = field(default_factory=list)
    _event_num: int = 0
    _home_score: int = 0
    _away_score: int = 0
    _period: int = 0
    _clock_remaining: int = 0
    _period_points: dict[int, dict[int, int]] = field(default_factory=dict)

    # -- low level -----------------------------------------------------------
    def _emit(
        self,
        msg_type: int,
        action_type: int,
        *,
        desc: str = "",
        desc_team: int = 0,
        neutral_desc: str | None = None,
        p1: int = 0,
        t1: int = 0,
        p2: int = 0,
        t2: int = 0,
        p3: int = 0,
        t3: int = 0,
        scoring: bool = False,
    ) -> None:
        self._event_num += 1
        row: dict[str, object] = dict.fromkeys(PBP_COLUMNS)
        row["GAME_ID"] = self.game_id
        row["EVENTNUM"] = self._event_num
        row["EVENTMSGTYPE"] = msg_type
        row["EVENTMSGACTIONTYPE"] = action_type
        row["PERIOD"] = self._period
        row["WCTIMESTRING"] = "8:00 PM"
        row["PCTIMESTRING"] = _clock(self._clock_remaining)
        if desc and desc_team == self.home_team_id:
            row["HOMEDESCRIPTION"] = desc
        elif desc:
            row["VISITORDESCRIPTION"] = desc
        row["NEUTRALDESCRIPTION"] = neutral_desc
        row["PLAYER1_ID"] = p1
        row["PLAYER2_ID"] = p2
        row["PLAYER3_ID"] = p3
        for slot, pid, tid in ((1, p1, t1), (2, p2, t2), (3, p3, t3)):
            if pid:
                row[f"PLAYER{slot}_NAME"] = f"Player {pid}"
                row[f"PLAYER{slot}_TEAM_ID"] = tid
                row[f"PERSON{slot}TYPE"] = 4 if tid == self.home_team_id else 5
        if scoring:
            row["SCORE"] = f"{self._away_score} - {self._home_score}"
            margin = self._home_score - self._away_score
            row["SCOREMARGIN"] = "TIE" if margin == 0 else str(margin)
        row["VIDEO_AVAILABLE_FLAG"] = 0
        self._rows.append([row[c] for c in PBP_COLUMNS])

    def _advance(self, seconds: int = 12) -> None:
        self._clock_remaining = max(self._clock_remaining - seconds, 0)

    def _credit(self, team_id: int, points: int) -> None:
        if team_id == self.home_team_id:
            self._home_score += points
        else:
            self._away_score += points
        bucket = self._period_points.setdefault(self._period, {})
        bucket[team_id] = bucket.get(team_id, 0) + points

    # -- period structure -------------------------------------------------
    def start_period(self, period: int) -> GameBuilder:
        self._period = period
        self._clock_remaining = REGULATION_SECONDS if period <= 4 else OVERTIME_SECONDS
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(
            period, f"OT{period - 4}"
        )
        self._emit(START_PERIOD, 0, neutral_desc=f"Start of {ordinal} Period")
        if period == 1 or period >= 5:
            self._emit(
                JUMP_BALL,
                0,
                neutral_desc="Jump Ball",
                p1=self.home_players[0],
                t1=self.home_team_id,
                p2=self.away_players[0],
                t2=self.away_team_id,
                p3=self.home_players[0],
                t3=self.home_team_id,
            )
        return self

    def end_period(self) -> GameBuilder:
        self._clock_remaining = 0
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(
            self._period, f"OT{self._period - 4}"
        )
        self._emit(END_PERIOD, 0, neutral_desc=f"End of {ordinal} Period")
        return self

    # -- plays ----------------------------------------------------------
    def make(
        self, team_id: int, player: int, points: int = 2, assist: int = 0
    ) -> GameBuilder:
        self._advance()
        self._credit(team_id, points)
        text = f"Player {player} {points}PT Shot"
        if assist:
            text += f" ({assist} AST)"
        self._emit(
            MADE_SHOT,
            1 if points == 2 else 79,
            desc=text,
            desc_team=team_id,
            p1=player,
            t1=team_id,
            p2=assist,
            t2=team_id if assist else 0,
            scoring=True,
        )
        return self

    def miss(self, team_id: int, player: int, points: int = 2) -> GameBuilder:
        self._advance()
        self._emit(
            MISSED_SHOT,
            1 if points == 2 else 79,
            desc=f"MISS Player {player} {points}PT Shot",
            desc_team=team_id,
            p1=player,
            t1=team_id,
        )
        return self

    def rebound(
        self, team_id: int, player: int, offensive: bool = False
    ) -> GameBuilder:
        self._advance(2)
        kind = "Off" if offensive else "Def"
        self._emit(
            REBOUND,
            0,
            desc=f"Player {player} REBOUND ({kind})",
            desc_team=team_id,
            p1=player,
            t1=team_id,
        )
        return self

    def turnover(self, team_id: int, player: int, steal_by: int = 0) -> GameBuilder:
        self._advance()
        other = self.away_team_id if team_id == self.home_team_id else self.home_team_id
        self._emit(
            TURNOVER,
            1 if not steal_by else 2,
            desc=f"Player {player} Turnover",
            desc_team=team_id,
            p1=player,
            t1=team_id,
            p2=steal_by,
            t2=other if steal_by else 0,
        )
        return self

    def foul(self, team_id: int, player: int, drew: int) -> GameBuilder:
        self._advance(1)
        other = self.away_team_id if team_id == self.home_team_id else self.home_team_id
        self._emit(
            FOUL,
            2,  # shooting foul
            desc=f"Player {player} S.FOUL",
            desc_team=team_id,
            p1=player,
            t1=team_id,
            p2=drew,
            t2=other,
        )
        return self

    def free_throw(
        self, team_id: int, player: int, index: int, of: int, made: bool = True
    ) -> GameBuilder:
        self._advance(0)
        action = {
            (1, 1): 10,
            (1, 2): 11,
            (2, 2): 12,
            (1, 3): 13,
            (2, 3): 14,
            (3, 3): 15,
        }[(index, of)]
        prefix = "" if made else "MISS "
        if made:
            self._credit(team_id, 1)
        self._emit(
            FREE_THROW,
            action,
            desc=f"{prefix}Player {player} Free Throw {index} of {of}",
            desc_team=team_id,
            p1=player,
            t1=team_id,
            scoring=made,
        )
        return self

    def technical_free_throw(
        self, team_id: int, player: int, made: bool = True
    ) -> GameBuilder:
        self._advance(0)
        prefix = "" if made else "MISS "
        if made:
            self._credit(team_id, 1)
        self._emit(
            FREE_THROW,
            16,
            desc=f"{prefix}Player {player} Free Throw Technical",
            desc_team=team_id,
            p1=player,
            t1=team_id,
            scoring=made,
        )
        return self

    def technical_foul(self, team_id: int, player: int) -> GameBuilder:
        self._advance(1)
        self._emit(
            FOUL,
            11,
            desc=f"Player {player} Technical",
            desc_team=team_id,
            p1=player,
            t1=team_id,
        )
        return self

    def substitution(
        self, team_id: int, out_player: int, in_player: int
    ) -> GameBuilder:
        self._advance(0)
        roster = (
            self.home_players if team_id == self.home_team_id else self.away_players
        )
        roster[:] = [in_player if p == out_player else p for p in roster]
        self._emit(
            SUBSTITUTION,
            0,
            desc=f"SUB: Player {in_player} FOR Player {out_player}",
            desc_team=team_id,
            p1=out_player,
            t1=team_id,
            p2=in_player,
            t2=team_id,
        )
        return self

    def timeout(self, team_id: int) -> GameBuilder:
        self._advance(0)
        self._emit(TIMEOUT, 1, desc="Timeout", desc_team=team_id)
        return self

    # -- output -------------------------------------------------------
    def pbp_payload(self) -> dict[str, object]:
        return {
            "resource": "playbyplayv2",
            "parameters": {"GameID": self.game_id},
            "resultSets": [
                {
                    "name": "PlayByPlay",
                    "headers": PBP_COLUMNS,
                    "rowSet": list(self._rows),
                }
            ],
        }

    def final_score(self) -> dict[int, int]:
        return {
            self.home_team_id: self._home_score,
            self.away_team_id: self._away_score,
        }

    def period_scores(self) -> dict[int, list[int]]:
        periods = sorted(self._period_points)
        return {
            team: [self._period_points.get(p, {}).get(team, 0) for p in periods]
            for team in (self.home_team_id, self.away_team_id)
        }


def _empty_shot_chart() -> dict[str, object]:
    return {
        "resource": "shotchartdetail",
        "parameters": {},
        "resultSets": [
            {
                "name": "Shot_Chart_Detail",
                "headers": ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "LOC_X", "LOC_Y"],
                "rowSet": [],
            }
        ],
    }


@dataclass
class GameSpec:
    builder: GameBuilder
    game_date: str
    season: str = "2023-24"
    season_type: str = "Regular Season"
    starters: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    days_rest: dict[int, int] | None = None
    # optional deliberate metadata corruption for failure tests
    final_score_override: dict[int, int] | None = None


def write_snapshot(path: str | Path, specs: list[GameSpec]) -> Path:
    root = Path(path)
    (root / "pbp").mkdir(parents=True, exist_ok=True)
    (root / "game_details").mkdir(parents=True, exist_ok=True)
    overrides: dict[str, dict[int, dict[int, list[int]]]] = {}
    games_meta: list[dict[str, object]] = []

    for spec in specs:
        b = spec.builder
        (root / "pbp" / f"stats_{b.game_id}.json").write_text(
            json.dumps(b.pbp_payload()), encoding="utf-8"
        )
        for side in ("home", "away"):
            (root / "game_details" / f"stats_{side}_shots_{b.game_id}.json").write_text(
                json.dumps(_empty_shot_chart()), encoding="utf-8"
            )
        if spec.starters:
            overrides[b.game_id] = spec.starters
        days_rest = spec.days_rest or {b.home_team_id: 2, b.away_team_id: 2}
        final_score = spec.final_score_override or b.final_score()
        games_meta.append(
            {
                "game_id": b.game_id,
                "game_date": spec.game_date,
                "season": spec.season,
                "season_type": spec.season_type,
                "home_team_id": b.home_team_id,
                "away_team_id": b.away_team_id,
                "days_rest": {str(k): v for k, v in days_rest.items()},
                "reconciliation": {
                    "final_score": {str(k): v for k, v in final_score.items()},
                    "period_scores": {str(k): v for k, v in b.period_scores().items()},
                },
            }
        )

    if overrides:
        (root / "overrides").mkdir(exist_ok=True)
        (root / "overrides" / "missing_period_starters.json").write_text(
            json.dumps(
                {
                    gid: {
                        str(period): {str(team): ids for team, ids in teams.items()}
                        for period, teams in periods.items()
                    }
                    for gid, periods in overrides.items()
                }
            ),
            encoding="utf-8",
        )

    (root / "courtgraph_snapshot.json").write_text(
        json.dumps({"snapshot_format": "stats_nba_pbpstats/v1", "games": games_meta}),
        encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------- #
# Named scenarios
# --------------------------------------------------------------------------- #

HOME_TEAM = 1610612739
AWAY_TEAM = 1610612744
HOME = [101, 102, 103, 104, 105]
AWAY = [201, 202, 203, 204, 205]
HOME_BENCH = [106, 107, 108, 109, 110]
AWAY_BENCH = [206, 207, 208, 209, 210]


def _fresh_builder(game_id: str) -> GameBuilder:
    return GameBuilder(
        game_id=game_id,
        home_team_id=HOME_TEAM,
        away_team_id=AWAY_TEAM,
        home_players=list(HOME),
        away_players=list(AWAY),
    )


def _all_period_starters(
    periods: list[int],
    home_by_period: dict[int, list[int]],
    away_by_period: dict[int, list[int]],
) -> dict[int, dict[int, list[int]]]:
    return {
        p: {HOME_TEAM: list(home_by_period[p]), AWAY_TEAM: list(away_by_period[p])}
        for p in periods
    }


def ordinary_game(
    game_id: str = "0022300001", game_date: str = "2023-10-24"
) -> GameSpec:
    """Two regulation periods, an offensive rebound, a mid-game substitution."""

    b = _fresh_builder(game_id)
    b.start_period(1)
    b.make(HOME_TEAM, 102, assist=101)
    b.make(AWAY_TEAM, 202, assist=201)
    b.miss(HOME_TEAM, 103)
    b.rebound(HOME_TEAM, 104, offensive=True)  # offensive rebound -> same possession
    b.make(HOME_TEAM, 104)
    b.turnover(AWAY_TEAM, 205, steal_by=105)
    b.make(HOME_TEAM, 105, points=3, assist=103)
    b.miss(AWAY_TEAM, 203)
    b.rebound(HOME_TEAM, 101)
    b.make(HOME_TEAM, 101)
    b.make(AWAY_TEAM, 204, assist=203)
    b.end_period()

    b.start_period(2)
    b.make(AWAY_TEAM, 201, points=3)
    b.substitution(HOME_TEAM, 105, 106)
    b.make(HOME_TEAM, 102)
    b.turnover(AWAY_TEAM, 202)
    b.make(HOME_TEAM, 106, assist=101)
    b.miss(AWAY_TEAM, 203)
    b.rebound(AWAY_TEAM, 204)
    b.make(AWAY_TEAM, 204)
    b.make(HOME_TEAM, 103)
    b.make(AWAY_TEAM, 205, assist=201)
    b.end_period()

    starters = _all_period_starters(
        [1, 2],
        {1: HOME, 2: HOME},
        {1: AWAY, 2: AWAY},
    )
    return GameSpec(
        builder=b,
        game_date=game_date,
        starters=starters,
        days_rest={HOME_TEAM: 3, AWAY_TEAM: 2},
    )


def free_throw_game(
    game_id: str = "0022300002", game_date: str = "2023-10-26"
) -> GameSpec:
    """One period featuring a shooting foul + two free throws and a technical."""

    b = _fresh_builder(game_id)
    b.start_period(1)
    b.make(HOME_TEAM, 101, assist=102)  # HOME won the tip -> HOME first
    # AWAY possession: HOME fouls the AWAY shooter -> AWAY shoots two FTs
    b.foul(HOME_TEAM, 103, drew=202)
    b.free_throw(AWAY_TEAM, 202, 1, 2, made=True)
    b.free_throw(AWAY_TEAM, 202, 2, 2, made=False)
    b.rebound(HOME_TEAM, 104)  # HOME rebounds the missed 2nd FT -> HOME possession
    b.make(HOME_TEAM, 104)
    # AWAY possession, interrupted by a technical on AWAY; HOME shoots the tech FT
    b.technical_foul(AWAY_TEAM, 205)
    b.technical_free_throw(HOME_TEAM, 101, made=True)
    b.make(AWAY_TEAM, 201, assist=203)
    b.miss(HOME_TEAM, 102)
    b.rebound(AWAY_TEAM, 203)
    b.make(AWAY_TEAM, 203)
    b.make(HOME_TEAM, 105, assist=104)
    b.end_period()
    starters = _all_period_starters([1], {1: HOME}, {1: AWAY})
    return GameSpec(builder=b, game_date=game_date, starters=starters)


def overtime_game(
    game_id: str = "0022300003", game_date: str = "2023-10-28"
) -> GameSpec:
    """One regulation period plus an overtime period."""

    b = _fresh_builder(game_id)
    b.start_period(4)
    b.make(HOME_TEAM, 101)
    b.make(AWAY_TEAM, 201)
    b.make(HOME_TEAM, 102)
    b.make(AWAY_TEAM, 202)
    b.turnover(HOME_TEAM, 103)
    b.make(AWAY_TEAM, 203)
    b.make(HOME_TEAM, 104, points=3)
    b.miss(AWAY_TEAM, 204)
    b.rebound(HOME_TEAM, 105)
    b.make(HOME_TEAM, 105)
    b.make(AWAY_TEAM, 205, points=3, assist=201)
    b.end_period()

    b.start_period(5)
    b.make(HOME_TEAM, 101, assist=102)  # HOME won the OT tip -> HOME first
    b.make(AWAY_TEAM, 201)
    b.substitution(AWAY_TEAM, 205, 206)  # sub at a dead ball -> not a split
    b.make(HOME_TEAM, 102)
    b.turnover(AWAY_TEAM, 206, steal_by=101)
    b.make(HOME_TEAM, 103, points=3)
    b.make(AWAY_TEAM, 201)
    b.end_period()

    starters = _all_period_starters([4, 5], {4: HOME, 5: HOME}, {4: AWAY, 5: AWAY})
    return GameSpec(builder=b, game_date=game_date, starters=starters)


def split_lineup_game(
    game_id: str = "0022300004", game_date: str = "2023-10-30"
) -> GameSpec:
    """A substitution *between live events inside a possession* (offensive rebound
    then a sub then the putback) -- that possession must be quarantined."""

    b = _fresh_builder(game_id)
    b.start_period(1)
    b.make(HOME_TEAM, 101)
    b.make(AWAY_TEAM, 201)
    b.miss(HOME_TEAM, 102)
    b.rebound(HOME_TEAM, 103, offensive=True)
    b.substitution(HOME_TEAM, 104, 106)  # sub mid-possession
    b.make(HOME_TEAM, 103)
    b.make(AWAY_TEAM, 202)
    b.make(HOME_TEAM, 106)
    b.make(AWAY_TEAM, 203)
    b.end_period()
    starters = _all_period_starters([1], {1: HOME}, {1: AWAY})
    return GameSpec(builder=b, game_date=game_date, starters=starters)


def returning_player_split_game(
    game_id: str = "0022300006", game_date: str = "2023-11-03"
) -> GameSpec:
    """Possessions 3 (HOME) and 4 (AWAY) each sub a player out, let a substitute
    take part in live play, then sub the original back **before the shot** --
    the possession's ten never differs first-vs-last but is still ambiguous, so
    both must be excluded. Accepted possessions 1-2 and 5-6 share the same five
    but sit on opposite sides of that excluded gap: they must **not** merge.
    """

    b = _fresh_builder(game_id)
    b.start_period(1)
    # spell 1 (accepted): starters on both sides
    b.make(HOME_TEAM, 101, assist=102)
    b.make(AWAY_TEAM, 201, assist=202)
    # possession 3 (HOME): out-and-back around live play -> excluded
    b.miss(HOME_TEAM, 102)
    b.rebound(HOME_TEAM, 103, offensive=True)
    b.substitution(HOME_TEAM, 104, 106)
    b.miss(HOME_TEAM, 106)
    b.rebound(HOME_TEAM, 105, offensive=True)
    b.substitution(HOME_TEAM, 106, 104)
    b.make(HOME_TEAM, 103)
    # possession 4 (AWAY): same shape -> excluded
    b.miss(AWAY_TEAM, 202)
    b.rebound(AWAY_TEAM, 203, offensive=True)
    b.substitution(AWAY_TEAM, 204, 206)
    b.miss(AWAY_TEAM, 206)
    b.rebound(AWAY_TEAM, 205, offensive=True)
    b.substitution(AWAY_TEAM, 206, 204)
    b.make(AWAY_TEAM, 203)
    # spell 2 (accepted): identical starting fives return
    b.make(HOME_TEAM, 102, assist=101)
    b.make(AWAY_TEAM, 201, assist=203)
    b.end_period()
    starters = _all_period_starters([1], {1: HOME}, {1: AWAY})
    return GameSpec(builder=b, game_date=game_date, starters=starters)


def noncontiguous_lineup_game(
    game_id: str = "0022300005", game_date: str = "2023-11-01"
) -> GameSpec:
    """The HOME starting five plays, is broken up by a substitution, then the
    identical five returns later. The two spells must be **separate** stints."""

    b = _fresh_builder(game_id)
    b.start_period(1)
    # spell 1: starters 101-105
    b.make(HOME_TEAM, 101, assist=102)
    b.make(AWAY_TEAM, 201)
    b.make(HOME_TEAM, 103)
    b.make(AWAY_TEAM, 202)
    # break the five up
    b.substitution(HOME_TEAM, 105, 106)
    b.make(HOME_TEAM, 106)
    b.make(AWAY_TEAM, 203)
    b.turnover(HOME_TEAM, 101)
    b.make(AWAY_TEAM, 204)
    # bring the identical starting five back
    b.substitution(HOME_TEAM, 106, 105)
    b.make(HOME_TEAM, 102, assist=101)
    b.make(AWAY_TEAM, 205)
    b.make(HOME_TEAM, 104)
    b.make(AWAY_TEAM, 201)
    b.end_period()
    starters = _all_period_starters([1], {1: HOME}, {1: AWAY})
    return GameSpec(builder=b, game_date=game_date, starters=starters)


def rotation_season(n_games: int = 4) -> list[GameSpec]:
    """Several small games with rotating lineups -- enough distinct stints to fit
    the existing chemistry model end to end."""

    specs: list[GameSpec] = []
    for g in range(n_games):
        gid = f"002230010{g}"
        b = _fresh_builder(gid)
        for period in (1, 2):
            b.start_period(period)
            home_on = list(HOME)
            away_on = list(AWAY)
            for step in range(9):
                shooter_h = home_on[step % 5]
                shooter_a = away_on[(step + g) % 5]
                b.make(
                    HOME_TEAM,
                    shooter_h,
                    points=2 + (step % 2),
                    assist=home_on[(step + 1) % 5],
                )
                b.make(AWAY_TEAM, shooter_a, points=2, assist=away_on[(step + 2) % 5])
                if step % 2 == 1:
                    out_h = home_on[step % 5]
                    in_h = HOME_BENCH[(step + g) % 5]
                    if in_h not in home_on:
                        b.substitution(HOME_TEAM, out_h, in_h)
                        home_on[step % 5] = in_h
                    out_a = away_on[(step + 1) % 5]
                    in_a = AWAY_BENCH[(step + g) % 5]
                    if in_a not in away_on:
                        b.substitution(AWAY_TEAM, out_a, in_a)
                        away_on[(step + 1) % 5] = in_a
            b.end_period()
        specs.append(
            GameSpec(
                builder=b,
                game_date=f"2023-11-{2 * g + 1:02d}",
                starters=_all_period_starters(
                    [1, 2], {1: HOME, 2: HOME}, {1: AWAY, 2: AWAY}
                ),
                days_rest={HOME_TEAM: 2 + g % 3, AWAY_TEAM: 1 + g % 4},
            )
        )
    return specs
