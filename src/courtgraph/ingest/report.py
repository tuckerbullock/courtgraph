"""A single readable HTML report for one ``courtgraph ingest`` run.

Reads the run's ``manifest.json`` / ``stints.jsonl`` / ``quarantine.jsonl`` and,
when the snapshot carries a ``display_names.json`` sidecar, resolves team and
player ids to names. Self-contained (inline CSS, no scripts, no external
assets). Shows what was reconstructed, the score check, the stints with real
lineups, and every exclusion -- and states plainly that a small run like this
is not evidence of predictive accuracy.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtgraph.ingest._paths import (
    OutputPathError,
    assert_not_symlink,
    reject_overlap,
    writable,
)

_CSS = "\n".join(
    (
        "body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;",
        "margin:2rem auto;max-width:60rem;color:#1a1a1a;padding:0 1rem}",
        "h1{font-size:1.5rem;margin-bottom:.2rem}",
        "h2{font-size:1.2rem;margin-top:2rem;border-bottom:2px solid #ddd;",
        "padding-bottom:.2rem}",
        "h3{font-size:1rem;margin:1.2rem 0 .4rem}",
        ".banner{background:#fff4e5;border:1px solid #f0c060;border-radius:6px;",
        "padding:.8rem 1rem;margin:1rem 0}",
        ".muted{color:#666}code{background:#f2f2f2;padding:.1rem .3rem;",
        "border-radius:3px}",
        "table{border-collapse:collapse;margin:.4rem 0;font-size:.92rem}",
        "th,td{border:1px solid #ccc;padding:.25rem .55rem;text-align:left}",
        "th{background:#f5f5f5}",
        "td.n{text-align:right;font-variant-numeric:tabular-nums}",
        ".ok{color:#1a7f37;font-weight:600}.warn{color:#b35900;font-weight:600}",
        ".pill{display:inline-block;background:#eee;border-radius:10px;",
        "padding:0 .5rem;font-size:.85rem}",
    )
)


@dataclass(frozen=True)
class _Names:
    teams: dict[str, str]
    players: dict[str, str]

    def team(self, team_id: int | str) -> str:
        return self.teams.get(str(team_id), f"team {team_id}")

    def player(self, player_id: int | str) -> str:
        return self.players.get(str(player_id), str(player_id))

    def lineup(self, ids: list[int]) -> str:
        return ", ".join(sorted(self.player(p) for p in ids))


def _load_names(snapshot_dir: Path | None) -> _Names:
    if snapshot_dir is not None:
        sidecar = snapshot_dir / "display_names.json"
        if sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return _Names(
                teams={str(k): str(v) for k, v in data.get("teams", {}).items()},
                players={str(k): str(v) for k, v in data.get("players", {}).items()},
            )
    return _Names(teams={}, players={})


def _esc(value: Any) -> str:
    return html.escape(str(value))


def render_report(
    ingest_out_dir: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
    title: str = "CourtGraph ingest — real-game demonstration",
) -> str:
    out = Path(ingest_out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stints = [
        json.loads(line)
        for line in (out / "stints.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = _load_names(Path(snapshot_dir) if snapshot_dir is not None else None)

    stints_by_game: dict[str, list[dict[str, Any]]] = {}
    for stint in stints:
        stints_by_game.setdefault(stint["game_id"], []).append(stint)

    parts: list[str] = [
        f"<style>{_CSS}</style>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='muted'>Generated {_esc(manifest['created_utc'])} · "
        f"parser {_esc(manifest['parser']['tool'])} "
        f"{_esc(manifest['parser']['version'])} (file mode) · "
        f"policy {_esc(manifest['policy']['policy_version'])}</p>",
        "<div class='banner'><b>Real NBA play-by-play, small demonstration.</b> "
        "Source: a local <code>SRC-SHUFINSKIY</code> archive (a re-packaging of "
        "stats.nba.com / data.nba.com; <code>DATA_SOURCES.md</code> §1 — "
        "local-dev-only, not redistributable). The score-check source is stated "
        "per game below (an operator-supplied official box score when provided, "
        "otherwise the data.nba.com game feed — a second NBA surface, not an "
        "independent provider). A handful of games is <b>not</b> evidence of "
        "predictive accuracy, calibration, or data quality at scale.</div>",
    ]

    provenance = manifest.get("source_provenance") or {}
    if provenance:
        rows = [
            ["converter version", provenance.get("converter_version", "?")],
            [
                "pinned shufinskiy commit",
                provenance.get("pinned_commit") or "(not recorded)",
            ],
        ]
        for name, digest in sorted(
            (provenance.get("consumed_csv_sha256") or {}).items()
        ):
            rows.append([f"sha256 {name}", digest])
        parts.append("<h2>Source provenance</h2>")
        parts.append(_table(["item", "value"], rows))

    totals = manifest["totals"]
    parts.append("<h2>Run totals</h2>")
    parts.append(
        _table(
            ["metric", "value"],
            [
                ["games in", totals["games_in"]],
                ["games accepted", totals["games_accepted"]],
                ["games quarantined", totals["games_quarantined"]],
                ["possessions reconstructed", totals["possessions_reconstructed"]],
                ["possessions accepted", totals["possessions_accepted"]],
                ["possessions excluded", totals["possessions_excluded"]],
                ["stints emitted", totals["stints_emitted"]],
            ],
            numeric={1},
        )
    )

    for game in manifest["games"]:
        parts.extend(_render_game(game, stints_by_game.get(game["game_id"], []), names))

    parts.append("<h2>Limitations of this demonstration</h2>")
    parts.append(
        "<ul>"
        "<li>One archive, one series — nothing here is validated at season scale.</li>"
        "<li>Unless an operator supplies official box-score totals, the score "
        "check is <b>within-NBA</b> (stats.nba.com possessions vs the "
        "data.nba.com feed); a truly independent lineage is still unavailable "
        "(<code>DATA_SOURCES.md</code> §5.2).</li>"
        "<li>Rest days use the validated <code>GAME_DATE</code>; box-score "
        "minutes and lineup minutes are not reconciled.</li>"
        "<li>Games <code>pbpstats</code> cannot order, or whose period starters "
        "need a box-score request, are quarantined here rather than patched — "
        "adding <code>overrides/</code> files would recover them.</li>"
        "<li>No model is fit or evaluated in this report.</li>"
        "</ul>"
    )
    return "\n".join(parts)


def _render_game(
    game: dict[str, Any], stints: list[dict[str, Any]], names: _Names
) -> list[str]:
    gid = game["game_id"]
    recon = game.get("reconciliation") or {}
    official = recon.get("final_score_official", {})
    derived = recon.get("final_score_derived", {})
    out: list[str] = ["<h2>Game " + _esc(gid) + "</h2>"]

    if game["status"] != "accepted":
        excl = game["excluded_possessions"]
        detail = _esc(excl[0]["detail"]) if excl else ""
        out.append(
            f"<p><span class='warn'>quarantined</span> — "
            f"<code>{_esc(game['quarantine_reason'])}</code>. {detail}</p>"
        )
        return out

    teams = sorted(
        {int(s["offense_team_id"]) for s in stints}
        | {int(s["defense_team_id"]) for s in stints}
    )
    home = teams[0] if teams else 0
    away = teams[1] if len(teams) > 1 else 0
    for stint in stints:
        if stint["home_offense"]:
            home, away = int(stint["offense_team_id"]), int(stint["defense_team_id"])
            break
    playoff = any(s.get("playoff") for s in stints)
    pills = (
        f"<span class='pill'>{game['reconstructed_possessions']} reconstructed</span> "
        f"<span class='pill'>{game['accepted_possessions']} accepted</span> "
        f"<span class='pill'>{game['stints_emitted']} stints</span>"
    )
    out.append(
        f"<p><b>{_esc(names.team(away))}</b> at <b>{_esc(names.team(home))}</b> · "
        f"{_esc(game['game_date'])} · {_esc(game['season'])}"
        f"{' playoffs' if playoff else ''} · {pills}</p>"
    )

    matched = recon.get("final_score_matched")
    verdict = (
        "<span class='ok'>matches</span>"
        if matched
        else "<span class='warn'>does not match</span>"
    )
    score_source = recon.get("official_score_source", "unspecified")
    rows = []
    for team in sorted(official, key=lambda k: -official[k]):
        rows.append(
            [
                names.team(int(team)),
                official.get(team, "?"),
                derived.get(team, "?"),
                int(derived.get(team, 0)) - int(official.get(team, 0)),
            ]
        )
    out.append(f"<h3>Score check — reconstructed vs recorded total: {verdict}</h3>")
    out.append(f"<p class='muted'>Score-check source: {_esc(score_source)}</p>")
    out.append(
        _table(
            ["team", "recorded total", "reconstructed", "delta"],
            rows,
            numeric={1, 2, 3},
        )
    )
    per_period = recon.get("period_score_delta", {})
    bad_periods = [p for p, d in per_period.items() if any(v != 0 for v in d.values())]
    out.append(
        "<p class='muted'>Per-period deltas: "
        + (
            "all zero."
            if not bad_periods
            else "non-zero in period(s) " + ", ".join(sorted(bad_periods)) + "."
        )
        + "</p>"
    )

    out.append("<h3>Stints — busiest lineups (offensive possessions)</h3>")
    ranked = sorted(stints, key=lambda s: -s["offensive_possessions"])[:8]
    out.append(
        _table(
            ["off. team", "lineup", "poss", "pts", "pts/100", "gt wt"],
            [
                [
                    names.team(s["offense_team_id"]),
                    names.lineup(s["offense_player_ids"]),
                    s["offensive_possessions"],
                    s["points_scored"],
                    round(100 * s["points_scored"] / s["offensive_possessions"], 1),
                    s["garbage_time_weight"],
                ]
                for s in ranked
            ],
            numeric={2, 3, 4, 5},
        )
    )
    poss: Counter[int] = Counter()
    pts: Counter[int] = Counter()
    for stint in stints:
        poss[int(stint["offense_team_id"])] += stint["offensive_possessions"]
        pts[int(stint["offense_team_id"])] += stint["points_scored"]
    gt = sum(1 for s in stints if s["garbage_time_weight"] < 1.0)
    out.append(
        _table(
            ["team", "stint offensive possessions", "stint-attributed points"],
            [[names.team(t), poss[t], pts[t]] for t in sorted(poss)],
            numeric={1, 2},
        )
    )
    out.append(
        f"<p class='muted'>{gt} stint(s) carry a garbage-time weight below 1.0. "
        "Stint-attributed points are below the final score because possessions "
        "with a mid-possession substitution (below) are reconciled but not "
        "assigned to a stint.</p>"
    )

    excl = game["excluded_possessions"]
    if excl:
        by_reason = Counter(e["reason"] for e in excl)
        out.append("<h3>Excluded possessions</h3>")
        out.append(
            _table(
                ["reason", "count"],
                [[r, c] for r, c in sorted(by_reason.items())],
                numeric={1},
            )
        )
        out.append(
            "<details><summary class='muted'>per-possession detail</summary>"
            + _table(
                ["period", "possession", "reason"],
                [
                    [e.get("period", ""), e.get("possession_number", ""), e["reason"]]
                    for e in excl
                ],
                numeric={0, 1},
            )
            + "</details>"
        )
    return out


def _table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    numeric: set[int] | None = None,
) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f"<td class='n'>{_esc(v)}</td>" if i in numeric else f"<td>{_esc(v)}</td>"
            for i, v in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def write_report(
    ingest_out_dir: str | Path,
    report_path: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
) -> Path:
    path = Path(report_path)
    # The report is derived output: never write it through a symlink, never
    # inside the immutable snapshot, and never on top of an ingest output file.
    assert_not_symlink(path)
    if snapshot_dir is not None:
        reject_overlap(
            Path(snapshot_dir), path, in_label="--snapshot-dir", out_label="--report"
        )
    resolved = path.resolve()
    protected_outputs = {
        (Path(ingest_out_dir) / name).resolve()
        for name in ("manifest.json", "stints.jsonl", "quarantine.jsonl", ".gitignore")
    }
    if resolved in protected_outputs:
        raise OutputPathError(f"--report would overwrite an ingest output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writable(path).write_text(
        render_report(ingest_out_dir, snapshot_dir=snapshot_dir), encoding="utf-8"
    )
    return path
