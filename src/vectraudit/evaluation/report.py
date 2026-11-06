"""Render check results as a self-contained HTML report or a Markdown summary.

Every value from a run (query ids, document ids, labels, backend names) is escaped: run
files can come from untrusted CI jobs. The HTML report has no scripts or external assets.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit import __version__
from vectraudit.evaluation.compare import execution_label
from vectraudit.evaluation.evidence import describe_evidence

NONE = "\u2013"  # en dash for missing values

_VERDICT_TEXT = {
    "pass": "Pass",
    "fail": "Fail",
    "error": "Error",
    "not_comparable": "Not comparable",
    "accepted": "Fail, accepted by exception",
}
_VERDICT_EMOJI = {
    "pass": "✅",
    "fail": "❌",
    "error": "⚠️",
    "not_comparable": "⏸️",
    "accepted": "🟡",
}
_METRIC_NAMES = {
    "recall": "Recall@k",
    "precision": "Precision@k",
    "mrr": "MRR@k",
    "ndcg": "nDCG@k",
    "ann_recall": "ANN recall@k",
}


def _num(x: Any, digits: int = 4, signed: bool = False) -> str:
    if x is None:
        return NONE
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _ids(ids: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" … (+{len(ids) - limit})" if len(ids) > limit else "")


def _metric_rows(metrics: Mapping[str, Any]) -> list[list[str]]:
    return [
        [
            _METRIC_NAMES.get(name, name),
            _num(m["baseline"]),
            _num(m["candidate"]),
            _num(m["delta"], signed=True),
            f"[{_num(m['ci_low'], signed=True)}, {_num(m['ci_high'], signed=True)}]",
            str(m["n"]),
            str(m.get("n_families", m["n"])),
        ]
        for name, m in metrics.items()
    ]


PAIRED_NOTE = (
    "Latency recorded interleaved (paired execution): each question ran on both sides back to "
    "back, alternating which went first, so drift during the run affected both alike."
)


def _latency_rows(latency: Mapping[str, Any] | None) -> list[list[str]]:
    if not latency:
        return []
    rows = []
    for p in ("p50", "p95"):
        change = latency.get(f"{p}_change_pct")
        rows.append(
            [
                p,
                f"{latency['baseline'][p]:.2f} ms",
                f"{latency['candidate'][p]:.2f} ms",
                NONE if change is None else f"{change:+.1f}%",
            ]
        )
    return rows


# --- Markdown ----------------------------------------------------------------------------


def _accepted_note(reason: Mapping[str, Any]) -> str:
    """`` (accepted by X until 2026-11-01: why)`` for a failure an exception covers."""
    accepted = reason.get("accepted")
    if not accepted:
        return ""
    return (
        f" (accepted by {accepted['accepted_by']} until {str(accepted['expires_at'])[:10]}: "
        f"{accepted['reason']})"
    )


def _md(text: Any) -> str:
    """Escape a value for a Markdown table cell (GitHub renders inline HTML)."""
    s = html.escape(str(text), quote=False)
    return s.replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_md(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


_TOLERANCE = {
    "within_tolerance": "shown within tolerance",
    "inconclusive": "inconclusive (interval reaches past the allowed drop)",
    "regression": "beyond tolerance",
}


def tolerance_line(result: Mapping[str, Any]) -> str | None:
    """``Non-inferiority: nDCG shown within tolerance; recall inconclusive (...).``"""
    parts = [
        f"{_METRIC_NAMES.get(name, name)} {_TOLERANCE[m['tolerance']['decision']]}"
        for name, m in (result.get("metrics") or {}).items()
        if m.get("tolerance")
    ]
    if not parts:
        return None
    return (
        "Non-inferiority (pass only when the whole 95% interval stays above the allowed "
        "drop): " + "; ".join(parts) + "."
    )


def cohort_line(result: Mapping[str, Any]) -> str | None:
    """``Cohorts: 4 of 6 judged on 8 rules, 1 failed; no multiplicity control.``"""
    judging = result.get("cohort_judging")
    if not judging or not judging["judged_rules"]:
        return None
    control = {
        "bonferroni": "each rule widened for the number judged (Bonferroni)",
        "holm": "p-values adjusted for the number judged (Holm)",
    }.get(judging["control"], "no multiplicity control")
    return (
        f"Cohorts: {judging['judged_cohorts']} of {judging['cohorts']} judged on "
        f"{judging['judged_rules']} rules, {judging['failed_rules']} failed; {control}."
    )


def render_markdown(result: Mapping[str, Any], *, report_url: str | None = None) -> str:
    """A compact summary for ``$GITHUB_STEP_SUMMARY`` or a pull-request comment."""
    verdict = result["verdict"]
    b, c = result["baseline"], result["candidate"]
    out = [
        f"## {_VERDICT_EMOJI[verdict]} Retrieval check: {_VERDICT_TEXT[verdict]}",
        "",
        f"Baseline **{_md(b['label'] or b['run_id'])}** ({_md(b['backend'])}) vs candidate "
        f"**{_md(c['label'] or c['run_id'])}** ({_md(c['backend'])}), "
        f"{result['n_queries']} query rows / "
        f"{result.get('n_families', result['n_queries'])} families, k = {_md(c['k'])}.",
        "",
    ]
    if result.get("repeat_quality"):
        out += [_repeat_note(result), ""]
    release = _release_line(result)
    if release:
        out += [release, ""]
    if result["reasons"]:
        out += [f"- {_md(r['message'])}{_accepted_note(r)}" for r in result["reasons"]] + [""]
    if result.get("critical_queries"):
        out += _critical_section(result, markdown=True)
    findings = result.get("findings") or []
    if findings:
        out += ["**Diagnosis**", ""]
        for f in findings[:3]:
            line = f"- _{_md(f['level'])}_: {_md(f['title'])}"
            if f.get("suggestion"):
                line += f". {_md(f['suggestion'])}"
            out.append(line)
        if len(findings) > 3:
            out.append(f"- …and {len(findings) - 3} more in the full report")
        out.append("")
    if result["metrics"]:
        out += [
            _md_table(
                ["Metric", "Baseline", "Candidate", "Change", "95% CI", "Rows", "Families"],
                _metric_rows(result["metrics"]),
            ),
            "",
        ]
    latency = _latency_rows(result["latency"])
    if latency:
        out += [_md_table(["Latency", "Baseline", "Candidate", "Change"], latency), ""]
        if (result["latency"] or {}).get("paired"):
            out += [PAIRED_NOTE, ""]
    if tolerance := tolerance_line(result):
        out += [tolerance, ""]
    if cohorts := cohort_line(result):
        out += [cohorts, ""]
    queries = result.get("queries")
    if queries and queries["regressed"]:
        out += [
            f"**{queries['regressed_count']} regressed, {queries['improved_count']} improved** "
            f"on {_METRIC_NAMES.get(queries['primary_metric'], queries['primary_metric'])}. "
            "Worst:",
            "",
            _md_table(
                ["Query", "Baseline", "Candidate", "Lost ids"],
                [
                    [
                        q["query_id"],
                        _num(q["baseline"], 3),
                        _num(q["candidate"], 3),
                        _ids(q["lost_ids"]),
                    ]
                    for q in queries["regressed"][:10]
                ],
            ),
            "",
        ]
    if result["warnings"]:
        out += ["<details><summary>Warnings</summary>", ""]
        out += [f"- {_md(w)}" for w in result["warnings"]] + ["", "</details>", ""]
    if report_url:
        out += [f"[Full report]({report_url})", ""]
    out.append(f"<sub>vectraudit {__version__}</sub>")
    return "\n".join(out) + "\n"


# --- HTML --------------------------------------------------------------------------------

_CSS = """
:root { --bg:#ffffff; --fg:#1b1f24; --muted:#5b636d; --line:#d9dde3; --card:#f6f8fa;
  --pass:#1a7f37; --fail:#cf222e; --error:#9a6700; --nc:#6e7781; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1;
  --line:#30363d; --card:#161b22; --pass:#3fb950; --fail:#f85149; --error:#d29922; --nc:#8b949e; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.muted { color:var(--muted); }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; color:#fff;
  font-weight:600; font-size:13px; vertical-align:middle; }
.pass{background:var(--pass)} .fail{background:var(--fail)} .error{background:var(--error)}
.not_comparable{background:var(--nc)} .accepted{background:var(--error)}
.runs { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card dl { display:grid; grid-template-columns:max-content 1fr; gap:2px 12px; margin:0; }
.card dt { color:var(--muted); } .card dd { margin:0; overflow-wrap:anywhere; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
  vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.ids { font-family:ui-monospace,monospace; font-size:12px; overflow-wrap:anywhere; }
ul.reasons li { margin:4px 0; }
.finding { border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:8px 0; }
.finding ul { margin:6px 0 0; padding-left:20px; } .finding .tip { margin:6px 0 0; }
.level { display:inline-block; font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; padding:1px 8px; border-radius:999px; border:1px solid currentColor; }
.level.verified { color:var(--pass); } .level.supported { color:var(--error); }
.level.hypothesis { color:var(--nc); }
pre { background:var(--card); padding:12px; border-radius:8px; overflow-x:auto; font-size:12px; }
"""


def _e(x: Any) -> str:
    return html.escape(NONE if x is None else str(x))


def _html_table(
    header: Sequence[str], rows: Sequence[Sequence[str]], ids_cols: tuple[int, ...] = ()
) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="ids">{_e(c)}</td>' if i in ids_cols else f"<td>{_e(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def _run_card(title: str, run: Mapping[str, Any]) -> str:
    fields = [
        ("Label", run.get("label")),
        ("Backend", run.get("backend")),
        ("Target", json.dumps(run.get("target")) if run.get("target") else None),
        ("Parameters", json.dumps(run.get("params"))),
        ("k", run.get("k")),
        ("Snapshot", run.get("snapshot")),
        ("Commit", run.get("commit_sha")),
        ("Created", run.get("created_at")),
        ("Status", run.get("status")),
        ("Run id", run.get("run_id")),
    ]
    rows = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in fields)
    link = run.get("ci_run_url")
    ci = f'<p><a href="{_e(link)}">CI run</a></p>' if link else ""
    return f'<div class="card"><h3 style="margin:0 0 8px">{_e(title)}</h3><dl>{rows}</dl>{ci}</div>'


def _release_line(result: Mapping[str, Any], *, plain: bool = False) -> str:
    """Which release configuration and policy decided a ``vectraudit release check``."""
    release = result.get("release")
    if not release:
        return ""
    policy = release["policy"]
    text = (
        f"Release configuration {release['name']} v{release['version']} · policy "
        f"{policy['name']} v{policy['version']} ({policy['sha256'][:12]}) · "
        f"{'exploratory' if release['split'] == 'explore' else 'held-out'} benchmark "
        f"{release['benchmark_sha256'][:12]}"
    )
    return text if plain else _md(text)


_CRITICAL_HEADER = ["Query", "Document", "Maximum rank", "Worst rank", "Executions", "Verdict"]
_CRITICAL_FAILURE_HEADER = ["Query", "Document", "Execution", "Observed rank", "Observed ranking"]


def _critical_rows(result: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for rule in result["critical_queries"]:
        total = rule.get("executions", 0)
        failed = rule.get("failed_executions", 0)
        if not total:
            executions = "No repeat evidence"
        elif failed:
            executions = f"{failed} of {total} failed"
        else:
            executions = f"all {total} held"
        rows.append(
            [
                str(rule["query_id"]),
                str(rule["doc_id"]),
                str(rule["max_rank"]),
                str(rule["rank"]) if rule["rank"] is not None else "Not observed",
                executions,
                str(rule["verdict"]),
            ]
        )
    return rows


def _critical_failure_rows(result: Mapping[str, Any]) -> list[list[str]]:
    """Which executions broke a critical rule, with the ranking each one returned."""
    return [
        [
            str(rule["query_id"]),
            str(rule["doc_id"]),
            execution_label(failure),
            str(failure["rank"]) if failure["rank"] is not None else "Not observed",
            _ids(failure["ids"], limit=10) or "(no results)",
        ]
        for rule in result["critical_queries"]
        for failure in rule.get("failures") or []
    ]


def _critical_section(result: Mapping[str, Any], *, markdown: bool) -> list[str]:
    title = "Critical questions (every candidate execution)"
    table = _md_table if markdown else _html_table
    out = [f"**{title}**" if markdown else f"<h2>{_e(title)}</h2>"]
    if markdown:
        out.append("")
    out.append(table(_CRITICAL_HEADER, _critical_rows(result)))
    failures = _critical_failure_rows(result)
    if failures:
        caption = "Failing executions"
        if markdown:
            out += ["", f"_{caption}_", "", _md_table(_CRITICAL_FAILURE_HEADER, failures)]
        else:
            out += [f"<h3>{caption}</h3>", _html_table(_CRITICAL_FAILURE_HEADER, failures, (4,))]
    if markdown:
        out.append("")
    return out


def _repeat_note(result: Mapping[str, Any]) -> str:
    protocol = result["repeat_quality"]
    return (
        f"Quality averages every repeat within each query "
        f"(baseline {protocol['baseline_repeats']}, candidate {protocol['candidate_repeats']}), "
        "then weights query families equally. Repeats do not increase the independent sample size. "
        "Ranked-ID lists and primary stage evidence show repeat 1; inspect individual repeats "
        "for intermittent losses. Critical and permission rules still check every execution."
    )


def render_html(result: Mapping[str, Any]) -> str:
    """A self-contained HTML investigation report."""
    verdict = result["verdict"]
    b, c = result["baseline"], result["candidate"]
    parts = [
        f'<h1>Retrieval check <span class="badge {_e(verdict)}">'
        f"{_e(_VERDICT_TEXT[verdict])}</span></h1>",
        f'<p class="muted">{result["n_queries"]} paired rows / '
        f"{result.get('n_families', result['n_queries'])} families · k = {_e(c['k'])} · "
        f"vectraudit {_e(__version__)}</p>",
        *(
            [f'<p class="muted">{_e(_release_line(result, plain=True))}</p>']
            if "release" in result
            else []
        ),
        f'<div class="runs">{_run_card("Baseline", b)}{_run_card("Candidate", c)}</div>',
    ]
    if result.get("repeat_quality"):
        parts.append(f'<p class="muted">{_e(_repeat_note(result))}</p>')
    if result["reasons"]:
        items = "".join(
            f'<li><span class="badge {_e(r["verdict"])}">{_e(_VERDICT_TEXT[r["verdict"]])}'
            f"</span> {_e(r['message'])}{_e(_accepted_note(r))}</li>"
            for r in result["reasons"]
        )
        parts.append(f'<h2>Why</h2><ul class="reasons">{items}</ul>')
    if result.get("critical_queries"):
        parts += _critical_section(result, markdown=False)
    if result.get("findings"):
        parts.append("<h2>Diagnosis</h2>")
        parts.append(
            '<p class="muted">Verified: measured in these runs. Supported: consistent with a '
            "measured change. Hypothesis: worth checking, not confirmed.</p>"
        )
        cards = []
        for f in result["findings"]:
            facts = "".join(f"<li>{_e(x)}</li>" for x in f["facts"])
            tip = f'<p class="tip">→ {_e(f["suggestion"])}</p>' if f.get("suggestion") else ""
            cards.append(
                f'<div class="finding"><span class="level {_e(f["level"])}">'
                f"{_e(f['level'])}</span> <strong>{_e(f['title'])}</strong>"
                f"<ul>{facts}</ul>{tip}</div>"
            )
        parts.append("".join(cards))
    if result["metrics"]:
        parts.append("<h2>Quality</h2>")
        if tolerance := tolerance_line(result):
            parts.append(f'<p class="muted">{html.escape(tolerance)}</p>')
        parts.append(
            _html_table(
                [
                    "Metric",
                    "Baseline",
                    "Candidate",
                    "Change",
                    "95% CI of change",
                    "Rows",
                    "Families",
                ],
                _metric_rows(result["metrics"]),
            )
        )
    latency = result.get("latency")
    if latency:
        note = "" if latency["compared"] else " (not compared; see warnings)"
        parts.append(f"<h2>Latency{_e(note)}</h2>")
        parts.append(
            _html_table(["Percentile", "Baseline", "Candidate", "Change"], _latency_rows(latency))
        )
        if latency.get("paired"):
            parts.append(f"<p>{_e(PAIRED_NOTE)}</p>")
    queries = result.get("queries")
    if queries:
        metric = _METRIC_NAMES.get(queries["primary_metric"], queries["primary_metric"])
        parts.append(
            f"<h2>Queries by {_e(metric)}</h2><p>{queries['regressed_count']} regressed · "
            f"{queries['improved_count']} improved · {queries['unchanged_count']} unchanged</p>"
        )
        if queries["regressed"]:
            parts.append(
                _html_table(
                    [
                        "Query",
                        "Tags",
                        "Baseline",
                        "Candidate",
                        "Change",
                        "Lost ids",
                        "New ids",
                        "Candidate evidence",
                    ],
                    [
                        [
                            q["query_id"],
                            ", ".join(q["tags"]),
                            _num(q["baseline"], 3),
                            _num(q["candidate"], 3),
                            _num(q["delta"], 3, signed=True),
                            _ids(q["lost_ids"], 10),
                            _ids(q["new_ids"], 10),
                            describe_evidence(q.get("candidate_evidence")) or NONE,
                        ]
                        for q in queries["regressed"]
                    ],
                    ids_cols=(5, 6),
                )
            )
    if result["cohorts"]:
        rows = []
        for tag, cohort in result["cohorts"].items():
            for name, m in cohort["metrics"].items():
                rows.append(
                    [
                        tag,
                        str(cohort["n"]),
                        str(cohort.get("n_families", cohort["n"])),
                        "yes" if cohort["judged"] else "no (too few measured families)",
                        _METRIC_NAMES.get(name, name),
                        _num(m["baseline"]),
                        _num(m["candidate"]),
                        _num(m["delta"], signed=True),
                    ]
                )
        parts.append("<h2>Cohorts</h2>")
        if cohorts := cohort_line(result):
            parts.append(f'<p class="muted">{html.escape(cohorts)}</p>')
        parts.append(
            _html_table(
                ["Tag", "Rows", "Families", "Judged", "Metric", "Baseline", "Candidate", "Change"],
                rows,
            )
        )
    if result["config_diff"]:
        parts.append("<h2>What changed</h2>")
        parts.append(
            _html_table(
                ["Setting", "Baseline", "Candidate"],
                [
                    [d["path"], json.dumps(d["baseline"]), json.dumps(d["candidate"])]
                    for d in result["config_diff"]
                ],
            )
        )
    if result["warnings"]:
        items = "".join(f"<li>{_e(w)}</li>" for w in result["warnings"])
        parts.append(f"<h2>Warnings</h2><ul>{items}</ul>")
    parts.append("<h2>Policy</h2>")
    parts.append(f"<pre>{_e(json.dumps(result['policy'], indent=2))}</pre>")
    body = "\n".join(parts)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'\">"
        f"<title>Retrieval check: {_e(_VERDICT_TEXT[verdict])}</title>"
        f"<style>{_CSS}</style></head><body><main>\n{body}\n</main></body></html>\n"
    )
