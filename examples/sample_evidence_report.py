"""One shareable page from the synthetic release demo: fail, explain, fix, protect.

Run the demo first, then build the page from its output folder:

    uv run python examples/release_demo.py --out /tmp/vectraudit-release-demo
    uv run python examples/sample_evidence_report.py /tmp/vectraudit-release-demo \
        --out sample-evidence-report.html

The page is self-contained (no scripts, no external requests) so it can be emailed or
attached to a review. It is a synthetic demonstration and says so at the top: it shows
how Vectraudit reasons about a release, not results on anyone's data, and its latency values
carry no performance claim. It embeds the two release reports the demo produced.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from vectraudit.evaluation.report import render_html

STYLE = """
:root{--bg:#fbfaf7;--fg:#1c1b19;--muted:#6b675f;--line:#e4e0d8;--card:#fff;--fail:#b42318;
--pass:#067647;--warn:#9a6700;--accent:#0f766e}
@media (prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ecebe7;--muted:#a19d94;
--line:#2c2b28;--card:#1c1b19;--fail:#f97066;--pass:#47cd89;--warn:#f5b54a;--accent:#2dd4bf}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:8px 0 12px}h2{font-size:20px;margin:36px 0 8px}
.banner{border:1px solid var(--warn);color:var(--warn);border-radius:10px;padding:10px 14px;
font-size:14px}.muted{color:var(--muted)}.steps{display:grid;gap:12px;margin-top:20px}
.step{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.step b.n{color:var(--accent);margin-right:6px}.verdict{font-weight:600}
.fail{color:var(--fail)}.pass{color:var(--pass)}
table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;
padding:6px 8px;border-bottom:1px solid var(--line)}
iframe{width:100%;height:640px;border:1px solid var(--line);border-radius:12px;
background:#fff}code{font-size:14px}
"""


def _load(folder: Path, name: str) -> dict[str, Any]:
    path = folder / name / "report.json"
    try:
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: run examples/release_demo.py first ({exc})") from exc
    return result


def _ndcg(result: dict[str, Any]) -> tuple[float, float]:
    stats = (result.get("metrics") or {}).get("ndcg") or {}
    return float(stats.get("baseline", 0.0)), float(stats.get("candidate", 0.0))


def _critical(result: dict[str, Any]) -> dict[str, Any]:
    rules = result.get("critical_queries") or []
    if not rules:
        raise SystemExit("The demo reports have no critical question; rerun the demo.")
    found: dict[str, Any] = rules[0]
    return found


def _embed(result: dict[str, Any], title: str) -> str:
    """The full report in a sandboxed frame; ``srcdoc`` keeps the page one file."""
    return (
        f'<iframe title="{html.escape(title)}" sandbox '
        f'srcdoc="{html.escape(render_html(result), quote=True)}"></iframe>'
    )


def build(folder: Path) -> str:
    """The page's HTML from a ``release_demo.py`` output folder."""
    candidate, fixed = _load(folder, "candidate"), _load(folder, "fixed")
    base_ndcg, cand_ndcg = _ndcg(candidate)
    _, fixed_ndcg = _ndcg(fixed)
    rule = _critical(candidate)
    failure = (rule.get("failures") or [{}])[0]
    returned = ", ".join(f"<code>{html.escape(i)}</code>" for i in failure.get("ids") or [])
    question, doc = html.escape(rule["query_id"]), html.escape(rule["doc_id"])
    executions = rule.get("executions") or 0
    e = html.escape
    steps = [
        (
            "The change looks better on average",
            f"A candidate retrieval configuration lifts mean nDCG from {base_ndcg:.3f} to "
            f"{cand_ndcg:.3f} on {candidate.get('n_queries')} benchmark questions. A "
            "dashboard of averages would approve it.",
        ),
        (
            "The release check catches what the average hides",
            f"The team marked <code>{question}</code> as critical: <code>{doc}</code> must "
            f"rank first on every execution. Each question ran {executions} times. On "
            f"repeat {failure.get('repeat', '?')} the candidate returned {returned or 'nothing'}"
            f" and lost the required document. Verdict: "
            f'<span class="verdict fail">{e(candidate["verdict"].upper())}</span>, '
            "CI exit code 3, the pull request is blocked.",
        ),
        (
            "The fix is measured, not assumed",
            f"A fixed configuration keeps the quality gain (nDCG {fixed_ndcg:.3f}) and returns "
            f"<code>{doc}</code> first on all {executions} executions. Verdict: "
            f'<span class="verdict pass">{e(fixed["verdict"].upper())}</span>, exit code 0.',
        ),
        (
            "The next release is protected",
            "The passing configuration is exported as a release check (benchmark, policy and "
            "pipeline settings, each with a version hash) and runs on every pull request in "
            "the team's CI, judged by the same rules as here. The same failure cannot ship "
            "again unnoticed; a deliberate trade-off needs an audited, expiring exception.",
        ),
    ]
    step_html = "".join(
        f'<div class="step"><b class="n">{i}.</b><b>{e(title)}</b><p>{body}</p></div>'
        for i, (title, body) in enumerate(steps, 1)
    )
    summary = (
        "<table><tr><th>Configuration</th><th>Mean nDCG</th><th>Critical question</th>"
        "<th>Verdict</th></tr>"
        f"<tr><td>Current</td><td>{base_ndcg:.3f}</td><td>held</td><td>reference</td></tr>"
        f"<tr><td>Candidate</td><td>{cand_ndcg:.3f}</td><td>lost on "
        f"{rule.get('failed_executions', 0)} of {executions} executions</td>"
        f'<td class="fail">{e(candidate["verdict"])} (exit 3)</td></tr>'
        f"<tr><td>Fixed</td><td>{fixed_ndcg:.3f}</td><td>held on all {executions} "
        "executions</td>"
        f'<td class="pass">{e(fixed["verdict"])} (exit 0)</td></tr></table>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Vectraudit sample evidence report</title>"
        f"<style>{STYLE}</style></head><body><main>"
        '<p class="banner"><b>Synthetic demonstration.</b> Generated from a fixture with '
        "deliberately simple data, not from any user's application. It shows how "
        "Vectraudit reasons about a release; the numbers are not benchmark results and the "
        "latency values carry no performance claim.</p>"
        "<h1>A retrieval change that looked better, and why it should not ship</h1>"
        '<p class="muted">Explain a failure, verify the fix, protect the next release.</p>'
        f'<div class="steps">{step_html}</div>'
        f"<h2>At a glance</h2>{summary}"
        "<h2>The blocked change: full release report</h2>"
        + _embed(candidate, "Candidate release report")
        + "<h2>The fix: full release report</h2>"
        + _embed(fixed, "Fixed release report")
        + '<h2>On your own application</h2><p class="muted">With your own application and '
        "questions: a reviewed benchmark graded by your domain owner, an investigation of a "
        "real failure or planned change, the fix measured under a policy you agree, and the "
        "same check installed in your CI. Your data stays on your machines unless you choose "
        "to share release decisions with a hosted project.</p>"
        "</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("demo", type=Path, help="Output folder of release_demo.py")
    parser.add_argument("--out", type=Path, default=Path("sample-evidence-report.html"))
    args = parser.parse_args(argv)
    args.out.write_text(build(args.demo), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
