"""Synthetic demonstration of the strategy's §11 story, end to end through product code.

    "Our assistant answered an account-recovery question incorrectly. The required exception
    was in a retrieved source but missing from the model context. Vectraudit showed the loss.
    We tested a packing change, checked the other important questions, and saved the verified
    configuration. The next release now checks this case automatically."

Run it, then open the page it writes:

    uv run python examples/context_story_demo.py --out /tmp/vectraudit-context-story

Everything here is synthetic: a tiny support assistant whose context packing has a fixed
character budget, documents written for the demo, and a review file standing in for a
team's human answer review. It shows how Vectraudit reasons, not results on anyone's data.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from vectraudit.evaluation.answers import attach_file
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import load_dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines.investigate import available_fixes, investigate, try_fix, validate_fix
from vectraudit.pipelines.manifest import load_manifest, open_adapter
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

DOCUMENTS = {
    "account-guide": "To recover your account, open the sign-in page, choose Forgot password and "
    "follow the emailed link within 24 hours.",
    "recovery-exception": "Accounts with two-factor authentication cannot be recovered by email. "
    "Exception: they need the backup codes or an identity check by support, which takes up to "
    "three business days.",
    "billing-faq": "Invoices are issued on the first day of each month and can be downloaded "
    "from the billing page.",
    "shipping": "Orders ship within two business days.",
}
RANKINGS = {
    "account-recovery": ["account-guide", "recovery-exception"],
    "account-recovery-2fa": ["recovery-exception", "account-guide"],
    "invoices": ["billing-faq", "shipping"],
    "shipping-times": ["shipping", "billing-faq"],
}
ASSISTANT = """\
from vectraudit.pipelines.models import ContextItem, Hit, ModelContext, PipelineResult

DOCUMENTS = {documents!r}
RANKINGS = {rankings!r}


def assistant(request):
    \"\"\"Retrieve, then pack documents into the prompt until the character budget runs out.\"\"\"
    ids = RANKINGS[request.query_id][: request.k]
    budget = request.params.get("context_budget", 160)
    items = []
    for doc in ids:
        if budget <= 0:
            break
        text = DOCUMENTS[doc]
        items.append(ContextItem(id=doc, truncated=len(text) > budget))
        budget -= len(text)
    return PipelineResult(
        hits=[Hit(id=doc) for doc in ids],
        context=ModelContext(items=items, method="reported"),
    )
"""
QUESTIONS = [
    {
        "query_id": "account-recovery",
        "text": "How do I recover my account?",
        "relevant": {"account-guide": 1, "recovery-exception": 1},
        "required_evidence": [["account-guide", "recovery-exception"]],
        "tags": ["account"],
    },
    {
        "query_id": "invoices",
        "text": "Where are my invoices?",
        "relevant": {"billing-faq": 1},
        "required_evidence": [["billing-faq"]],
    },
    {
        "query_id": "shipping-times",
        "text": "How long does shipping take?",
        "relevant": {"shipping": 1},
        "required_evidence": [["shipping"]],
    },
    {
        "query_id": "account-recovery-2fa",
        "text": "I lost my phone; how do I get back into my account?",
        "relevant": {"account-guide": 1, "recovery-exception": 1},
        "required_evidence": [["recovery-exception"]],
        "split": "heldout",
    },
]
REVIEW = [  # Stands in for the team's human answer review; synthetic.
    {
        "query_id": "account-recovery",
        "evaluator": "support-review",
        "kind": "human",
        "rubric": "v1",
        "verdict": "incorrect",
    },
    {
        "query_id": "invoices",
        "evaluator": "support-review",
        "kind": "human",
        "rubric": "v1",
        "verdict": "correct",
    },
    {
        "query_id": "shipping-times",
        "evaluator": "support-review",
        "kind": "human",
        "rubric": "v1",
        "verdict": "correct",
    },
]
FIX = {
    "name": "pack whole documents",
    "addresses": ["lost_in_context"],
    "params": {"context_budget": 1000},
}
POLICY = {
    "metrics": {"ndcg": {"max_drop": 0.0}},
    "min_queries": 1,
    "critical_context": [{"query_id": "account-recovery"}],
}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rehearse(root: Path) -> dict[str, Any]:
    """Run the story; returns the facts the page and tests read."""
    root.mkdir(parents=True, exist_ok=False)
    (root / "support_assistant.py").write_text(
        ASSISTANT.format(documents=DOCUMENTS, rankings=RANKINGS), encoding="utf-8"
    )
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in QUESTIONS) + "\n")
    (root / "answer-review.jsonl").write_text("\n".join(json.dumps(r) for r in REVIEW) + "\n")
    _write(root / "policy.json", POLICY)
    _write(
        root / "manifest.json",
        {
            "dataset": "questions.jsonl",
            "k": 2,
            "repeats": 2,
            "corpus_version": "synthetic-support-v1",
            "candidates": [
                {
                    "name": "current",
                    "pipeline_version": "synthetic-1",
                    "adapter": {"kind": "python", "target": "support_assistant:assistant"},
                    "fixes": [FIX],
                }
            ],
        },
    )
    spec = load_manifest(root / "manifest.json")
    candidate = spec.candidates[0]
    pipeline = open_adapter(candidate.adapter, root)
    dataset = load_dataset(root / "questions.jsonl")
    # Day-to-day runs use the exploratory questions; held-out ones are kept for validation.
    explore = parse_dataset(
        render_dataset([r for r in dataset_rows(dataset) if r.get("split") != "heldout"])
    )
    policy = parse_policy(POLICY)

    def record(params: dict[str, Any], label: str) -> dict[str, Any]:
        return execute_pipeline(
            explore,
            pipeline,
            PipelineOptions(k=spec.k, repeats=spec.repeats, warmup=0, label=label, params=params),
            config={"adapter": "python", "name": label, "params": params},
            corpus_version=spec.corpus_version,
        )

    # 1. The incident: a wrong answer, explained by what reached the model.
    current = attach_file(record(candidate.params, "current"), root / "answer-review.jsonl")
    _write(root / "current.run.json", current)
    incident = next(q for q in current["queries"] if q["query_id"] == "account-recovery")
    investigation = investigate(current, "account-recovery", available_fixes(candidate, root))
    _write(root / "investigation.json", investigation)

    # 2. Try the declared packing fix, then validate it under the policy on held-out questions.
    trial = try_fix(
        dataset,
        pipeline,
        candidate,
        candidate.fixes[0],
        ["account-recovery"],
        k=spec.k,
        repeats=spec.repeats,
        sample=10,
        corpus_version=spec.corpus_version,
    )
    trial.pop("runs")
    _write(root / "trial.json", trial)
    validation = validate_fix(
        dataset,
        pipeline,
        candidate,
        candidate.fixes[0],
        parse_policy({**POLICY, "critical_context": [{"query_id": "account-recovery-2fa"}]}),
        k=spec.k,
        repeats=spec.repeats,
        corpus_version=spec.corpus_version,
    )
    validation.pop("runs")
    _write(root / "validation.json", validation)

    # 3. Protect: the release check now requires the exception to reach the model.
    fixed = record(validation["promote"]["params"], "fixed")
    _write(root / "fixed.run.json", fixed)
    approved = compare_runs(current, fixed, policy)
    _write(root / "release-fixed.json", approved)
    regressed = record({"context_budget": 160}, "next release, budget reverted")
    blocked = compare_runs(fixed, regressed, policy)
    _write(root / "release-regressed.json", blocked)
    return {
        "recall": incident["metrics"]["recall"],
        "sufficiency": incident["sufficiency"],
        "diagnosis": incident["answer"]["diagnosis"],
        "losses": investigation["losses"],
        "fixes": [f["name"] for f in investigation["fixes"]],
        "trial": {k: trial[k] for k in ("recovered_focus", "regressed", "questions")},
        "validation": validation["status"],
        "approved": approved["verdict"],
        "blocked": blocked["verdict"],
        "blocked_reasons": [r["message"] for r in blocked["reasons"] if r["verdict"] == "fail"],
    }


def _report_style() -> str:
    spec = importlib.util.spec_from_file_location(
        "sample_evidence_report", Path(__file__).with_name("sample_evidence_report.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    style: str = module.STYLE
    return style


def page(facts: dict[str, Any]) -> str:
    """One self-contained page telling the story from the demo's own results."""
    e = html.escape
    loss = next(x for x in facts["losses"] if x["class"] == "lost_in_context")
    steps = [
        (
            "The answer was wrong, and retrieval looked fine",
            f"Human review judged the account-recovery answer incorrect. Document recall was "
            f"{facts['recall']:.2f}: both the recovery guide and its two-factor exception were "
            "retrieved and delivered. A retrieval dashboard would show nothing wrong.",
        ),
        (
            "Vectraudit showed where the evidence was lost",
            f"The required evidence was <b>delivered: {e(facts['sufficiency']['delivered'])}</b> "
            f"but <b>in the model context: {e(facts['sufficiency']['context'])}</b>. "
            f"<code>{e(loss['doc_id'])}</code> was {e(loss.get('detail', 'lost'))} by context "
            f"packing. Diagnosis: <b>{e(facts['diagnosis']['class'].replace('_', ' '))}</b>, the "
            "first observable loss, not a claimed root cause.",
        ),
        (
            "A declared fix was tried, then validated",
            f"The pipeline declares one fix for this class: <i>{e(', '.join(facts['fixes']))}</i>. "
            f"Run interleaved with the current configuration, it recovered "
            f"{facts['trial']['recovered_focus']} of 1 failing question with "
            f"{len(facts['trial']['regressed'])} regressions across {facts['trial']['questions']} "
            f"exploratory questions. On held-out questions under the release policy: "
            f"<b>{e(facts['validation'].replace('_', ' '))}</b>.",
        ),
        (
            "The next release checks this case automatically",
            "The policy now requires the account-recovery evidence to reach the model in full on "
            f'every execution. The fixed configuration: <span class="pass">{e(facts["approved"])}'
            '</span>. A later release that reverted the packing budget: <span class="fail">'
            f"{e(facts['blocked'])}</span>: {e(facts['blocked_reasons'][0])}",
        ),
    ]
    body = "".join(
        f'<div class="step"><b class="n">{i}.</b><b>{e(title)}</b><p>{text}</p></div>'
        for i, (title, text) in enumerate(steps, 1)
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Vectraudit: evidence that never reached the model</title><style>{_report_style()}"
        "</style></head><body><main>"
        '<p class="banner"><b>Synthetic demonstration.</b> A tiny support assistant written for '
        "this demo, with a synthetic review file standing in for a team's answer review. It "
        "shows how Vectraudit reasons, not results on anyone's data.</p>"
        "<h1>The right document was retrieved. The model never saw the part that mattered.</h1>"
        '<p class="muted">Explain the failure, test a repair, protect the next release.</p>'
        f'<div class="steps">{body}</div>'
        '<h2>What a pilot adds</h2><p class="muted">The same loop on your assistant: your '
        "questions and required evidence, your reviewers' judgments, fixes your pipeline "
        "supports, and the check in your CI. Documents and prompts stay on your machines.</p>"
        "</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="New folder for the results.")
    args = parser.parse_args(argv)
    facts = rehearse(args.out)
    (args.out / "context-story.html").write_text(page(facts), encoding="utf-8")
    print(
        f"Recall {facts['recall']:.2f}; context {facts['sufficiency']['context']}; "
        f"diagnosis {facts['diagnosis']['class']}"
    )
    print(
        f"Trial: recovered {facts['trial']['recovered_focus']} of 1, "
        f"{len(facts['trial']['regressed'])} regressions; validation {facts['validation']}"
    )
    print(f"Release with the fix: {facts['approved']}; reverted release: {facts['blocked']}")
    print(f"Wrote {args.out / 'context-story.html'}")
    return 0 if (facts["approved"], facts["blocked"]) == ("pass", "fail") else 1


if __name__ == "__main__":
    sys.exit(main())
