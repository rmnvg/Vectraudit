"""Investigate one failed question, try a declared fix, and validate it under the policy.

The loop the strategy calls "explain a real failure, test a repair, protect every release":

1. :func:`investigate` reads one question of a run: what was lost and where, the recorded
   facts, possible causes (labelled as hypotheses), cautions, and which of the pipeline's
   *declared* fixes address the losses. Nothing is suggested that the pipeline did not
   declare, and nothing is offered for permission or no-answer failures.
2. :func:`try_fix` runs the current and the fixed configuration interleaved on the failing
   questions plus a seeded sample of other exploratory questions, and reports which
   failures recover and which questions regress. It is exploratory evidence, never a
   release decision, and it never touches held-out questions.
3. :func:`validate_fix` runs both configurations on the protected suite (held-out questions)
   and judges the change with the full release policy. Only a pass there reads
   "approved under policy".
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.exceptions import failing_rules
from vectraudit.evaluation.policy import Policy, policy_sha256
from vectraudit.pipelines.budget import combine
from vectraudit.pipelines.manifest import Candidate, Fix
from vectraudit.pipelines.runner import (
    PairedSide,
    PipelineOptions,
    RetryPolicy,
    execute_paired,
)
from vectraudit.pipelines.runtime import Pipeline

EPS = 1e-9
HYPOTHESES = {
    "not_observed": (
        "The document was in no recorded stage. It may be missing from the indexed corpus or "
        "its version, excluded before the first recorded stage, ranked beyond the candidate "
        "depth, or poorly matched by the query. Check the corpus and index before blaming "
        "embeddings."
    ),
    "lost_in_context": (
        "Retrieval delivered it, but context packing left it out or trimmed it: a token "
        "budget, chunk limit or ordering rule."
    ),
    "answered_wrong_despite_evidence": (
        "Enough evidence reached the model and the answer was still judged wrong. The "
        "prompt, the model or conflicting context may be responsible; the evidence does not "
        "say which."
    ),
    "permission": (
        "A document this user must not see was delivered. Correct the access rule or its "
        "inputs; this is not a quality trade-off."
    ),
    "no_answer": (
        "No document should answer this question, yet results were delivered. A score or "
        "relevance threshold, or abstention logic, may be missing."
    ),
}
STAGE_HYPOTHESES = {
    "rerank": "The reranker scored it below the documents it kept.",
    "filter": (
        "The filter excluded it. Check whether that is intended, such as permission "
        "filtering, before treating it as a defect."
    ),
    "truncate": "It ranked below the cut-off.",
    "fusion": "Fusion weighting pushed it out of the combined ranking.",
}
CAUTIONS = {
    "permission": "Disabling or loosening permissions is never a deployable repair.",
    "filter": "A loss at a filter may be intended access control, not a defect.",
    "not_observed": (
        "A document that never entered retrieval needs corpus or index evidence first; a "
        "different embedding model is rarely the first fix."
    ),
}


def query_losses(query: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every recorded loss for one question of a run, in path order."""
    out: list[dict[str, Any]] = []
    if query.get("error"):
        return [{"class": "error", "code": (query["error"] or {}).get("code")}]
    for doc in query.get("forbidden_ids") or []:
        out.append({"class": "permission", "doc_id": doc})
    if query.get("no_answer_correct") is False:
        out.append({"class": "no_answer", "doc_ids": list(query.get("ids") or [])})
    evidence = query.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    for failure in evidence.get("failures") or []:
        if failure.get("stage"):
            out.append(
                {
                    "class": "lost_downstream",
                    "doc_id": failure["doc_id"],
                    "stage": failure["stage"],
                    "stage_kind": failure.get("kind"),
                }
            )
        else:
            out.append({"class": "not_observed", "doc_id": failure["doc_id"]})
    context = evidence.get("context")
    if isinstance(context, Mapping):
        for loss in context.get("losses") or []:
            out.append(
                {"class": "lost_in_context", "doc_id": loss["doc_id"], "detail": loss["kind"]}
            )
    diagnosis = ((query.get("answer") or {}).get("diagnosis") or {}).get("class")
    if diagnosis == "answered_wrong_despite_evidence":
        out.append({"class": diagnosis})
    return out


def _key(loss: Mapping[str, Any]) -> tuple[str, str]:
    return str(loss["class"]), str(loss.get("doc_id", ""))


def addresses(fix: Fix, loss: Mapping[str, Any]) -> bool:
    """Whether a declared fix targets this loss (its class, or class and stage kind)."""
    for item in fix.addresses:
        kind, _, stage = item.partition(":")
        if kind == loss["class"] and (not stage or stage == loss.get("stage_kind")):
            return True
    return False


def builtin_fixes(candidate: Candidate, root: Path | None = None) -> list[Fix]:
    """Fixes the built-in hybrid baseline supports, from its current parameters.

    Vector fusion is offered only when every corpus document has a vector (``root`` is the
    manifest folder holding the corpus): without vectors the baseline cannot run it.
    """
    if candidate.adapter.kind != "hybrid":
        return []
    params = candidate.params
    depth = params.get("depth")
    current = depth if isinstance(depth, int) else 50
    fixes = [
        Fix(
            name="raise candidate depth",
            addresses=["not_observed"],
            params={"depth": min(10000, max(2 * current, 100))},
        )
    ]
    if params.get("reranker", "none") == "none":
        fixes.append(
            Fix(
                name="use the phrase reranker",
                addresses=["lost_downstream:truncate"],
                params={"reranker": "phrase"},
            )
        )
    if not params.get("vector_weight") and _has_vectors(candidate, root):
        fixes.append(
            Fix(
                name="add vector similarity to fusion",
                addresses=["not_observed", "lost_downstream:truncate"],
                params={"vector_weight": 0.5},
            )
        )
    return fixes


def _has_vectors(candidate: Candidate, root: Path | None) -> bool:
    if root is None or not candidate.adapter.corpus:
        return False
    try:
        lines = (root / candidate.adapter.corpus).read_text(encoding="utf-8").splitlines()
        return all("vector" in json.loads(line) for line in lines if line.strip())
    except (OSError, ValueError):
        return False


def available_fixes(candidate: Candidate, root: Path | None = None) -> list[Fix]:
    """Declared fixes, then built-in ones whose names the declaration does not reuse."""
    declared = {fix.name for fix in candidate.fixes}
    builtin = builtin_fixes(candidate, root)
    return [*candidate.fixes, *(f for f in builtin if f.name not in declared)]


def investigate(
    run: Mapping[str, Any],
    query_id: str,
    fixes: Sequence[Fix] = (),
    *,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What one question lost, the facts, possible causes and the declared fixes to try."""
    by_id = {q["query_id"]: q for q in run["queries"]}
    if query_id not in by_id:
        raise ValueError(f"question {query_id!r} is not in this run")
    query = by_id[query_id]
    losses = query_losses(query)
    uncertainty = []
    evidence = query.get("evidence")
    if not isinstance(evidence, Mapping):
        uncertainty.append("The run recorded no evidence for this question; losses are unknown.")
    elif not evidence.get("stages"):
        uncertainty.append(
            "No stage trace was recorded, so a missing document cannot be located in the pipeline."
        )
    coverage = [
        (obs.get("coverage") or {}).get("stage_trace") for obs in query.get("observations") or []
    ]
    if "partial" in coverage:
        uncertainty.append("The stage trace is partial; losses on unrecorded edges are unknown.")
    if len({tuple(r) for r in query.get("repeat_ids") or []}) > 1:
        uncertainty.append(
            "Repeated executions returned different rankings; this evidence is repeat 1."
        )
    context = (evidence or {}).get("context") if isinstance(evidence, Mapping) else None
    if isinstance(context, Mapping) and context.get("unknown"):
        uncertainty.append(
            "Whether " + ", ".join(context["unknown"]) + " reached the model is unknown."
        )
    hypotheses = []
    cautions = []
    for loss in losses:
        kind = loss["class"]
        text = (
            f"{loss.get('doc_id')}: dropped at {loss['stage']}. "
            + STAGE_HYPOTHESES.get(str(loss.get("stage_kind")), "")
            if kind == "lost_downstream"
            else HYPOTHESES.get(kind)
        )
        if text and text.strip() not in hypotheses:
            hypotheses.append(text.strip())
        caution = CAUTIONS.get("filter" if loss.get("stage_kind") == "filter" else kind)
        if caution and caution not in cautions:
            cautions.append(caution)
    suggested = []
    for fix in fixes:
        targets = [loss for loss in losses if addresses(fix, loss)]
        if targets:
            suggested.append(
                {
                    "name": fix.name,
                    "params": dict(fix.params),
                    "addresses": [_key(t) for t in targets],
                }
            )
    unaddressed = sorted(
        {loss["class"] for loss in losses if not any(addresses(fix, loss) for fix in fixes)}
    )
    regression = None
    if baseline is not None:
        before = {q["query_id"]: q for q in baseline["queries"]}.get(query_id)
        if before is not None:
            old = {_key(loss) for loss in query_losses(before)}
            regression = {
                "new_losses": [loss for loss in losses if _key(loss) not in old],
                "baseline_losses": sorted(old),
            }
    return {
        "format": "vectraudit.investigation",
        "format_version": 1,
        "run_id": run.get("run_id"),
        "query_id": query_id,
        "metrics": query.get("metrics"),
        "ids": query.get("ids"),
        "losses": losses,
        "regression": regression,
        "hypotheses": hypotheses,
        "hypotheses_note": (
            "Possible causes consistent with the recorded facts; none is established."
        ),
        "cautions": cautions,
        "uncertainty": uncertainty,
        "fixes": suggested,
        "unaddressed": unaddressed,
    }


def _subset(dataset: Dataset, ids: Sequence[str]) -> Dataset:
    wanted = set(ids)
    rows = [r for r in dataset_rows(dataset) if r["query_id"] in wanted]
    return parse_dataset(render_dataset(rows, dataset.meta))


def _options(
    k: int, params: Mapping[str, Any], repeats: int, retry: RetryPolicy | None, label: str
) -> PipelineOptions:
    return PipelineOptions(
        k=k,
        params=dict(params),
        repeats=repeats,
        warmup=0,
        label=label,
        retry=retry or RetryPolicy(),
    )


def try_plan(dataset: Dataset, focus: Sequence[str], sample: int, seed: int) -> list[str]:
    """The questions an exploratory try runs: the focus, then a seeded sample of the rest."""
    explore = [q.query_id for q in dataset.queries if q.split == "explore"]
    missing = [qid for qid in focus if qid not in {q.query_id for q in dataset.queries}]
    if missing:
        raise ValueError(f"questions not in the benchmark: {', '.join(missing)}")
    held = [qid for qid in focus if qid not in explore]
    if held:
        raise ValueError(
            "held-out questions are reserved for validation and cannot be used to try a fix: "
            + ", ".join(held)
        )
    if not focus:
        raise ValueError("name at least one failing question to try the fix on")
    if sample < 0:
        raise ValueError("sample must be >= 0")
    others = sorted(set(explore) - set(focus))
    picked = random.Random(seed).sample(others, min(sample, len(others)))
    return list(dict.fromkeys(focus)) + sorted(picked)


def try_fix(
    dataset: Dataset,
    pipeline: Pipeline,
    candidate: Candidate,
    fix: Fix,
    focus: Sequence[str],
    *,
    k: int,
    repeats: int = 1,
    sample: int = 20,
    seed: int = 0,
    retry: RetryPolicy | None = None,
    config: dict[str, Any] | None = None,
    corpus_version: str | None = None,
) -> dict[str, Any]:
    """Run current and fixed configurations interleaved on exploratory questions.

    Reports, for the focus questions, which targeted losses recovered, and for every
    question whether the fix introduced a regression: lower nDCG, a new loss, a failed
    safety assertion or an error. Exploratory: it is not a release decision.
    """
    ids = try_plan(dataset, focus, sample, seed)
    subset = _subset(dataset, ids)
    fixed_params = {**candidate.params, **fix.params}
    base_run, fixed_run = execute_paired(
        subset,
        PairedSide(
            pipeline,
            _options(k, candidate.params, repeats, retry, candidate.name),
            config,
            corpus_version,
        ),
        PairedSide(
            pipeline,
            _options(k, fixed_params, repeats, retry, f"{candidate.name} + {fix.name}"),
            {**(config or {}), "fix": fix.name},
            corpus_version,
        ),
    )
    before = {q["query_id"]: q for q in base_run["queries"]}
    after = {q["query_id"]: q for q in fixed_run["queries"]}
    rows: list[dict[str, Any]] = []
    for qid in ids:
        b, c = before[qid], after[qid]
        old_losses, new_losses = query_losses(b), query_losses(c)
        targeted = [loss for loss in old_losses if addresses(fix, loss)]
        # Compare per document: a loss that moved to a later stage is progress, not a
        # recovery and not a new failure.
        old_docs = {_key(loss)[1] for loss in old_losses}
        now: dict[str, dict[str, Any]] = {}
        for loss in new_losses:
            now.setdefault(_key(loss)[1], loss)
        recovered = [loss for loss in targeted if _key(loss)[1] not in now]

        def where(loss: Mapping[str, Any]) -> tuple[str, Any]:
            return str(loss["class"]), loss.get("stage")

        moved = [
            {"from": loss, "to": now[doc]}
            for loss in targeted
            if (doc := _key(loss)[1]) in now and where(now[doc]) != where(loss)
        ]
        introduced = [
            loss
            for loss in new_losses
            if _key(loss)[1] not in old_docs
            or (not _key(loss)[1] and _key(loss) not in {_key(x) for x in old_losses})
        ]
        bn = (b.get("metrics") or {}).get("ndcg")
        cn = (c.get("metrics") or {}).get("ndcg")
        regressed = bool(
            c.get("error") or introduced or (bn is not None and cn is not None and cn < bn - EPS)
        )
        rows.append(
            {
                "query_id": qid,
                "focus": qid in focus,
                "ndcg_before": bn,
                "ndcg_after": cn,
                "targeted_losses": targeted,
                "recovered": recovered,
                "moved": moved,
                "introduced": introduced,
                "regressed": regressed,
            }
        )
    focus_rows = [r for r in rows if r["focus"]]
    return {
        "format": "vectraudit.fix_trial",
        "format_version": 1,
        "status": "exploratory",
        "status_label": "Exploratory: shows recovery and regressions on a sample; not a "
        "release decision. Validate on held-out questions under the policy.",
        "candidate": candidate.name,
        "fix": {"name": fix.name, "params": dict(fix.params)},
        "dataset_sha256": dataset.sha256,
        "questions": len(ids),
        "focus": list(dict.fromkeys(focus)),
        "seed": seed,
        "recovered_focus": sum(
            1
            for r in focus_rows
            if r["targeted_losses"] and len(r["recovered"]) == len(r["targeted_losses"])
        ),
        "untargeted_focus": [r["query_id"] for r in focus_rows if not r["targeted_losses"]],
        "regressed": [r["query_id"] for r in rows if r["regressed"]],
        "rows": rows,
        "usage": combine([base_run["usage"], fixed_run["usage"]]),
        "runs": {"current": base_run, "fixed": fixed_run},
    }


def validate_fix(
    dataset: Dataset,
    pipeline: Pipeline,
    candidate: Candidate,
    fix: Fix,
    policy: Policy | None,
    *,
    k: int,
    repeats: int = 1,
    retry: RetryPolicy | None = None,
    config: dict[str, Any] | None = None,
    corpus_version: str | None = None,
) -> dict[str, Any]:
    """Judge the fix with the release policy on held-out questions, against the current one.

    Without a policy the result is exploratory. Without held-out questions there is nothing
    unconsulted to validate on, so validation refuses rather than reuse tuning questions.
    """
    held = [q.query_id for q in dataset.queries if q.split == "heldout"]
    if not held:
        raise ValueError(
            "The benchmark has no held-out questions to validate on; mark some questions "
            '"split": "heldout" that were not used to choose the fix.'
        )
    suite = _subset(dataset, held)
    fixed_params = {**candidate.params, **fix.params}
    base_run, fixed_run = execute_paired(
        suite,
        PairedSide(
            pipeline,
            _options(k, candidate.params, repeats, retry, candidate.name),
            config,
            corpus_version,
        ),
        PairedSide(
            pipeline,
            _options(k, fixed_params, repeats, retry, f"{candidate.name} + {fix.name}"),
            {**(config or {}), "fix": fix.name},
            corpus_version,
        ),
    )
    check = None
    status = "exploratory_only"
    if policy is not None:
        restricted = replace(
            policy,
            critical_queries=tuple(r for r in policy.critical_queries if r.query_id in set(held)),
        )
        result = compare_runs(base_run, fixed_run, restricted)
        failing = [r for r in result["reasons"] if r["verdict"] == "fail"]
        check = {
            "verdict": result["verdict"],
            "failing_rules": failing_rules(result),
            "reasons": [r["message"] for r in result["reasons"]][:10],
            "inconclusive": bool(failing) and all(r.get("inconclusive") for r in failing),
        }
        status = "approved_under_policy" if result["verdict"] == "pass" else "not_approved"
    labels = {
        "exploratory_only": (
            "Measured on held-out questions without a release policy; not approved."
        ),
        "approved_under_policy": "Approved under the release policy on held-out questions.",
        "not_approved": "Not approved under the release policy on held-out questions.",
    }
    return {
        "format": "vectraudit.fix_validation",
        "format_version": 1,
        "status": status,
        "status_label": labels[status],
        "approved": status == "approved_under_policy",
        "candidate": candidate.name,
        "fix": {"name": fix.name, "params": dict(fix.params)},
        "policy_sha256": None if policy is None else policy_sha256(policy),
        "questions": len(held),
        "check": check,
        "promote": {
            "name": f"{candidate.name} + {fix.name}",
            "params": fixed_params,
            "note": "Add as a manifest candidate, then export the release check as usual.",
        },
        "usage": combine([base_run["usage"], fixed_run["usage"]]),
        "runs": {"current": base_run, "fixed": fixed_run},
    }
