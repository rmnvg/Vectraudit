"""Compare a candidate run with its baseline under a policy, and decide.

Verdicts, in order of precedence:

- ``error``: a run is not complete (failed, partial or cancelled). Never passes.
- ``not_comparable``: different dataset version, k, metric contract or query set; too few
  queries; or a metric the policy requires was not measured in both runs.
- ``fail``: a policy rule is broken.
- ``pass``: every rule holds.

Quality deltas are paired per query (candidate - baseline), averaged within each family,
then averaged and bootstrapped across families with seeded 95% intervals.
Latency is compared only between runs from matching runner environments,
unless the policy allows otherwise.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from vectraudit.evaluation.diagnose import diagnose
from vectraudit.evaluation.families import family_count, family_id, family_means
from vectraudit.evaluation.metrics import QUALITY_METRICS, SEMANTIC_METRICS
from vectraudit.evaluation.policy import DEFAULT_POLICY, MetricRule, Policy

CHECK_FORMAT = "vectraudit.check"
CHECK_FORMAT_VERSION = 1
SMALL_SAMPLE = 30
MANY_COHORT_RULES = 5
"""From this many cohort rules judged without control, warn that some fail by chance."""
EPS = 1e-9
BOOTSTRAP_MAX_CELLS = 1_000_000
MAX_CRITICAL_FAILURES = 10
_ENVIRONMENT_KEYS = ("system", "machine", "cpu_count", "python")


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_COMPARABLE = "not_comparable"
    ACCEPTED = "accepted"
    """Only after audited exceptions accept every failing rule (see exceptions.py)."""


_PRECEDENCE = [Verdict.ERROR, Verdict.NOT_COMPARABLE, Verdict.FAIL]


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Policy = DEFAULT_POLICY,
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Compare two validated run records and return the check result (plain JSON)."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    check = _Check(baseline, candidate, policy, seed, n_bootstrap)
    return check.result()


class _Check:
    def __init__(
        self,
        base: Mapping[str, Any],
        cand: Mapping[str, Any],
        policy: Policy,
        seed: int,
        n_bootstrap: int,
    ) -> None:
        self.base, self.cand, self.policy = base, cand, policy
        self.rng = np.random.default_rng(seed)
        self.n_bootstrap = n_bootstrap
        self.reasons: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.out: dict[str, Any] = {
            "format": CHECK_FORMAT,
            "format_version": CHECK_FORMAT_VERSION,
            "verdict": None,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "policy": policy.as_dict(),
            "baseline": _run_brief(base),
            "candidate": _run_brief(cand),
            "n_queries": 0,
            "n_families": 0,
            "sampling_unit": "query_family",
            "metrics": {},
            "latency": None,
            "cohorts": {},
            "cohort_judging": None,
            "queries": None,
            "findings": [],
            "config_diff": config_diff(base, cand),
        }

    def flag(self, verdict: Verdict, message: str, rule: str | None = None) -> None:
        """Record a reason. A failing reason carries ``rule``: a stable key naming exactly
        which policy rule failed, which an audited exception can name (see exceptions.py)."""
        reason: dict[str, Any] = {"verdict": verdict.value, "message": message}
        if rule is not None:
            reason["rule"] = rule
        self.reasons.append(reason)

    def result(self) -> dict[str, Any]:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        if self._runs_complete() and self._comparable():
            pairs = self._pairs()
            self.out["n_queries"] = len(pairs)
            self.out["n_families"] = family_count(b for b, _ in pairs)
            self._safety(pairs)
            self._critical_queries()
            self._critical_context()
            self._quality(pairs)
            self._cohorts(pairs)
            self._latency()
        verdicts = {Verdict(r["verdict"]) for r in self.reasons}
        verdict = next((v for v in _PRECEDENCE if v in verdicts), Verdict.PASS)
        self.out["verdict"] = verdict.value
        if self.out["metrics"]:
            self.out["findings"] = diagnose(self.out, self.base, self.cand, pairs)
        return self.out

    def _safety(self, pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        """Safety assertions are absolute gates, even when the baseline also violates them."""
        for base, candidate in pairs:
            if base.get("permission_checked") != candidate.get("permission_checked"):
                self.flag(Verdict.NOT_COMPARABLE, "Permission measurement coverage changed.")
            if (base.get("no_answer_correct") is None) != (
                candidate.get("no_answer_correct") is None
            ):
                self.flag(Verdict.NOT_COMPARABLE, "No-answer measurement coverage changed.")
            if candidate.get("forbidden_ids"):
                self.flag(
                    Verdict.FAIL,
                    f"Forbidden documents delivered for {candidate['query_id']}.",
                    f"safety.forbidden:{candidate['query_id']}",
                )
            self._permission_path(candidate)
            self._failed_attempts(candidate)
            self._context_coverage(base, candidate)
            if candidate.get("no_answer_correct") is False:
                self.flag(
                    Verdict.FAIL,
                    f"No-answer assertion failed for {candidate['query_id']}.",
                    f"safety.no_answer:{candidate['query_id']}",
                )

    def _context_coverage(self, base: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
        """A candidate that stops recording model context cannot pass the context-leak check
        its baseline could be judged by."""
        before, after = base.get("permission_path"), candidate.get("permission_path")
        could = isinstance(before, Mapping) and before.get("context_recorded") is True
        can = isinstance(after, Mapping) and after.get("context_recorded") is True
        if could and not can:
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"The candidate no longer records the model context in full for "
                f"{candidate['query_id']}, so forbidden documents reaching the model cannot be "
                "checked as they were in the baseline.",
            )

    def _failed_attempts(self, candidate: Mapping[str, Any]) -> None:
        """A forbidden document served before a failure counts, even if a retry was clean."""
        leaked: set[str] = set()
        for attempt in candidate.get("attempts") or []:
            result = attempt.get("result") if isinstance(attempt, Mapping) else None
            if not isinstance(result, Mapping) or attempt.get("status") != "failed":
                continue
            leaked |= set(result.get("forbidden_ids") or [])
            path = result.get("permission_path")
            if isinstance(path, Mapping):
                leaked |= {d["doc_id"] for d in path["documents"] if d.get("context") is True}
                boundary = set(self.policy.forbidden_stages)
                leaked |= {d["doc_id"] for d in path["documents"] if boundary & set(d["kinds"])}
        if leaked:
            self.flag(
                Verdict.FAIL,
                f"Forbidden documents reached the user in a failed attempt for "
                f"{candidate['query_id']} that was later retried: "
                + ", ".join(sorted(leaked))
                + ".",
                f"safety.forbidden:{candidate['query_id']}",
            )

    def _permission_path(self, candidate: Mapping[str, Any]) -> None:
        """Forbidden documents upstream of the results: the model context always fails; the
        policy's ``forbidden_stages`` fail too, and cannot pass on a run without a trace."""
        query_id = candidate["query_id"]
        path = candidate.get("permission_path")
        rule = f"safety.forbidden:{query_id}"
        boundary = set(self.policy.forbidden_stages)
        if isinstance(path, Mapping):
            in_context = [d["doc_id"] for d in path["documents"] if d.get("context") is True]
            if in_context:
                self.flag(
                    Verdict.FAIL,
                    f"Forbidden documents reached the model context for {query_id}: "
                    + ", ".join(in_context)
                    + ".",
                    rule,
                )
            for doc in path["documents"]:
                reached = sorted(boundary & set(doc["kinds"]))
                if reached:
                    self.flag(
                        Verdict.FAIL,
                        f"Forbidden document {doc['doc_id']} reached a "
                        + " and ".join(reached)
                        + f" stage for {query_id}, past the policy's permission boundary.",
                        rule,
                    )
        if (
            boundary
            and candidate.get("permission_checked")
            and not (isinstance(path, Mapping) and path.get("stages_recorded"))
        ):
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"The policy forbids forbidden documents in {', '.join(sorted(boundary))} "
                f"stages, but no stage trace shows where they travelled for {query_id}.",
            )

    def _critical_queries(self) -> None:
        """Judge every recorded candidate execution, independently of aggregate quality.

        A critical document must meet its rank on each repeat, including repeats of earlier
        failed attempts that a checkpoint retry replaced. Runs without per-repeat rankings
        (recorded before this check existed) are insufficient evidence, never a pass.
        """
        if not self.policy.critical_queries:
            return
        queries = {q["query_id"]: q for q in self.cand["queries"]}
        repeats = (self.cand.get("timing") or {}).get("repeats")
        assertions = []
        for rule in self.policy.critical_queries:
            query = queries.get(rule.query_id)
            rank = None
            executions: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            if query is None:
                verdict, detail = Verdict.NOT_COMPARABLE, "query was not measured"
            elif rule.max_rank > self.cand["search"]["k"]:
                verdict, detail = Verdict.NOT_COMPARABLE, "max_rank exceeds the run's k"
            elif query.get("error") or query.get("ids") is None:
                verdict, detail = Verdict.ERROR, "query has no successful measurement"
            elif (missing := _missing_repeat_evidence(query, repeats)) is not None:
                verdict, detail = Verdict.NOT_COMPARABLE, missing
            else:
                executions = _executions(query)
                for execution in executions:
                    ids = execution["ids"]
                    execution["rank"] = ids.index(rule.doc_id) + 1 if rule.doc_id in ids else None
                failures = [e for e in executions if e["rank"] is None or e["rank"] > rule.max_rank]
                ranks = [e["rank"] for e in executions]
                rank = None if None in ranks else max(ranks)
                verdict = Verdict.FAIL if failures else Verdict.PASS
                detail = _critical_detail(failures, len(executions))
            assertions.append(
                {
                    "query_id": rule.query_id,
                    "doc_id": rule.doc_id,
                    "max_rank": rule.max_rank,
                    "rank": rank,
                    "verdict": verdict.value,
                    "executions": len(executions),
                    "failed_executions": len(failures),
                    "failures": failures[:MAX_CRITICAL_FAILURES],
                }
            )
            if verdict != Verdict.PASS:
                self.flag(
                    verdict,
                    f"Critical query {rule.query_id}: document {rule.doc_id} "
                    f"must appear in the top {rule.max_rank} on every execution; {detail}.",
                    f"critical:{rule.query_id}:{rule.doc_id}" if verdict == Verdict.FAIL else None,
                )
        self.out["critical_queries"] = assertions

    def _critical_context(self) -> None:
        """Required evidence must reach the model in full on every candidate execution.

        A demonstrated loss on any execution fails. Context that was not recorded, or whose
        completeness is unknown, is insufficient evidence: not comparable, never a pass.
        """
        if not self.policy.critical_context:
            return
        queries = {q["query_id"]: q for q in self.cand["queries"]}
        repeats = (self.cand.get("timing") or {}).get("repeats")
        results = []
        for rule in self.policy.critical_context:
            query = queries.get(rule.query_id)
            states: list[str] = []
            if query is None:
                verdict, detail = Verdict.NOT_COMPARABLE, "the question was not measured"
            elif query.get("error"):
                verdict, detail = Verdict.ERROR, "the question has no successful measurement"
            elif not isinstance(query.get("repeat_sufficiency"), list):
                verdict, detail = (
                    Verdict.NOT_COMPARABLE,
                    "the run records no per-execution evidence sufficiency; declare "
                    "required_evidence for the question and re-run",
                )
            elif isinstance(repeats, int) and len(query["repeat_sufficiency"]) != repeats:
                verdict, detail = Verdict.NOT_COMPARABLE, "sufficiency is missing for some repeats"
            else:
                states = [row.get("context") for row in query["repeat_sufficiency"]]
                lost = states.count("insufficient")
                unknown = len(states) - lost - states.count("sufficient")
                if lost:
                    verdict = Verdict.FAIL
                    detail = (
                        f"the required evidence did not reach the model intact on {lost} of "
                        f"{len(states)} executions"
                    )
                elif unknown:
                    verdict = Verdict.NOT_COMPARABLE
                    detail = (
                        f"whether it reached the model is unknown on {unknown} of "
                        f"{len(states)} executions (context unrecorded or incomplete)"
                    )
                else:
                    verdict, detail = Verdict.PASS, f"held on all {len(states)} executions"
            results.append(
                {
                    "query_id": rule.query_id,
                    "verdict": verdict.value,
                    "executions": len(states),
                    "context": states,
                    "detail": detail,
                }
            )
            if verdict != Verdict.PASS:
                self.flag(
                    verdict,
                    f"Critical context {rule.query_id}: its required evidence must reach the "
                    f"model in full on every execution; {detail}.",
                    f"critical_context:{rule.query_id}" if verdict == Verdict.FAIL else None,
                )
        self.out["critical_context"] = results

    # --- gates -----------------------------------------------------------------------

    def _runs_complete(self) -> bool:
        ok = True
        for role, run in (("baseline", self.base), ("candidate", self.cand)):
            if run["status"] != "complete":
                ok = False
                detail = ""
                if run.get("error"):
                    detail = f": {run['error'].get('message')}"
                elif run["status"] == "partial":
                    n = sum(bool(q["error"]) for q in run["queries"])
                    detail = f": {n} queries failed"
                self.flag(Verdict.ERROR, f"The {role} run is {run['status']}{detail}.")
        return ok

    def _comparable(self) -> bool:
        b, c = self.base, self.cand
        before = len(self.reasons)
        if b["dataset"]["sha256"] != c["dataset"]["sha256"]:
            self.flag(
                Verdict.NOT_COMPARABLE,
                "The runs used different dataset versions "
                f"({b['dataset']['sha256'][:12]} vs {c['dataset']['sha256'][:12]}).",
            )
        if b["search"]["k"] != c["search"]["k"]:
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"The runs used different k ({b['search']['k']} vs {c['search']['k']}).",
            )
        if b["metric_contract"]["version"] != c["metric_contract"]["version"]:
            self.flag(
                Verdict.NOT_COMPARABLE,
                "The runs use different metric contract versions; re-record both sides "
                "to compare quality across all repeats.",
            )
        from vectraudit.evaluation.runfile import RunFileError, check_run

        for role, run in (("baseline", b), ("candidate", c)):
            version = run["metric_contract"]["version"]
            if version not in (1, 2):
                self.flag(Verdict.NOT_COMPARABLE, "Unsupported metric contract version.")
            # Every run is validated, not only files loaded from disk: a verdict must never
            # rest on duplicate rows, non-finite metrics or a "complete" run with failures.
            try:
                check_run(run)
            except RunFileError as exc:
                self.flag(Verdict.NOT_COMPARABLE, f"The {role} run is not a valid run: {exc}")
        if len(self.reasons) > before:
            return False
        if b["metric_contract"]["version"] == c["metric_contract"]["version"] == 2:
            br = (b.get("timing") or {}).get("repeats")
            cr = (c.get("timing") or {}).get("repeats")
            self.out["repeat_quality"] = {
                "aggregation": "mean_per_query",
                "baseline_repeats": br,
                "candidate_repeats": cr,
                "ranking_display": "first_repeat",
            }
            if br != cr:
                self.flag(
                    Verdict.NOT_COMPARABLE,
                    "Repeat counts differ; re-record under the same repeat protocol.",
                )
        if len(self.reasons) > before:
            return False
        if {q["query_id"] for q in b["queries"]} != {q["query_id"] for q in c["queries"]}:
            self.flag(Verdict.NOT_COMPARABLE, "The runs measured different query sets.")
            return False
        paired = self._pairs()
        if any(family_id(bq) != family_id(cq) for bq, cq in paired):
            self.flag(Verdict.NOT_COMPARABLE, "Query family assignments differ between runs.")
            return False
        # The same dataset labels the same queries on both sides; a query scored on only one
        # side would let a regression drop out of the comparison unseen.
        uneven = [
            bq["query_id"]
            for bq, cq in paired
            if (bq.get("metrics") is None) != (cq.get("metrics") is None)
        ]
        if uneven:
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"{len(uneven)} queries were scored in only one run (for example "
                f"{uneven[0]}); quality cannot be compared on different query sets.",
            )
            return False
        n = family_count(b["queries"])
        self.out["n_queries"] = len(paired)
        self.out["n_families"] = n
        if n < self.policy.min_queries:
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"{n} query families ({len(paired)} rows) is below the policy minimum "
                f"of {self.policy.min_queries}.",
            )
            # Still inspect hard safety assertions; insufficient evidence never passes.
        if not n:
            return False
        if n < len(paired):
            self.warnings.append(
                f"{len(paired)} query rows represent {n} independent query families. "
                "Quality means weight families equally; intervals resample whole families."
            )
        self._context_warnings(n)
        return True

    def _context_warnings(self, n: int) -> None:
        b, c = self.base, self.cand
        if n == 1:
            self.warnings.append(
                "One independent family cannot estimate between-family uncertainty; "
                "a zero-width bootstrap interval is not evidence of certainty."
            )
        if n < SMALL_SAMPLE:
            self.warnings.append(
                f"Only {n} independent query families: sampling evidence is limited and one "
                f"family can move the mean by up to {1 / n:.2f}. Treat deltas as indicative."
            )
        bs, cs = b["backend"].get("snapshot"), c["backend"].get("snapshot")
        if bs is None or cs is None:
            self.warnings.append(
                "A backend has no immutable snapshot id, so concurrent corpus updates "
                "can't be ruled out. Freeze or version the corpus while evaluating."
            )
        elif bs != cs:
            self.warnings.append(
                "The backends hold different corpus snapshots. Expected after a re-index; "
                "otherwise the corpus changed between runs."
            )
        if b["backend"].get("distance") != c["backend"].get("distance"):
            self.warnings.append(
                f"Distance differs ({b['backend'].get('distance')} vs "
                f"{c['backend'].get('distance')}); compare rankings, not scores."
            )
        nondeterministic = sum(bool(q.get("nondeterministic")) for q in c["queries"])
        if nondeterministic:
            self.warnings.append(
                f"{nondeterministic} candidate queries returned different results across repeats."
            )

    # --- quality ---------------------------------------------------------------------

    def _pairs(self) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        by_id = {q["query_id"]: q for q in self.cand["queries"]}
        return [(q, by_id[q["query_id"]]) for q in self.base["queries"]]

    def _quality(self, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        available = {}
        for name in QUALITY_METRICS:
            stats = self._metric_stats(pairs, name)
            if stats is not None:
                available[name] = stats
        self.out["metrics"] = available
        if (
            not available
            and (self.policy.critical_queries or self.policy.critical_context)
            and not self.policy.metrics
            and self.policy.max_regressed_queries is None
            and not self.policy.cohorts.metrics
        ):
            return
        safety_pairs = [
            b
            for b, c in pairs
            if (b.get("no_answer_correct") is not None and c.get("no_answer_correct") is not None)
            or bool(b.get("permission_checked") and c.get("permission_checked"))
        ]
        if (
            not available
            and family_count(safety_pairs) >= self.policy.min_queries
            and not any(rule.required for rule in self.policy.metrics.values())
        ):
            self.out["safety"] = {
                "n": len(safety_pairs),
                "n_families": family_count(safety_pairs),
                "absolute_assertions": True,
            }
            return
        if not available:
            self.flag(
                Verdict.NOT_COMPARABLE,
                "No quality metric was measured in both runs: add relevance labels or an "
                "exact reference.",
            )
            return
        for name, stats in available.items():
            if stats["n"] < len(pairs):
                self.warnings.append(
                    f"{name} was measured on {stats['n']}/{len(pairs)} paired queries; "
                    "unmeasured queries do not contribute to this metric."
                )
            if stats["n_families"] < SMALL_SAMPLE <= self.out["n_families"]:
                self.warnings.append(
                    f"Only {stats['n_families']} measured families for {name}: "
                    "treat its interval as indicative."
                )
        judged = 0
        for name, rule in self.policy.metrics.items():
            if name not in available:
                if rule.required:
                    self.flag(Verdict.NOT_COMPARABLE, _unavailable(name))
                continue
            if available[name]["n_families"] < self.policy.min_queries:
                self.flag(
                    Verdict.NOT_COMPARABLE,
                    f"{name} has {available[name]['n']} measured query pairs in "
                    f"{available[name]['n_families']} families, below the policy "
                    f"minimum of {self.policy.min_queries}. Add labels or an exact reference.",
                )
                continue
            judged += 1
            self._apply_rule(name, rule, available[name], "", "metric")
        if self.policy.is_default and judged == 0:
            self.flag(Verdict.NOT_COMPARABLE, "The default policy found no metric to judge.")
        self._regressed_queries(pairs, available)

    def _metric_stats(
        self, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], name: str
    ) -> dict[str, Any] | None:
        base_vals, cand_vals = [], []
        for b, c in pairs:
            bv, cv = _value(b, name), _value(c, name)
            if bv is not None and cv is not None:
                base_vals.append((family_id(b), bv))
                cand_vals.append((family_id(b), cv))
        if not base_vals:
            return None
        base_means, cand_means = family_means(base_vals), family_means(cand_vals)
        deltas = np.asarray(cand_means) - np.asarray(base_means)
        low, high = self._bootstrap(deltas)
        return {
            "n": len(base_vals),
            "n_families": len(deltas),
            "baseline": statistics.fmean(base_means),
            "candidate": statistics.fmean(cand_means),
            "delta": float(deltas.mean()),
            "ci_low": low,
            "ci_high": high,
        }

    def _bootstrap(self, deltas: np.ndarray[Any, Any]) -> tuple[float, float]:
        if len(deltas) == 1:
            return float(deltas[0]), float(deltas[0])
        # Preserve all seeded resamples, but never allocate a queries by resamples matrix.
        batch = max(1, BOOTSTRAP_MAX_CELLS // len(deltas))
        means = np.empty(self.n_bootstrap, dtype=np.float64)
        for start in range(0, self.n_bootstrap, batch):
            stop = min(self.n_bootstrap, start + batch)
            idx = self.rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
            means[start:stop] = deltas[idx].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def _apply_rule(
        self, name: str, rule: MetricRule, stats: dict[str, Any], where: str, scope: str
    ) -> None:
        label = f"{where}{name}"
        if rule.max_drop is not None and self.policy.non_inferiority and scope == "metric":
            self._non_inferiority(name, rule.max_drop, stats, label)
        elif rule.max_drop is not None:
            drop = -stats["delta"]
            worst_case = -stats["ci_high"]
            exceeded = (
                worst_case > rule.max_drop + EPS
                if self.policy.require_significance
                else drop > rule.max_drop + EPS
            )
            if exceeded:
                self.flag(
                    Verdict.FAIL,
                    f"{label} dropped by {drop:.4f} ({stats['baseline']:.4f} → "
                    f"{stats['candidate']:.4f}, 95% CI of change [{stats['ci_low']:+.4f}, "
                    f"{stats['ci_high']:+.4f}]); the policy allows {rule.max_drop:.4f}.",
                    f"{scope}:{name}.max_drop",
                )
        if rule.min is not None and stats["candidate"] < rule.min - EPS:
            self.flag(
                Verdict.FAIL,
                f"{label} is {stats['candidate']:.4f}, below the policy minimum {rule.min:.4f}.",
                f"{scope}:{name}.min",
            )

    def _non_inferiority(self, name: str, margin: float, stats: dict[str, Any], label: str) -> None:
        """Within tolerance only if the whole 95% interval of the change is above -margin."""
        low, high = stats["ci_low"], stats["ci_high"]
        if low > -margin + EPS:
            decision = "within_tolerance"
        elif high < -margin - EPS:
            decision = "regression"
        else:
            decision = "inconclusive"
        stats["tolerance"] = {"decision": decision, "margin": margin}
        interval = f"95% CI of change [{low:+.4f}, {high:+.4f}]"
        if decision == "regression":
            self.flag(
                Verdict.FAIL,
                f"{label} dropped beyond tolerance: {stats['baseline']:.4f} → "
                f"{stats['candidate']:.4f}, {interval} lies entirely below -{margin:.4f}.",
                f"metric:{name}.max_drop",
            )
        elif decision == "inconclusive":
            self.flag(
                Verdict.FAIL,
                f"{label} is not shown to be within tolerance (inconclusive): {interval} "
                f"reaches below -{margin:.4f}. More query families would narrow it.",
                f"metric:{name}.max_drop",
            )
            self.reasons[-1]["inconclusive"] = True

    def _regressed_queries(
        self,
        pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        available: Mapping[str, Any],
    ) -> None:
        primary = self.policy.primary_metric or next(
            (m for m in (*SEMANTIC_METRICS[::-1], "ann_recall") if m in available), None
        )
        if primary is None or primary not in available:
            if self.policy.max_regressed_queries is not None:
                self.flag(Verdict.NOT_COMPARABLE, _unavailable(primary or "a quality metric"))
            return
        if (
            self.policy.max_regressed_queries is not None
            and available[primary]["n_families"] < self.policy.min_queries
        ):
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"Regressed-query rule needs {self.policy.min_queries} measured families "
                f"for {primary}; only {available[primary]['n_families']} are available.",
            )
            return
        changes = []
        for b, c in pairs:
            bv, cv = _value(b, primary), _value(c, primary)
            if bv is None or cv is None:
                continue
            changes.append(
                {
                    "query_id": b["query_id"],
                    "family_id": family_id(b),
                    "tags": b.get("tags", []),
                    "filtered": bool(b.get("filtered")),
                    "baseline": bv,
                    "candidate": cv,
                    "delta": cv - bv,
                    "lost_ids": [i for i in b["ids"] if i not in set(c["ids"])],
                    "new_ids": [i for i in c["ids"] if i not in set(b["ids"])],
                    "baseline_ids": b["ids"],
                    "candidate_ids": c["ids"],
                    "baseline_evidence": b.get("evidence"),
                    "candidate_evidence": c.get("evidence"),
                    **(
                        {
                            "baseline_repeat_ids": b.get("repeat_ids"),
                            "candidate_repeat_ids": c.get("repeat_ids"),
                            "baseline_repeat_metrics": b.get("repeat_metrics"),
                            "candidate_repeat_metrics": c.get("repeat_metrics"),
                        }
                        if self.cand["metric_contract"]["version"] == 2
                        else {}
                    ),
                }
            )
        regressed = sorted((x for x in changes if x["delta"] < -EPS), key=lambda x: x["delta"])
        improved = sum(x["delta"] > EPS for x in changes)
        self.out["queries"] = {
            "primary_metric": primary,
            "regressed_count": len(regressed),
            "regressed_families": family_count(regressed),
            "improved_count": improved,
            "unchanged_count": len(changes) - len(regressed) - improved,
            "regressed": regressed[:100],
        }
        limit = self.policy.max_regressed_queries
        if limit is not None and len(regressed) > limit:
            self.flag(
                Verdict.FAIL,
                f"{len(regressed)} queries regressed on {primary}; the policy allows {limit}.",
                "regressed_queries",
            )

    def _cohorts(self, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        """Judge each tag as a cohort, then report how many cohort rules were judged.

        With ``cohorts.control = "bonferroni"`` a cohort ``max_drop`` fails only when its 95%
        interval, widened to 1 - 0.05/m for the m cohort ``max_drop`` rules judged, lies
        entirely beyond the allowed drop: a normal interval on the family-mean change with the
        family standard error. ``min`` rules are absolute floors and are not adjusted.
        """
        tags = sorted({t for b, _ in pairs for t in b.get("tags", [])})
        rule = self.policy.cohorts
        prepared = []
        for tag in tags:
            members = [(b, c) for b, c in pairs if tag in b.get("tags", [])]
            metrics = {}
            for name in QUALITY_METRICS:
                stats = self._metric_stats(members, name)
                if stats is not None:
                    metrics[name] = stats
            n_families = family_count(b for b, _ in members)
            enough_queries = n_families >= rule.min_queries
            judged_metrics = []
            if enough_queries:
                for name, metric_rule in rule.metrics.items():
                    if name not in metrics:
                        if metric_rule.required:
                            self.flag(
                                Verdict.NOT_COMPARABLE, f"cohort {tag!r}: {_unavailable(name)}"
                            )
                    elif metrics[name]["n_families"] < rule.min_queries:
                        self.warnings.append(
                            f"Cohort {tag!r}: {name} reported without judging because only "
                            f"{metrics[name]['n_families']} measured families are available "
                            f"(minimum {rule.min_queries})."
                        )
                    else:
                        judged_metrics.append(name)
            prepared.append((tag, members, metrics, n_families, enough_queries, judged_metrics))
        drop_rules = sum(
            1
            for *_, judged in prepared
            for name in judged
            if rule.metrics[name].max_drop is not None
        )
        adjust = rule.control == "bonferroni" and drop_rules > 1
        failures_before = len(self.reasons)
        for tag, members, metrics, n_families, enough_queries, judged_metrics in prepared:
            for name in judged_metrics:
                metric_rule = rule.metrics[name]
                if adjust and metric_rule.max_drop is not None:
                    self._apply_adjusted(tag, name, metric_rule, members, metrics[name], drop_rules)
                    metric_rule = MetricRule(min=metric_rule.min, required=metric_rule.required)
                self._apply_rule(
                    name, metric_rule, metrics[name], f"cohort {tag!r}: ", f"cohort:{tag}"
                )
            self.out["cohorts"][tag] = {
                "n": len(members),
                "n_families": n_families,
                "judged": bool(judged_metrics) if rule.metrics else enough_queries,
                "judged_metrics": judged_metrics,
                "metrics": metrics,
            }
        judged_rules = sum(len(judged) for *_, judged in prepared)
        failed = sum(
            1
            for r in self.reasons[failures_before:]
            if r["verdict"] == Verdict.FAIL.value and str(r.get("rule", "")).startswith("cohort:")
        )
        self.out["cohort_judging"] = {
            "cohorts": len(tags),
            "judged_cohorts": sum(1 for *_, judged in prepared if judged),
            "judged_rules": judged_rules,
            "failed_rules": failed,
            "control": rule.control,
        }
        if rule.control == "none" and judged_rules >= MANY_COHORT_RULES:
            self.warnings.append(
                f"{judged_rules} cohort rules were judged one by one: even an unchanged system "
                "fails some by chance. Measure that with `vectraudit calibrate`, or set "
                'cohorts.control to "bonferroni" in the policy.'
            )

    def _apply_adjusted(
        self,
        tag: str,
        name: str,
        rule: MetricRule,
        members: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        stats: dict[str, Any],
        m: int,
    ) -> None:
        deltas = self._family_deltas(members, name)
        confidence = 1 - 0.05 / m
        z = statistics.NormalDist().inv_cdf(1 - 0.05 / (2 * m))
        se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        low, high = stats["delta"] - z * se, stats["delta"] + z * se
        stats["ci_adjusted"] = {"low": low, "high": high, "confidence": confidence, "rules": m}
        assert rule.max_drop is not None
        if -high > rule.max_drop + EPS:
            self.flag(
                Verdict.FAIL,
                f"cohort {tag!r}: {name} dropped by {-stats['delta']:.4f} "
                f"({stats['baseline']:.4f} → {stats['candidate']:.4f}); even the "
                f"{confidence:.2%} interval adjusted for {m} cohort rules "
                f"[{low:+.4f}, {high:+.4f}] exceeds the allowed {rule.max_drop:.4f}.",
                f"cohort:{tag}:{name}.max_drop",
            )

    def _family_deltas(
        self, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], name: str
    ) -> list[float]:
        base_vals, cand_vals = [], []
        for b, c in pairs:
            bv, cv = _value(b, name), _value(c, name)
            if bv is not None and cv is not None:
                base_vals.append((family_id(b), bv))
                cand_vals.append((family_id(b), cv))
        pairs_of_means = zip(family_means(base_vals), family_means(cand_vals), strict=True)
        return [c - b for b, c in pairs_of_means]

    # --- latency ---------------------------------------------------------------------

    def _latency(self) -> None:
        bl = self.base["summary"].get("latency_ms") if self.base.get("summary") else None
        cl = self.cand["summary"].get("latency_ms") if self.cand.get("summary") else None
        if not bl or not cl:
            return
        rule = self.policy.latency
        if any(_traced(r) for r in (self.base, self.cand)):
            # Span timestamps record production load, not a controlled measurement.
            self.out["latency"] = {
                "baseline": bl,
                "candidate": cl,
                "p50_change_pct": None,
                "p95_change_pct": None,
                "same_environment": False,
                "compared": False,
                "paired": False,
            }
            self.warnings.append(
                "Latency was not judged: an imported run's latency comes from application "
                "traces recorded under unknown load, not from a controlled measurement."
            )
            return
        envs = [_environment(r) for r in (self.base, self.cand)]
        same_env = envs[0] == envs[1]
        workers = [_concurrency(r) for r in (self.base, self.cand)]
        # Latency under load is a different measurement from serial latency.
        compared = workers[0] == workers[1] and (same_env or not rule.require_same_environment)
        change = {p: (cl[p] - bl[p]) / bl[p] * 100 if bl[p] > 0 else None for p in ("p50", "p95")}
        self.out["latency"] = {
            "baseline": bl,
            "candidate": cl,
            "p50_change_pct": change["p50"],
            "p95_change_pct": change["p95"],
            "same_environment": same_env,
            "compared": compared,
            # Recorded interleaved, question by question: drift fell on both sides alike.
            "paired": _paired(self.base, self.cand),
        }
        warmups = [(r.get("timing") or {}).get("warmup") for r in (self.base, self.cand)]
        if all(isinstance(w, int) for w in warmups) and warmups[0] != warmups[1]:
            self.warnings.append(
                f"Warm-up differed (baseline {warmups[0]}, candidate {warmups[1]} calls): the "
                "first questions of one side may have met colder caches."
            )
        if workers[0] != workers[1]:
            self.warnings.append(
                f"Latency was not compared: the runs used different concurrency "
                f"({workers[0]} vs {workers[1]} queries in flight)."
            )
        elif not compared:
            self.warnings.append(
                "Latency was not compared: the runs come from different runner environments "
                f"({_env_label(envs[0])} vs {_env_label(envs[1])})."
            )
        elif not same_env:
            self.warnings.append("Latency compared across different runner environments.")
        overlapped = [_overlapped(r) for r in (self.base, self.cand)]
        if any(overlapped):
            sides = " and ".join(
                f"{side} beside {n} other{'s' * (n != 1)}"
                for side, n in zip(("baseline", "candidate"), overlapped, strict=True)
                if n
            )
            self.warnings.append(
                f"Latency includes extra load: evaluations ran at the same time in the "
                f"workbench ({sides}). Re-run alone before relying on latency."
            )
        if rule.p95_max_ms is not None and cl["p95"] > rule.p95_max_ms + EPS:
            self.flag(
                Verdict.FAIL,
                f"Candidate p95 latency is {cl['p95']:.2f} ms, above the policy limit "
                f"{rule.p95_max_ms:.2f} ms.",
                "latency.p95_max_ms",
            )
        if not compared:
            return
        for p, limit in (("p50", rule.p50_max_increase_pct), ("p95", rule.p95_max_increase_pct)):
            pct = change[p]
            rise_ms = cl[p] - bl[p]
            if (
                limit is not None
                and pct is not None
                and pct > limit + EPS
                and rise_ms > rule.min_increase_ms + EPS
            ):
                self.flag(
                    Verdict.FAIL,
                    f"{p} latency rose {pct:.1f}% ({bl[p]:.2f} → {cl[p]:.2f} ms); "
                    f"the policy allows {limit:.1f}%.",
                    f"latency.{p}_increase",
                )


def _value(q: Mapping[str, Any], name: str) -> float | None:
    if name == "ann_recall":
        v = q.get("ann_recall")
    else:
        metrics = q.get("metrics")
        v = metrics.get(name) if metrics else None
    return None if v is None else float(v)


def _missing_repeat_evidence(query: Mapping[str, Any], repeats: Any) -> str | None:
    """Why a query's per-repeat rankings cannot support a critical check, if they cannot."""
    recorded = query.get("repeat_ids")
    if recorded is None:
        return (
            "the run has no per-repeat rankings (recorded by an older version); "
            "re-run the candidate"
        )
    if isinstance(repeats, int) and len(recorded) != repeats:
        return f"only {len(recorded)} of {repeats} repeat rankings were recorded"
    if not recorded or recorded[0] != query["ids"]:
        return "the per-repeat rankings do not match the recorded ranking"
    return None


def _executions(query: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every observed ranking: repeats of earlier failed attempts, then the final attempt.

    ``attempt`` is set only when the query was retried, so plain runs read ``repeat n``.
    """
    final = query.get("attempt_count")
    retried = isinstance(final, int) and final > 1
    out: list[dict[str, Any]] = []
    for attempt in query.get("attempts") or []:
        result = attempt.get("result") or {}
        if attempt.get("status") == "failed":
            out += [
                {"attempt": attempt.get("attempt"), "repeat": n, "ids": ids}
                for n, ids in enumerate(result.get("repeat_ids") or [], 1)
            ]
    out += [
        {"attempt": final if retried else None, "repeat": n, "ids": ids}
        for n, ids in enumerate(query["repeat_ids"], 1)
    ]
    return out


def execution_label(execution: Mapping[str, Any]) -> str:
    """``repeat 2``, or ``attempt 1, repeat 2`` when the query was retried."""
    repeat = f"repeat {execution['repeat']}"
    attempt = execution.get("attempt")
    return repeat if attempt is None else f"attempt {attempt}, {repeat}"


def _critical_detail(failures: Sequence[Mapping[str, Any]], total: int) -> str:
    if not failures:
        return f"held on all {total} executions"
    first = failures[0]
    observed = (
        f"observed rank {first['rank']}"
        if first["rank"] is not None
        else "document was not retrieved"
    )
    where = execution_label(first)
    return f"failed on {len(failures)} of {total} executions; first at {where}: {observed}"


def _unavailable(name: str) -> str:
    need = "an exact reference" if name == "ann_recall" else "relevance labels"
    return f"The policy requires {name}, which needs {need} in both runs."


def _environment(run: Mapping[str, Any]) -> dict[str, Any]:
    runner = (run.get("provenance") or {}).get("runner") or {}
    return {k: runner.get(k) for k in _ENVIRONMENT_KEYS}


def _concurrency(run: Mapping[str, Any]) -> int:
    value = (run.get("timing") or {}).get("concurrency", 1)
    return value if isinstance(value, int) else 1


def _paired(base: Mapping[str, Any], cand: Mapping[str, Any]) -> bool:
    """Whether the two runs were recorded interleaved with each other (``execute_paired``)."""
    bt, ct = base.get("timing") or {}, cand.get("timing") or {}
    return bool(
        bt.get("order") == ct.get("order") == "interleaved"
        and bt.get("paired_with") == cand.get("run_id")
        and ct.get("paired_with") == base.get("run_id")
    )


def _traced(run: Mapping[str, Any]) -> bool:
    """Whether the run's latency was imported from application traces, not measured."""
    return (run.get("timing") or {}).get("source") == "trace"


def _overlapped(run: Mapping[str, Any]) -> int:
    """How many other workbench evaluations ran during this one (0 when unrecorded)."""
    value = (run.get("timing") or {}).get("overlapped_runs", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _env_label(env: Mapping[str, Any]) -> str:
    return f"{env['system']}/{env['machine']}, {env['cpu_count']} CPUs, Python {env['python']}"


def _run_brief(run: Mapping[str, Any]) -> dict[str, Any]:
    backend = run.get("backend") or {}
    provenance = run.get("provenance") or {}
    return {
        "run_id": run.get("run_id"),
        "label": run.get("label"),
        "created_at": run.get("created_at"),
        "status": run.get("status"),
        "backend": backend.get("backend"),
        "target": backend.get("target"),
        "snapshot": backend.get("snapshot"),
        "k": run["search"]["k"],
        "params": run["search"]["params"],
        "commit_sha": provenance.get("commit_sha"),
        "ci_run_url": (provenance.get("ci") or {}).get("run_url"),
        "summary": run.get("summary"),
    }


def config_diff(base: Mapping[str, Any], cand: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Settings that differ between the runs: backend config, search, connection, labels."""
    diffs = []
    for section in ("backend", "search", "connection", "labels"):
        b = _flatten(_section(base, section), section)
        c = _flatten(_section(cand, section), section)
        for path in sorted(set(b) | set(c)):
            if b.get(path) != c.get(path):
                diffs.append({"path": path, "baseline": b.get(path), "candidate": c.get(path)})
    return diffs


def _section(run: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    if section == "labels":
        return (run.get("provenance") or {}).get("labels") or {}
    data = run.get(section) or {}
    # The corpus size is reported in the run summary, not as a configuration change.
    return {k: v for k, v in data.items() if not (section == "backend" and k == "count")}


def _flatten(obj: Any, prefix: str) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}"))
        return out
    return {prefix: obj}
