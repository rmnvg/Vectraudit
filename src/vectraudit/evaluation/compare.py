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
EPS = 1e-9
BOOTSTRAP_MAX_CELLS = 1_000_000
MAX_CRITICAL_FAILURES = 10
_ENVIRONMENT_KEYS = ("system", "machine", "cpu_count", "python")


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_COMPARABLE = "not_comparable"


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
        self.reasons: list[dict[str, str]] = []
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
            "queries": None,
            "findings": [],
            "config_diff": config_diff(base, cand),
        }

    def flag(self, verdict: Verdict, message: str) -> None:
        self.reasons.append({"verdict": verdict.value, "message": message})

    def result(self) -> dict[str, Any]:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        if self._runs_complete() and self._comparable():
            pairs = self._pairs()
            self.out["n_queries"] = len(pairs)
            self.out["n_families"] = family_count(b for b, _ in pairs)
            self._safety(pairs)
            self._critical_queries()
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
                    Verdict.FAIL, f"Forbidden documents delivered for {candidate['query_id']}."
                )
            if candidate.get("no_answer_correct") is False:
                self.flag(Verdict.FAIL, f"No-answer assertion failed for {candidate['query_id']}.")

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
                )
        self.out["critical_queries"] = assertions

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
            self.flag(Verdict.NOT_COMPARABLE, "The runs use different metric contract versions.")
        if len(self.reasons) > before:
            return False
        if {q["query_id"] for q in b["queries"]} != {q["query_id"] for q in c["queries"]}:
            self.flag(Verdict.NOT_COMPARABLE, "The runs measured different query sets.")
            return False
        paired = self._pairs()
        if any(family_id(bq) != family_id(cq) for bq, cq in paired):
            self.flag(Verdict.NOT_COMPARABLE, "Query family assignments differ between runs.")
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
            and self.policy.critical_queries
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
            self._apply_rule(name, rule, available[name], "")
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
        self, name: str, rule: MetricRule, stats: Mapping[str, Any], where: str
    ) -> None:
        label = f"{where}{name}"
        if rule.max_drop is not None:
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
                )
        if rule.min is not None and stats["candidate"] < rule.min - EPS:
            self.flag(
                Verdict.FAIL,
                f"{label} is {stats['candidate']:.4f}, below the policy minimum {rule.min:.4f}.",
            )

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
            )

    def _cohorts(self, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        tags = sorted({t for b, _ in pairs for t in b.get("tags", [])})
        rule = self.policy.cohorts
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
                        self._apply_rule(name, metric_rule, metrics[name], f"cohort {tag!r}: ")
            self.out["cohorts"][tag] = {
                "n": len(members),
                "n_families": n_families,
                "judged": bool(judged_metrics) if rule.metrics else enough_queries,
                "judged_metrics": judged_metrics,
                "metrics": metrics,
            }

    # --- latency ---------------------------------------------------------------------

    def _latency(self) -> None:
        bl = self.base["summary"].get("latency_ms") if self.base.get("summary") else None
        cl = self.cand["summary"].get("latency_ms") if self.cand.get("summary") else None
        if not bl or not cl:
            return
        rule = self.policy.latency
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
        }
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
        if rule.p95_max_ms is not None and cl["p95"] > rule.p95_max_ms + EPS:
            self.flag(
                Verdict.FAIL,
                f"Candidate p95 latency is {cl['p95']:.2f} ms, above the policy limit "
                f"{rule.p95_max_ms:.2f} ms.",
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
