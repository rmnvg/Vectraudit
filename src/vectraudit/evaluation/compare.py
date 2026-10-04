"""Compare a candidate run with its baseline under a policy, and decide.

Verdicts, in order of precedence:

- ``error``: a run is not complete (failed, partial or cancelled). Never passes.
- ``not_comparable``: different dataset version, k, metric contract or query set; too few
  queries; or a metric the policy requires was not measured in both runs.
- ``fail``: a policy rule is broken.
- ``pass``: every rule holds.

Quality deltas are paired per query (candidate - baseline) with seeded bootstrap 95%
intervals. Latency is compared only between runs from matching runner environments,
unless the policy allows otherwise.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from vectraudit.evaluation.diagnose import diagnose
from vectraudit.evaluation.metrics import QUALITY_METRICS, SEMANTIC_METRICS
from vectraudit.evaluation.policy import DEFAULT_POLICY, MetricRule, Policy

CHECK_FORMAT = "vectraudit.check"
CHECK_FORMAT_VERSION = 1
SMALL_SAMPLE = 30
EPS = 1e-9
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
            self._quality(pairs)
            self._cohorts(pairs)
            self._latency()
        verdicts = {Verdict(r["verdict"]) for r in self.reasons}
        verdict = next((v for v in _PRECEDENCE if v in verdicts), Verdict.PASS)
        self.out["verdict"] = verdict.value
        if self.out["metrics"]:
            self.out["findings"] = diagnose(self.out, self.base, self.cand, pairs)
        return self.out

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
        n = len(b["queries"])
        if n < self.policy.min_queries:
            self.flag(
                Verdict.NOT_COMPARABLE,
                f"{n} queries is below the policy minimum of {self.policy.min_queries}.",
            )
            return False
        self._context_warnings(n)
        return True

    def _context_warnings(self, n: int) -> None:
        b, c = self.base, self.cand
        if n < SMALL_SAMPLE:
            self.warnings.append(
                f"Only {n} queries: intervals are wide and one query moves the mean by "
                f"{1 / n:.2f}. Treat deltas as indicative."
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
        if not available:
            self.flag(
                Verdict.NOT_COMPARABLE,
                "No quality metric was measured in both runs: add relevance labels or an "
                "exact reference.",
            )
            return
        judged = 0
        for name, rule in self.policy.metrics.items():
            if name not in available:
                if rule.required:
                    self.flag(Verdict.NOT_COMPARABLE, _unavailable(name))
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
                base_vals.append(bv)
                cand_vals.append(cv)
        if not base_vals:
            return None
        deltas = np.asarray(cand_vals) - np.asarray(base_vals)
        low, high = self._bootstrap(deltas)
        return {
            "n": len(deltas),
            "baseline": statistics.fmean(base_vals),
            "candidate": statistics.fmean(cand_vals),
            "delta": float(deltas.mean()),
            "ci_low": low,
            "ci_high": high,
        }

    def _bootstrap(self, deltas: np.ndarray[Any, Any]) -> tuple[float, float]:
        if len(deltas) == 1:
            return float(deltas[0]), float(deltas[0])
        idx = self.rng.integers(0, len(deltas), size=(self.n_bootstrap, len(deltas)))
        means = deltas[idx].mean(axis=1)
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
        changes = []
        for b, c in pairs:
            bv, cv = _value(b, primary), _value(c, primary)
            if bv is None or cv is None:
                continue
            changes.append(
                {
                    "query_id": b["query_id"],
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
            judged = len(members) >= rule.min_queries
            self.out["cohorts"][tag] = {"n": len(members), "judged": judged, "metrics": metrics}
            if not judged:
                continue
            for name, metric_rule in rule.metrics.items():
                if name in metrics:
                    self._apply_rule(name, metric_rule, metrics[name], f"cohort {tag!r}: ")
                elif metric_rule.required:
                    self.flag(Verdict.NOT_COMPARABLE, f"cohort {tag!r}: {_unavailable(name)}")

    # --- latency ---------------------------------------------------------------------

    def _latency(self) -> None:
        bl = self.base["summary"].get("latency_ms") if self.base.get("summary") else None
        cl = self.cand["summary"].get("latency_ms") if self.cand.get("summary") else None
        if not bl or not cl:
            return
        rule = self.policy.latency
        envs = [_environment(r) for r in (self.base, self.cand)]
        same_env = envs[0] == envs[1]
        compared = same_env or not rule.require_same_environment
        change = {p: (cl[p] - bl[p]) / bl[p] * 100 if bl[p] > 0 else None for p in ("p50", "p95")}
        self.out["latency"] = {
            "baseline": bl,
            "candidate": cl,
            "p50_change_pct": change["p50"],
            "p95_change_pct": change["p95"],
            "same_environment": same_env,
            "compared": compared,
        }
        if not compared:
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
            if limit is not None and pct is not None and pct > limit + EPS:
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


def _unavailable(name: str) -> str:
    need = "an exact reference" if name == "ann_recall" else "relevance labels"
    return f"The policy requires {name}, which needs {need} in both runs."


def _environment(run: Mapping[str, Any]) -> dict[str, Any]:
    runner = (run.get("provenance") or {}).get("runner") or {}
    return {k: runner.get(k) for k in _ENVIRONMENT_KEYS}


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
