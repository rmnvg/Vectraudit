"""The review sample: uniform, reproducible, never held-out, and refused when edited."""

import copy
import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation import review_sample as R
from vectraudit.evaluation.dataset import parse_dataset

runner = CliRunner()


def dataset(n_families=80, *, per_family=2, heldout=(), extra=()):
    rows = []
    for f in range(n_families):
        for j in range(per_family):
            rows.append(
                {
                    "query_id": f"q{f}-{j}",
                    "text": f"question {f} {j}",
                    "family_id": f"fam{f}",
                    "split": "heldout" if f in heldout else "explore",
                }
            )
    rows += list(extra)
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def run_for(ds, *, errored=(), missing=()):
    entries = []
    for q in ds.queries:
        if q.query_id in missing:
            continue
        entries.append(
            {
                "query_id": q.query_id,
                "family_id": q.family_id,
                "ids": ["d"],
                "error": {"code": "timeout"} if q.query_id in errored else None,
            }
        )
    return {"run_id": "r1", "dataset": {"sha256": ds.sha256}, "queries": entries}


# --- drawing ------------------------------------------------------------------------


def test_the_draw_is_pinned_so_it_never_changes_between_versions() -> None:
    frame = [f"fam{i}" for i in range(20)]
    assert R.draw(frame, 5, 0) == ["fam10", "fam11", "fam15", "fam7", "fam9"]  # recorded 2026-11-09
    assert R.draw(frame, 5, 1) == ["fam10", "fam11", "fam15", "fam2", "fam6"]


def test_the_draw_ignores_input_order_and_duplicates() -> None:
    frame = [f"fam{i}" for i in range(50)]
    assert R.draw(frame, 10, 3) == R.draw(list(reversed(frame)), 10, 3)
    assert R.draw(frame + frame, 10, 3) == R.draw(frame, 10, 3)


def test_every_family_is_equally_likely_to_be_chosen() -> None:
    frame = [f"fam{i}" for i in range(40)]
    counts: Counter[str] = Counter()
    for seed in range(2000):
        counts.update(R.draw(frame, 10, seed))
    expected = 2000 * 10 / 40  # 500 each
    assert all(abs(c - expected) < 75 for c in counts.values())  # about 3.5 standard deviations
    assert len(counts) == 40


# --- the frame ------------------------------------------------------------------------


def test_the_frame_is_exploratory_families_that_did_not_error() -> None:
    ds = dataset(10, heldout={0, 1}, extra=[{"query_id": "solo", "text": "alone"}])
    run = run_for(ds, errored={"q2-0"}, missing={"q3-1"})
    frame = R.frame_of(run, ds)
    assert "fam0" not in frame
    assert "fam1" not in frame  # held out
    assert "fam2" not in frame  # one member errored: judged completely or not at all
    assert "fam3" not in frame  # one member missing from the run
    assert frame["fam4"] == ["q4-0", "q4-1"]
    assert frame["solo"] == ["solo"]  # a question without a family is its own unit


# A family cannot cross the explore/held-out boundary: parse_dataset refuses it, so the frame
# never has to split one.


# --- the file -------------------------------------------------------------------------


def test_a_sample_is_built_written_and_read_back() -> None:
    ds = dataset(80)
    run = run_for(ds)
    content = R.build_sample(run, ds, size=40, seed=7)
    assert content["unit"] == "family"
    assert len(content["chosen"]) == 40
    assert set(content["questions"]) == set(content["chosen"])
    assert all(len(ids) == 2 for ids in content["questions"].values())
    assert content["dataset_sha256"] == ds.sha256
    assert "text" not in json.dumps(content).replace('"unit"', "")  # identifiers only
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = R.write_sample(content, Path(d) / "s.json")
        read = R.read_sample(path)
    assert read.chosen == tuple(content["chosen"])
    assert read.seed == 7
    assert len(read.sha256) == 64
    assert R.problems_with(read, run) == []


def test_a_sample_that_is_too_small_or_leaves_nothing_to_correct_is_refused() -> None:
    ds = dataset(60)
    run = run_for(ds)
    with pytest.raises(R.ReviewSampleError, match="too small") as e:
        R.build_sample(run, ds, size=10)
    assert "at least 30" in e.value.hint
    with pytest.raises(R.ReviewSampleError, match="leaves none to correct"):
        R.build_sample(run, ds, size=60)
    with pytest.raises(R.ReviewSampleError, match=r"different|not recorded"):
        R.build_sample(run, dataset(61), size=30)


def _unchosen(c):
    return next(f for f in c["frame"] if f not in c["chosen"])


def test_an_edited_sample_is_refused(tmp_path) -> None:
    ds = dataset(80)
    content = R.build_sample(run_for(ds), ds, size=40, seed=2)
    path = tmp_path / "s.json"

    def attempt(edit):
        changed = copy.deepcopy(content)
        edit(changed)
        R.write_sample(changed, path)
        return R.read_sample(path)

    # Swapping a chosen family for a hand-picked one is not what the seed draws.
    with pytest.raises(R.ReviewSampleError, match="not what the recorded seed draws"):
        attempt(
            lambda c: c["chosen"].__setitem__(
                0, next(f for f in c["frame"] if f not in c["chosen"])
            )
        )
    # Shrinking the frame to make a favourable draw changes its digest.
    with pytest.raises(R.ReviewSampleError, match="edited"):
        attempt(lambda c: c["frame"].remove(c["frame"][0]))
    with pytest.raises(R.ReviewSampleError, match=r"not a vectraudit\.review_sample"):
        attempt(lambda c: c.__setitem__("format", "other"))
    with pytest.raises(R.ReviewSampleError, match="lacks the field"):
        attempt(lambda c: c.pop("seed"))
    with pytest.raises(R.ReviewSampleError, match="Cannot read"):
        R.read_sample(tmp_path / "missing.json")


def test_a_sample_for_another_run_is_reported(tmp_path) -> None:
    ds = dataset(80)
    run = run_for(ds)
    path = R.write_sample(R.build_sample(run, ds, size=40), tmp_path / "s.json")
    sample = R.read_sample(path)
    other = run_for(dataset(80, per_family=3))
    assert any("different dataset" in p for p in R.problems_with(sample, other))
    errored = run_for(ds, errored={"q5-0", "q6-1"})
    assert any("missing from this run or errored" in p for p in R.problems_with(sample, errored))


# --- the command ------------------------------------------------------------------------


def test_the_command_writes_a_sample_and_refuses_bad_ones(tmp_path) -> None:
    ds = dataset(80)
    (tmp_path / "ds.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"query_id": q.query_id, "text": q.text, "family_id": q.family_id, "split": q.split}
            )
            for q in ds.queries
        )
    )
    from vectraudit.evaluation.dataset import load_dataset

    loaded = load_dataset(tmp_path / "ds.jsonl")
    (tmp_path / "run.json").write_text(json.dumps(_valid_run(loaded)))
    base = [
        "workbench",
        "review-sample",
        str(tmp_path / "run.json"),
        "--dataset",
        str(tmp_path / "ds.jsonl"),
        "--out",
        str(tmp_path / "sample.json"),
    ]
    done = runner.invoke(app, [*base, "--size", "40", "--seed", "3"])
    assert done.exit_code == 0, done.output
    assert "Drew 40 of 80 families (80 questions)" in done.output
    again = json.loads((tmp_path / "sample.json").read_text())
    assert again["chosen"] == R.draw(again["frame"], 40, 3)
    small = runner.invoke(app, [*base, "--size", "5"])
    assert small.exit_code == 2
    assert "too small" in small.output


def _valid_run(ds):
    """A schema-valid run record for the dataset (every question answered, nothing failed)."""
    from vectraudit.connectors.base import BackendInfo, Capabilities, Connector, Distance
    from vectraudit.evaluation.runner import RunOptions, execute_run

    class Stub(Connector):
        name = "stub"

        def describe(self):
            return BackendInfo("stub", Distance.L2, 3, 1)

        def capabilities(self):
            return Capabilities()

        def _search(self, request):
            return ["d1"]

    import numpy as np

    vectors = parse_dataset(
        "\n".join(
            json.dumps(
                {"query_id": q.query_id, "vector": [1.0, 0.0, 0.0], "family_id": q.family_id}
            )
            for q in ds.queries
        )
    )
    del np
    run = execute_run(vectors, Stub(), RunOptions(k=1, warmup=0))
    run["dataset"]["sha256"] = ds.sha256
    return run
