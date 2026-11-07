"""The corpus atlas: categories, held-out hiding, sampling invariants and the binary payload."""

import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.core import atlas as A
from vectraudit.core.atlas import Category as C
from vectraudit.pipelines.web import create_workbench

runner = CliRunner()
D = 8


def corpus(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [f"d{i}" for i in range(n)], rng.standard_normal((n, D)).astype("float32")


def entry(qid, ids, staged=(), error=None):
    stages = [{"hits": [{"id": d} for d in staged]}] if staged else []
    return {"query_id": qid, "ids": None if error else ids, "error": error,
            "evidence": {"stages": stages}}  # fmt: skip


def run(*entries, run_id="r"):
    return {"run_id": run_id, "label": run_id, "queries": list(entries)}


def question(qid, vec, relevant=(), **kw):
    return A.AtlasQuestion(qid, vec, frozenset(relevant), **kw)


def by_id(atlas):
    ids = atlas.sidecar["points"]["ids"]
    return {i: C(int(c)) for i, c in zip(ids, atlas.category, strict=True)}


@pytest.fixture
def scene():
    ids, vectors = corpus()
    qv = vectors[:6] + 0.01
    questions = [
        question("q-pass", qv[0], ["d1"]),
        question("q-fail-never", qv[1], ["d2"]),
        question("q-fail-dropped", qv[2], ["d3"]),
        question("q-regress", qv[3], ["d4", "d5"]),
        question("q-none", qv[4], no_answer=True),
        question("q-unjudged", qv[5]),
    ]
    selected = run(
        entry("q-pass", ["d1", "d9"]),
        entry("q-fail-never", ["d9"]),
        entry("q-fail-dropped", ["d9"], staged=["d3", "d9"]),
        entry("q-regress", ["d4"]),
        entry("q-none", []),
    )
    baseline = run(entry("q-regress", ["d4", "d5"]), entry("q-pass", ["d1"]), run_id="base")
    return ids, vectors, questions, selected, baseline


# --- categories -------------------------------------------------------------


def test_every_point_is_categorised_from_the_run(scene) -> None:
    ids, vectors, questions, selected, baseline = scene
    atlas = A.build_atlas(
        ids, vectors, questions, selected_run=selected, baseline_run=baseline,
        hubs={"d20"}, flagged={"d30": "corpus scan: x", "d2": "poison suite: y"},
    )  # fmt: skip
    cat = by_id(atlas)
    assert cat["q-pass"] is C.QUESTION_PASSED
    assert cat["q-fail-never"] is C.QUESTION_FAILED
    assert cat["q-fail-dropped"] is C.QUESTION_FAILED
    assert cat["q-regress"] is C.QUESTION_REGRESSED
    assert cat["q-none"] is C.QUESTION_NO_ANSWER
    assert cat["q-unjudged"] is C.QUESTION_UNJUDGED
    assert cat["d1"] is C.RELEVANT
    assert cat["d3"] is C.LOST_DROPPED  # in a recorded stage, missing from what was delivered
    assert cat["d5"] is C.LOST_NEVER_RETRIEVED
    assert cat["d2"] is C.FLAGGED  # flagged outranks lost
    assert cat["d20"] is C.HUB
    assert cat["d30"] is C.FLAGGED
    assert cat["d100"] is C.DOCUMENT
    assert atlas.sidecar["counts"]["by_category"]["lost_dropped"] == 1


def test_without_a_run_nothing_is_called_lost_or_failed(scene) -> None:
    ids, vectors, questions, _, _ = scene
    cat = by_id(A.build_atlas(ids, vectors, questions))
    assert cat["d2"] is C.RELEVANT
    assert cat["q-pass"] is C.QUESTION_UNJUDGED
    assert cat["q-none"] is C.QUESTION_NO_ANSWER


def test_an_errored_or_unanswered_question_is_failed_never_passed(scene) -> None:
    ids, vectors, questions, _, _ = scene
    errored = run(entry("q-pass", [], error={"code": "timeout"}))
    cat = by_id(A.build_atlas(ids, vectors, questions, selected_run=errored))
    assert cat["q-pass"] is C.QUESTION_FAILED
    assert cat["q-fail-never"] is C.QUESTION_UNJUDGED  # not in the run: unknown, not failed


# --- held-out ---------------------------------------------------------------


def heldout_scene():
    ids, vectors = corpus()
    qs = [
        question("open", vectors[0], ["d1"]),
        question("secret", vectors[1], ["d2"], heldout=True),
    ]
    sel = run(entry("open", ["d1"]), entry("secret", ["d9"]))
    return ids, vectors, qs, sel


def test_heldout_questions_are_hidden_by_default() -> None:
    ids, vectors, qs, sel = heldout_scene()
    atlas = A.build_atlas(ids, vectors, qs, selected_run=sel)
    cat = by_id(atlas)
    assert "secret" not in cat
    assert atlas.sidecar["counts"]["heldout_hidden"] == 1
    # The held-out question's label and failure colour no document.
    assert cat["d2"] is C.DOCUMENT
    assert atlas.sidecar["heldout"]["shown"] is False


def test_shown_heldout_questions_are_uncoloured_and_colour_nothing() -> None:
    ids, vectors, qs, sel = heldout_scene()
    atlas = A.build_atlas(ids, vectors, qs, selected_run=sel, show_heldout=True)
    cat = by_id(atlas)
    assert cat["secret"] is C.QUESTION_UNJUDGED  # it failed in the run, and is not drawn as failed
    assert cat["d2"] is C.DOCUMENT  # nor is its relevant document drawn as lost or relevant
    assert cat["open"] is C.QUESTION_PASSED
    assert atlas.sidecar["counts"]["heldout_shown"] == 1


# --- sampling ---------------------------------------------------------------


def test_sampling_keeps_every_question_and_notable_document(scene) -> None:
    ids, vectors, questions, selected, baseline = scene
    hubs = {f"d{i}" for i in range(50, 70)}
    atlas = A.build_atlas(
        ids, vectors, questions, selected_run=selected, baseline_run=baseline, hubs=hubs,
        max_points=40, seed=3,
    )  # fmt: skip
    cat = by_id(atlas)
    assert {q.query_id for q in questions} <= set(cat)
    assert hubs <= set(cat)
    assert {"d1", "d3", "d5"} <= set(cat)
    plain = [i for i, c in cat.items() if c is C.DOCUMENT]
    notable = len(cat) - len(plain)
    assert len(plain) == max(0, 40 - notable)
    assert atlas.sidecar["sampling"]["over_cap"] is (len(cat) > 40)


def test_the_cap_may_be_exceeded_by_notable_points_but_never_drops_one(scene) -> None:
    ids, vectors, questions, selected, _ = scene
    atlas = A.build_atlas(ids, vectors, questions, selected_run=selected, max_points=1)
    assert atlas.sidecar["sampling"]["over_cap"] is True
    assert len(atlas) >= len(questions)
    assert atlas.sidecar["counts"]["plain_documents_shown"] == 0


def test_the_payload_is_deterministic_and_seeded(scene) -> None:
    ids, vectors, questions, selected, _ = scene
    make = lambda seed: A.build_atlas(  # noqa: E731
        ids, vectors, questions, selected_run=selected, max_points=60, seed=seed
    )
    a, b, other = make(1), make(1), make(2)
    assert a.payload() == b.payload()
    assert a.sidecar["bin"]["sha256"] == hashlib.sha256(a.payload()).hexdigest()
    assert a.sidecar["points"]["ids"] == b.sidecar["points"]["ids"]
    assert a.sidecar["bin"]["sha256"] != other.sidecar["bin"]["sha256"]


def test_a_question_is_placed_where_its_vector_projects(scene) -> None:
    ids, vectors, questions, _, _ = scene
    atlas = A.build_atlas(ids, vectors, questions)
    pca = A.fit_pca(vectors, 2)
    want = pca.transform(questions[0].vector[None, :])[0]
    at = atlas.sidecar["points"]["ids"].index("q-pass")
    np.testing.assert_allclose([atlas.x[at], atlas.y[at]], want, rtol=1e-4, atol=1e-4)


# --- payload ----------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8])
def test_payload_round_trips_for_any_length(n) -> None:
    rng = np.random.default_rng(n)
    x, y = rng.standard_normal(n).astype("float32"), rng.standard_normal(n).astype("float32")
    cat = rng.integers(0, 11, n).astype("uint8")
    row = rng.integers(0, 10**6, n).astype("uint32")
    data = A.encode_payload(x, y, cat, row)
    assert len(data) % 4 == 0
    rx, ry, rc, rr = A.decode_payload(data, n)
    np.testing.assert_array_equal(rx, x)
    np.testing.assert_array_equal(ry, y)
    np.testing.assert_array_equal(rc, cat)
    np.testing.assert_array_equal(rr, row)
    with pytest.raises(A.AtlasError):
        A.decode_payload(data[:-4], n)


def test_the_sidecar_describes_the_payload(scene) -> None:
    ids, vectors, questions, selected, _ = scene
    atlas = A.build_atlas(ids, vectors, questions, selected_run=selected, name="demo")
    s = atlas.sidecar
    assert (s["format"], s["format_version"], s["local_only"]) == ("vectraudit.atlas", 1, True)
    assert s["points"]["n"] == s["bin"]["n"] == len(atlas) == len(s["points"]["ids"])
    assert len(s["points"]["titles"]) == len(atlas)
    assert s["bin"]["bytes"] == len(atlas.payload())
    assert [item["id"] for item in s["legend"]] == list(range(11))
    assert "distorts distance" in s["projection"]["note"]
    json.dumps(s, allow_nan=False)


def test_unusable_inputs_are_refused() -> None:
    ids, vectors = corpus(10)
    bad = vectors.copy()
    bad[0, 0] = np.nan
    for args in (
        (ids, bad, []),
        (ids[:5], vectors, []),
        ([*ids, "d0"], np.vstack([vectors, vectors[:1]]), []),
        (["a", "b"], np.ones((2, 1), dtype="float32"), []),
    ):
        with pytest.raises(A.AtlasError):
            A.build_atlas(*args)
    for cap in (0, A.HARD_MAX_POINTS + 1):
        with pytest.raises(A.AtlasError, match="max_points"):
            A.build_atlas(ids, vectors, [], max_points=cap)
    with pytest.raises(A.AtlasError, match="dimension"):
        A.build_atlas(ids, vectors, [question("q", np.ones(D + 1))])


def test_questions_without_vectors_are_counted_not_drawn() -> None:
    ids, vectors = corpus(10)
    atlas = A.build_atlas(ids, vectors, [question("q", None), question("r", vectors[0])])
    assert atlas.sidecar["counts"]["questions_without_vectors"] == 1
    assert "q" not in atlas.sidecar["points"]["ids"]


def test_titles_are_omitted_for_very_large_atlases(monkeypatch) -> None:
    monkeypatch.setattr(A, "TITLES_MAX_POINTS", 5)
    ids, vectors = corpus(10)
    atlas = A.build_atlas(ids, vectors, [], doc_titles=[f"t{i}" for i in range(10)])
    assert atlas.sidecar["points"]["titles"] is None
    assert atlas.sidecar["points"]["titles_omitted"] is True


# --- the command and the route ----------------------------------------------


def write_files(tmp_path, n=120):
    ids, vectors = corpus(n)
    rows = [
        {"doc_id": d, "text": f"Title of {d}\nbody", "metadata": {"title": f"T {d}"},
         "vector": v.tolist()}
        for d, v in zip(ids, vectors, strict=True)
    ]  # fmt: skip
    (tmp_path / "corpus.jsonl").write_text("\n".join(map(json.dumps, rows)))
    queries = [
        {"query_id": "a", "vector": vectors[0].tolist(), "relevant": {"d0": 1}},
        {"query_id": "h", "vector": vectors[1].tolist(), "relevant": {"d1": 1}, "split": "heldout"},
    ]
    (tmp_path / "ds.jsonl").write_text("\n".join(map(json.dumps, queries)))


def atlas_cmd(tmp_path, *extra):
    return runner.invoke(
        app,
        ["workbench", "atlas", str(tmp_path / "corpus.jsonl"), str(tmp_path / "ds.jsonl"),
         "--out", str(tmp_path / "wb"), "--name", "demo", *extra],
    )  # fmt: skip


def test_the_command_writes_a_bundle_and_hides_heldout(tmp_path) -> None:
    write_files(tmp_path)
    out = atlas_cmd(tmp_path)
    assert out.exit_code == 0, out.output
    side = json.loads((tmp_path / "wb" / "atlas-demo.json").read_text())
    data = (tmp_path / "wb" / "atlas-demo.bin").read_bytes()
    assert hashlib.sha256(data).hexdigest() == side["bin"]["sha256"]
    assert side["bin"]["file"] == "atlas-demo.bin"
    assert "h" not in side["points"]["ids"]
    assert "1 held-out" in out.output
    assert side["points"]["titles"][0] == "T d0"
    assert atlas_cmd(tmp_path, "--show-heldout").exit_code == 0
    shown = json.loads((tmp_path / "wb" / "atlas-demo.json").read_text())
    assert "h" in shown["points"]["ids"]


def test_a_large_cap_needs_its_flag_and_a_bad_name_is_refused(tmp_path) -> None:
    write_files(tmp_path)
    refused = atlas_cmd(tmp_path, "--max-points", "60000")
    assert refused.exit_code == 2
    assert "--large-atlas" in refused.output
    assert atlas_cmd(tmp_path, "--max-points", "60000", "--large-atlas").exit_code == 0
    assert atlas_cmd(tmp_path, "--max-points", "2000000", "--large-atlas").exit_code == 2
    assert atlas_cmd(tmp_path, "--name", "../x").exit_code == 2


def test_a_corpus_without_vectors_says_how_to_get_them(tmp_path) -> None:
    write_files(tmp_path)
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"doc_id": "d0", "text": "hello"}))
    out = atlas_cmd(tmp_path)
    assert out.exit_code == 2
    assert "--embedder" in " ".join(out.output.split())


def test_the_workbench_serves_the_points_and_nothing_else(tmp_path) -> None:
    write_files(tmp_path)
    assert atlas_cmd(tmp_path).exit_code == 0
    root = tmp_path / "wb"
    client = TestClient(create_workbench(root))
    r = client.get("/api/workbench/atlas/demo")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.content == (root / "atlas-demo.bin").read_bytes()
    assert client.get("/api/workbench/atlas/missing").status_code == 404
    assert client.get("/api/workbench/atlas/.hidden").status_code == 404  # not a valid name
    (tmp_path / "secret.bin").write_bytes(b"x")
    (root / "atlas-link.bin").symlink_to(tmp_path / "secret.bin")
    assert client.get("/api/workbench/atlas/link").status_code == 404
    # The sidecar is listed like any workbench artifact.
    names = [a["name"] for a in client.get("/api/workbench").json()["artifacts"]]
    assert "atlas-demo.json" in names


# --- the frontend's fixture ---------------------------------------------------


def test_the_frontend_fixture_matches_what_the_builder_writes() -> None:
    """frontend/src/lib/fixtures/atlas.{json,bin} are real output; regenerate them if this fails."""
    from pathlib import Path

    fixtures = Path(__file__).parents[1] / "frontend" / "src" / "lib" / "fixtures"
    fixture = json.loads((fixtures / "atlas.json").read_text())
    data = (fixtures / "atlas.bin").read_bytes()
    ids, vectors = corpus(30)
    current = A.build_atlas(ids, vectors, [question("q", vectors[0], ["d1"])], name="fixture")
    assert set(fixture) == set(current.sidecar)
    assert set(fixture["points"]) == set(current.sidecar["points"])
    assert set(fixture["regions"][0]) >= set(current.sidecar["regions"][0]) - {"pool"}
    assert hashlib.sha256(data).hexdigest() == fixture["bin"]["sha256"]
    x, y, category, row = A.decode_payload(data, fixture["bin"]["n"])
    assert len(x) == len(y) == len(category) == len(row) == len(fixture["points"]["ids"])
    assert fixture["format"] == "vectraudit.atlas"
