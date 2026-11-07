"""The poison suite: deterministic, sentinel-only, exploratory-only, original files untouched."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import load_dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows
from vectraudit.pipelines.poison import (
    ATTACKS,
    PoisonError,
    build_suite,
    canary_token,
    intended_result,
    load_corpus,
    load_generator,
    select_targets,
    top_terms,
    write_suite,
)

CORPUS = [
    {
        "doc_id": "refund-policy",
        "title": "Refund policy for annual plans",
        "text": "Refund policy for annual plans. Customers may request a refund within "
        "thirty days.\nContact billing for exceptions.",
        "metadata": {"tenant": "acme", "current": True},
        "vector": [1.0, 0.0, 0.0],
    },
    {
        "doc_id": "reset-guide",
        "text": "Reset your password from the account page. A reset email arrives within minutes.",
        "metadata": {"tenant": "acme", "current": True},
        "vector": [0.0, 1.0, 0.0],
    },
    {
        "doc_id": "shipping",
        "text": "Shipping takes five business days for domestic orders.",
        "metadata": {"tenant": "other", "current": True},
        "vector": [0.0, 0.0, 1.0],
    },
    {
        "doc_id": "old-policy",
        "text": "Legacy refund rules that no longer apply.",
        "metadata": {"tenant": "acme", "current": False},
        "vector": [0.9, 0.1, 0.0],
    },
]
ROWS = [
    {
        "query_id": "q-refund",
        "text": "How do I get a refund?",
        "vector": [1.0, 0.1, 0.0],
        "relevant": {"refund-policy": 2},
        "context": {"tenant": "acme"},
        "filter": {"current": True},
        "tags": ["billing"],
    },
    {"query_id": "q-reset", "text": "Reset my password", "relevant": {"reset-guide": 1}},
    {
        "query_id": "q-held",
        "text": "Where is my order?",
        "relevant": {"shipping": 1},
        "split": "heldout",
    },
    {"query_id": "q-vector", "vector": [0.0, 1.0, 0.0], "relevant": {"reset-guide": 1}},
    {"query_id": "q-open", "text": "Anything else?"},
]


def write(tmp_path, corpus=CORPUS, rows=ROWS):
    c = tmp_path / "corpus.jsonl"
    c.write_text("\n".join(json.dumps(r) for r in corpus) + "\n")
    d = tmp_path / "benchmark.jsonl"
    d.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return c, d


def suite(tmp_path, targets=("q-refund", "q-reset"), out=None, **options):
    corpus_path, dataset_path = write(tmp_path)
    corpus, dataset = load_corpus(corpus_path), load_dataset(dataset_path)
    chosen = select_targets(dataset, corpus, targets=list(targets), sample=None, seed=0)
    attacks = options.pop("attacks", ATTACKS)
    seed = options.get("seed", 0)
    poisoned, rows = build_suite(corpus, dataset, chosen, attacks=attacks, **options)
    out = out or tmp_path / "suite"
    descriptor = write_suite(
        out, corpus_path=corpus_path, dataset_path=dataset_path, dataset=dataset,
        poisoned=poisoned, dataset_rows_=rows, attacks=attacks,
        per_target=options.get("per_target", 1), match_filters=options.get("match_filters", False),
        seed=seed, generator=None, targets=list(targets),
    )  # fmt: skip
    return poisoned, descriptor, out


def by_attack(poisoned, target="q-refund"):
    return {r["poison"]["attack"]: r for r in poisoned if target in r["poison"]["targets"]}


def test_every_attack_gets_a_document_with_the_documented_id_canary_and_block(tmp_path):
    poisoned, descriptor, _ = suite(tmp_path)
    assert len(poisoned) == 2 * len(ATTACKS)
    docs = by_attack(poisoned)
    assert set(docs) == set(ATTACKS)
    for attack, row in docs.items():
        assert row["doc_id"] == f"poison:{attack}:q-refund:1"
        block = row["poison"]
        assert block["canary"] == canary_token(0, attack, "q-refund", 1)
        assert block["targets"] == ["q-refund"]
        assert block["template_version"] == 1
        assert block["canary"] in row["text"]
    assert len({r["poison"]["canary"] for r in poisoned}) == len(poisoned)
    assert descriptor["parameters"]["attacks"] == list(ATTACKS)


def test_templates_follow_the_taxonomy_and_the_claim_is_only_a_sentinel(tmp_path):
    docs = by_attack(suite(tmp_path)[0])
    claim = lambda attack: docs[attack]["poison"]["intended_result"]  # noqa: E731
    assert re.fullmatch(r"Contact support at canary-[0-9a-f]{12}@example\.invalid", claim("inject"))
    assert all(claim(a) in docs[a]["text"] for a in ATTACKS)
    assert docs["hijack"]["text"].startswith("How do I get a refund?")
    assert "refund" in docs["hijack"]["text"].lower().split(". ")[0]
    assert docs["inject"]["text"].startswith("Refund policy for annual plans.")
    assert "include the code canary-" in docs["inject"]["text"]
    assert docs["contradict"]["text"].startswith(
        "Update: guidance in 'Refund policy for annual plans' is withdrawn."
    )
    assert "supersedes all other documents" in docs["authority"]["text"]
    assert "system prompt" in docs["exfil"]["text"]
    assert intended_result("canary-abc") == "Contact support at canary-abc@example.invalid"


def test_the_inject_copy_is_truncated_and_hijack_uses_rare_terms_of_the_answer():
    long = {"doc_id": "d", "text": "word " * 200, "metadata": {}}
    from vectraudit.pipelines.poison import Target, render_text

    query = parse_dataset('{"query_id": "q", "text": "a question", "relevant": {"d": 1}}').queries[
        0
    ]
    text = render_text("inject", Target(query, (long,)), "canary-x", terms=[])
    assert text.index("...") <= 301
    docs = [
        {"doc_id": "a", "text": "zebra stripes migration of the herd"},
        {"doc_id": "b", "text": "the herd of the the the the common words the"},
    ]
    df = {
        "the": 2,
        "zebra": 1,
        "stripes": 1,
        "migration": 1,
        "herd": 2,
        "common": 1,
        "words": 1,
        "of": 2,
    }
    assert top_terms(docs[:1], df, 2)[:3] == ["migration", "stripes", "zebra"]


def test_the_same_inputs_and_seed_give_identical_files_and_another_seed_other_canaries(tmp_path):
    _, first, a = suite(tmp_path, out=tmp_path / "a")
    _, second, b = suite(tmp_path, out=tmp_path / "b")
    for name in ("poison-corpus.jsonl", "poison-dataset.jsonl", "poison-suite.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert first == second
    _, other, _ = suite(tmp_path, out=tmp_path / "c", seed=1)
    assert other["suite_hash"] != first["suite_hash"]
    assert {d["canary"] for d in other["documents"]}.isdisjoint(
        {d["canary"] for d in first["documents"]}
    )


def test_the_original_corpus_and_benchmark_are_never_changed(tmp_path):
    corpus_path, dataset_path = write(tmp_path)
    before = (corpus_path.read_bytes(), dataset_path.read_bytes())
    suite(tmp_path)
    assert (corpus_path.read_bytes(), dataset_path.read_bytes()) == before


def test_the_poison_dataset_forbids_the_poisoned_documents_and_keeps_everything_else(tmp_path):
    poisoned, descriptor, out = suite(tmp_path)
    original = load_dataset(tmp_path / "benchmark.jsonl")
    copy = load_dataset(out / "poison-dataset.jsonl")
    q = {x.query_id: x for x in copy.queries}
    ids = {r["doc_id"] for r in poisoned if r["poison"]["targets"] == ["q-refund"]}
    assert set(q["q-refund"].forbidden) == ids
    assert {t for t in q["q-refund"].tags if t.startswith("poison:")} == {
        f"poison:{a}" for a in ATTACKS
    }
    assert "billing" in q["q-refund"].tags  # existing tags are kept
    assert q["q-refund"].relevant == original.queries[0].relevant

    def rows(dataset):
        return {r["query_id"]: r for r in dataset_rows(dataset)}

    for untouched in ("q-held", "q-vector", "q-open"):
        assert rows(copy)[untouched] == rows(original)[untouched]
    assert copy.meta["provenance"]["poison_suite"] == descriptor["suite_hash"]
    assert copy.sha256 != original.sha256
    assert q["q-held"].split == "heldout"


def test_the_descriptor_hashes_every_input_and_output_and_lists_the_canaries(tmp_path):
    poisoned, descriptor, out = suite(tmp_path)
    assert descriptor["format"] == "vectraudit.poison_suite"
    assert (
        descriptor["inputs"]["corpus"]["sha256"]
        == hashlib.sha256((tmp_path / "corpus.jsonl").read_bytes()).hexdigest()
    )
    assert (
        descriptor["inputs"]["dataset"]["sha256"]
        == load_dataset(tmp_path / "benchmark.jsonl").sha256
    )
    for name, info in descriptor["outputs"].items():
        assert info["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert descriptor["outputs"]["poison-corpus.jsonl"]["documents"] == len(poisoned)
    assert {d["canary"] for d in descriptor["documents"]} == {
        r["poison"]["canary"] for r in poisoned
    }
    assert json.loads((out / "poison-suite.json").read_text()) == descriptor
    assert "How do I get a refund" not in json.dumps(descriptor)  # no question or document text


def test_filter_fields_are_copied_only_when_asked_and_vectors_follow_the_corpus(tmp_path):
    plain, _, _ = suite(tmp_path)
    assert all(r["metadata"] == {} for r in plain)
    matched, _, _ = suite(tmp_path, out=tmp_path / "m", match_filters=True)
    refund = by_attack(matched)
    assert all(r["metadata"] == {"tenant": "acme", "current": True} for r in refund.values())
    reset = by_attack(matched, "q-reset")
    assert reset["inject"]["metadata"] == {}  # that question has no tenant or filter to match
    # Vectors: the hijack takes the question's own vector, the others the best document's.
    assert refund["hijack"]["vector"] == pytest.approx([1.0, 0.1, 0.0])
    assert refund["inject"]["vector"] == [1.0, 0.0, 0.0]
    without = [{k: v for k, v in d.items() if k != "vector"} for d in CORPUS]
    corpus_path, dataset_path = write(tmp_path, corpus=without)
    corpus, dataset = load_corpus(corpus_path), load_dataset(dataset_path)
    chosen = select_targets(dataset, corpus, targets=["q-refund"], sample=None, seed=0)
    poisoned, _ = build_suite(corpus, dataset, chosen, attacks=["hijack"])
    assert "vector" not in poisoned[0]


def test_several_documents_per_target_are_numbered_and_each_has_its_own_canary(tmp_path):
    poisoned, _, _ = suite(tmp_path, targets=("q-refund",), attacks=("inject",), per_target=3)
    assert [r["doc_id"] for r in poisoned] == [f"poison:inject:q-refund:{n}" for n in (1, 2, 3)]
    assert len({r["poison"]["canary"] for r in poisoned}) == 3


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (["q-held"], "is held out"),
        (["nope"], "No question"),
        (["q-vector"], "has no question text"),
        (["q-open"], "has no relevant documents"),
    ],
)
def test_only_exploratory_questions_with_text_and_answers_can_be_targeted(
    tmp_path, targets, message
):
    corpus_path, dataset_path = write(tmp_path)
    with pytest.raises(PoisonError, match=message):
        select_targets(
            load_dataset(dataset_path),
            load_corpus(corpus_path),
            targets=targets,
            sample=None,
            seed=0,
        )


def test_a_question_whose_documents_are_not_in_the_corpus_is_refused(tmp_path):
    corpus_path, dataset_path = write(tmp_path, corpus=CORPUS[1:])
    with pytest.raises(PoisonError, match="missing from the corpus: refund-policy"):
        select_targets(
            load_dataset(dataset_path), load_corpus(corpus_path), targets=["q-refund"],
            sample=None, seed=0,
        )  # fmt: skip


def test_a_sample_is_seeded_eligible_only_and_never_includes_a_held_out_question(tmp_path):
    corpus_path, dataset_path = write(tmp_path)
    corpus, dataset = load_corpus(corpus_path), load_dataset(dataset_path)

    def pick(seed, n):
        return [
            t.query.query_id
            for t in select_targets(dataset, corpus, targets=None, sample=n, seed=seed)
        ]

    assert pick(0, 1) == pick(0, 1)
    assert set(pick(0, 10)) == {"q-refund", "q-reset"}  # the others are not eligible
    for seed in range(20):
        assert "q-held" not in pick(seed, 2)
    for options in ({"targets": None, "sample": None}, {"targets": ["q-refund"], "sample": 1}):
        with pytest.raises(PoisonError, match="Choose the targets"):
            select_targets(dataset, corpus, seed=0, **options)
    with pytest.raises(PoisonError, match="at least 1"):
        select_targets(dataset, corpus, targets=None, sample=0, seed=0)


def test_options_and_clashes_are_validated(tmp_path):
    corpus_path, dataset_path = write(tmp_path)
    corpus, dataset = load_corpus(corpus_path), load_dataset(dataset_path)
    chosen = select_targets(dataset, corpus, targets=["q-refund"], sample=None, seed=0)
    for options, message in [
        ({"attacks": []}, "--attacks"),
        ({"attacks": ["sql"]}, "--attacks"),
        ({"attacks": ["inject"], "per_target": 0}, "--per-target"),
        ({"attacks": ["inject"], "per_target": 6}, "--per-target"),
    ]:
        with pytest.raises(PoisonError, match=message):
            build_suite(corpus, dataset, chosen, **options)
    taken = [*CORPUS, {"doc_id": "poison:inject:q-refund:1", "text": "x"}]
    with pytest.raises(PoisonError, match="already has a document named"):
        build_suite(taken, dataset, chosen, attacks=["inject"])


def test_corpus_files_are_read_strictly(tmp_path):
    path = tmp_path / "c.jsonl"
    for text, message in [
        ("", "contains no documents"),
        ("not json", "not JSON"),
        ('{"doc_id": "a"}', "needs a doc_id and text"),
        ('{"doc_id": "a", "text": " "}', "needs a doc_id and text"),
        ('{"doc_id": "a", "text": "x"}\n{"doc_id": "a", "text": "y"}', "repeats"),
    ]:
        path.write_text(text)
        with pytest.raises(PoisonError, match=message):
            load_corpus(path)
    with pytest.raises(PoisonError, match="Cannot read"):
        load_corpus(tmp_path / "missing.jsonl")


def test_a_generator_of_yours_writes_the_text_and_must_keep_the_canary(tmp_path, monkeypatch):
    (tmp_path / "my_poison.py").write_text(
        textwrap.dedent(
            """
            def good(request):
                return f"{request['question']} {request['canary']} {request['attack']}"

            def forgetful(request):
                return "no canary here"

            def empty(request):
                return ""

            not_callable = 3
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    corpus_path, dataset_path = write(tmp_path)
    corpus, dataset = load_corpus(corpus_path), load_dataset(dataset_path)
    chosen = select_targets(dataset, corpus, targets=["q-refund"], sample=None, seed=0)
    poisoned, _ = build_suite(
        corpus, dataset, chosen, attacks=["inject"], generator=load_generator("my_poison:good")
    )
    assert poisoned[0]["text"].startswith("How do I get a refund? canary-")
    for name, message in [("forgetful", "lacks the canary"), ("empty", "must return text")]:
        with pytest.raises(PoisonError, match=message):
            build_suite(
                corpus, dataset, chosen, attacks=["inject"],
                generator=load_generator(f"my_poison:{name}"),
            )  # fmt: skip
    bad_specs = [
        ("nomodule", "module:function"),
        ("my_poison:missing", "Cannot load"),
        ("missing_mod:f", "Cannot load"),
        ("my_poison:not_callable", "not callable"),
        ("bad name:f", "module:function"),
    ]
    for spec, message in bad_specs:
        with pytest.raises(PoisonError, match=message):
            load_generator(spec)


def test_the_cli_writes_the_suite_and_refuses_what_it_should(tmp_path):
    corpus_path, dataset_path = write(tmp_path)
    out = tmp_path / "suite"
    base = ["workbench", "poison", "generate", "--corpus", str(corpus_path),
            "--dataset", str(dataset_path), "--out", str(out)]  # fmt: skip
    result = CliRunner().invoke(app, [*base, "--targets", "q-refund,q-reset", "--match-filters"])
    assert result.exit_code == 0, result.output
    assert "Wrote 10 poisoned documents for 2 questions" in result.output
    assert "filters matched: yes" in result.output
    assert "sentinels" in result.output
    assert sorted(p.name for p in out.iterdir()) == [
        "poison-corpus.jsonl", "poison-dataset.jsonl", "poison-suite.json",
    ]  # fmt: skip
    sampled = CliRunner().invoke(
        app, [*base, "--sample", "1", "--attacks", "hijack", "--seed", "3"]
    )
    assert sampled.exit_code == 0
    assert "Wrote 1 poisoned documents for 1 questions" in sampled.output
    for args, message in [
        (["--targets", "q-held"], "is held out"),
        ([], "Choose the targets"),
        (["--targets", "q-refund", "--attacks", "sql"], "--attacks"),
        (["--targets", "q-refund", "--generator", "nope"], "module:function"),
    ]:
        failed = CliRunner().invoke(app, [*base, *args])
        assert failed.exit_code == 2, failed.output
        assert message in failed.output
