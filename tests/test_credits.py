"""docs/credits.md names real files, and each credited module says where its method came from."""

import re
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).parents[1]
CREDITS = (ROOT / "docs" / "credits.md").read_text()
# Modules that reimplement a published method or a tool's behaviour (not every module in a phase).
REIMPLEMENTS = {
    "src/vectraudit/core/fusion.py": ("ranx", "Cormack"),
    "src/vectraudit/pipelines/fusion_replay.py": ("ranx",),
    "src/vectraudit/pipelines/fusion_grid.py": ("ranx", "Cormack"),
    "src/vectraudit/evaluation/significance.py": ("ranx", "Tukey"),
    "src/vectraudit/evaluation/ragchecker_import.py": ("RAGChecker",),
    "src/vectraudit/pipelines/claim_review.py": ("RAGChecker",),
    "src/vectraudit/pipelines/ragchecker_input.py": ("RAGChecker",),
    "src/vectraudit/evaluation/corpus_scan.py": ("promptfoo",),
    "src/vectraudit/pipelines/poison.py": ("promptfoo",),
    "src/vectraudit/pipelines/poison_run.py": ("promptfoo",),
    "src/vectraudit/pipelines/canary.py": ("promptfoo",),
    "src/vectraudit/evaluation/promptfoo_redteam_import.py": ("promptfoo",),
    "src/vectraudit/core/filtering.py": ("VectorDBBench",),
    "src/vectraudit/core/selectivity.py": ("VectorDBBench",),
    "src/vectraudit/evaluation/capacity.py": ("VectorDBBench",),
    "src/vectraudit/evaluation/vdbbench_import.py": ("VectorDBBench",),
    "src/vectraudit/evaluation/external_reference.py": ("VectorDBBench",),
    "src/vectraudit/core/atlas.py": ("embedding-atlas",),
    "src/vectraudit/core/coverage.py": ("k-means++",),
    "src/vectraudit/evaluation/ppi.py": ("ARES", "PPI++"),
    "src/vectraudit/evaluation/trec.py": ("ir-measures", "pytrec_eval"),
    "src/vectraudit/experiments/mteb_export.py": ("MTEB",),
    "src/vectraudit/experiments/beir.py": ("BEIR",),
    "src/vectraudit/evaluation/measure_names.py": ("ir-measures",),
}


def test_every_file_the_credits_page_names_exists() -> None:
    paths = re.findall(r"`((?:core|pipelines|evaluation|experiments|frontend)[\w/.*-]*)`", CREDITS)
    assert paths
    for name in paths:
        base = ROOT / "frontend" if name.startswith("frontend") else ROOT / "src" / "vectraudit"
        pattern = name.removeprefix("frontend/")
        assert list(base.glob(pattern)), f"credits.md names {name}, which does not exist"


def test_each_reimplementing_module_credits_its_source_in_its_docstring() -> None:
    for path, names in REIMPLEMENTS.items():
        source = (ROOT / path).read_text()
        docstring = source[3 : source.index('"""', 3)]
        for name in names:
            assert name.lower() in docstring.lower(), f"{path} does not credit {name}"
        relative = path.removeprefix("src/vectraudit/")
        listed = re.findall(r"`((?:core|pipelines|evaluation|experiments)/[\w/.*-]*)`", CREDITS)
        assert any(fnmatch(relative, pattern) for pattern in listed), f"{path} is not in credits.md"


def test_the_credits_page_says_nothing_runs_from_these_projects() -> None:
    assert "Nothing below is a runtime dependency" in CREDITS
    assert "never runs it" in CREDITS
