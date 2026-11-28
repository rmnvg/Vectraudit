"""Links in the documentation lead somewhere: the file exists and so does the heading.

The README is also the PyPI project page, where relative links break, so its links to the
repository are absolute and are checked against the files in this checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BLOB = "https://github.com/rmnvg/Vectraudit/blob/main/"
RAW = "https://raw.githubusercontent.com/rmnvg/Vectraudit/main/"
LINK = re.compile(r"\]\(([^)\s]+)\)|src=\"([^\"]+)\"")
FENCE = re.compile(r"```.*?```", re.S)
DOCS = sorted([*ROOT.glob("docs/*.md"), ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md"])


def links(path: Path) -> list[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return [a or b for a, b in LINK.findall(text)]


def anchors(path: Path) -> set[str]:
    """GitHub's heading anchors: lowercase, punctuation dropped, spaces to hyphens."""
    seen: dict[str, int] = {}
    out = set()
    for line in FENCE.sub("", path.read_text(encoding="utf-8")).splitlines():
        if not line.startswith("#"):
            continue
        text = line.lstrip("#").strip().lower()
        slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def check(source: Path, target: Path, anchor: str | None) -> str | None:
    if not target.exists():
        return f"{source.name}: {target.relative_to(ROOT)} does not exist"
    if anchor and target.suffix == ".md" and anchor not in anchors(target):
        return f"{source.name}: no heading #{anchor} in {target.relative_to(ROOT)}"
    return None


def test_readme_links_are_absolute_and_resolve():
    problems = []
    for url in links(ROOT / "README.md"):
        if url.startswith(("https://github.com/rmnvg/Vectraudit/actions", "https://img.shields")):
            continue
        if url.startswith(BLOB) or url.startswith(RAW):
            rest = url.removeprefix(BLOB).removeprefix(RAW)
            path, _, anchor = rest.partition("#")
            problems.append(check(ROOT / "README.md", ROOT / path, anchor or None))
        elif not url.startswith("https://"):
            problems.append(f"README.md: relative link {url} breaks on PyPI")
    assert [p for p in problems if p] == []


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    problems = []
    for url in links(doc):
        if url.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = url.partition("#")
        target = (doc.parent / path).resolve() if path else doc
        problems.append(check(doc, target, anchor or None))
    assert [p for p in problems if p] == []


def test_every_guide_is_in_the_site_navigation():
    """A page left out of mkdocs.yml's nav is built but unreachable from the site's menu."""
    import yaml

    def pages(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for item in node:
                yield from pages(item)
        elif isinstance(node, dict):
            for value in node.values():
                yield from pages(value)

    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    listed = set(pages(config["nav"]))
    on_disk = {p.name for p in (ROOT / "docs").glob("*.md")}
    assert on_disk - listed == set(), "add these pages to mkdocs.yml's nav"
    assert listed - on_disk == set(), "mkdocs.yml's nav names pages that do not exist"
