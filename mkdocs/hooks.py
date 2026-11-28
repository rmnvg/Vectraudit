"""MkDocs hook: links from docs/ to files outside it point at GitHub on the site.

The Markdown in docs/ links to ../examples/..., ../CONTRIBUTING.md and similar, which work
when the files are read on GitHub; the site only holds docs/, so those links are rewritten
to the same files on GitHub when the site is built. The Markdown itself is not changed.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

REPO = "https://github.com/rmnvg/Vectraudit/blob/main/"
LINK = re.compile(r"\]\((\.\./[^)\s#]+)(#[^)\s]*)?\)")


def on_page_markdown(markdown: str, page: Any, **_: Any) -> str:
    here = posixpath.dirname(page.file.src_uri)

    def outside(match: re.Match[str]) -> str:
        target = posixpath.normpath(posixpath.join("docs", here, match.group(1)))
        if target.startswith("docs/"):
            return match.group(0)  # still inside docs/: MkDocs resolves it
        return f"]({REPO}{target}{match.group(2) or ''})"

    return LINK.sub(outside, markdown)
