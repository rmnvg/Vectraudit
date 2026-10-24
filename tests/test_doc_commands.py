"""Every ``vectraudit`` command in the docs exists, with the options it is shown with.

A pilot engineer follows the documentation literally. A renamed subcommand or option in a
fenced ``bash`` block fails them before anything else works, and nothing else catches it.
This walks the real command tree (nothing is executed) for each documented invocation.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.main

from vectraudit.cli import app

ROOT = Path(__file__).parents[1]
DOCS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
FENCE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.S)
SEPARATORS = {"&&", "||", ";", "|", ">", ">>", "2>&1"}


def documented() -> list[tuple[str, int, list[str]]]:
    """``(file, line, tokens)`` for every ``vectraudit`` invocation in a shell block."""
    found = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for block in FENCE.finditer(text):
            start = text.count("\n", 0, block.start(1)) + 1
            lines = block.group(1).split("\n")
            joined, first = "", None
            for offset, raw in enumerate(lines):
                line = raw.split(" #", 1)[0].rstrip()
                if line.lstrip().startswith("#"):
                    continue
                if first is None:
                    first = start + offset
                joined += line[:-1] + " " if line.endswith("\\") else line
                if line.endswith("\\"):
                    continue
                for tokens in _commands(joined):
                    found.append((path.name, first, tokens))
                joined, first = "", None
    return found


def _commands(line: str) -> list[list[str]]:
    try:
        tokens = shlex.split(line)
    except ValueError:
        return []
    out, current = [], []
    for token in [*tokens, ";"]:
        if token in SEPARATORS:
            if current:
                out.append(current)
            current = []
        else:
            current.append(token)
    commands = []
    for words in out:
        if "vectraudit" in words:
            commands.append(words[words.index("vectraudit") + 1 :])
    return commands


def resolve(tokens: list[str]) -> tuple[Any, list[str], str]:
    """The command the tokens name, its remaining tokens and its dotted name.

    Typer bundles its own Click, so groups are recognised by their ``commands`` mapping.
    """
    command: Any = typer.main.get_command(app)
    name = "vectraudit"
    rest = list(tokens)
    while getattr(command, "commands", None) and rest and not rest[0].startswith("-"):
        child = command.commands.get(rest[0])
        if child is None:
            raise LookupError(f"no command `{name} {rest[0]}`")
        name, command, rest = f"{name} {rest[0]}", child, rest[1:]
    return command, rest, name


def options(command: Any) -> set[str]:
    names = {"--help", "-h"}
    for param in command.params:
        names.update(getattr(param, "opts", []))
        names.update(getattr(param, "secondary_opts", []))
    return names


def shown_in_the_workbench() -> list[tuple[str, int, list[str]]]:
    """Commands the workbench UI tells users to run (quoted ``vectraudit …`` strings)."""
    found = []
    for path in sorted((ROOT / "frontend" / "src").rglob("*.ts*")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"[\"']vectraudit ([a-z][^\"'$]*)[\"']", text):
            if match.group(1).rstrip().endswith(":"):
                continue  # a log message ("vectraudit view crashed:"), not a command
            line = text.count("\n", 0, match.start()) + 1
            found += [(path.name, line, tokens) for tokens in _commands(match.group(0)[1:-1])]
    return found


COMMANDS = documented() + shown_in_the_workbench()


def test_the_docs_show_vectraudit_commands():
    assert len(COMMANDS) > 50  # guards the extractor itself


@pytest.mark.parametrize(
    ("where", "tokens"),
    [(f"{f}:{line}", tokens) for f, line, tokens in COMMANDS],
    ids=[f"{f}:{line}" for f, line, _ in COMMANDS],
)
def test_documented_command_exists_with_its_options(where, tokens):
    if not tokens or tokens[0] in ("--version", "--help"):
        return
    try:
        command, rest, name = resolve(tokens)
    except LookupError as exc:
        pytest.fail(f"{where}: {exc}")
    if getattr(command, "commands", None):
        pytest.fail(f"{where}: `{name}` needs a subcommand")
    known = options(command)
    unknown = [
        t.split("=", 1)[0]
        for t in rest
        if t.startswith("-") and t != "-" and t.split("=", 1)[0] not in known
    ]
    assert not unknown, f"{where}: `{name}` has no option {', '.join(unknown)}"


def test_the_checker_catches_a_renamed_command_and_a_misspelled_option():
    with pytest.raises(LookupError, match="workbench acces"):
        resolve(["workbench", "acces", "changes.jsonl"])
    command, rest, _ = resolve(["workbench", "access", "a", "b", "--fail-on-leeks"])
    assert rest[-1] == "--fail-on-leeks"
    assert "--fail-on-leeks" not in options(command)
    assert "--fail-on-leak" in options(command)
    assert _commands("cd x && uv run vectraudit check a.json b.json --html r.html | tee log") == [
        ["check", "a.json", "b.json", "--html", "r.html"]
    ]
