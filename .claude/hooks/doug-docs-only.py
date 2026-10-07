#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: doug-docs-only.py
"""PreToolUse guard for the ``doug-documenter`` subagent.

Doug may edit documentation only. This hook reads the Claude Code hook payload
from stdin, takes ``tool_input.file_path``, resolves it against the repository
root and refuses anything outside ``docs/``, ``latex/``, ``README.md``,
``configs/train_complete_and_commented.yaml`` and ``experiments/*/README.md``.

Exit codes follow the PreToolUse contract: ``0`` allows the call, ``2`` blocks it
and shows stderr to the subagent. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: Directories Doug may write inside, relative to the repository root.
ALLOWED_DIRECTORIES = ("docs", "latex")

#: Individual files Doug may write, relative to the repository root.
ALLOWED_FILES = ("README.md", "configs/train_complete_and_commented.yaml")


def repository_root() -> Path:
    """The project root.

    ``CLAUDE_PROJECT_DIR`` when Claude Code sets it, otherwise derived from this
    script's own location (``<root>/.claude/hooks/doug-docs-only.py``) so the
    check never depends on the working directory.
    """
    from_env = os.environ.get("CLAUDE_PROJECT_DIR")
    if from_env:
        return Path(from_env).resolve()
    return Path(__file__).resolve().parents[2]


def refuse(message: str) -> int:
    """Block the tool call, telling the subagent why."""
    print(f"doug-docs-only: {message}", file=sys.stderr)
    return 2


def is_allowed(relative: Path) -> bool:
    """Whether a repository-relative path is documentation Doug may edit."""
    parts = relative.parts
    if parts and parts[0] in ALLOWED_DIRECTORIES:
        return True
    if relative.as_posix() in ALLOWED_FILES:
        return True
    # An experiment's write-up, and only that file of the experiment.
    return (len(parts) == 3 and parts[0] == "experiments"
            and parts[2] == "README.md")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return refuse(f"could not parse the hook payload ({error}).")

    if not isinstance(payload, dict):
        return refuse("the hook payload was not a JSON object.")

    tool_input = payload.get("tool_input")
    file_path = (tool_input.get("file_path")
                 if isinstance(tool_input, dict) else None)
    if not file_path:
        # Nothing addresses a file: there is nothing to judge.
        return 0

    root = repository_root()
    target = Path(file_path)
    if not target.is_absolute():
        target = root / target
    # resolve() follows symlinks: the repository is reachable through more
    # than one path on a Dropbox-synced machine.
    target = target.resolve()

    allowed = ("docs/, latex/, README.md, "
               "configs/train_complete_and_commented.yaml and "
               "experiments/*/README.md")
    try:
        relative = target.relative_to(root)
    except ValueError:
        return refuse(f"{target} is outside the repository ({root}). "
                      f"Doug Documenter may only edit {allowed}.")

    if is_allowed(relative):
        return 0
    return refuse(
        f"refusing to edit {relative.as_posix()}. Doug Documenter may only "
        f"edit {allowed}. A change needed anywhere else (a docstring under "
        f"src/, a test, pyproject.toml) must be reported to the main "
        f"conversation for a human to make.")


if __name__ == "__main__":
    sys.exit(main())
