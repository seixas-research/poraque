# -*- coding: utf-8 -*-
# file: logging.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The one way a Poraquê command writes its output.

Every console command --- ``poraque-train``, ``poraque-inference``,
``poraque-atoms``, ``poraque-vasp``, ``poraque-mp``, ``poraque-committee`` and
``poraque-active-learning`` --- reports through a :class:`Logger`, so their
output shares one layout that a person can read and a script can split:

* the start-up **banner** (:func:`poraque.environment.banner_lines`): the logo,
  the version, and the ``[SYSTEM]``, ``[PYTHON]`` and ``[DEPENDENCIES]`` blocks
  that record what produced the output;
* a **title** between two rules, naming what the command is doing;
* a sequence of **blocks**, each opened by a header ``[NAME]`` at column 0 and
  followed by its body, indented by :data:`INDENT`: ``key: value`` lines,
  tables, or free text. A block ends where the next header begins; a blank line
  separates them.

That is the whole format. A block is the unit of meaning (the cache, the
model, the training progress, the results of one split, the resource profile),
so finding a block in a long log is a search for its header, and a parser
only has to recognize lines that start with ``[``. :func:`split_blocks` is the
reference reader.

A :class:`Logger` is also **callable**, ``log("text")``, and writes the text
verbatim. That is the contract every function that reports progress already
had, with ``log=print`` as its default, so library code keeps taking a plain
callable and stays independent of this module; only the commands, which own
the output, build a :class:`Logger`. Library code that wants a header without
depending on one calls :func:`header`.

The output goes to the terminal, to a file, or to both, line by line and
flushed at once, so a long run can be followed while it runs. Under
distributed training every rank but the first is :attr:`~Logger.silent`: four
ranks writing one file truncate and interleave each other.
"""

import os
import sys

#: Indentation of a block's body, under its ``[NAME]`` header at column 0.
INDENT = "  "

#: The rule that frames a title.
RULE = "=" * 78


def header(name):
    """
    The header line of a block: ``[NAME]``, upper case.

    Parameters
    ----------
    name : str
        The block's name; brackets around it are tolerated and not doubled.

    Returns
    -------
    str
    """
    return f"[{str(name).strip().strip('[]').upper()}]"


def indent(lines, level=1):
    """Indent each of ``lines`` by ``level`` times :data:`INDENT`."""
    prefix = INDENT * int(level)
    return [prefix + line if line else line for line in lines]


def field_lines(mapping, level=1):
    """
    ``key: value`` lines with the colons aligned; ``None`` values are skipped.

    Parameters
    ----------
    mapping : dict
    level : int, optional
        Indentation level of the lines.

    Returns
    -------
    list of str
    """
    items = [(str(key), value) for key, value in mapping.items()
             if value is not None]
    if not items:
        return []
    width = max(len(key) for key, _ in items)
    return indent([f"{key + ':':<{width + 1}s} {value}" for key, value in items],
                  level)


def table_lines(headers, rows, align=None, level=1):
    """
    A fixed-width table: a heading, a dashed rule, and one line per row.

    Parameters
    ----------
    headers : sequence of str
    rows : iterable of sequence
        Cells are converted with :func:`str`.
    align : str, optional
        One character per column, ``"<"`` (left) or ``">"`` (right). Defaults
        to the first column left-aligned and the rest right-aligned.
    level : int, optional
        Indentation level of the table.

    Returns
    -------
    list of str
    """
    body = [[str(cell) for cell in row] for row in rows]
    align = align or ("<" + ">" * (len(headers) - 1))
    widths = [max([len(str(name))] + [len(row[i]) for row in body])
              for i, name in enumerate(headers)]
    heading = "  ".join(f"{str(name):{a}{w}s}"
                        for name, a, w in zip(headers, align, widths))
    lines = [heading.rstrip(), "-" * len(heading)]
    lines += ["  ".join(f"{cell:{a}{w}s}"
                        for cell, a, w in zip(row, align, widths)).rstrip()
              for row in body]
    return indent(lines, level)


def split_blocks(text):
    """
    Split a Logger's output into its blocks: the reference reader.

    Parameters
    ----------
    text : str
        The whole output.

    Returns
    -------
    list of tuple
        ``(name, lines)`` per block in order of appearance; ``lines`` are the
        block's body with the indentation removed and trailing blank lines
        dropped. Text before the first header (the logo, a title) is the block
        named ``""``.
    """
    blocks, name, lines = [], "", []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("[") and stripped.endswith("]") \
                and not line.startswith(" "):
            blocks.append((name, lines))
            name, lines = stripped[1:-1], []
            continue
        lines.append(stripped[len(INDENT):] if stripped.startswith(INDENT)
                     else stripped)
    blocks.append((name, lines))
    cleaned = []
    for block_name, block_lines in blocks:
        while block_lines and not block_lines[-1].strip():
            block_lines.pop()
        if block_name or block_lines:
            cleaned.append((block_name, block_lines))
    return cleaned


class Logger:
    """
    Write a command's output to the terminal and, optionally, a file.

    Parameters
    ----------
    path : str, optional
        Log file, truncated when opened; its directory is created. ``None``
        writes to the terminal only.
    terminal : bool, optional
        Echo every line to standard output (default ``True``).
    silent : bool, optional
        Write nothing at all, and open no file. What every rank but the first
        gets under distributed training.
    stream : file-like, optional
        The terminal stream; :data:`sys.stdout` by default, read at write time
        so a test's capture sees the output.

    Examples
    --------
    >>> log = Logger()
    >>> log.section("model")
    <BLANKLINE>
    [MODEL]
    >>> log.fields({"width": 64, "modes": 24})
      width: 64
      modes: 24
    """

    def __init__(self, path=None, terminal=True, silent=False, stream=None):
        self.silent = bool(silent)
        self.path = None if self.silent else path
        self.terminal = bool(terminal)
        self._stream = stream
        self.handle = None
        #: The block being written, as its header; ``None`` before the first.
        self.current = None
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            self.handle = open(self.path, "w", encoding="utf-8")

    # -- the primitive ---------------------------------------------------- #
    def __call__(self, message=""):
        """Write ``message`` verbatim, one or more lines."""
        if self.silent:
            return
        text = str(message)
        if self.terminal:
            stream = self._stream or sys.stdout
            try:
                stream.write(text + "\n")
                stream.flush()
            except BrokenPipeError:
                # The reader went away (`| head`): stop echoing, keep the file.
                self.terminal = False
        if self.handle is not None:
            self.handle.write(text + "\n")
            self.handle.flush()

    def lines(self, lines):
        """Write each of ``lines`` verbatim."""
        for line in lines:
            self(line)

    # -- structure -------------------------------------------------------- #
    def banner(self):
        """The logo, the version and the environment blocks."""
        from ..environment import banner_lines

        self.lines(banner_lines())

    def title(self, text):
        """``text`` between two rules: what the command is doing."""
        self("")
        self(RULE)
        self(INDENT + str(text))
        self(RULE)

    def section(self, name):
        """Open a block: a blank line, then its ``[NAME]`` header."""
        self.current = header(name)
        self("")
        self(self.current)

    def text(self, message, level=1):
        """A body line, indented ``level`` times."""
        self.lines(indent(str(message).splitlines() or [""], level))

    def fields(self, mapping, level=1):
        """``key: value`` lines with aligned colons; ``None`` is skipped."""
        self.lines(field_lines(mapping, level))

    def table(self, headers, rows, align=None, level=1):
        """A fixed-width table; see :func:`table_lines`."""
        self.lines(table_lines(headers, rows, align=align, level=level))

    def block(self, name, mapping=None, lines=None):
        """
        A whole block at once: the header, then ``mapping`` as fields and
        ``lines`` as indented text.
        """
        self.section(name)
        if mapping:
            self.fields(mapping)
        for line in lines or ():
            self.text(line)

    # -- lifetime --------------------------------------------------------- #
    def close(self):
        """Close the file, if one is open. The terminal is left alone."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
