# -*- coding: utf-8 -*-
# file: environment.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
The start-up banner: what Poraquê is, and what it is running on.

Its own module for two reasons.

**Nothing here belongs on the import path.** Reporting the version of ase,
numpy, scipy, matplotlib, torch, yaml and pytest used to require importing all
seven in ``poraque/__init__.py`` -- so ``import poraque`` pulled in the whole
scientific stack *and pytest*, a test-only dependency, before the caller had
asked for anything. The package docstring said in as many words that ASE and
PyTorch are not imported there because the field and energy layers do not need
them, three lines below the imports that did it. They are now taken inside
:func:`banner_lines`, at the moment the banner is actually wanted.

**A banner is output, not initialisation.** It is written by the console
commands, and a training run writes it through its own logger, so the
environment that produced a result is recorded in the log beside the result.
"""

import os
import platform
from socket import gethostname
from sys import executable as __python_executable__
from sys import version as __python_version__

from .version import __version__


def _describe_dependency(name):
    """
    ``"<version>    [<install dir>]"`` for a module, or ``"not installed"``.

    Guarded so the banner reports an absent dependency instead of crashing the
    console command that wanted to print it -- pytest in particular is a
    test-only dependency a production install may not carry.
    """
    from importlib import import_module

    try:
        module = import_module(name)
    except ImportError:
        return "not installed"
    version = getattr(module, "__version__", "unknown")
    location = os.path.dirname(getattr(module, "__file__", "") or "")
    return f"{version}    [{location}]"


def _current_user():
    """The login name, without assuming ``USER`` is set (containers, CI)."""
    try:
        import getpass

        return os.environ.get("USER") or getpass.getuser()
    except Exception:
        return "unknown"


def banner_lines():
    """
    The start-up banner, as a list of lines.

    Returned rather than printed so the same text can go to the terminal, to a
    log file, or to both. A run's log is the record of what produced it, and
    the environment -- interpreter, torch build, working directory -- is
    exactly what a reader needs when a result has to be reproduced or
    explained months later.

    The dependency imports are inside :func:`_describe_dependency` on purpose:
    see the module docstring.

    Returns
    -------
    list of str
    """
    return [
        "                                                 ",
        "    ████▄ ▄███▄ ████▄  ▀▀█▄ ▄████ ██ ██ ▄█▀█▄    ",
        "    ██ ██ ██ ██ ██ ▀▀ ▄█▀██ ██ ██ ██ ██ ██▄█▀    ",
        "    ████▀ ▀███▀ ██    ▀█▄██ ▀████ ▀██▀█ ▀█▄▄▄    ",
        "    ██                         ██                ",
        "    ▀▀                         ▀▀                ",
        "                                                 ",
        f"    version: {__version__}                       ",
        "        developed by: Leandro Seixas Rocha      ",
        "        homepage: https://github.com/seixas-research/poraque",
        "                                                  ",
        "------------------------------------------------------------",
        "                                                  ",
        "System:",
        f" ├── architecture: {platform.machine()}",
        f" ├── platform: {platform.system()}",
        f" ├── user: {_current_user()}",
        f" ├── hostname: {gethostname()}",
        f" ├── cwd: {os.getcwd()}",
        f" └── PID: {os.getpid()}",
        "                                               ",
        "Python:",
        f" ├── version: {__python_version__}      ",
        f" └── executable: {__python_executable__}      ",
        "                                               ",
        "Dependencies:",
        f" ├── ase version: {_describe_dependency('ase')}",
        f" ├── numpy version: {_describe_dependency('numpy')}",
        f" ├── scipy version: {_describe_dependency('scipy')}",
        f" ├── matplotlib version: {_describe_dependency('matplotlib')}",
        f" ├── torch version: {_describe_dependency('torch')}",
        f" ├── yaml version: {_describe_dependency('yaml')}",
        f" └── pytest version: {_describe_dependency('pytest')}",
        "                                               ",
    ]


def banner(emit=print):
    """
    Write the banner.

    Parameters
    ----------
    emit : callable, optional
        Sink for each line. Defaults to :func:`print`; a training run passes
        its ``Tee``, so the environment is recorded in the log as well as
        shown on screen.

    Notes
    -----
    Called by the console commands, **not** on import. Thirty lines of
    environment on ``import poraque`` landed in front of every library user,
    every notebook cell and every test run -- and, because it happened before
    a run could open its log file, the one place the information is actually
    wanted never received it.
    """
    for line in banner_lines():
        emit(line)
