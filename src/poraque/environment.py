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

**A banner is output, not initialization.** It is written by the console
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


def _describe_torch():
    """
    ``"<version> (<cuda build>)    [<install dir>]"``, or ``"not installed"``.

    Torch gets its own describer because two of its fields answer questions no
    other dependency raises, and both were learned the expensive way on a
    cluster:

    * **which CUDA the wheel was built against.** ``2.13.0+cu130`` and
      ``2.7.1+cu126`` are the same version number to
      :func:`_describe_dependency` and completely different binaries; the
      second runs on a V100 and the first cannot.
    * **where it was imported from.** A ``torch`` in
      ``~/.local/lib/pythonX.Y/site-packages`` outranks the one in the active
      conda environment or venv, so the environment a job activated is not
      necessarily the one it ran. ``PYTHONNOUSERSITE=1`` is the fix; this line
      is how the problem becomes visible in the log instead of becoming a
      four-hour CPU run.
    """
    from importlib import import_module

    try:
        module = import_module("torch")
    except ImportError:
        return "not installed"
    version = getattr(module, "__version__", "unknown")
    build = getattr(getattr(module, "version", None), "cuda", None)
    location = os.path.dirname(getattr(module, "__file__", "") or "")
    return f"{version} (cuda {build or 'none'})    [{location}]"


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
    from .utils.logging import INDENT, field_lines, header

    dependencies = {
        "ase": _describe_dependency("ase"),
        "numpy": _describe_dependency("numpy"),
        "scipy": _describe_dependency("scipy"),
        "matplotlib": _describe_dependency("matplotlib"),
        "torch": _describe_torch(),
        "yaml": _describe_dependency("yaml"),
        "pytest": _describe_dependency("pytest"),
    }
    return [
        "",
        "    ████▄ ▄███▄ ████▄  ▀▀█▄ ▄████ ██ ██ ▄█▀█▄",
        "    ██ ██ ██ ██ ██ ▀▀ ▄█▀██ ██ ██ ██ ██ ██▄█▀",
        "    ████▀ ▀███▀ ██    ▀█▄██ ▀████ ▀██▀█ ▀█▄▄▄",
        "    ██                         ██",
        "    ▀▀                         ▀▀",
        "",
        "-" * 64,
        f"{INDENT}version:      {__version__}",
        f"{INDENT}developed by: Leandro Seixas Rocha",
        f"{INDENT}homepage:     https://poraque.seixas.dev",
        f"{INDENT}repository:   https://github.com/seixas-research/poraque",
        "-" * 64,
        "",
        header("system"),
        *field_lines({"architecture": platform.machine(),
                      "platform": platform.system(),
                      "user": _current_user(),
                      "hostname": gethostname(),
                      "cwd": os.getcwd(),
                      "PID": os.getpid()}),
        "",
        header("python"),
        *field_lines({"version": __python_version__,
                      "executable": __python_executable__}),
        "",
        header("dependencies"),
        *field_lines(dependencies),
    ]


def banner(emit=print):
    """
    Write the banner.

    Parameters
    ----------
    emit : callable, optional
        Sink for each line. Defaults to :func:`print`; a training run passes
        its :class:`~poraque.utils.logging.Logger`, so the environment is recorded in the log as well as
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
