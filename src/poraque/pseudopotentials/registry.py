# -*- coding: utf-8 -*-
# file: registry.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Local registry of bundled ``.upf`` pseudopotentials.

Poraquê ships a small library of norm-conserving pseudopotentials (PseudoDojo /
ONCVPSP, UPF v2 format) under a top-level ``pseudos/`` directory, organized by
exchange-correlation functional::

    pseudos/
        LDA/   H.upf  ...  Si.upf  ...
        PBE/   H.upf  ...  Si.upf  ...

This module locates that directory and maps a ``(symbol, functional)`` pair to
the matching file, so that pseudopotential selection follows the calculator's
chosen functional automatically (LDA orbitals use the LDA-generated
pseudopotentials, PBE orbitals the PBE ones). The search path is, in order:

1. the ``PORAQUE_PSEUDO_DIR`` environment variable, if set;
2. a ``pseudos/`` directory found by walking up from this file (the repository
   layout used for development and tests).
"""

import os
from functools import lru_cache
from pathlib import Path

from .upf import read_upf

# Recognized functionals and their canonical subdirectory names.
_FUNCTIONAL_ALIASES = {
    "LDA": "LDA",
    "PZ": "LDA",
    "PW": "LDA",
    "SLA": "LDA",
    "PBE": "PBE",
    "GGA": "PBE",
    "PBESOL": "PBE",
}


def normalize_functional(functional):
    """
    Map a functional name onto a bundled subdirectory (``"LDA"`` or ``"PBE"``).

    Parameters
    ----------
    functional : str
        Functional name (case-insensitive), e.g. ``"lda"``, ``"PBE"``.

    Returns
    -------
    str
        The canonical subdirectory name.
    """
    key = str(functional).strip().upper()
    if key not in _FUNCTIONAL_ALIASES:
        raise KeyError(
            f"No bundled pseudopotentials for functional {functional!r}; "
            f"expected one of {sorted(set(_FUNCTIONAL_ALIASES.values()))}."
        )
    return _FUNCTIONAL_ALIASES[key]


@lru_cache(maxsize=1)
def find_pseudo_dir():
    """
    Locate the bundled ``pseudos/`` directory.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the ``LDA/`` and ``PBE/`` subfolders.

    Raises
    ------
    FileNotFoundError
        If no ``pseudos/`` directory can be found.
    """
    env = os.environ.get("PORAQUE_PSEUDO_DIR")
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path
        raise FileNotFoundError(
            f"PORAQUE_PSEUDO_DIR points to a missing directory: {env!r}."
        )
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pseudos"
        if candidate.is_dir() and (candidate / "LDA").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate a bundled 'pseudos/' directory. Set the "
        "PORAQUE_PSEUDO_DIR environment variable to point at it."
    )


def pseudo_path(symbol, functional="LDA"):
    """
    Return the path to the ``.upf`` file for ``symbol`` and ``functional``.

    Parameters
    ----------
    symbol : str
        Chemical symbol (e.g. ``"Si"``).
    functional : str, optional
        Exchange-correlation functional (default ``"LDA"``).

    Returns
    -------
    pathlib.Path
        Path to the matching UPF file.
    """
    sub = normalize_functional(functional)
    path = find_pseudo_dir() / sub / f"{symbol}.upf"
    if not path.is_file():
        raise FileNotFoundError(
            f"No {sub} pseudopotential for element {symbol!r} (looked for {path})."
        )
    return path


@lru_cache(maxsize=None)
def _read_registry_pseudo(symbol, functional):
    return read_upf(str(pseudo_path(symbol, functional)))


def registry_pseudopotential(symbol, functional="LDA"):
    """
    Load the bundled UPF pseudopotential for an element and functional.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    functional : str, optional
        Exchange-correlation functional (default ``"LDA"``).

    Returns
    -------
    UPFLocalPseudopotential
        The parsed local pseudopotential (results are cached per element).
    """
    return _read_registry_pseudo(symbol, normalize_functional(functional))
