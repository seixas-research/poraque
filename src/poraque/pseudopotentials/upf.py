# -*- coding: utf-8 -*-
# file: upf.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Reader for Unified Pseudopotential Format (``.upf``) files.

The UPF format is the standard pseudopotential container used by Quantum
ESPRESSO and distributed by libraries such as `PseudoDojo
<http://www.pseudo-dojo.org>`_. A UPF v2 file is an XML-like document whose
``<PP_HEADER>`` records the element, valence charge and exchange-correlation
functional, and whose ``<PP_MESH>``/``<PP_LOCAL>`` blocks tabulate the radial
mesh :math:`r` (Bohr) and the **local** ionic potential :math:`V_\\text{loc}(r)`
(stored in Rydberg, by the Quantum ESPRESSO convention).

Poraquê uses the *local* channel of these norm-conserving pseudopotentials to
build the external potential for valence-only KS-DFT. The (Kleinman-Bylander)
nonlocal projectors present in the file are parsed for metadata but not yet
applied — they are reported as a zero nonlocal energy term in the calculator's
energy accounting, leaving an explicit hook for a future nonlocal implementation.

The reader is deliberately tolerant: the ``PP_INFO``/``PP_INPUTFILE`` preambles
of real ONCVPSP files are not strictly valid XML, so headers and data blocks are
extracted with targeted regular expressions rather than a full XML parse.
"""

import re

import numpy as np

from .base import LocalPseudopotential

# Local potentials in UPF files are stored in Rydberg; Poraquê works in Hartree.
RYDBERG_TO_HARTREE = 0.5


class UPFLocalPseudopotential(LocalPseudopotential):
    """
    Local pseudopotential tabulated on a radial mesh read from a UPF file.

    The spherically symmetric local potential :math:`V_\\text{loc}(r)` (Hartree)
    is interpolated from the file's radial mesh. Beyond the tabulated range the
    physical :math:`-Z_v/r` Coulomb tail is used so that the potential stays well
    defined everywhere on the real-space grid.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    z_valence : float
        Number of valence electrons (effective ionic charge).
    r : array_like
        Radial mesh (Bohr), strictly increasing.
    v_loc : array_like
        Local potential on ``r`` (Hartree).
    functional : str, optional
        Exchange-correlation functional recorded in the file (e.g. ``"PBE"``).
    meta : dict, optional
        Remaining header metadata (number of projectors, relativistic flag, ...).
    """

    def __init__(self, symbol, z_valence, r, v_loc, functional=None, meta=None):
        super().__init__(symbol, z_valence)
        self.r = np.asarray(r, dtype=float)
        self.v_loc = np.asarray(v_loc, dtype=float)
        self.functional = functional
        self.meta = dict(meta or {})
        order = np.argsort(self.r)
        self.r = self.r[order]
        self.v_loc = self.v_loc[order]
        self._r_max = float(self.r[-1])

    def radial_potential(self, r):
        r = np.asarray(r, dtype=float)
        # Interpolate inside the mesh; use the Coulomb tail outside it.
        v = np.interp(r, self.r, self.v_loc)
        tail = -self.z_valence / np.maximum(r, 1e-12)
        return np.where(r > self._r_max, tail, v)

    def __repr__(self):
        return (f"UPFLocalPseudopotential(symbol={self.symbol!r}, "
                f"z_valence={self.z_valence}, functional={self.functional!r}, "
                f"mesh={self.r.size})")


def _parse_header(text):
    """Extract the ``<PP_HEADER>`` attributes as a ``dict``."""
    match = re.search(r"<PP_HEADER\b(.*?)/?>", text, re.DOTALL)
    if match is None:
        raise ValueError("No <PP_HEADER> block found; not a UPF v2 file.")
    body = match.group(1)
    attrs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', body))
    return {k: v.strip() for k, v in attrs.items()}


def _parse_block(text, tag):
    """Return the whitespace-separated float array inside ``<tag> ... </tag>``."""
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.DOTALL)
    if match is None:
        raise ValueError(f"No <{tag}> block found in UPF file.")
    return np.array(match.group(1).split(), dtype=float)


def _normalize_functional(raw):
    """Map a UPF ``functional`` string onto ``"LDA"`` or ``"PBE"`` when possible."""
    if not raw:
        return None
    token = raw.strip().upper()
    if token in ("LDA", "PBE", "PBESOL"):
        return token
    # ONCVPSP encodes the four-part dft string, e.g. "SLA  PW   NOGX NOGC" (LDA)
    # or "SLA  PW   PBX  PBC" (PBE). Detect the gradient-corrected exchange.
    if "PBX" in token or "PBE" in token:
        return "PBE"
    if "NOGX" in token and "NOGC" in token:
        return "LDA"
    return raw.strip()


def read_upf(path):
    """
    Read a local pseudopotential from a UPF (``.upf``) file.

    Parameters
    ----------
    path : str
        Path to a UPF v2 file (PseudoDojo / Quantum ESPRESSO format).

    Returns
    -------
    UPFLocalPseudopotential
        The local channel of the pseudopotential, ready to be sampled on a grid.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    header = _parse_header(text)
    symbol = header.get("element", "").strip()
    if not symbol:
        raise ValueError(f"UPF header in {path!r} has no 'element'.")
    z_valence = float(header.get("z_valence", "nan"))
    if not np.isfinite(z_valence):
        raise ValueError(f"UPF header in {path!r} has no valid 'z_valence'.")
    functional = _normalize_functional(header.get("functional"))

    r = _parse_block(text, "PP_R")
    v_loc_ry = _parse_block(text, "PP_LOCAL")
    if r.size != v_loc_ry.size:
        # Trim to the common length (some files pad differently); be defensive.
        n = min(r.size, v_loc_ry.size)
        r, v_loc_ry = r[:n], v_loc_ry[:n]
    v_loc = v_loc_ry * RYDBERG_TO_HARTREE

    meta = {
        "pseudo_type": header.get("pseudo_type"),
        "relativistic": header.get("relativistic"),
        "is_paw": header.get("is_paw"),
        "is_ultrasoft": header.get("is_ultrasoft"),
        "core_correction": header.get("core_correction"),
        "number_of_proj": header.get("number_of_proj"),
        "l_max": header.get("l_max"),
        "rho_cutoff": header.get("rho_cutoff"),
    }
    return UPFLocalPseudopotential(symbol, z_valence, r, v_loc,
                                   functional=functional, meta=meta)
