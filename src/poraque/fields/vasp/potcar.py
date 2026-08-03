# -*- coding: utf-8 -*-
# file: potcar.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Reader for VASP ``POTCAR`` pseudopotential files.

A ``POTCAR`` is the concatenation of one dataset per species, each opening with
a title line and closing with ``End of Dataset``. This module extracts the
header quantities that define the *ionic* problem seen by the valence
electrons:

``ZVAL``
    Valence charge of the pseudo-ion — the charge that enters the local
    external potential.
``ENMAX``
    Recommended plane-wave cutoff (eV); the largest ``ENMAX`` over the species
    present is the natural fallback when the ``INCAR`` has no ``ENCUT``.
``RCORE``
    Outermost pseudization radius (a.u.). Outside it the local pseudopotential
    is the bare ``-Z_val e^2 / r`` Coulomb tail; inside it is softened. It
    therefore sets the natural width of the smeared pseudo-ion model used by
    :class:`poraque.fields.ExternalPotential`.

The ``local part`` block
------------------------
The block following the ``local part`` marker holds the tabulated
short-ranged local pseudopotential in reciprocal space. Its layout was
recovered from the VASP source (``pseudo.F``, the ``POTCAR`` reader)::

    READ(10,*) P(NTYP)%PSGMAX
    READ(10,*) (P(NTYP)%PSP(I,2), I=1,NPSPTS)
    DO I=1,NPSPTS
        P(NTYP)%PSP(I,1) = (P(NTYP)%PSGMAX/NPSPTS)*(I-1)
    ENDDO

so the first number after the marker is **PSGMAX**, the maximum wavevector of
the table (Å⁻¹), *not* the valence charge — a coincidence in some files, where
the two happen to look alike. It is followed by exactly ``NPSPTS = 1000``
values sampled on the **uniform** mesh

.. math:: q_i = \frac{\mathrm{PSGMAX}}{1000}\,(i-1), \qquad i = 1 \ldots 1000,

in units of eV·Å³. The values are the short-ranged remainder
:math:`v_{\rm short}(q)`, i.e. the local pseudopotential with its
:math:`-4\pi Z_{\rm val}e^2/q^2` Coulomb tail already subtracted; VASP adds
that tail back analytically in ``POTION``. See
:class:`poraque.fields.ExternalPotential` for the reconstruction.
"""

import re

import numpy as np

from .poscar import symbol_to_z

_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
_ZVAL_RE = re.compile(r"ZVAL\s*=\s*(" + _FLOAT + r")")
_ENMAX_RE = re.compile(r"ENMAX\s*=\s*(" + _FLOAT + r")")
_RCORE_RE = re.compile(r"RCORE\s*=\s*(" + _FLOAT + r")")
_TITEL_RE = re.compile(r"TITEL\s*=\s*(.+)")
_LEXCH_RE = re.compile(r"LEXCH\s*=\s*(\S+)")
_NUMERIC_LINE_RE = re.compile(r"^\s*(?:" + _FLOAT + r"\s*)+$")


class PotcarSingle:
    """
    One species' dataset inside a ``POTCAR``.

    Attributes
    ----------
    symbol : str
        POTCAR variant symbol, e.g. ``"Si"``, ``"Fe_pv"``, ``"Ga_d"``.
    element : str
        Bare chemical symbol (``"Fe_pv"`` -> ``"Fe"``).
    zval : float
        Valence (pseudo-ion) charge in units of ``+e``.
    enmax : float or None
        Recommended plane-wave cutoff in eV.
    rcore : float or None
        Outermost pseudization radius in Bohr (as written by VASP).
    functional : str or None
        ``LEXCH`` tag: ``"PE"`` (PBE), ``"CA"`` (LDA), ``"91"`` (PW91), ...
    local_part : numpy.ndarray or None
        Raw floats of the ``local part`` block: ``PSGMAX`` followed by the
        ``NPSPTS`` table values. Prefer :attr:`psgmax` and
        :attr:`local_potential`.
    """

    #: Number of tabulated points, ``NPSPTS`` in ``pseudo_struct.F``.
    NPSPTS = 1000

    def __init__(self, symbol, zval, enmax=None, rcore=None, functional=None,
                 titel=None, local_part=None):
        self.symbol = str(symbol)
        self.zval = float(zval)
        self.enmax = None if enmax is None else float(enmax)
        self.rcore = None if rcore is None else float(rcore)
        self.functional = functional
        self.titel = titel
        self.local_part = local_part

    @property
    def psgmax(self):
        """
        Maximum wavevector of the tabulated local potential, Å⁻¹.

        ``None`` unless the POTCAR was read with ``parse_tables=True``.
        """
        if self.local_part is None or len(self.local_part) < 1:
            return None
        return float(self.local_part[0])

    @property
    def local_potential(self):
        r"""
        Short-ranged local pseudopotential :math:`v_{\rm short}(q)`, eV·Å³.

        The :math:`-4\pi Z_{\rm val}e^2/q^2` Coulomb tail has been removed;
        :class:`poraque.fields.ExternalPotential` adds it back analytically.

        Returns
        -------
        numpy.ndarray or None
            The tabulated values, or ``None`` if the tables were not parsed.
            A well-formed POTCAR yields exactly ``NPSPTS`` of them; check
            :attr:`has_local_table` before relying on the length.
        """
        if self.local_part is None or len(self.local_part) < 2:
            return None
        return np.asarray(self.local_part[1:1 + self.NPSPTS], dtype=float)

    @property
    def has_local_table(self):
        """
        Whether a **complete** local-potential table was read.

        A truncated block — an abridged test fixture, a partial download —
        parses without error but cannot be splined onto the ``PSGMAX`` mesh.
        Callers must gate on this rather than on
        ``local_potential is not None``, so that an incomplete table falls back
        to an analytic model instead of raising from inside the interpolator.
        """
        values = self.local_potential
        return (self.psgmax is not None and values is not None
                and len(values) == self.NPSPTS)

    @property
    def local_q_grid(self):
        r"""
        Wavevectors of :attr:`local_potential`, Å⁻¹.

        Uniform, ``q_i = PSGMAX * (i-1) / NPSPTS``, exactly as ``pseudo.F``
        constructs ``PSP(:,1)``. The returned length always matches
        :attr:`local_potential`, so the two can be paired directly even for a
        truncated table.
        """
        values = self.local_potential
        if self.psgmax is None or values is None:
            return None
        return (self.psgmax / self.NPSPTS) * np.arange(len(values), dtype=float)

    @property
    def pscore(self):
        r"""
        ``PSCORE`` = :math:`v_{\rm short}(q\to0)`, eV·Å³.

        The :math:`\mathbf{G}=0` limit of the short-ranged part, which VASP
        uses for the ``PSCENC`` energy correction. It does not enter the
        potential itself, since :math:`V(\mathbf{G}=0)` is set to zero.
        """
        values = self.local_potential
        return None if values is None else float(values[0])

    @property
    def element(self):
        """Bare chemical symbol, stripped of the POTCAR variant suffix."""
        return self.symbol.split("_")[0]

    @property
    def atomic_number(self):
        """Atomic number ``Z`` of the element."""
        return symbol_to_z(self.element)

    @property
    def rcore_angstrom(self):
        """:attr:`rcore` converted to Ångström, or ``None``."""
        from ..constants import BOHR_TO_ANGSTROM

        return None if self.rcore is None else self.rcore * BOHR_TO_ANGSTROM

    @classmethod
    def from_block(cls, text, parse_tables=False):
        """
        Parse one dataset block.

        Parameters
        ----------
        text : str
            Text of a single species dataset.
        parse_tables : bool, optional
            Also extract the raw ``local part`` float table.

        Returns
        -------
        PotcarSingle
        """
        titel_match = _TITEL_RE.search(text)
        if titel_match:
            titel = titel_match.group(1).strip()
            # "PAW_PBE Si 05Jan2001" -> "Si"
            tokens = titel.split()
            symbol = tokens[1] if len(tokens) > 1 else tokens[0]
        else:
            # Fall back to the first non-empty line: "  PAW_PBE Si 05Jan2001".
            first = next(line for line in text.splitlines() if line.strip())
            tokens = first.split()
            titel = first.strip()
            symbol = tokens[1] if len(tokens) > 1 else tokens[0]

        zval_match = _ZVAL_RE.search(text)
        if zval_match is None:
            raise ValueError(f"No ZVAL found in POTCAR dataset for {symbol!r}.")

        enmax_match = _ENMAX_RE.search(text)
        rcore_match = _RCORE_RE.search(text)
        lexch_match = _LEXCH_RE.search(text)

        return cls(
            symbol=symbol,
            zval=_to_float(zval_match.group(1)),
            enmax=_to_float(enmax_match.group(1)) if enmax_match else None,
            rcore=_to_float(rcore_match.group(1)) if rcore_match else None,
            functional=lexch_match.group(1) if lexch_match else None,
            titel=titel,
            local_part=_parse_local_part(text) if parse_tables else None,
        )

    def __repr__(self):
        return (f"PotcarSingle({self.symbol!r}, ZVAL={self.zval:g}, "
                f"ENMAX={self.enmax})")


class Potcar(list):
    """
    A ``POTCAR`` file: an ordered list of :class:`PotcarSingle` datasets.

    The order matters — it must match the species order of the ``POSCAR``.
    """

    @classmethod
    def from_string(cls, text, parse_tables=False):
        """Parse a concatenated POTCAR from a string."""
        blocks = [block for block in text.split("End of Dataset") if block.strip()]
        if not blocks:
            raise ValueError("POTCAR contains no 'End of Dataset' markers.")
        return cls(PotcarSingle.from_block(block, parse_tables=parse_tables)
                   for block in blocks)

    @classmethod
    def from_file(cls, path, parse_tables=False):
        """Read a POTCAR from ``path``."""
        with open(path, "r", errors="replace") as handle:
            return cls.from_string(handle.read(), parse_tables=parse_tables)

    @property
    def symbols(self):
        """Species symbols in file order."""
        return [entry.symbol for entry in self]

    @property
    def elements(self):
        """Bare chemical symbols in file order."""
        return [entry.element for entry in self]

    @property
    def zval_map(self):
        """``{element: zval}`` mapping."""
        return {entry.element: entry.zval for entry in self}

    @property
    def rcore_map(self):
        """``{element: rcore_in_angstrom}`` mapping (entries may be ``None``)."""
        return {entry.element: entry.rcore_angstrom for entry in self}

    @property
    def enmax(self):
        """Largest recommended cutoff (eV) over all species, or ``None``."""
        values = [entry.enmax for entry in self if entry.enmax is not None]
        return max(values) if values else None

    def matches(self, poscar):
        """
        Check that this POTCAR's species order matches a :class:`Poscar`.

        Parameters
        ----------
        poscar : Poscar
            Structure to compare against.

        Returns
        -------
        bool
        """
        return self.elements == [s.split("_")[0] for s in poscar.symbols]

    def __repr__(self):
        return f"Potcar({', '.join(self.symbols)})"


def _to_float(token):
    """Parse a Fortran-style float (``1.0D+03``)."""
    return float(str(token).replace("D", "E").replace("d", "e"))


def _parse_local_part(text):
    """
    Extract the raw floats of the ``local part`` block.

    Returns
    -------
    numpy.ndarray or None
        Every float between the ``local part`` marker and the next
        non-numeric line. The first entry is ``ZVAL``; the remainder is the
        tabulated ``V_loc(q)``. ``None`` when the marker is absent.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("local part"):
            start = index + 1
            break
    if start is None:
        return None

    values = []
    for line in lines[start:]:
        if not _NUMERIC_LINE_RE.match(line):
            break
        values.extend(_to_float(token) for token in line.split())
    return np.asarray(values, dtype=float) if values else None
