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

The ``PAW radial sets`` block
-----------------------------
Further down, a PAW dataset tabulates its one-centre quantities on a
logarithmic radial mesh: ``grid``, then ``aepotential``, then ``core
charge-density``, and later ``core charge-density (pseudized)``. Two
conventions in it are not written anywhere in the file, and both were pinned
by measurement rather than taken on trust:

* **the mesh is in Å.** The pseudized core is the all-electron core outside the
  partial-core radius ``RPACOR``, which the header states in atomic units, and
  the two tables join at ``RPACOR`` converted to Å --- within one mesh point
  for Pt, Ag, Ni, Pd and Si --- and not at ``RPACOR`` read as Å;
* **the values are** :math:`r^2\rho_{00}(r)`, the :math:`\ell = 0` component
  of the density times :math:`r^2`, with :math:`\rho = \rho_{00}Y_{00}` and
  :math:`Y_{00} = 1/\sqrt{4\pi}`. So
  :math:`N_{\rm core} = \sqrt{4\pi}\int r^2\rho_{00}\,dr`, and that
  integral returns exactly :math:`Z - Z_{\rm val}` --- 68.0000 for Pt, 36.0000
  for Ag and Pd, 18.0000 for Ni and Cu, 12.0000 for ``Fe_pv``, 2.0000 for O.
  Neither :math:`\int` of the raw values nor :math:`4\pi\int r^2` of them
  gives an integer for any of them.

:attr:`PotcarSingle.core_profile` applies both and returns the density itself,
in e/Å³.
"""

import gzip
import os
import re
import warnings

import numpy as np

from ..structure import element_of
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
    radial_sets : dict or None
        Raw ``PAW radial sets`` tables, ``{"grid", "core charge-density",
        "core charge-density (pseudized)"}``, as the file stores them. Prefer
        :attr:`core_profile`.
    """

    #: Number of tabulated points, ``NPSPTS`` in ``pseudo_struct.F``.
    NPSPTS = 1000

    def __init__(self, symbol, zval, enmax=None, rcore=None, functional=None,
                 titel=None, local_part=None, radial_sets=None):
        self.symbol = str(symbol)
        self.zval = float(zval)
        self.enmax = None if enmax is None else float(enmax)
        self.rcore = None if rcore is None else float(rcore)
        self.functional = functional
        self.titel = titel
        self.local_part = local_part
        self.radial_sets = radial_sets

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
    def core_profile(self):
        r"""
        The dataset's radial PAW core charge density, as a plain record.

        Both tables the dataset carries, on its own logarithmic mesh:
        ``core_density`` is the all-electron frozen core and
        ``pseudo_core_density`` the smooth partial core that replaces it inside
        ``RPACOR`` (and equals it outside). Converted from the file's
        :math:`r^2\rho_{00}(r)` to :math:`\rho(r)` in e/Å³ --- see the module
        docstring for how the convention and the unit were established.

        ``core_electrons`` is :math:`4\pi\int r^2\rho\,dr` by Simpson's rule
        on that mesh, and ``expected_core_electrons`` is
        :math:`Z - Z_{\rm val}`. They agree to 1e-4 on every dataset this was
        checked against, so a disagreement means a misread table, not physics.

        Returns
        -------
        dict or None
            ``element``, ``atomic_number``, ``symbol``, ``titel``, ``zval``,
            ``r`` (Å), ``core_density`` and ``pseudo_core_density`` (e/Å³,
            lists of float, the second ``None`` when the dataset has none),
            ``core_electrons`` and ``expected_core_electrons``. ``None`` unless
            the POTCAR was read with ``parse_tables=True`` and is a PAW dataset
            with both a mesh and an all-electron core on it.
        """
        tables = self.radial_sets or {}
        r = tables.get("grid")
        raw = tables.get("core charge-density")
        if r is None or raw is None or len(r) != len(raw) or len(r) < 3:
            return None

        from scipy.integrate import simpson

        r = np.asarray(r, dtype=float)
        norm = np.sqrt(4.0 * np.pi)

        def density(values):
            return np.asarray(values, dtype=float) / (norm * r ** 2)

        pseudo = tables.get("core charge-density (pseudized)")
        if pseudo is not None and len(pseudo) != len(r):
            pseudo = None
        return {
            "element": self.element,
            "atomic_number": int(self.atomic_number),
            "symbol": self.symbol,
            "titel": self.titel,
            "zval": self.zval,
            "r": r.tolist(),
            "core_density": density(raw).tolist(),
            "pseudo_core_density": (None if pseudo is None
                                    else density(pseudo).tolist()),
            "core_electrons": float(norm * simpson(np.asarray(raw), x=r)),
            "expected_core_electrons": float(self.atomic_number - self.zval),
        }

    @property
    def element(self):
        """Bare chemical symbol, stripped of the POTCAR variant suffix."""
        return element_of(self.symbol)

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
            Also extract the raw ``local part`` float table and the core
            densities of the ``PAW radial sets`` block.

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
            radial_sets=_parse_radial_sets(text) if parse_tables else None,
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
        """Read a POTCAR from ``path``, transparently handling ``.gz``/``.Z``."""
        return cls.from_string(read_potcar_text(path),
                               parse_tables=parse_tables)

    @classmethod
    def from_library(cls, directory, elements, parse_tables=True):
        """
        Assemble a ``POTCAR`` for ``elements`` from a library directory.

        A ``POTCAR`` library is what VASP ships: one directory per species,
        each holding that species' file. This builds the concatenation a
        calculation would have used, **in the order given**, which is the order
        the structure lists its species in — a ``POTCAR`` whose species order
        disagrees with the ``POSCAR`` describes a different system.

        Its reason for existing is data that has a structure but no
        pseudopotentials: a Materials Project charge density, or a local run
        whose ``POTCAR`` was stripped for licensing. With the library the exact
        tabulated local potential can be reconstructed; without it only a model
        form factor can.

        Parameters
        ----------
        directory : str or pathlib.Path
            Library root; see :func:`find_potcar` for the layouts recognised.
        elements : sequence of str
            Bare chemical symbols, in species order.
        parse_tables : bool, optional
            Read the ``local part`` tables too. On by default, because the
            exact potential is the only reason to consult a library at all.

        Returns
        -------
        Potcar

        Raises
        ------
        FileNotFoundError
            If any element is missing from the library.
        ValueError
            If an entry holds more than one dataset, or is for the wrong
            element.
        """
        entries = []
        for element in elements:
            path = find_potcar(directory, element)
            single = cls.from_string(read_potcar_text(path),
                                     parse_tables=parse_tables)
            if len(single) != 1:
                raise ValueError(
                    f"{path} holds {len(single)} datasets; a library entry "
                    f"must contain exactly one."
                )
            if single[0].element != element:
                raise ValueError(
                    f"{path} is a POTCAR for {single[0].element!r}, not "
                    f"{element!r}."
                )
            entries.append(single[0])
        return cls(entries)

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
        return self.elements == [element_of(s) for s in poscar.symbols]

    def __repr__(self):
        return f"Potcar({', '.join(self.symbols)})"


# ---------------------------------------------------------------------- #
# POTCAR libraries
# ---------------------------------------------------------------------- #
#: Filenames a library entry may use, in preference order.
POTCAR_NAMES = ("POTCAR", "POTCAR.gz", "POTCAR.Z")


def read_potcar_text(path):
    """Read a ``POTCAR``, transparently handling ``.gz`` compression."""
    path = str(path)
    if path.endswith((".gz", ".Z")):
        try:
            with gzip.open(path, "rt", errors="replace") as handle:
                return handle.read()
        except gzip.BadGzipFile as error:
            # A true `.Z` is LZW (Unix `compress`), which the gzip module
            # cannot decode; failing here with the real reason beats a
            # BadGzipFile three frames deep in a library scan.
            raise ValueError(
                f"{path} is not gzip data. `.Z` archives are LZW-compressed; "
                f"decompress the library once (`uncompress` or `gzip -d`) "
                f"and point potcar_dir at the result."
            ) from error
    with open(path, "r", errors="replace") as handle:
        return handle.read()


def find_potcar(directory, element):
    r"""
    Locate the ``POTCAR`` for ``element`` inside a library directory.

    Recognised layouts, in preference order:

    1. ``<dir>/<element>/POTCAR`` --- what VASP ships;
    2. ``<dir>/<element>_<variant>/POTCAR`` --- ``Pt_pv``, ``Fe_sv``, ...;
    3. ``<dir>/POTCAR.<element>`` or ``<dir>/<element>.POTCAR`` --- flat.

    Each accepts a ``.gz`` or ``.Z`` suffix.

    Parameters
    ----------
    directory : str or pathlib.Path
        Library root.
    element : str
        Bare chemical symbol.

    Returns
    -------
    str
        Path to the file.

    Raises
    ------
    FileNotFoundError
        When nothing matches, listing what the directory does contain.
    ValueError
        When only *variant* directories match and there is more than one. The
        choice between ``Fe`` and ``Fe_pv`` changes ``ZVAL`` and therefore
        every energy, so it is the user's to make, not a coin flip.
    """
    directory = str(directory)

    exact = os.path.join(directory, element)
    if os.path.isdir(exact):
        for name in POTCAR_NAMES:
            candidate = os.path.join(exact, name)
            if os.path.isfile(candidate):
                return candidate

    for stem in (f"POTCAR.{element}", f"{element}.POTCAR"):
        for suffix in ("", ".gz", ".Z"):
            candidate = os.path.join(directory, stem + suffix)
            if os.path.isfile(candidate):
                return candidate

    variants = sorted(
        entry for entry in os.listdir(directory)
        if entry.startswith(f"{element}_")
        and os.path.isdir(os.path.join(directory, entry))
        and any(os.path.isfile(os.path.join(directory, entry, name))
                for name in POTCAR_NAMES)
    )
    if len(variants) == 1:
        chosen = os.path.join(directory, variants[0])
        for name in POTCAR_NAMES:
            candidate = os.path.join(chosen, name)
            if os.path.isfile(candidate):
                warnings.warn(
                    f"No plain {element!r} POTCAR in {directory}; using the "
                    f"only variant present, {variants[0]!r}.",
                    RuntimeWarning, stacklevel=4,
                )
                return candidate
    if len(variants) > 1:
        raise ValueError(
            f"No plain {element!r} POTCAR in {directory}, and several "
            f"variants exist: {variants}. They differ in ZVAL and therefore "
            f"in every energy, so name the one you want by passing an "
            f"explicit potcar= file."
        )

    available = sorted(entry for entry in os.listdir(directory)
                       if not entry.startswith("."))[:20]
    raise FileNotFoundError(
        f"No POTCAR for {element!r} under {directory}. Expected "
        f"{element}/POTCAR, POTCAR.{element} or {element}.POTCAR. "
        f"The directory contains: {available}"
    )


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
        non-numeric line. The first entry is ``PSGMAX``, the maximum
        wavevector of the table (see the module docstring); the remainder is
        the tabulated ``V_loc(q)``. ``None`` when the marker is absent.
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


#: ``PAW radial sets`` tables :attr:`PotcarSingle.core_profile` reads.
RADIAL_TABLES = ("grid", "core charge-density",
                 "core charge-density (pseudized)")


def _parse_radial_sets(text):
    """
    Extract the core-density tables of the ``PAW radial sets`` block.

    Each table is a title line followed by numeric lines up to the next
    non-numeric one, exactly as ``local part`` is. Titles are matched **whole**,
    not by prefix: ``core charge-density`` is a prefix of ``core charge-density
    (pseudized)``, and the reciprocal-space ``core charge-density (partial)``
    table earlier in the file is a different quantity on a different mesh ---
    which is also why the search starts at the block marker.

    Returns
    -------
    dict or None
        ``{title: numpy.ndarray}`` for the :data:`RADIAL_TABLES` present, or
        ``None`` for a dataset with no ``PAW radial sets`` block (an ultrasoft
        or norm-conserving potential).
    """
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.strip().lower() == "paw radial sets"), None)
    if start is None:
        return None

    tables = {}
    for index in range(start + 1, len(lines)):
        title = lines[index].strip()
        if title not in RADIAL_TABLES or title in tables:
            continue
        values = []
        for line in lines[index + 1:]:
            if not _NUMERIC_LINE_RE.match(line):
                break
            values.extend(_to_float(token) for token in line.split())
        if values:
            tables[title] = np.asarray(values, dtype=float)
    return tables or None
