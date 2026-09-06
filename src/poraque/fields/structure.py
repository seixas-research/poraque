# -*- coding: utf-8 -*-
# file: structure.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Code-agnostic crystal structure.

:class:`Structure` is the neutral geometry container every reader in
:mod:`poraque.fields.io` produces and every field carries. It deliberately
knows nothing about VASP, Quantum ESPRESSO or GPAW: format-specific parsing
and serialization live in the readers, so adding a code means adding a reader,
never touching this class or anything downstream of it.

Units are Ångström throughout, matching the rest of :mod:`poraque.fields`.
"""

import numpy as np

# Minimal symbol -> Z table (Z = 1..103), so the package needs no ASE at import.
_CHEMICAL_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr"
).split()
_ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(_CHEMICAL_SYMBOLS)}


def symbol_to_z(symbol):
    """
    Map a chemical symbol to its atomic number.

    Accepts the decorated forms every code uses for pseudopotential variants —
    VASP's ``Si_sv``/``Fe_pv``, Quantum ESPRESSO's ``Fe1``, ``O.pbe-n-kjpaw`` —
    by stripping everything after the first underscore or dot and any trailing
    digits.

    Parameters
    ----------
    symbol : str
        Chemical symbol, optionally decorated.

    Returns
    -------
    int
        Atomic number ``Z``.
    """
    element = element_of(symbol)
    element = element.capitalize() if len(element) > 1 else element.upper()
    try:
        return _ATOMIC_NUMBERS[element]
    except KeyError:
        raise ValueError(f"Unknown chemical symbol: {symbol!r}") from None


def element_of(symbol):
    """
    Bare element name of a possibly decorated symbol.

    ``Fe_pv`` (a VASP POTCAR variant), ``O.pbe-n-kjpaw`` (a Quantum ESPRESSO
    pseudopotential name), ``Fe1`` (a numbered species) and ``H.5`` (VASP's
    fractional hydrogen) all reduce to their element. This is the **one**
    stripping rule in the package: every valence-charge lookup, POTCAR match
    and per-element table keys on its result, so two call sites can never
    disagree about which element a symbol names.
    """
    return str(symbol).split("_")[0].split(".")[0].strip().rstrip("0123456789")


class Structure:
    """
    A periodic crystal structure.

    Parameters
    ----------
    cell : array_like
        ``(3, 3)`` lattice vectors in Ångström; rows are ``a1, a2, a3``.
    symbols : sequence of str
        One chemical symbol per *species*, e.g. ``["Si", "O"]``.
    counts : sequence of int
        Number of atoms of each species, aligned with ``symbols``.
    scaled_positions : array_like
        ``(natoms, 3)`` fractional coordinates.
    comment : str, optional
        Free-form label.
    selective_dynamics : array_like, optional
        ``(natoms, 3)`` boolean mask, when the source format carries one.

    Notes
    -----
    Atoms are grouped by species — the layout VASP requires and every other
    code tolerates — so a species is addressed by a contiguous slice; see
    :meth:`species_slices`.
    """

    def __init__(self, cell, symbols, counts, scaled_positions, comment="",
                 selective_dynamics=None):
        self.comment = str(comment).strip()
        self.cell = np.asarray(cell, dtype=float).reshape(3, 3)
        self.symbols = [str(s) for s in symbols]
        self.counts = [int(c) for c in counts]
        self.scaled_positions = np.asarray(scaled_positions, dtype=float).reshape(-1, 3)
        self.selective_dynamics = (
            None if selective_dynamics is None
            else np.asarray(selective_dynamics, dtype=bool).reshape(-1, 3)
        )

        if len(self.symbols) != len(self.counts):
            raise ValueError(
                f"{len(self.symbols)} species symbols but {len(self.counts)} counts."
            )
        if sum(self.counts) != len(self.scaled_positions):
            raise ValueError(
                f"Species counts sum to {sum(self.counts)} but "
                f"{len(self.scaled_positions)} positions were given."
            )

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def natoms(self):
        """Total number of atoms."""
        return int(sum(self.counts))

    @property
    def volume(self):
        """Cell volume in Å³."""
        return float(abs(np.linalg.det(self.cell)))

    @property
    def elements(self):
        """Bare element names, one per species."""
        return [element_of(symbol) for symbol in self.symbols]

    @property
    def symbols_per_atom(self):
        """List of length :attr:`natoms` with one symbol per atom."""
        out = []
        for symbol, count in zip(self.symbols, self.counts):
            out.extend([symbol] * count)
        return out

    @property
    def atomic_numbers(self):
        """``(natoms,)`` array of atomic numbers."""
        return np.array([symbol_to_z(s) for s in self.symbols_per_atom], dtype=int)

    @property
    def positions(self):
        """``(natoms, 3)`` Cartesian coordinates in Ångström."""
        return self.scaled_positions @ self.cell

    @property
    def formula(self):
        """Compact formula string, e.g. ``"Si2O4"``."""
        return "".join(f"{s}{c}" for s, c in zip(self.symbols, self.counts))

    def species_slices(self):
        """
        Yield ``(symbol, slice)`` pairs indexing the atoms of each species.

        Yields
        ------
        tuple of (str, slice)
        """
        start = 0
        for symbol, count in zip(self.symbols, self.counts):
            yield symbol, slice(start, start + count)
            start += count

    # ------------------------------------------------------------------ #
    # Interoperability
    # ------------------------------------------------------------------ #
    def to_ase(self):
        """Convert to an :class:`ase.Atoms` object (requires ASE)."""
        from ase import Atoms

        return Atoms(
            symbols=[element_of(s) for s in self.symbols_per_atom],
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            pbc=True,
        )

    @classmethod
    def from_ase(cls, atoms, comment=""):
        """
        Build a :class:`Structure` from an :class:`ase.Atoms` object.

        Atoms are reordered so each species is contiguous, as required by the
        species-block layout.
        """
        per_atom = list(atoms.get_chemical_symbols())
        order = np.argsort([_ATOMIC_NUMBERS.get(s, 0) for s in per_atom], kind="stable")

        symbols, counts = [], []
        for index in order:
            symbol = per_atom[index]
            if symbols and symbols[-1] == symbol:
                counts[-1] += 1
            else:
                symbols.append(symbol)
                counts.append(1)

        return cls(
            cell=np.asarray(atoms.get_cell()),
            symbols=symbols,
            counts=counts,
            scaled_positions=np.asarray(atoms.get_scaled_positions())[order],
            comment=comment or atoms.get_chemical_formula(),
        )

    def __len__(self):
        return self.natoms

    def __repr__(self):
        return f"{type(self).__name__}({self.formula}, volume={self.volume:.3f} A^3)"
