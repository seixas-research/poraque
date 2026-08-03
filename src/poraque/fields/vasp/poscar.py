# -*- coding: utf-8 -*-
# file: poscar.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Reader/writer for VASP ``POSCAR`` / ``CONTCAR`` structure files.

The parser is deliberately dependency-free (NumPy only) so that the
:mod:`poraque.fields` stack can be used on machines without ASE, while
:meth:`Poscar.to_ase` provides the bridge when ASE *is* available.
"""

import numpy as np

# Minimal symbol -> Z table (used when ASE is unavailable). Covers Z = 1..103.
_CHEMICAL_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr"
).split()
_ATOMIC_NUMBERS = {s: i for i, s in enumerate(_CHEMICAL_SYMBOLS)}


def symbol_to_z(symbol):
    """
    Map a chemical symbol to its atomic number.

    Accepts POTCAR-style decorated symbols (``"Si_sv"``, ``"Fe_pv"``,
    ``"Ga_d"``) by stripping everything after the first underscore.

    Parameters
    ----------
    symbol : str
        Chemical symbol, optionally decorated.

    Returns
    -------
    int
        Atomic number ``Z``.
    """
    element = str(symbol).split("_")[0].split(".")[0].strip()
    element = element.capitalize() if len(element) > 1 else element.upper()
    try:
        return _ATOMIC_NUMBERS[element]
    except KeyError:
        raise ValueError(f"Unknown chemical symbol: {symbol!r}") from None


class Poscar:
    """
    A VASP ``POSCAR``/``CONTCAR`` structure.

    Attributes
    ----------
    comment : str
        First line of the file (free-form system name).
    cell : numpy.ndarray
        ``(3, 3)`` lattice vectors in Ångström, **already multiplied** by the
        universal scaling factor (rows are ``a1, a2, a3``).
    symbols : list of str
        One chemical symbol per *species* (e.g. ``["Si", "O"]``).
    counts : list of int
        Number of atoms of each species, aligned with :attr:`symbols`.
    scaled_positions : numpy.ndarray
        ``(natoms, 3)`` fractional coordinates.
    selective_dynamics : numpy.ndarray or None
        ``(natoms, 3)`` boolean array when selective dynamics is active.
    """

    def __init__(self, cell, symbols, counts, scaled_positions,
                 comment="", selective_dynamics=None):
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

    def species_slices(self):
        """
        Yield ``(symbol, slice)`` pairs indexing the atoms of each species.

        Yields
        ------
        tuple of (str, slice)
            Species symbol and the slice selecting its atoms in
            :attr:`scaled_positions`.
        """
        start = 0
        for symbol, count in zip(self.symbols, self.counts):
            yield symbol, slice(start, start + count)
            start += count

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    @classmethod
    def from_string(cls, text, symbols=None):
        """
        Parse a POSCAR from a string.

        Parameters
        ----------
        text : str
            File contents.
        symbols : sequence of str, optional
            Species symbols, required only for VASP-4 style files that omit the
            symbol line.

        Returns
        -------
        Poscar
        """
        lines = [line.rstrip() for line in text.splitlines()]
        if len(lines) < 8:
            raise ValueError("POSCAR is too short to be valid.")

        comment = lines[0]

        # --- Universal scaling factor: 1 value, or 3 (VASP >= 6) ---
        scale_tokens = [float(t) for t in lines[1].split()]
        cell = np.array([[float(t) for t in lines[i].split()[:3]] for i in (2, 3, 4)])
        if len(scale_tokens) == 3:
            cell = cell * np.asarray(scale_tokens)[:, None]
        else:
            scale = scale_tokens[0]
            if scale < 0:
                # A negative value is the *target volume* of the cell.
                scale = (abs(scale) / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
            cell = cell * scale

        # --- Species line is optional (VASP 4 vs 5+) ---
        cursor = 5
        tokens = lines[cursor].split()
        if tokens and not _is_int(tokens[0]):
            file_symbols = tokens
            cursor += 1
        elif symbols is not None:
            file_symbols = list(symbols)
        else:
            raise ValueError(
                "POSCAR has no species-symbol line (VASP 4 format); pass "
                "`symbols=` explicitly or supply a POTCAR."
            )

        counts = [int(t) for t in lines[cursor].split()]
        cursor += 1
        natoms = sum(counts)

        # --- Optional selective dynamics flag ---
        selective = lines[cursor].strip()[:1] in ("S", "s")
        if selective:
            cursor += 1

        mode = lines[cursor].strip()[:1].lower()
        cursor += 1
        cartesian = mode in ("c", "k")

        coords = np.empty((natoms, 3), dtype=float)
        flags = np.ones((natoms, 3), dtype=bool) if selective else None
        for i in range(natoms):
            tokens = lines[cursor + i].split()
            coords[i] = [float(t) for t in tokens[:3]]
            if selective and len(tokens) >= 6:
                flags[i] = [t.upper().startswith("T") for t in tokens[3:6]]

        if cartesian:
            # Cartesian coordinates are also affected by the scaling factor,
            # which is already folded into `cell`; invert to fractional.
            scale_c = scale_tokens[0] if len(scale_tokens) == 1 and scale_tokens[0] > 0 else 1.0
            coords = (coords * scale_c) @ np.linalg.inv(cell)

        return cls(cell, file_symbols, counts, coords,
                   comment=comment, selective_dynamics=flags)

    @classmethod
    def from_file(cls, path, symbols=None):
        """Read a POSCAR/CONTCAR from ``path``."""
        with open(path, "r") as handle:
            return cls.from_string(handle.read(), symbols=symbols)

    def to_string(self, direct=True):
        """Serialize back to POSCAR format."""
        out = [self.comment or "generated by poraque", "   1.00000000000000"]
        for vector in self.cell:
            out.append("  {:>21.16f} {:>21.16f} {:>21.16f}".format(*vector))
        out.append("  " + "  ".join(f"{s:>3s}" for s in self.symbols))
        out.append("  " + "  ".join(f"{c:>3d}" for c in self.counts))
        out.append("Direct" if direct else "Cartesian")
        coords = self.scaled_positions if direct else self.positions
        for row in coords:
            out.append("  {:>19.16f} {:>19.16f} {:>19.16f}".format(*row))
        return "\n".join(out) + "\n"

    def write(self, path, direct=True):
        """Write this structure to ``path`` in POSCAR format."""
        with open(path, "w") as handle:
            handle.write(self.to_string(direct=direct))

    # ------------------------------------------------------------------ #
    # Interoperability
    # ------------------------------------------------------------------ #
    def to_ase(self):
        """Convert to an :class:`ase.Atoms` object (requires ASE)."""
        from ase import Atoms

        return Atoms(
            symbols=[s.split("_")[0] for s in self.symbols_per_atom],
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            pbc=True,
        )

    @classmethod
    def from_ase(cls, atoms, comment=""):
        """Build a :class:`Poscar` from an :class:`ase.Atoms` object."""
        symbols_per_atom = list(atoms.get_chemical_symbols())
        order = np.argsort([_ATOMIC_NUMBERS.get(s, 0) for s in symbols_per_atom],
                           kind="stable")
        sorted_symbols = [symbols_per_atom[i] for i in order]

        symbols, counts = [], []
        for symbol in sorted_symbols:
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

    def to_system(self, electrons=None):
        """
        Convert to a :class:`poraque.core.System` (atomic units, Bohr).

        Parameters
        ----------
        electrons : int, optional
            Electron count; defaults to the neutral all-electron value.
        """
        from ...core.system import System
        from ..constants import ANGSTROM_TO_BOHR

        return System(
            positions=self.positions * ANGSTROM_TO_BOHR,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell * ANGSTROM_TO_BOHR,
            pbc=True,
            electrons=electrons,
        )

    def __len__(self):
        return self.natoms

    def __repr__(self):
        formula = "".join(f"{s}{c}" for s, c in zip(self.symbols, self.counts))
        return f"Poscar({formula}, volume={self.volume:.3f} A^3)"


def _is_int(token):
    """True when ``token`` parses as an integer."""
    try:
        int(token)
    except ValueError:
        return False
    return True
