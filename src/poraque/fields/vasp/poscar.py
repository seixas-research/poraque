# -*- coding: utf-8 -*-
# file: poscar.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Reader/writer for VASP ``POSCAR`` / ``CONTCAR`` structure files.

:class:`Poscar` is a thin **serializer** on top of the code-agnostic
:class:`~poraque.fields.structure.Structure`: it adds VASP's text format and
nothing else. All geometry lives in the base class, so a Quantum ESPRESSO or
GPAW reader produces the same object type and everything downstream — grids,
fields, the ML dataset — is unaffected by which code the data came from.
"""

import numpy as np

# `symbol_to_z` is re-exported: `poraque.fields.vasp` publishes it, and
# `potcar` imports it from here. `element_of` rode along on this line without a
# single consumer -- it is available from `poraque.fields.structure`, which is
# where it is defined.
from ..structure import Structure, symbol_to_z  # noqa: F401  (re-export)


class Poscar(Structure):
    """
    A VASP ``POSCAR``/``CONTCAR`` structure.

    See :class:`~poraque.fields.structure.Structure` for the geometry API; this
    subclass only adds :meth:`from_string` / :meth:`to_string` and their file
    counterparts.
    """

    @classmethod
    def from_string(cls, text, symbols=None):
        """
        Parse a POSCAR from a string.

        Handles the universal scaling factor in all three of its forms (single
        positive value, single negative value meaning a *target volume*, and
        the three-value form accepted by VASP 6), optional selective dynamics,
        and both ``Direct`` and ``Cartesian`` coordinates.

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

        scale_tokens = [float(t) for t in lines[1].split()]
        cell = np.array([[float(t) for t in lines[i].split()[:3]] for i in (2, 3, 4)])
        if len(scale_tokens) == 3:
            cell = cell * np.asarray(scale_tokens)[:, None]
        else:
            scale = scale_tokens[0]
            if scale < 0:
                # A negative value is the target volume of the cell.
                scale = (abs(scale) / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
            cell = cell * scale

        # The species-symbol line is present in VASP 5+, absent in VASP 4.
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
            # Cartesian coordinates are scaled by the same factor, which is
            # already folded into `cell`; invert to fractional.
            scale_c = (scale_tokens[0] if len(scale_tokens) == 1
                       and scale_tokens[0] > 0 else 1.0)
            coords = (coords * scale_c) @ np.linalg.inv(cell)

        return cls(cell, file_symbols, counts, coords,
                   comment=comment, selective_dynamics=flags)

    @classmethod
    def from_file(cls, path, symbols=None):
        """Read a POSCAR/CONTCAR from ``path``, compressed or not."""
        from ..io.compressed import open_text

        with open_text(path) as handle:
            return cls.from_string(handle.read(), symbols=symbols)

    @classmethod
    def from_structure(cls, structure):
        """Wrap any :class:`Structure` so it can be written in VASP format."""
        if isinstance(structure, cls):
            return structure
        return cls(structure.cell, structure.symbols, structure.counts,
                   structure.scaled_positions, comment=structure.comment,
                   selective_dynamics=structure.selective_dynamics)

    def to_string(self, direct=True):
        """Serialize back to POSCAR format."""
        out = [self.comment or "generated by poraque", "   1.00000000000000"]
        for vector in self.cell:
            out.append("  {:>21.16f} {:>21.16f} {:>21.16f}".format(*vector))
        out.append("  " + "  ".join(f"{s:>3s}" for s in self.symbols))
        out.append("  " + "  ".join(f"{c:>3d}" for c in self.counts))
        flags = self.selective_dynamics
        if flags is not None:
            # Parsed on the way in, so it must survive the way out: dropping
            # the flags here silently unfroze a constrained relaxation.
            out.append("Selective dynamics")
        out.append("Direct" if direct else "Cartesian")
        coords = self.scaled_positions if direct else self.positions
        for index, row in enumerate(coords):
            line = "  {:>19.16f} {:>19.16f} {:>19.16f}".format(*row)
            if flags is not None:
                line += "   " + " ".join("T" if f else "F"
                                         for f in flags[index])
            out.append(line)
        return "\n".join(out) + "\n"

    def write(self, path, direct=True):
        """Write this structure to ``path`` in POSCAR format."""
        with open(path, "w") as handle:
            handle.write(self.to_string(direct=direct))


def _is_int(token):
    """True when ``token`` parses as an integer."""
    try:
        int(token)
    except ValueError:
        return False
    return True
