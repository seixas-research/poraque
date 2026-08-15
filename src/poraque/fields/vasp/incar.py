# -*- coding: utf-8 -*-
# file: incar.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Reader for VASP ``INCAR`` files.

Only the tags that influence the 3D field grid are interpreted
(:attr:`Incar.encut`, :attr:`Incar.prec`, and any explicit ``NGX*``/``NGX*F``);
every other tag is kept as a raw string so nothing is silently lost.
"""

import re


class Incar(dict):
    """
    A parsed ``INCAR`` file: a ``dict`` of upper-case tag -> raw string value.

    Values are stored verbatim; use :meth:`get_float`, :meth:`get_int` or the
    named properties for typed access.
    """

    @classmethod
    def from_string(cls, text):
        """
        Parse INCAR contents.

        Handles ``!``/``#`` comments, several ``TAG = VALUE`` pairs separated by
        semicolons on one line, and continuation-free multi-token values.

        Parameters
        ----------
        text : str
            File contents.

        Returns
        -------
        Incar
        """
        incar = cls()
        for raw_line in text.splitlines():
            line = re.split(r"[!#]", raw_line, maxsplit=1)[0].strip()
            if not line:
                continue
            for statement in line.split(";"):
                if "=" not in statement:
                    continue
                key, value = statement.split("=", 1)
                key = key.strip().upper()
                if key:
                    incar[key] = value.strip()
        return incar

    @classmethod
    def from_file(cls, path):
        """Read an INCAR from ``path``."""
        with open(path, "r") as handle:
            return cls.from_string(handle.read())

    # ------------------------------------------------------------------ #
    # Typed access
    # ------------------------------------------------------------------ #
    def get_float(self, key, default=None):
        """Return tag ``key`` as ``float`` (``default`` when absent/unparsable)."""
        try:
            return float(str(self[key]).split()[0])
        except (KeyError, IndexError, ValueError):
            return default

    def get_int(self, key, default=None):
        """Return tag ``key`` as ``int`` (``default`` when absent/unparsable)."""
        value = self.get_float(key, None)
        return default if value is None else int(value)

    @property
    def encut(self):
        """Plane-wave cutoff ``ENCUT`` in eV, or ``None`` if unset."""
        return self.get_float("ENCUT")

    @property
    def prec(self):
        """Lower-case ``PREC`` setting; defaults to ``'normal'``."""
        return str(self.get("PREC", "normal")).strip().lower()

    @property
    def coarse_shape(self):
        """
        Explicit ``(NGX, NGY, NGZ)`` from the INCAR, or ``None``.

        All three tags must be present for a shape to be returned.
        """
        return self._shape_tags("NGX", "NGY", "NGZ")

    @property
    def fine_shape(self):
        """
        Explicit ``(NGXF, NGYF, NGZF)`` from the INCAR, or ``None``.

        This is the grid VASP uses for the charge density, i.e. the grid a
        ``CHGCAR`` is written on. When present it must always win over any
        ``ENCUT``-derived estimate.
        """
        return self._shape_tags("NGXF", "NGYF", "NGZF")

    def _shape_tags(self, *keys):
        values = [self.get_int(key) for key in keys]
        if any(v is None for v in values):
            return None
        return tuple(int(v) for v in values)

    def __repr__(self):
        return f"Incar({len(self)} tags, ENCUT={self.encut}, PREC={self.prec!r})"
