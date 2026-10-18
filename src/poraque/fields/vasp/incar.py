# -*- coding: utf-8 -*-
# file: incar.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Reader for VASP ``INCAR`` files.

Raw tag values are preserved, with typed access to mesh, electronic and
convergence controls. Solver-specific validation lives in ElectronicSettings.
"""

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
        semicolons on one line, quoted strings and backslash continuations.

        Parameters
        ----------
        text : str
            File contents.

        Returns
        -------
        Incar
        """
        incar = cls()
        for raw_line in text.replace('\\\n', ' ').splitlines():
            # SYSTEM strings may contain literal semicolons/comment markers.
            statements, current, quote = [], [], None
            for char in raw_line:
                if char in ('"', "'"):
                    if quote is None:
                        quote = char
                    elif quote == char:
                        quote = None
                if quote is None and char in '!#':
                    break
                if quote is None and char == ';':
                    statements.append(''.join(current))
                    current = []
                else:
                    current.append(char)
            statements.append(''.join(current))
            if quote is not None:
                raise ValueError('Unterminated quoted INCAR string')
            for statement in statements:
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
            return float(str(self[key]).split()[0].replace("D", "E").replace("d", "e"))
        except (KeyError, IndexError, ValueError):
            return default

    def get_int(self, key, default=None):
        """Return tag ``key`` as ``int`` (``default`` when absent/unparsable)."""
        value = self.get_float(key, None)
        return default if value is None else int(value)

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Parse VASP/Fortran logical values without treating '.FALSE.' as true."""
        if key not in self:
            return default
        value = str(self[key]).strip().strip('.').upper()
        if value in ('T', 'TRUE'):
            return True
        if value in ('F', 'FALSE'):
            return False
        raise ValueError(f'Invalid logical {key}={self[key]}')

    def numbers(self, key: str) -> list[float]:
        """Expand numeric lists, including VASP repetition syntax (e.g. 4*0)."""
        result = []
        for token in str(self[key]).replace(',', ' ').split():
            parts = token.split('*')
            if len(parts) > 2:
                raise ValueError(f'Invalid numeric list {key}')
            number = float(parts[-1].replace('D', 'E').replace('d', 'e'))
            repeat = int(parts[0]) if len(parts) == 2 else 1
            if repeat < 0:
                raise ValueError('Negative repetition count')
            result.extend([number] * repeat)
        return result

    @property
    def ismear(self) -> int:
        """Occupation scheme; VASP's default is first-order Methfessel-Paxton."""
        return self.get_int('ISMEAR', 1)

    @property
    def sigma(self) -> float:
        """Smearing width in eV (VASP default 0.2)."""
        return self.get_float('SIGMA', 0.2)

    @property
    def ediff(self) -> float:
        """Electronic convergence threshold in eV (default 1e-4)."""
        return self.get_float('EDIFF', 1e-4)

    @property
    def nelm(self) -> int:
        """Maximum SCF iterations, retained as metadata for fixed-density runs."""
        return self.get_int('NELM', 60)

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
