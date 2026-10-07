# -*- coding: utf-8 -*-
# file: paw_setup.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The PAW dataset of one ``POTCAR`` element, read whole.

:mod:`poraque.fields.vasp.potcar` reads what the grid needs: the local
potential, the valence charge, the core profile and the projector channels.
The one-center energy of an atom needs the rest of the dataset --- the
partial waves, the core densities, the radial potentials, the bare one-center
matrix :math:`D^0_{ij}` --- and this module reads it, as the file stores it,
without interpreting it. :mod:`poraque.physics.paw` gives the tables their
meaning.

What the file holds, after the header parameters:

``Non local Part`` (one per projector angular momentum)
    ``l  nproj  rmax``, then the ``nproj x nproj`` matrix :math:`D^0_{ij}`
    (eV) for that ``l``, then each projector's ``Reciprocal Space Part`` and
    ``Real Space Part``.
``PAW radial sets``
    ``npoints  step``, then a sequence of titled tables: ``augmentation
    charges (non sperical)`` and ``uccopancies in atom`` (``nch x nch``, over
    every channel in projector order), ``grid`` (Å), ``aepotential``, ``core
    charge-density``, ``kinetic energy-density``, ``mkinetic energy-density
    pseudized``, ``local pseudopotential core``, ``pspotential valence only``,
    ``core charge-density (pseudized)``, and one ``pseudo wavefunction`` / ``ae
    wavefunction`` pair per channel.

The tables are stored as the file stores them --- the radial densities as
:math:`r^2\rho(r)`, the partial waves as :math:`r\phi(r)` --- and the units are
the file's; the consumer converts.
"""

from dataclasses import dataclass, field

import numpy as np

#: Titles inside ``PAW radial sets`` that occur once.
SINGLE_TABLES = (
    "augmentation charges (non sperical)", "uccopancies in atom", "grid",
    "aepotential", "core charge-density", "kinetic energy-density",
    "mkinetic energy-density pseudized", "local pseudopotential core",
    "pspotential valence only", "core charge-density (pseudized)",
)


def _floats(line):
    """The numbers on a line, or ``None`` when it is not a line of numbers."""
    tokens = line.split()
    if not tokens:
        return None
    try:
        return [float(token) for token in tokens]
    except ValueError:
        return None


def _header_value(text, key):
    """``KEY = value`` from the PSCTR header, as a string; ``None`` if absent."""
    for line in text.splitlines():
        stripped = line.strip()
        for part in stripped.split(";"):
            part = part.strip()
            if part.startswith(key) and "=" in part:
                name, value = part.split("=", 1)
                if name.strip() == key:
                    return value.split()[0] if value.split() else None
    return None


@dataclass
class PawSetup:
    """
    One element's PAW dataset, as read from its ``POTCAR`` block.

    Attributes
    ----------
    title : str
        ``TITEL``, e.g. ``"PAW_PBE C 08Apr2002"``.
    zval : float
        Valence charge.
    parameters : dict
        Header values: ``RCORE``, ``RDEP``, ``RDEPT``, ``RAUG``, ``RPACOR``,
        ``EATOM`` (eV), ``LEXCH`` and the like, as floats where numeric.
    channels : list of int
        The ``l`` of every projector channel, in file order.
    d0 : numpy.ndarray
        ``(nch, nch)`` bare one-center matrix in eV, block-diagonal in ``l``.
    tables : dict
        The single ``PAW radial sets`` tables by title, each a 1-D array (the
        two ``nch x nch`` tables reshaped).
    partial_waves : list of tuple
        ``(pseudo, ae)`` arrays per channel, in file order.
    reciprocal : dict
        The reciprocal-space tables before the projectors, by title:
        ``local part`` (``[PSGMAX, v_1..v_1000]``, eV Å³, including
        :math:`+4\pi Z_v e^2/q^2`), ``core charge-density (partial)`` and
        ``atomic pseudo charge-density`` (1000 values on the same q grid,
        electrons).
    compensation_radius : float
        The second number after ``PAW radial sets``, Å: the radius of the
        compensation charge (VASP's ``PSDMAX``).
    error_table : tuple of numpy.ndarray or None
        ``(ENCUT, error)`` of the header's basis-set error table, eV.
    """

    title: str
    zval: float
    parameters: dict = field(default_factory=dict)
    channels: list = field(default_factory=list)
    d0: np.ndarray = None
    tables: dict = field(default_factory=dict)
    partial_waves: list = field(default_factory=list)
    reciprocal: dict = field(default_factory=dict)
    compensation_radius: float = None
    error_table: tuple = None

    @property
    def element(self):
        from ..structure import element_of

        return element_of(self.title.split()[1])

    @property
    def grid(self):
        """The radial grid, Å."""
        return self.tables["grid"]

    @property
    def augmentation_charges(self):
        """``(nch, nch)``: the ``augmentation charges (non sperical)`` table."""
        n = len(self.channels)
        return self.tables["augmentation charges (non sperical)"].reshape(n, n)

    @property
    def atomic_occupancies(self):
        """``(nch, nch)``: the ``uccopancies in atom`` table."""
        n = len(self.channels)
        return self.tables["uccopancies in atom"].reshape(n, n)

    @classmethod
    def from_block(cls, text):
        """
        Parse one element's ``POTCAR`` block.

        Raises
        ------
        ValueError
            For a block with no ``PAW radial sets`` (an ultrasoft or
            norm-conserving dataset), or whose tables do not add up.
        """
        lines = text.splitlines()
        title = lines[0].strip()
        for line in lines:
            if "TITEL" in line:
                title = line.split("=", 1)[1].strip()
                break
        zval = float(_header_value(text, "ZVAL"))
        parameters = {}
        for key in ("RCORE", "RDEP", "RDEPT", "RAUG", "RPACOR", "RMAX", "EATOM",
                    "QCUT", "QGAM", "POMASS", "EAUG", "DEXC", "LEXCH",
                    "ICORE"):
            value = _header_value(text, key)
            if value is None:
                continue
            try:
                parameters[key] = float(value)
            except ValueError:
                parameters[key] = value

        # The basis-set error table of the header: NDATA values on
        # ENCUT = START * STEP**k (eV).
        ndata, step = _header_value(text, "NDATA"), None
        for line in lines:
            if line.strip().startswith("STEP") and "=" in line:
                step = [float(v) for v in line.split("=", 1)[1].split()[:2]]
                break
        error_table = None
        end_header = next((i for i, line in enumerate(lines)
                           if "END of PSCTR" in line), None)
        if ndata is not None and step is not None and end_header is not None:
            values = []
            for line in lines[:end_header]:
                numbers = _floats(line)
                if numbers is not None and len(numbers) > 2:
                    values.extend(numbers)
            ndata = int(float(ndata))
            if len(values) >= ndata:
                cutoffs = step[0] * step[1] ** np.arange(ndata)
                error_table = (cutoffs, np.array(values[-ndata:]))

        # Everything after the header is a run of titled number blocks.
        runs, run_title, run_values = [], None, []
        for line in lines[(end_header or 0) + 1:]:
            stripped = line.strip()
            if stripped.startswith("End of Dataset"):
                break
            if not stripped or stripped.startswith("("):
                continue
            numbers = _floats(line)
            if numbers is None:
                if run_title is not None:
                    runs.append((run_title, np.array(run_values)))
                run_title, run_values = stripped, []
            else:
                run_values.extend(numbers)
        if run_title is not None:
            runs.append((run_title, np.array(run_values)))

        reciprocal, channels, blocks = {}, [], []
        tables, waves = {}, []
        radial, npoints, rcomp = False, None, None
        for name, values in runs:
            if name == "PAW radial sets":
                radial = True
                npoints, rcomp = int(values[0]), float(values[1])
                continue
            if not radial:
                if name == "Non local Part":
                    l, nproj = int(values[0]), int(values[1])
                    channels.extend([l] * nproj)
                    blocks.append(values[3:3 + nproj * nproj].reshape(nproj,
                                                                      nproj))
                elif name not in ("Reciprocal Space Part", "Real Space Part"):
                    reciprocal.setdefault(name, values)
                continue
            if name in ("pseudo wavefunction", "ae wavefunction"):
                waves.append((name, values))
            else:
                tables[name] = values
        if not radial:
            raise ValueError(f"{title}: no 'PAW radial sets' block; this is "
                             f"not a PAW dataset.")

        d0 = np.zeros((len(channels), len(channels)))
        start = 0
        for block in blocks:
            n = block.shape[0]
            d0[start:start + n, start:start + n] = block
            start += n

        for name in ("grid", "aepotential", "core charge-density"):
            if name not in tables or tables[name].size != npoints:
                raise ValueError(f"{title}: the '{name}' table has "
                                 f"{tables.get(name, np.zeros(0)).size} values "
                                 f"for a {npoints}-point grid.")
        pseudo = [array for name, array in waves if name == "pseudo wavefunction"]
        ae = [array for name, array in waves if name == "ae wavefunction"]
        if len(pseudo) != len(channels) or len(ae) != len(channels):
            raise ValueError(f"{title}: {len(pseudo)} pseudo and {len(ae)} AE "
                             f"partial waves for {len(channels)} channels.")
        return cls(title=title, zval=zval, parameters=parameters,
                   channels=channels, d0=d0, tables=tables,
                   partial_waves=list(zip(pseudo, ae)),
                   reciprocal=reciprocal, compensation_radius=rcomp,
                   error_table=error_table)

    @classmethod
    def from_file(cls, path):
        """Every PAW dataset of a (possibly concatenated) ``POTCAR``."""
        from .potcar import read_potcar_text

        text = read_potcar_text(path)
        blocks = [block for block in text.split("End of Dataset") if block.strip()]
        return [cls.from_block(block + "End of Dataset") for block in blocks]
