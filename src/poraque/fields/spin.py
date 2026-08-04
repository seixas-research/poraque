# -*- coding: utf-8 -*-
# file: spin.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Spin-polarised densities (``ISPIN = 2``).

A spin-polarised ``CHGCAR`` carries **two** grid blocks rather than one:

.. code-block:: text

    <POSCAR header>
    NGXF NGYF NGZF
    <block 1>                  rho_up + rho_down   (total)
    augmentation occupancies   ...
    NGXF NGYF NGZF
    <block 2>                  rho_up - rho_down   (magnetisation)

:class:`SpinDensity` stores exactly that pair, in that order, because it is
what VASP writes and what VASP reads back. The
:math:`(\rho_\uparrow, \rho_\downarrow)` view is derived on demand
(:attr:`up`, :attr:`down`) rather than stored, so a round trip through disk
cannot silently change convention.

Why the total/magnetisation basis is also the right one to *learn*
------------------------------------------------------------------
The two channels are wildly different in scale — :math:`m` integrates to the
cell's magnetic moment, often a few :math:`\mu_B` against hundreds of
electrons in :math:`\rho` — and a network predicting
:math:`(\rho_\uparrow, \rho_\downarrow)` would have to produce that small
difference as a cancellation between two large numbers. Predicting
:math:`(\rho, m)` puts the small quantity on its own channel where its own
error can be measured, and keeps the electron count in a single channel that
:meth:`~poraque.fields.ChargeDensity.normalized` can fix.

A non-magnetic system is representable here: :math:`m \equiv 0`. That is a
useful property rather than a wasted channel, because it makes an
``ISPIN = 2`` model a strict generalisation of an ``ISPIN = 1`` one, and lets
a mixed dataset train a single operator.
"""

import numpy as np

from .density import ChargeDensity
from .grid import FieldGrid
from .vasp.volumetric import read_volumetric, write_volumetric


def is_spin_polarized(path):
    """
    Whether a ``CHGCAR``-format file carries a second (magnetisation) block.

    Reads the file rather than trusting a neighbouring ``INCAR``: the question
    is what this file contains, and a directory can hold an ``INCAR`` from a
    different run.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    bool
    """
    _, _, extra = read_volumetric(path, read_all=True)
    return len(extra) >= 1


def spin_from_incar(path, default=False):
    """
    ``ISPIN`` from an ``INCAR``, as a boolean.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``INCAR`` to read.
    default : bool, optional
        Returned when the file has no ``ISPIN`` tag — VASP's own default is
        ``ISPIN = 1``, so ``False``.

    Returns
    -------
    bool
    """
    from .vasp.incar import Incar

    value = Incar.from_file(path).get_int("ISPIN")
    return default if value is None else value == 2


class SpinDensity:
    r"""
    Valence density of a spin-polarised calculation.

    Parameters
    ----------
    total : array_like
        :math:`\rho_\uparrow + \rho_\downarrow` in e/Å³, shape ``grid.shape``.
    magnetization : array_like
        :math:`\rho_\uparrow - \rho_\downarrow` in e/Å³, same shape.
    grid : FieldGrid
        Shared mesh.
    structure : Poscar
        Geometry, written into the file header.
    metadata : dict, optional
        Free-form provenance.

    Attributes
    ----------
    n_channels : int
        Always ``2``. Present so the ML layer can size an operator from the
        field rather than from a flag that might disagree with it.

    Examples
    --------
    >>> density = SpinDensity.read("CHGCAR")             # doctest: +SKIP
    >>> density.electron_count(), density.magnetic_moment()   # doctest: +SKIP
    (32.0, 2.0)
    """

    name = "spin-polarised valence charge density"
    default_filename = "CHGCAR"
    unit = "e/Ang^3"
    volume_scaled = True
    n_channels = 2

    def __init__(self, total, magnetization, grid, structure, metadata=None):
        self.total = np.asarray(total, dtype=float)
        self.magnetization = np.asarray(magnetization, dtype=float)
        self.grid = grid
        self.structure = structure
        self.metadata = dict(metadata or {})

        for label, values in (("total", self.total),
                              ("magnetization", self.magnetization)):
            if values.shape != tuple(grid.shape):
                raise ValueError(
                    f"SpinDensity: {label} has shape {values.shape}, which "
                    f"does not match the grid {tuple(grid.shape)}."
                )

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    @property
    def data(self):
        r"""
        ``(2, Nx, Ny, Nz)`` stack of :math:`(\rho, m)`.

        The layout the ML layer consumes: channel first, matching
        ``torch.nn.Conv3d``.
        """
        return np.stack([self.total, self.magnetization], axis=0)

    @property
    def up(self):
        r""":math:`\rho_\uparrow = (\rho + m)/2`."""
        return 0.5 * (self.total + self.magnetization)

    @property
    def down(self):
        r""":math:`\rho_\downarrow = (\rho - m)/2`."""
        return 0.5 * (self.total - self.magnetization)

    def as_charge_density(self):
        """
        The total density alone, as a :class:`~poraque.fields.ChargeDensity`.

        Everything in :mod:`poraque.physics` that predates spin support takes
        the total density, so this is the adapter that lets a spin-polarised
        prediction be fed to it unchanged. Note that an ``LSDA``/spin-polarised
        exchange-correlation energy needs both channels and is **not**
        obtainable this way; see :func:`~poraque.physics.energy.xc_energy`.
        """
        return ChargeDensity(self.total, self.grid, self.structure,
                             metadata=dict(self.metadata))

    # ------------------------------------------------------------------ #
    # Integrals
    # ------------------------------------------------------------------ #
    def electron_count(self):
        """:math:`\\int\\rho\\,d^3r`, the valence electron count."""
        return float(self.grid.integrate(self.total))

    def magnetic_moment(self):
        r""":math:`\int m\,d^3r`, the cell's magnetic moment in :math:`\mu_B`."""
        return float(self.grid.integrate(self.magnetization))

    def normalized(self, n_electrons, clip_negative=True):
        r"""
        Rescale the **total** channel to ``n_electrons``.

        The magnetisation is scaled by the same factor rather than left alone,
        so that :math:`\rho_\uparrow` and :math:`\rho_\downarrow` stay
        non-negative and their ratio — the physical polarisation at each point
        — is untouched. Rescaling :math:`\rho` alone would change the local
        polarisation :math:`m/\rho` everywhere, which is not a normalisation
        but a different prediction.

        See :meth:`~poraque.fields.ChargeDensity.normalized` for why the
        rescaling is needed at all.

        Parameters
        ----------
        n_electrons : float
            Target valence electron count.
        clip_negative : bool, optional
            Clip :math:`\rho_\uparrow` and :math:`\rho_\downarrow` — not
            :math:`\rho` and :math:`m` — at zero first. Those are the two
            quantities that are physically non-negative; :math:`m` legitimately
            takes either sign.

        Returns
        -------
        SpinDensity

        Raises
        ------
        ValueError
            If the total density integrates to zero.
        """
        up, down = self.up, self.down
        if clip_negative:
            up = np.clip(up, 0.0, None)
            down = np.clip(down, 0.0, None)

        total = up + down
        current = float(self.grid.integrate(total))
        if abs(current) < 1e-30:
            raise ValueError(
                "The total density integrates to zero, so it cannot be "
                "normalized to a finite electron count."
            )

        factor = float(n_electrons) / current
        metadata = dict(self.metadata)
        metadata["electron_count_before_normalization"] = current
        return SpinDensity(total * factor, (up - down) * factor, self.grid,
                           self.structure, metadata=metadata)

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    @classmethod
    def read(cls, path, grid=None):
        """
        Read a spin-polarised ``CHGCAR``.

        Parameters
        ----------
        path : str or pathlib.Path
        grid : FieldGrid, optional
            Shared mesh; built from the file header when omitted.

        Returns
        -------
        SpinDensity

        Raises
        ------
        ValueError
            When the file has only one grid block. A non-spin-polarised file
            read as spin-polarised would otherwise silently acquire a zero
            magnetisation, which is a physical claim the file does not make.
        """
        structure, raw, extra = read_volumetric(path, read_all=True)
        if not extra:
            raise ValueError(
                f"{path} carries a single grid block, so it is not a "
                f"spin-polarised CHGCAR. Read it with ChargeDensity.read, or "
                f"check ISPIN in the INCAR that produced it."
            )

        if grid is None:
            grid = FieldGrid(raw.shape, structure.cell)
        elif tuple(grid.shape) != raw.shape:
            raise ValueError(
                f"{path}: grid shape {raw.shape} does not match the supplied "
                f"shared grid {tuple(grid.shape)}."
            )

        volume = grid.volume
        return cls(raw / volume, extra[0] / volume, grid, structure,
                   metadata={"source": str(path), "ispin": 2})

    @classmethod
    def from_up_down(cls, up, down, grid, structure, metadata=None):
        r"""Build from :math:`(\rho_\uparrow, \rho_\downarrow)`."""
        up = np.asarray(up, dtype=float)
        down = np.asarray(down, dtype=float)
        return cls(up + down, up - down, grid, structure, metadata=metadata)

    @classmethod
    def from_channels(cls, data, grid, structure, metadata=None):
        r"""Build from a ``(2, Nx, Ny, Nz)`` stack of :math:`(\rho, m)`."""
        values = np.asarray(data, dtype=float)
        if values.shape != (2,) + tuple(grid.shape):
            raise ValueError(
                f"Expected a (2, {', '.join(map(str, grid.shape))}) stack, got "
                f"{values.shape}."
            )
        return cls(values[0], values[1], grid, structure, metadata=metadata)

    def write(self, path=None, comment=None, columns=5, width=17, decimals=11,
              augmentation=None):
        """
        Write both blocks in VASP's spin-polarised ``CHGCAR`` layout.

        Parameters
        ----------
        path : str or pathlib.Path, optional
        comment : str, optional
        columns, width, decimals : int, optional
            Passed through to
            :func:`~poraque.fields.vasp.volumetric.write_volumetric`.
        augmentation : sequence of str, optional
            PAW records, written between the two blocks exactly as VASP does.

        Returns
        -------
        str
            The path written.
        """
        path = str(path if path is not None else self.default_filename)
        if comment is None:
            formula = "".join(f"{s}{c}" for s, c in
                              zip(self.structure.symbols, self.structure.counts))
            comment = f"{formula}  CHGCAR: {self.name} [{self.unit}]"

        volume = self.grid.volume
        write_volumetric(path, self.structure, self.total * volume,
                         comment=comment, columns=columns, width=width,
                         decimals=decimals, augmentation=augmentation)

        # The second block repeats the grid header and carries no structure
        # header -- the file has exactly one. Written through the same
        # column-positional path as the first block, since VASP reads it with
        # the same non-advancing (1X,E17.11) and a value one column off fails
        # the whole file.
        with open(path, "a") as handle:
            handle.write("  {:d}  {:d}  {:d}\n".format(*self.grid.shape))
            handle.write(_format_block(self.magnetization * volume,
                                       columns, width, decimals))
        return path

    def __repr__(self):
        return (f"SpinDensity(shape={tuple(self.grid.shape)}, "
                f"electrons={self.electron_count():.4f}, "
                f"moment={self.magnetic_moment():.4f} mu_B)")


def _format_block(values, columns, width, decimals):
    """One grid block, in the same layout :func:`write_volumetric` produces."""
    from .vasp.volumetric import fortran_exponential

    flat = np.asarray(values, dtype=float).ravel(order="F")
    lines = []
    for start in range(0, flat.size, columns):
        chunk = flat[start:start + columns]
        lines.append("".join(
            " " + fortran_exponential(value, decimals=decimals, width=width)
            for value in chunk))
    return "\n".join(lines) + "\n"
