# -*- coding: utf-8 -*-
# file: hartree.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The Hartree potential as a field, solved from the density.

:math:`v_{\rm H}` is **not learned**. It follows from :math:`\rho` by Poisson's
equation, which is an exact linear relation:

.. math::

    \nabla^2 v_{\rm H} = -4\pi e^2\rho
    \quad\Longleftrightarrow\quad
    v_{\rm H}(\mathbf G) = \frac{4\pi e^2\,\rho(\mathbf G)}{G^2},
    \qquad v_{\rm H}(\mathbf G = 0) = 0 .

On a periodic plane-wave grid the reciprocal-space form is not an
approximation to the real-space Laplacian --- it *is* the solution, exact for
every band-limited density, and costs two FFTs. Training a second operator to
predict it would be strictly worse: it would introduce error into a quantity
that has none, and would not be guaranteed to satisfy the very equation that
defines it.

The :math:`\mathbf G = 0` term is set to zero rather than diverging. That is
the neutralizing-background convention, and it is the same one
:class:`~poraque.fields.ExternalPotential` uses --- which is what makes the two
potentials addable, and what makes
:func:`~poraque.physics.energy.hartree_energy` of a uniform density come out at
exactly zero rather than infinite.

    from poraque.fields import HartreePotential

    v_hartree = HartreePotential.from_density(rho)
    v_hartree.write("LOCPOT")
"""

import numpy as np

from .base import ScalarField


class HartreePotential(ScalarField):
    r"""
    Classical electrostatic potential of the electrons, in eV.

    Notes
    -----
    Written by default to ``LOCPOT``, VASP's name for a local potential on the
    FFT grid. Unlike ``CHGCAR`` a ``LOCPOT`` stores the potential **directly**,
    not multiplied by the cell volume, so :attr:`volume_scaled` is false and
    the value on disk is the value in eV. Getting this backwards produces a
    file that opens without complaint and is wrong by a factor of the cell
    volume.

    .. warning::

       This is :math:`v_{\rm H}` alone --- the electron-electron term. VASP's
       ``LOCPOT`` conventionally holds :math:`v_{\rm H} + v_{\rm ext}`, and
       with ``LVHAR = .TRUE.`` the Hartree part only. Add
       :class:`~poraque.fields.ExternalPotential` if the total local potential
       is what is wanted; both use the same :math:`\mathbf G = 0` convention,
       so they may simply be summed.
    """

    name = "Hartree potential"
    default_filename = "LOCPOT"
    unit = "eV"
    volume_scaled = False

    @classmethod
    def from_density(cls, density, grid=None, structure=None, metadata=None):
        r"""
        Solve Poisson's equation for ``density``.

        Parameters
        ----------
        density : ChargeDensity, SpinDensity or array_like
            Electron density in e/Å³. A
            :class:`~poraque.fields.SpinDensity` contributes through its
            **total** channel: the Hartree term is the classical repulsion of
            the whole charge, and is blind to how it is polarised.
        grid : FieldGrid, optional
            Required when ``density`` is a bare array.
        structure : Structure, optional
            Required when ``density`` is a bare array.
        metadata : dict, optional

        Returns
        -------
        HartreePotential

        Raises
        ------
        ValueError
            When a bare array is passed without a grid and a structure. The
            field would otherwise have no cell to be periodic in.
        """
        from ..physics.energy import hartree_potential

        values, grid, structure = _unpack(density, grid, structure)
        payload = {"source": "poisson", "derived_from": "charge density"}
        payload.update(metadata or {})
        return cls(hartree_potential(values, grid), grid, structure,
                   metadata=payload)

    @classmethod
    def compute(cls, *args, **kwargs):
        """Alias for :meth:`from_density`, for symmetry with the other fields."""
        return cls.from_density(*args, **kwargs)

    def total_with(self, external):
        r"""
        :math:`v_{\rm H} + V_{\rm ext}`, the total local potential.

        Parameters
        ----------
        external : ExternalPotential
            Must share this field's grid.

        Returns
        -------
        HartreePotential
            Carrying the sum. The class is reused because the sum is still a
            local potential on the same mesh in the same units; the metadata
            records what it holds.

        Raises
        ------
        ValueError
            On a grid mismatch.
        """
        if tuple(external.grid.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"The external potential is on a {tuple(external.grid.shape)} "
                f"grid and the Hartree potential on {tuple(self.grid.shape)}; "
                f"they must share one mesh to be added."
            )
        metadata = dict(self.metadata)
        metadata["source"] = "hartree + external"
        return type(self)(self.data + np.asarray(external.data, dtype=float),
                          self.grid, self.structure, metadata=metadata)


def _unpack(density, grid, structure):
    """Pull ``(values, grid, structure)`` out of a field or a bare array."""
    values = getattr(density, "total", None)
    if values is None:
        values = getattr(density, "data", density)

    grid = grid if grid is not None else getattr(density, "grid", None)
    structure = (structure if structure is not None
                 else getattr(density, "structure", None))

    if grid is None or structure is None:
        raise ValueError(
            "A bare density array needs an explicit grid= and structure=; "
            "pass a ChargeDensity instead and both come with it."
        )
    return np.asarray(values, dtype=float), grid, structure
