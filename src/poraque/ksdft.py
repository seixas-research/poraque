# -*- coding: utf-8 -*-
# file: ksdft.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""High-level Kohn-Sham DFT driver."""

from .backends.numpy import NumpyBackend
from .core import Grid, SolverSettings, System
from .engine import KSDFTEngine
from .functionals import LDA
from .potentials import build_external_potential


def run_ksdft(atoms, grid_shape=(32, 32, 32), charge=0, xc=None, hartree=True,
              external_kind="soft", external_kwargs=None, mixing=0.3,
              max_iter=100, tolerance=1e-6):
    """
    Run a Kohn-Sham DFT calculation from an ASE ``Atoms`` object.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure (Ångström units).
    grid_shape : tuple of int, optional
        Real-space grid shape.
    charge : int, optional
        Net charge of the system.
    xc : Functional, optional
        Exchange-correlation functional (default :class:`~poraque.functionals.LDA`).
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    external_kind : str, optional
        External-potential model.
    external_kwargs : dict, optional
        Extra keyword arguments for the external-potential builder.
    mixing : float, optional
        Linear density-mixing parameter.
    max_iter : int, optional
        Maximum number of SCF iterations.
    tolerance : float, optional
        Convergence tolerance on the density residual.

    Returns
    -------
    Result
        The converged KS-DFT result (with orbitals in ``result.state``).
    """
    system = System.from_ase(atoms, charge=charge)
    grid = Grid(grid_shape, system.cell, system.pbc)
    v_ext = build_external_potential(
        grid, system, kind=external_kind, **(external_kwargs or {})
    )
    settings = SolverSettings(max_iter=max_iter, mixing=mixing, tolerance=tolerance)
    engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(), settings,
                         xc=xc if xc is not None else LDA(), hartree=hartree)
    return engine.run()
