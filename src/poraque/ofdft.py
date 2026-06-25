# -*- coding: utf-8 -*-
# file: ofdft.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

from .calculator import Poraque
from .core import Grid, System, SolverSettings
from .functionals import External, Hartree, LDA, TFvW
from .potentials import build_external_potential


def run_ofdft(atoms, grid_shape=(24, 24, 24), functionals=None, charge=0,
              mixing=0.1, max_iter=100, tolerance=1e-6, external_kind="soft",
              external_kwargs=None):
    """
    Convenience driver to run an OF-DFT calculation from an ASE ``Atoms`` object.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure (Ångström units).
    grid_shape : tuple of int, optional
        Real-space grid shape.
    functionals : list of Functional, optional
        Explicit functional stack. If ``None``, a default
        ``TFvW + Hartree + LDA + External`` stack is built automatically.
    charge : int, optional
        Net charge of the system.
    mixing : float, optional
        Initial descent step size.
    max_iter : int, optional
        Maximum number of minimization iterations.
    tolerance : float, optional
        Convergence tolerance on the projected-gradient residual.
    external_kind : str, optional
        External-potential model (see
        :func:`~poraque.potentials.build_external_potential`).
    external_kwargs : dict, optional
        Extra keyword arguments for the external-potential builder.

    Returns
    -------
    Result
        The converged OF-DFT result.
    """
    system = System.from_ase(atoms, charge=charge)
    grid = Grid(grid_shape, system.cell, system.pbc)

    if functionals is None:
        v_ext = build_external_potential(
            grid, system, kind=external_kind, **(external_kwargs or {})
        )
        functionals = [TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)]

    settings = SolverSettings(max_iter=max_iter, mixing=mixing, tolerance=tolerance)
    calc = Poraque(system, grid, functionals, settings=settings)
    return calc.calculate()
