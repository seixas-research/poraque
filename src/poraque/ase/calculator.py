# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""ASE :class:`~ase.calculators.calculator.Calculator` interface for Poraquê.

The :class:`PoraqueASE` calculator converts an ASE ``Atoms`` object into the
internal :class:`~poraque.core.System`/:class:`~poraque.core.Grid` objects,
assembles an OF-DFT (or, optionally, KS-DFT) functional stack, runs the engine,
and exposes the results as standard ASE properties (energy in eV, forces in
eV/Å).
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..calculator import Poraque
from ..core import Grid, System
from ..core.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from ..functionals import External, Hartree, LDA, TFvW
from ..potentials import build_external_potential


class PoraqueASE(Calculator):
    """
    ASE calculator backed by the Poraquê OF-DFT engine.

    Parameters
    ----------
    grid_shape : tuple of int, optional
        Number of real-space grid points ``(Nx, Ny, Nz)``.
    kinetic : Functional, optional
        Kinetic energy functional. Defaults to ``TFvW(lambda_vw=1.0)``.
    xc : Functional or None, optional
        Exchange-correlation functional. Defaults to :class:`~poraque.functionals.LDA`.
        Pass ``None`` to disable XC.
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    external_kind : str, optional
        External-potential model passed to
        :func:`~poraque.potentials.build_external_potential` (default ``"soft"``).
    external_kwargs : dict, optional
        Extra keyword arguments for the external-potential builder.
    charge : int, optional
        Net charge of the system.
    backend : str, optional
        Numerical backend name (default ``"numpy"``).
    settings : SolverSettings, optional
        Solver configuration.
    fd_step : float, optional
        Finite-difference displacement (Å) used for numerical forces.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, grid_shape=(24, 24, 24), kinetic=None, xc="lda",
                 hartree=True, external_kind="soft", external_kwargs=None,
                 charge=0, backend="numpy", settings=None, fd_step=0.01, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.grid_shape = grid_shape
        self.kinetic = kinetic
        self.xc = xc
        self.hartree = hartree
        self.external_kind = external_kind
        self.external_kwargs = external_kwargs or {}
        self.charge = charge
        self.backend_name = backend
        self.solver_settings = settings
        self.fd_step = fd_step

    def _build_functionals(self, system, grid):
        """Assemble the functional list for a given system/grid."""
        kinetic = self.kinetic if self.kinetic is not None else TFvW(lambda_vw=1.0)
        functionals = [kinetic]
        if self.hartree:
            functionals.append(Hartree())
        if self.xc is not None:
            functionals.append(LDA() if self.xc == "lda" else self.xc)
        v_ext = build_external_potential(
            grid, system, kind=self.external_kind, **self.external_kwargs
        )
        functionals.append(External(v_ext))
        return functionals

    def _single_point_energy(self, atoms):
        """Total energy (Hartree) for an ASE ``Atoms`` object."""
        system = System.from_ase(atoms, charge=self.charge)
        grid = Grid(self.grid_shape, system.cell, system.pbc)
        functionals = self._build_functionals(system, grid)
        calc = Poraque(system, grid, functionals,
                       backend=self.backend_name, settings=self.solver_settings)
        return calc.calculate()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        result = self._single_point_energy(self.atoms)
        energy_ev = result.total_energy * HARTREE_TO_EV

        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        self.results["density"] = result.density
        self.results["converged"] = result.converged

        if "forces" in properties:
            self.results["forces"] = self._numerical_forces(self.atoms)

    def _numerical_forces(self, atoms):
        """Central finite-difference forces in eV/Å."""
        forces = np.zeros((len(atoms), 3))
        delta = self.fd_step  # Å
        # Hartree/Bohr -> eV/Å
        conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM
        for a in range(len(atoms)):
            for c in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                pos = atoms.get_positions()
                pp = pos.copy()
                pm = pos.copy()
                pp[a, c] += delta
                pm[a, c] -= delta
                plus.set_positions(pp)
                minus.set_positions(pm)
                e_plus = self._single_point_energy(plus).total_energy
                e_minus = self._single_point_energy(minus).total_energy
                # delta is in Å; convert derivative to atomic units then to eV/Å.
                dEdx_au = (e_plus - e_minus) / (2 * delta / BOHR_TO_ANGSTROM)
                forces[a, c] = -dEdx_au * conv
        return forces
