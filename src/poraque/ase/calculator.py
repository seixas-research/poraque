# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Unified ASE :class:`~ase.calculators.calculator.Calculator` for Poraquê.

:class:`Poraque` is the single user-facing calculator. It converts an ASE
``Atoms`` object into the internal :class:`~poraque.core.System` /
:class:`~poraque.core.Grid` objects and dispatches to either the orbital-free
(``mode='of'``) or Kohn-Sham (``mode='ks'``) engine, exposing the results as
standard ASE properties (energy in eV, forces in eV/Å). The choice of method,
the real-space/plane-wave grid, Brillouin-zone sampling, and pseudopotentials
are all configured through the constructor.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..calculator import KSDFTCalculator, OFDFTCalculator
from ..core import Grid, System
from ..core.kpoints import gamma_only, monkhorst_pack_kpoints
from ..core.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from ..functionals import External, Hartree, LDA, TFvW
from ..potentials import build_external_potential
from ..pseudopotentials import build_pseudopotential_potential


class Poraque(Calculator):
    """
    Unified ASE calculator backed by the Poraquê DFT engines.

    Parameters
    ----------
    mode : {"ks", "of"}, optional
        Electronic-structure method: Kohn-Sham (``"ks"``, default) or
        orbital-free (``"of"``) DFT.
    grid_shape : tuple of int, optional
        Real-space grid points ``(Nx, Ny, Nz)``. Ignored when ``ecut`` is set.
    ecut : float, optional
        Plane-wave kinetic-energy cutoff (Hartree). When given, the grid density
        is chosen automatically via :meth:`~poraque.core.Grid.from_ecut`.
    kpts : tuple of int or array_like, optional
        Brillouin-zone sampling. A ``(n1, n2, n3)`` tuple builds a
        Monkhorst-Pack grid (via :mod:`ase.dft.kpoints`); an explicit
        ``(Nk, 3)`` array is used as fractional k-points. ``None`` (default) is
        the Gamma point. Only used in ``mode='ks'``.
    pseudopotentials : "auto" or dict or LocalPseudopotential, optional
        Pseudopotential prescription. When given, only valence electrons are
        treated explicitly and the external potential is built from the local
        pseudopotentials (see :mod:`poraque.pseudopotentials`). When ``None``
        (default), an all-electron regularized nuclear potential is used.
    xc : Functional or "lda" or None, optional
        Exchange-correlation functional (default :class:`~poraque.functionals.LDA`).
        ``None`` disables XC.
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    kinetic : Functional, optional
        OF-DFT kinetic functional (default ``TFvW(lambda_vw=1.0)``).
    external_kind : str, optional
        All-electron external-potential model used when ``pseudopotentials`` is
        ``None`` (default ``"soft"``).
    external_kwargs : dict, optional
        Extra keyword arguments for the external-potential builder.
    charge : int, optional
        Net charge of the system.
    backend : str, optional
        Numerical backend name (default ``"numpy"``).
    settings : SolverSettings, optional
        Solver configuration.
    fd_step : float, optional
        Finite-difference displacement (Å) for numerical forces.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, mode="ks", grid_shape=(32, 32, 32), ecut=None,
                 kpts=None, pseudopotentials=None, xc="lda", hartree=True,
                 kinetic=None, external_kind="soft", external_kwargs=None,
                 charge=0, backend="numpy", settings=None, fd_step=0.01, **kwargs):
        Calculator.__init__(self, **kwargs)
        mode = mode.lower()
        if mode not in ("ks", "of"):
            raise ValueError(f"mode must be 'ks' or 'of', got {mode!r}")
        self.mode = mode
        self.grid_shape = grid_shape
        self.ecut = ecut
        self.kpts = kpts
        self.pseudopotentials = pseudopotentials
        self.xc = xc
        self.hartree = hartree
        self.kinetic = kinetic
        self.external_kind = external_kind
        self.external_kwargs = external_kwargs or {}
        self.charge = charge
        self.backend_name = backend
        self.solver_settings = settings
        self.fd_step = fd_step

    # -- construction helpers -------------------------------------------------

    def _build_grid(self, system):
        """Build the real-space grid from ``ecut`` or an explicit shape."""
        if self.ecut is not None:
            return Grid.from_ecut(system.cell, self.ecut, pbc=system.pbc)
        return Grid(self.grid_shape, system.cell, system.pbc)

    def _build_external(self, system, grid):
        """
        Build the external potential and finalize the electron count.

        With pseudopotentials, the explicit electron count is reduced to the
        sum of valence charges; otherwise the all-electron nuclear potential is
        used and the electron count is left untouched.
        """
        if self.pseudopotentials is not None:
            v_ext, n_valence = build_pseudopotential_potential(
                grid, system, self.pseudopotentials
            )
            system.electrons = int(round(n_valence)) - int(self.charge)
            return v_ext
        return build_external_potential(
            grid, system, kind=self.external_kind, **self.external_kwargs
        )

    def _build_kpoints(self):
        """Resolve the ``kpts`` argument into (kpoints, weights)."""
        if self.kpts is None:
            return gamma_only()
        kpts = np.asarray(self.kpts)
        if kpts.ndim == 1:  # a Monkhorst-Pack size, e.g. (4, 4, 4)
            return monkhorst_pack_kpoints(tuple(int(k) for k in kpts))
        weights = np.full(len(kpts), 1.0 / len(kpts))  # explicit k-point list
        return kpts, weights

    def _ofdft_functionals(self, system, grid, v_ext):
        """Assemble the OF-DFT functional stack."""
        kinetic = self.kinetic if self.kinetic is not None else TFvW(lambda_vw=1.0)
        functionals = [kinetic]
        if self.hartree:
            functionals.append(Hartree())
        if self.xc is not None:
            functionals.append(LDA() if self.xc == "lda" else self.xc)
        functionals.append(External(v_ext))
        return functionals

    def _xc_functional(self):
        """Resolve the XC functional argument for KS-DFT."""
        if self.xc is None:
            return None
        return LDA() if self.xc == "lda" else self.xc

    # -- core evaluation ------------------------------------------------------

    def _single_point(self, atoms):
        """Run a single-point calculation and return the :class:`Result`."""
        system = System.from_ase(atoms, charge=self.charge)
        grid = self._build_grid(system)
        v_ext = self._build_external(system, grid)

        if self.mode == "of":
            functionals = self._ofdft_functionals(system, grid, v_ext)
            driver = OFDFTCalculator(system, grid, functionals,
                                     backend=self.backend_name,
                                     settings=self.solver_settings)
            return driver.calculate()

        kpoints, kweights = self._build_kpoints()
        driver = KSDFTCalculator(system, grid, v_ext,
                                 xc=self._xc_functional(), hartree=self.hartree,
                                 kpoints=kpoints, kweights=kweights,
                                 backend=self.backend_name,
                                 settings=self.solver_settings)
        return driver.calculate()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        result = self._single_point(self.atoms)
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
        conv = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # Hartree/Bohr -> eV/Å
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
                e_plus = self._single_point(plus).total_energy
                e_minus = self._single_point(minus).total_energy
                dEdx_au = (e_plus - e_minus) / (2 * delta / BOHR_TO_ANGSTROM)
                forces[a, c] = -dEdx_au * conv
        return forces
