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

import logging
import warnings

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..calculator import KSDFTCalculator, OFDFTCalculator
from ..core import Grid, System, reporting
from ..core.kpoints import gamma_only, monkhorst_pack_kpoints
from ..core.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from ..functionals import External, Hartree, LDA, TFvW
from ..potentials import build_external_potential
from ..pseudopotentials import build_pseudopotential_potential

logger = logging.getLogger(__name__)

# Accepted spellings of the (mandatory) plane-wave basis for KS-DFT.
_PW_ALIASES = {"pw", "planewave", "plane-wave", "plane_wave"}


class Poraque(Calculator):
    """
    Unified ASE calculator backed by the Poraquê DFT engines.

    Parameters
    ----------
    mode : {"ks", "of"}, optional
        Electronic-structure method: Kohn-Sham (``"ks"``, default) or
        orbital-free (``"of"``) DFT.
    basis : {"pw"}, optional
        Single-particle basis set. Poraquê's Kohn-Sham engine is solved in the
        **plane-wave basis** dual to the real-space grid (the kinetic operator is
        applied spectrally as ``½|G + k|²``), so ``"pw"`` is the only accepted
        value and is enforced for ``mode='ks'``.
    grid_shape : tuple of int, optional
        Real-space grid points ``(Nx, Ny, Nz)``. Used only when ``ecut`` is not
        set; when both are given, the explicit ``grid_shape`` is overridden by
        the automatically optimized ``ecut``-based grid (with a logged warning).
    ecut : float, optional
        Plane-wave kinetic-energy cutoff (Hartree). When given, the real-space
        grid is generated automatically via
        :meth:`~poraque.core.Grid.from_ecut`, sized to resolve the cutoff.
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

    def __init__(self, mode="ks", basis="pw", grid_shape=None, ecut=None,
                 kpts=None, pseudopotentials=None, pseudo_functional=None,
                 xc="lda", hartree=True,
                 kinetic=None, external_kind="soft", external_kwargs=None,
                 charge=0, backend="numpy", settings=None, fd_step=0.01,
                 verbose=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        mode = mode.lower()
        if mode not in ("ks", "of"):
            raise ValueError(f"mode must be 'ks' or 'of', got {mode!r}")
        self.mode = mode

        # Plane waves are mandatory for KS-DFT (the kinetic operator is applied
        # spectrally in reciprocal space).
        basis_norm = str(basis).lower()
        if mode == "ks" and basis_norm not in _PW_ALIASES:
            raise ValueError(
                f"KS-DFT requires the plane-wave basis (basis='pw'); got {basis!r}."
            )
        self.basis = "pw" if basis_norm in _PW_ALIASES else basis_norm

        # Track whether the grid was explicitly requested so we can warn when an
        # ``ecut`` silently overrides it.
        self._grid_shape_explicit = grid_shape is not None
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else (32, 32, 32)
        self.ecut = ecut
        self.kpts = kpts
        self.pseudopotentials = pseudopotentials
        self.pseudo_functional = pseudo_functional
        self.xc = xc
        self.hartree = hartree
        self.kinetic = kinetic
        self.external_kind = external_kind
        self.external_kwargs = external_kwargs or {}
        self.charge = charge
        self.backend_name = backend
        self.solver_settings = settings
        self.fd_step = fd_step
        self.verbose = verbose

    # -- construction helpers -------------------------------------------------

    def _build_grid(self, system):
        """
        Build the real-space / plane-wave grid.

        When a plane-wave cutoff ``ecut`` is given, the grid is generated
        automatically and sized to resolve that cutoff
        (:meth:`~poraque.core.Grid.from_ecut`). If the user *also* supplied an
        explicit ``grid_shape``, the automatically optimized grid takes
        precedence and a warning is logged.
        """
        if self.ecut is not None:
            grid = Grid.from_ecut(system.cell, self.ecut, pbc=system.pbc)
            if self._grid_shape_explicit and tuple(self.grid_shape) != grid.shape:
                msg = (
                    f"Ignoring the manually supplied grid_shape={self.grid_shape}: "
                    f"the plane-wave cutoff ecut={self.ecut} Ha requires "
                    f"grid_shape={grid.shape} to be resolved; using the "
                    f"automatically optimized grid."
                )
                warnings.warn(msg, stacklevel=2)
                logger.warning(msg)
            grid.basis = "plane waves"
            return grid
        grid = Grid(self.grid_shape, system.cell, system.pbc)
        grid.basis = "plane waves"
        return grid

    def _pseudo_functional(self):
        """Resolve the functional used to pick UPF pseudopotentials."""
        if self.pseudo_functional is not None:
            return self.pseudo_functional
        # Infer from the XC choice: anything PBE-flavored -> PBE, else LDA.
        xc = self.xc
        name = ""
        if isinstance(xc, str):
            name = xc
        elif xc is not None:
            name = getattr(xc, "name", "") or type(xc).__name__
        return "PBE" if "pbe" in name.lower() else "LDA"

    def _build_external(self, system, grid):
        """
        Build the external potential and finalize the electron count.

        With pseudopotentials, the explicit electron count is reduced to the
        sum of valence charges; otherwise the all-electron nuclear potential is
        used and the electron count is left untouched.
        """
        if self.pseudopotentials is not None:
            v_ext, n_valence = build_pseudopotential_potential(
                grid, system, self.pseudopotentials,
                functional=self._pseudo_functional(),
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

    def _single_point(self, atoms, verbose=False):
        """Run a single-point calculation and return the :class:`Result`."""
        system = System.from_ase(atoms, charge=self.charge)
        grid = self._build_grid(system)
        v_ext = self._build_external(system, grid)

        if verbose:
            # Report the automatically generated grid and the material structure
            # (the electron count is final only after the external potential is
            # built, hence printing the system here).
            print(reporting.format_grid(grid))
            print(reporting.format_system(system))

        if self.mode == "of":
            functionals = self._ofdft_functionals(system, grid, v_ext)
            driver = OFDFTCalculator(system, grid, functionals,
                                     backend=self.backend_name,
                                     settings=self.solver_settings,
                                     verbose=verbose)
            return driver.calculate()

        kpoints, kweights = self._build_kpoints()
        driver = KSDFTCalculator(system, grid, v_ext,
                                 xc=self._xc_functional(), hartree=self.hartree,
                                 kpoints=kpoints, kweights=kweights,
                                 basis=self.basis,
                                 backend=self.backend_name,
                                 settings=self.solver_settings,
                                 verbose=verbose)
        return driver.calculate()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        result = self._single_point(self.atoms, verbose=self.verbose)
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
