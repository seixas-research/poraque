# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Low-level method drivers.

These thin, ``System``/``Grid``-level drivers wrap the engines and own the
boilerplate (initial-density guess, backend resolution). They are mode-specific
on purpose; the *unified*, user-facing entry point is the ASE calculator
:class:`poraque.ase.Poraque`, which selects between them via ``mode='of'`` and
``mode='ks'``.
"""

from .core import Grid, System, Density, SolverSettings
from .backends.base import Backend
from .backends.numpy import NumpyBackend
from .engine import OFDFTEngine, KSDFTEngine
import numpy as np


def _resolve_backend(backend):
    """Return a Backend instance from a name or a pre-built instance."""
    if isinstance(backend, Backend):
        return backend
    if backend == 'numpy':
        return NumpyBackend()
    raise ValueError(f"Unknown backend: {backend!r}")


class OFDFTCalculator:
    """
    Orbital-free DFT driver over a fixed functional stack.

    Parameters
    ----------
    system : System
        Atomic structure.
    grid : Grid
        Real-space grid.
    functionals : list of Functional
        The energy functionals to minimize (kinetic + Hartree + XC + external).
    backend : str or Backend, optional
        Numerical backend (default ``"numpy"``).
    settings : SolverSettings, optional
        Minimizer configuration.
    """
    def __init__(self, system, grid, functionals, backend='numpy', settings=None):
        self.system = system
        self.grid = grid
        self.functionals = functionals
        self.backend = _resolve_backend(backend)
        self.settings = settings if settings is not None else SolverSettings()

    def calculate(self, initial_density=None):
        """Run the OF-DFT minimization and return a :class:`~poraque.core.Result`."""
        if initial_density is None:
            # Uniform density guess.
            n_init = np.ones(self.grid.shape) * (self.system.electrons / self.grid.volume)
            initial_density = Density(self.grid, n_init)

        engine = OFDFTEngine(
            system=self.system,
            grid=self.grid,
            functionals=self.functionals,
            backend=self.backend,
            settings=self.settings,
        )
        return engine.run(initial_density)


class KSDFTCalculator:
    """
    Kohn-Sham DFT driver with optional Brillouin-zone sampling.

    Parameters
    ----------
    system : System
        Atomic structure.
    grid : Grid
        Real-space grid.
    v_ext : numpy.ndarray
        External (electron-nucleus / pseudopotential) potential on the grid.
    xc : Functional, optional
        Exchange-correlation functional (default :class:`~poraque.functionals.LDA`).
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    kpoints : array_like, optional
        ``(Nk, 3)`` fractional k-points (default: the Gamma point).
    kweights : array_like, optional
        k-point weights (default uniform).
    backend : str or Backend, optional
        Numerical backend (default ``"numpy"``).
    settings : SolverSettings, optional
        SCF configuration.
    """
    def __init__(self, system, grid, v_ext, xc=None, hartree=True,
                 kpoints=None, kweights=None, backend='numpy', settings=None):
        self.system = system
        self.grid = grid
        self.v_ext = v_ext
        self.xc = xc
        self.hartree = hartree
        self.kpoints = kpoints
        self.kweights = kweights
        self.backend = _resolve_backend(backend)
        self.settings = settings if settings is not None else SolverSettings()

    def calculate(self, initial_density=None):
        """Run the KS-DFT SCF loop and return a :class:`~poraque.core.Result`."""
        engine = KSDFTEngine(
            self.system, self.grid, self.v_ext, self.backend, self.settings,
            xc=self.xc, hartree=self.hartree,
            kpoints=self.kpoints, kweights=self.kweights,
        )
        return engine.run(initial_density)
