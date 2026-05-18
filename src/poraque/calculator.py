# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from .core import Grid, System, Density, SolverSettings
from .backends.numpy import NumpyBackend
from .engine import OFDFTEngine

class Poraque:
    """
    High-level API for Poraquê calculations.
    """
    def __init__(self, system, grid, functionals, backend='numpy', settings=None):
        self.system = system
        self.grid = grid
        self.functionals = functionals
        
        if backend == 'numpy':
            self.backend = NumpyBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        self.settings = settings if settings is not None else SolverSettings()

    def calculate(self, initial_density=None):
        """
        Run the calculation.
        """
        if initial_density is None:
            # Uniform density guess
            n_init = np.ones(self.grid.shape) * (self.system.electrons / self.grid.volume)
            initial_density = Density(self.grid, n_init)
            
        engine = OFDFTEngine(
            system=self.system,
            grid=self.grid,
            functionals=self.functionals,
            backend=self.backend,
            settings=self.settings
        )
        
        return engine.run(initial_density)

import numpy as np # Needed for the uniform density guess
