# -*- coding: utf-8 -*-
# file: ofdft.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from .calculator import Poraque
from .core import Grid, System, SolverSettings
from .backends.numpy import NumpyBackend
from .functionals import ThomasFermi, Hartree, External

def run_ofdft(atoms, grid_shape=(20, 20, 20), functionals=None, mixing=0.1, max_iter=100):
    """
    Convenience function to run an OF-DFT calculation from an ASE Atoms object.
    """
    system = System.from_ase(atoms)
    grid = Grid(grid_shape, atoms.get_cell(), atoms.get_pbc())
    
    if functionals is None:
        # Default: TF + Hartree
        # Note: In a real scenario, we'd also need external potential.
        # This is just a placeholder convenience wrapper.
        functionals = [ThomasFermi(), Hartree()]
        
    settings = SolverSettings(max_iter=max_iter, mixing=mixing)
    backend = NumpyBackend()
    
    calc = Poraque(system, grid, functionals, backend=backend, settings=settings)
    return calc.calculate()
