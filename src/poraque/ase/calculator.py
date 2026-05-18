# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from ase.calculators.calculator import Calculator, all_changes
from ..calculator import Poraque
from ..core import System, Grid, Density

class PoraqueASE(Calculator):
    """
    ASE Calculator for Poraquê.
    """
    implemented_properties = ['energy', 'forces'] # forces not yet implemented in engine

    def __init__(self, grid_shape, functionals, backend='numpy', settings=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.grid_shape = grid_shape
        self.functionals = functionals
        self.backend_name = backend
        self.solver_settings = settings

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Convert ASE atoms to Poraquê System
        system = System.from_ase(self.atoms)
        
        # 2. Create Grid
        grid = Grid(self.grid_shape, self.atoms.get_cell(), self.atoms.get_pbc())
        
        # 3. Setup Poraquê calculation
        calc = Poraque(
            system=system,
            grid=grid,
            functionals=self.functionals,
            backend=self.backend_name,
            settings=self.solver_settings
        )
        
        # 4. Run
        result = calc.calculate()
        
        # 5. Store results
        self.results['energy'] = result.total_energy
        # self.results['forces'] = ... # To be implemented
