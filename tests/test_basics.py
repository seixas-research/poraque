# -*- coding: utf-8 -*-

import unittest
import numpy as np
from poraque.core import Grid, System, Density, SolverSettings
from poraque.backends.numpy import NumpyBackend
from poraque.functionals import ThomasFermi, Hartree, External
from poraque.engine import OFDFTEngine
from poraque.calculator import Poraque

class TestBasics(unittest.TestCase):
    def test_grid_integration(self):
        grid = Grid(shape=(10, 10, 10), cell=np.eye(3) * 10.0)
        field = np.ones(grid.shape)
        integral = grid.integrate(field)
        self.assertAlmostEqual(integral, 1000.0)

    def test_thomas_fermi_energy(self):
        grid = Grid(shape=(10, 10, 10), cell=np.eye(3) * 10.0)
        system = System(positions=[[5, 5, 5]], atomic_numbers=[1], cell=np.eye(3)*10.0)
        density = Density(grid, np.ones(grid.shape) * (1.0 / 1000.0))
        
        tf = ThomasFermi()
        backend = NumpyBackend()
        
        energy = tf.energy(density, system, grid, backend)
        expected = tf.C_TF * (0.001**(5/3)) * 1000.0
        self.assertAlmostEqual(energy, expected)

    def test_engine_smoke(self):
        grid = Grid(shape=(8, 8, 8), cell=np.eye(3) * 5.0)
        system = System(positions=[[2.5, 2.5, 2.5]], atomic_numbers=[1], cell=np.eye(3)*5.0)
        
        # Very simple setup
        functionals = [ThomasFermi()]
        backend = NumpyBackend()
        settings = SolverSettings(max_iter=5, mixing=0.1)
        
        calc = Poraque(system, grid, functionals, backend='numpy', settings=settings)
        result = calc.calculate()
        
        self.assertGreater(result.iterations, 0)
        self.assertEqual(len(result.history['energy']), result.iterations)

if __name__ == '__main__':
    unittest.main()
