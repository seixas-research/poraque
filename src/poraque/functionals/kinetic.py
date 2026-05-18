# -*- coding: utf-8 -*-
# file: kinetic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from .base import Functional

class ThomasFermi(Functional):
    """
    Thomas-Fermi kinetic energy functional.
    E[n] = 3/10 * (3*pi^2)^(2/3) * integral(n^(5/3))
    """
    def __init__(self):
        super().__init__("Thomas-Fermi")
        self.C_TF = 0.3 * (3 * np.pi**2)**(2/3)

    def energy(self, density, system, grid, backend):
        # n^(5/3)
        e_dens = self.C_TF * density.data**(5/3)
        return backend.integrate(e_dens, grid)

    def potential(self, density, system, grid, backend):
        # v = 5/3 * C_TF * n^(2/3)
        return (5/3) * self.C_TF * density.data**(2/3)


class VonWeizsaecker(Functional):
    """
    von Weizsäcker kinetic energy functional.
    E[n] = 1/8 * integral( |grad n|^2 / n )
    """
    def __init__(self):
        super().__init__("von Weizsäcker")

    def energy(self, density, system, grid, backend):
        # Using the form: E = -1/2 * integral( sqrt(n) * nabla^2 sqrt(n) )
        sqrt_n = np.sqrt(density.data)
        lap_sqrt_n = backend.laplacian(sqrt_n, grid)
        return -0.5 * backend.integrate(sqrt_n * lap_sqrt_n, grid)

    def potential(self, density, system, grid, backend):
        # v = -1/2 * (nabla^2 sqrt(n)) / sqrt(n)
        sqrt_n = np.sqrt(density.data)
        # Avoid division by zero
        safe_sqrt_n = np.where(sqrt_n > 1e-12, sqrt_n, 1e-12)
        lap_sqrt_n = backend.laplacian(sqrt_n, grid)
        return -0.5 * lap_sqrt_n / safe_sqrt_n
