# -*- coding: utf-8 -*-
# file: hartree.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from .base import Functional

class Hartree(Functional):
    """
    Hartree energy functional.
    E[n] = 1/2 * integral( n(r) * V_H(r) )
    """
    def __init__(self):
        super().__init__("Hartree")

    def energy(self, density, system, grid, backend):
        v_h = backend.poisson(density.data, grid)
        return 0.5 * backend.integrate(density.data * v_h, grid)

    def potential(self, density, system, grid, backend):
        return backend.poisson(density.data, grid)
