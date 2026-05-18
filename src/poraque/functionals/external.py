# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from .base import Functional

class External(Functional):
    """
    Functional for a static external potential.
    E[n] = integral( n(r) * v_ext(r) )
    """
    def __init__(self, v_ext):
        """
        Parameters
        ----------
        v_ext : array_like
            The external potential on the grid.
        """
        super().__init__("External")
        self.v_ext = v_ext

    def energy(self, density, system, grid, backend):
        return backend.integrate(density.data * self.v_ext, grid)

    def potential(self, density, system, grid, backend):
        return self.v_ext
