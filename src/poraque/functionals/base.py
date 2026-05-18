# -*- coding: utf-8 -*-
# file: base.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from abc import ABC, abstractmethod

class Functional(ABC):
    """
    Abstract base class for all energy functionals.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def energy(self, density, system, grid, backend):
        """
        Calculate the energy contribution E[n].
        """
        pass

    @abstractmethod
    def potential(self, density, system, grid, backend):
        """
        Calculate the functional derivative / local potential v(r) = delta E / delta n.
        """
        pass

    def __repr__(self):
        return f"Functional(name={self.name})"
