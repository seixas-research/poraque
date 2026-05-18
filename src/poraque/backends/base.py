# -*- coding: utf-8 -*-
# file: base.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from abc import ABC, abstractmethod

class Backend(ABC):
    """
    Abstract base class for numerical backends.
    """
    
    @abstractmethod
    def integrate(self, field, grid):
        """Integrate a scalar field over the grid."""
        pass

    @abstractmethod
    def gradient(self, field, grid):
        """Compute the gradient of a scalar field."""
        pass

    @abstractmethod
    def laplacian(self, field, grid):
        """Compute the Laplacian of a scalar field."""
        pass

    @abstractmethod
    def poisson(self, charge_density, grid):
        """Solve the Poisson equation: nabla^2 V = -4 * pi * n."""
        pass

    @abstractmethod
    def fft(self, field):
        """Perform a Fast Fourier Transform."""
        pass

    @abstractmethod
    def ifft(self, field):
        """Perform an Inverse Fast Fourier Transform."""
        pass

    @abstractmethod
    def dot(self, a, b):
        """Compute the dot product of two arrays."""
        pass

    @abstractmethod
    def norm(self, a):
        """Compute the norm of an array."""
        pass
