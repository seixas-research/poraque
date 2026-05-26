# -*- coding: utf-8 -*-
# file: results.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

class Result:
    """
    Represents the output of a calculation.
    """
    def __init__(self, energy, components, density, state=None, converged=False, iterations=0, history=None):
        """
        Initialize the Result.

        Parameters
        ----------
        energy : float
            Total energy.
        components : dict
            Energy components (kinetic, Hartree, XC, external, etc.).
        density : Density
            Converged electron density.
        state : State, optional
            Converged electronic state (orbitals, occupations).
        converged : bool
            Whether the calculation converged.
        iterations : int
            Number of iterations performed.
        history : dict, optional
            Convergence history (energy, residuals, etc.).
        """
        self.total_energy = energy
        self.energy_components = components
        self.density = density
        self.state = state
        self.converged = converged
        self.iterations = iterations
        self.history = history if history is not None else {}

    def __repr__(self):
        return (f"Result(energy={self.total_energy:.6f}, "
                f"converged={self.converged}, "
                f"iterations={self.iterations})")
