# -*- coding: utf-8 -*-
# file: engine.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from .core import Density, Result

class OFDFTEngine:
    """
    Engine for Orbital-Free DFT calculations.
    Performs direct minimization of the total energy.
    """
    def __init__(self, system, grid, functionals, backend, settings):
        self.system = system
        self.grid = grid
        self.functionals = functionals
        self.backend = backend
        self.settings = settings

    def compute_total_energy(self, density):
        """
        Compute total energy and its components.
        """
        total_e = 0.0
        components = {}
        for func in self.functionals:
            e = func.energy(density, self.system, self.grid, self.backend)
            components[func.name] = e
            total_e += e
        return total_e, components

    def compute_effective_potential(self, density):
        """
        Compute the effective potential v_eff = sum(v_i).
        """
        v_eff = np.zeros(self.grid.shape)
        for func in self.functionals:
            v_eff += func.potential(density, self.system, self.grid, self.backend)
        return v_eff

    def run(self, initial_density):
        """
        Run the minimization loop.
        """
        density = initial_density
        # Ensure initial normalization
        density.normalize(self.system.electrons)
        
        # Optimization variable: w = sqrt(n)
        w = np.sqrt(density.data)
        
        history = {'energy': [], 'residual': []}
        converged = False
        
        for i in range(self.settings.max_iter):
            # 1. Update density from w
            density.data = w**2
            
            # 2. Compute energy and potential
            energy, components = self.compute_total_energy(density)
            v_eff = self.compute_effective_potential(density)
            
            # 3. Gradient w.r.t w: g = 2 * w * v_eff
            # Note: We should subtract the chemical potential to stay on the N-surface
            # mu = <w | v_eff | w> / <w | w>
            mu = self.backend.integrate(w**2 * v_eff, self.grid) / self.system.electrons
            g = 2 * w * (v_eff - mu)
            
            # 4. Check convergence
            residual = self.backend.norm(g)
            history['energy'].append(energy)
            history['residual'].append(residual)
            
            if residual < self.settings.tolerance:
                converged = True
                break
                
            # 5. Simple steepest descent step
            w -= self.settings.mixing * g
            
            # 6. Re-normalize w
            norm_w = np.sqrt(self.backend.integrate(w**2, self.grid))
            w *= np.sqrt(self.system.electrons) / norm_w

        return Result(
            energy=energy,
            components=components,
            density=density,
            converged=converged,
            iterations=i+1,
            history=history
        )
