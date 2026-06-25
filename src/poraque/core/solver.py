# -*- coding: utf-8 -*-
# file: solver.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

class SolverSettings:
    """
    Generic solver configuration and utilities.
    """
    def __init__(self, max_iter=100, tolerance=1e-6, mixing=0.5,
                 algorithm='scf', max_line_search=20, cg_restart=20):
        """
        Initialize SolverSettings.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tolerance : float
            Convergence tolerance (energy or density residual).
        mixing : float
            Mixing parameter / initial step size (linear mixing, descent step).
        algorithm : str
            Optimization algorithm ('scf', 'minimization', etc.).
        max_line_search : int
            Maximum number of backtracking halvings per OF-DFT step.
        cg_restart : int
            Restart the conjugate-gradient direction to steepest descent every
            ``cg_restart`` iterations.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mixing = mixing
        self.algorithm = algorithm
        self.max_line_search = max_line_search
        self.cg_restart = cg_restart

    def __repr__(self):
        return f"SolverSettings(max_iter={self.max_iter}, tolerance={self.tolerance})"
