# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np

def point_charge_potential(grid, positions, charges, rc=0.5):
    """
    Compute the electrostatic potential of point charges on a grid.
    To avoid singularities, a regularized form can be used: 1 / sqrt(r^2 + rc^2)
    or just 1/r with a cutoff.
    
    For periodic systems, this would ideally be done via Ewald summation or similar,
    but here we provide a simple real-space summation for demonstration.
    """
    coords = grid.get_xyz()
    v_ext = np.zeros(grid.shape)
    
    for pos, q in zip(positions, charges):
        # Displacement vectors
        dr = coords - pos
        # Distance
        r = np.linalg.norm(dr, axis=-1)
        # Regularized potential: v = -q / sqrt(r^2 + rc^2)
        v_ext -= q / np.sqrt(r**2 + rc**2)
        
    return v_ext
