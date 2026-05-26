# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from scipy.special import erfc

def point_charge_potential(grid, positions, charges, rc=0.1):
    """
    Compute the electrostatic potential of point charges on a grid.
    To avoid singularities, a regularized form can be used 1/r with a cutoff.
    
    For periodic systems, this would ideally be done via Ewald summation or similar,
    but here we provide a simple real-space summation for demonstration.
    """
    coords = grid.get_xyz()
    v_ext = np.zeros(grid.shape)
    
    for pos, q in zip(positions, charges):
        # Displacement vectors
        dr = coords - pos

        # Distance array across the grid
        r = np.linalg.norm(dr, axis=-1)

        # Vectorized cutoff: replaces any value in 'r' that is less than 'rc' with 'rc'
        r_regularized = np.maximum(r, rc)

        # Calculate and accumulate the potential
        v_ext -= q / r_regularized        
    return v_ext


def ewald_potential(grid, positions, charges, cell, alpha=None, r_cut=None, k_cut=None):
    """
    Compute the electrostatic potential on a grid using Ewald summation.
    
    Parameters:
    - grid: Object with get_xyz() returning an array of coordinates
    - positions: Array of point charge coordinates (N, 3)
    - charges: Array of charge magnitudes (N,)
    - cell: (3, 3) array where ROWS are the lattice vectors a, b, c
    - alpha: Ewald splitting parameter
    - r_cut: Real-space cutoff distance
    - k_cut: Reciprocal-space cutoff (maximum k-vector magnitude)
    """
    coords = grid.get_xyz()
    v_total = np.zeros(grid.shape)
    
    # --- 1. Cell Geometry & Reciprocal Space Setup ---
    # H matrix: rows are real-space lattice vectors (a, b, c)
    H = np.array(cell, dtype=float)
    V_box = np.abs(np.linalg.det(H))
    H_inv = np.linalg.inv(H)
    
    # B matrix: rows are reciprocal lattice vectors (b1, b2, b3)
    # Defined such that H @ B.T = 2 * pi * Identity
    B = 2 * np.pi * H_inv.T
    
    # --- 2. Parameter Heuristics ---
    if alpha is None:
        # Scale based on the equivalent volumetric length
        alpha = 5.0 / (V_box**(1/3.0)) 
        
    if r_cut is None:
        # The maximum safe cutoff for the minimum image convention without looping
        # over neighboring cells is half the shortest perpendicular height of the cell.
        # The heights can be found via the magnitudes of the reciprocal vectors.
        perpendicular_heights = 2 * np.pi / np.linalg.norm(B, axis=1)
        r_cut = np.min(perpendicular_heights) / 2.0
        
    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi

    # --- 3. Real-Space Summation ---
    for pos, q in zip(positions, charges):
        # Raw displacement
        dr = coords - pos
        
        # Transform to fractional (crystal) coordinates
        s = np.dot(dr, H_inv)
        
        # Apply Minimum Image Convention in fractional space
        # (Rounding to nearest integer gives the closest periodic image)
        s_mic = s - np.round(s)
        
        # Transform back to Cartesian coordinates
        dr_mic = np.dot(s_mic, H)
        
        r = np.linalg.norm(dr_mic, axis=-1)
        r_safe = np.maximum(r, 1e-12) # Prevent division by zero
        
        mask = r_safe < r_cut
        v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]

    # --- 4. Reciprocal-Space Summation ---
    # Determine the maximum integer multiplier needed along each reciprocal axis
    # to guarantee we encompass the k_cut sphere
    n_max = np.ceil(k_cut / np.linalg.norm(B, axis=1)).astype(int)
    
    # Generate meshgrid of integer indices (n_x, n_y, n_z)
    nx = np.arange(-n_max[0], n_max[0] + 1)
    ny = np.arange(-n_max[1], n_max[1] + 1)
    nz = np.arange(-n_max[2], n_max[2] + 1)
    
    # Create an array of all (nx, ny, nz) combinations
    mesh_n = np.array(np.meshgrid(nx, ny, nz, indexing='ij')).reshape(3, -1).T
    
    # Calculate all physical k-vectors: k = n @ B
    k_vectors = np.dot(mesh_n, B)
    
    for k in k_vectors:
        k_sq = np.dot(k, k)
        
        # Skip the k=0 singularity and anything outside the spherical cutoff
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue
            
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))
        grid_phase = np.exp(1j * np.dot(coords, k))
        
        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha**2)) / k_sq
        v_total += np.real(prefactor * S_k * grid_phase)
        
    # --- 5. Constant / Background Terms ---
    net_charge = np.sum(charges)
    if not np.isclose(net_charge, 0.0):
        v_bg = - (np.pi / (alpha**2 * V_box)) * net_charge
        v_total += v_bg
        
    return v_total
