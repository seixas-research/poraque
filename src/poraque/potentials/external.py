# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from scipy.special import erfc
import warnings

def point_charge_potential(grid, positions, charges, rc=1e-6):
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

import numpy as np
from scipy.special import erfc
import warnings

def ewald_summation(grid, positions, charges, cell, alpha=None, r_cut=None, k_cut=None, mic=False):
    """
    Compute the electrostatic potential on a grid using Ewald summation 
    for an arbitrary periodic unit cell.
    
    Parameters:
    - grid: Object with get_xyz() returning an array of coordinates (..., 3)
    - positions: Array of point charge coordinates (N, 3)
    - charges: Array of charge magnitudes (N,)
    - cell: (3, 3) array where ROWS are the lattice vectors a1, a2, a3
    - alpha: Ewald splitting parameter (controls width of Gaussian clouds)
    - r_cut: Real-space cutoff distance
    - k_cut: Reciprocal-space cutoff (maximum k-vector magnitude)
    - mic: Boolean. If True, uses the fast fractional Minimum Image Convention 
           (assumes r_cut <= min_height/2 and cell is not severely skewed).
           If False (default), loops over neighboring cells in a bounding box 
           for guaranteed accuracy at any cutoff or cell skewness.
    """
    coords = grid.get_xyz()
    v_total = np.zeros(grid.shape)
    
    # --- 1. Cell Geometry & Reciprocal Space Setup ---
    H = np.array(cell, dtype=float)
    V_box = np.abs(np.linalg.det(H))
    H_inv = np.linalg.inv(H)
    
    # B matrix: rows are reciprocal lattice vectors
    B = 2 * np.pi * H_inv.T
    
    # Perpendicular heights of the real-space cell
    perpendicular_heights = 2 * np.pi / np.linalg.norm(B, axis=1)
    
    # --- 2. Parameter Heuristics ---
    if alpha is None:
        alpha = 5.0 / (V_box**(1/3.0)) 
        
    if r_cut is None:
        r_cut = np.min(perpendicular_heights) / 2.0
        
    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi

    # Safety check for the fast MIC method
    if mic and r_cut > np.min(perpendicular_heights) / 2.0 + 1e-6:
        warnings.warn("Warning: r_cut is larger than half the shortest cell height. "
                      "The MIC method (mic=True) may miss interacting periodic images. "
                      "Consider using mic=False.")

    # --- 3. Real-Space Summation ---
    if mic:
        # FAST METHOD: Fractional Rounding Minimum Image Convention
        for pos, q in zip(positions, charges):
            dr = coords - pos
            # Fractional coordinates -> Round to nearest -> Cartesian
            s = np.dot(dr, H_inv)
            s_mic = s - np.round(s)
            dr_mic = np.dot(s_mic, H)
            
            r = np.linalg.norm(dr_mic, axis=-1)
            r_safe = np.maximum(r, 1e-12)
            
            mask = r_safe < r_cut
            v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]
            
    else:
        # ROBUST METHOD: Bounding Box over neighboring periodic cells
        n_max_real = np.ceil(r_cut / perpendicular_heights).astype(int)
        
        nx_r = np.arange(-n_max_real[0], n_max_real[0] + 1)
        ny_r = np.arange(-n_max_real[1], n_max_real[1] + 1)
        nz_r = np.arange(-n_max_real[2], n_max_real[2] + 1)
        mesh_n_r = np.array(np.meshgrid(nx_r, ny_r, nz_r, indexing='ij')).reshape(3, -1).T
        
        shift_vectors = np.dot(mesh_n_r, H)
        
        for pos, q in zip(positions, charges):
            for shift in shift_vectors:
                dr = coords - (pos + shift)
                
                r = np.linalg.norm(dr, axis=-1)
                r_safe = np.maximum(r, 1e-12)
                
                mask = r_safe < r_cut
                v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]

    # --- 4. Reciprocal-Space Summation ---
    n_max_recip = np.ceil(k_cut / np.linalg.norm(B, axis=1)).astype(int)
    
    nx_k = np.arange(-n_max_recip[0], n_max_recip[0] + 1)
    ny_k = np.arange(-n_max_recip[1], n_max_recip[1] + 1)
    nz_k = np.arange(-n_max_recip[2], n_max_recip[2] + 1)
    mesh_n_k = np.array(np.meshgrid(nx_k, ny_k, nz_k, indexing='ij')).reshape(3, -1).T
    
    k_vectors = np.dot(mesh_n_k, B)
    
    for k in k_vectors:
        k_sq = np.dot(k, k)
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue
            
        # Structure factor
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))
        
        # Grid evaluation
        grid_phase = np.exp(1j * np.dot(coords, k))
        
        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha**2)) / k_sq
        v_total += np.real(prefactor * S_k * grid_phase)
        
    # --- 5. Constant / Background Terms ---
    net_charge = np.sum(charges)
    if not np.isclose(net_charge, 0.0):
        v_bg = - (np.pi / (alpha**2 * V_box)) * net_charge
        v_total += v_bg
        
    return v_total