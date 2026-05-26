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


def ewald_potential(grid, positions, charges, L, alpha=None, r_cut=None, k_cut=None):
    """
    Compute the electrostatic potential on a grid using Ewald summation 
    for a cubic periodic box.
    
    Parameters:
    - grid: Object with get_xyz() returning an (Nx, Ny, Nz, 3) array of coordinates
    - positions: Array of point charge coordinates (N, 3)
    - charges: Array of charge magnitudes (N,)
    - L: Length of the cubic box (scalar)
    - alpha: Ewald splitting parameter (controls width of Gaussian clouds)
    - r_cut: Real-space cutoff distance
    - k_cut: Reciprocal-space cutoff (maximum k-vector magnitude)
    """
    coords = grid.get_xyz()
    v_total = np.zeros(grid.shape)
    V_box = L**3
    
    # --- 1. Parameter Setup ---
    # Heuristics if parameters are not provided
    if alpha is None:
        alpha = 5.0 / L  
    if r_cut is None:
        r_cut = L / 2.0  # Minimum image convention max distance
    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi # Standard k-space convergence heuristic
        
    # --- 2. Real-Space Summation ---
    # In a full MD code, we loop over adjacent periodic cells (n_x, n_y, n_z).
    # For a minimum image convention (assuming r_cut <= L/2), n=0 is sufficient.
    for pos, q in zip(positions, charges):
        # Displacement with Minimum Image Convention
        dr = coords - pos
        dr = dr - L * np.round(dr / L)
        
        r = np.linalg.norm(dr, axis=-1)
        
        # Avoid division by zero exactly at point charge locations
        r_safe = np.maximum(r, 1e-12)
        
        # Real-space term using erfc
        mask = r_safe < r_cut
        v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]

    # --- 3. Reciprocal-Space Summation ---
    # Generate k-vectors for the cubic box: k = (2*pi/L) * (nx, ny, nz)
    n_max = int(np.ceil(k_cut * L / (2 * np.pi)))
    n_range = np.arange(-n_max, n_max + 1)
    
    # Create a grid of k-vectors
    kx, ky, kz = np.meshgrid(n_range, n_range, n_range, indexing='ij')
    k_vectors = (2 * np.pi / L) * np.stack((kx, ky, kz), axis=-1).reshape(-1, 3)
    
    for k in k_vectors:
        k_sq = np.dot(k, k)
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue  # Skip k=0 term and vectors outside the spherical cutoff
            
        # Structure factor: sum_i q_i * exp(-i * k * r_i)
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))
        
        # Potential evaluation at all grid coords: exp(i * k * r_grid)
        grid_phase = np.exp(1j * np.dot(coords, k))
        
        # Prefactor
        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha**2)) / k_sq
        
        # Add the real part of the Fourier term to the potential
        v_total += np.real(prefactor * S_k * grid_phase)
        
    # --- 4. Constant / Background Terms ---
    # Shift to account for the uniform neutralizing background if net charge != 0
    net_charge = np.sum(charges)
    if not np.isclose(net_charge, 0.0):
        v_bg = - (np.pi / (alpha**2 * V_box)) * net_charge
        v_total += v_bg
        
    # Note: If evaluating total energy, we would subtract the self-interaction term:
    # E_self = (alpha / sqrt(pi)) * sum(q_i**2). 
    # Because we are evaluating the potential on a grid, self-interaction only applies 
    # if evaluating exactly ON the charge coordinates, which we typically don't do on a mesh.

    return v_total
