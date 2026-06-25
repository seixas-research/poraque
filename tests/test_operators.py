# -*- coding: utf-8 -*-
# file: test_operators.py
"""Unit tests for numerical operators: gradients, Laplacians, Poisson."""

import numpy as np
import pytest

from poraque.core import Grid


@pytest.fixture
def plane_wave(grid):
    """A single plane wave f = sin(2*pi*x/L) and its analytic Laplacian."""
    coords = grid.get_xyz()
    x = coords[..., 0]
    L = 10.0
    k = 2 * np.pi / L
    f = np.sin(k * x)
    lap_analytic = -(k**2) * f
    grad_x_analytic = k * np.cos(k * x)
    return f, grad_x_analytic, lap_analytic


def test_fft_laplacian_matches_analytic(backend, grid, plane_wave):
    f, _, lap_analytic = plane_wave
    lap = backend.laplacian_fft(f, grid)
    # Spectral accuracy: essentially machine precision for a band-limited wave.
    assert np.allclose(lap, lap_analytic, atol=1e-8)


def test_fd_laplacian_matches_analytic(backend, grid, plane_wave):
    f, _, lap_analytic = plane_wave
    lap = backend.laplacian(f, grid)
    # 2nd-order finite difference -> looser tolerance.
    assert np.allclose(lap, lap_analytic, atol=5e-2)


def test_fd_gradient_matches_analytic(backend, grid, plane_wave):
    f, grad_x_analytic, _ = plane_wave
    grad = backend.gradient(f, grid)
    assert grad.shape == (3, *grid.shape)
    assert np.allclose(grad[0], grad_x_analytic, atol=5e-2)
    # No variation along y, z.
    assert np.allclose(grad[1], 0.0, atol=1e-8)
    assert np.allclose(grad[2], 0.0, atol=1e-8)


def test_fd_and_fft_laplacian_consistent(backend, grid, plane_wave):
    f, _, _ = plane_wave
    lap_fd = backend.laplacian(f, grid)
    lap_fft = backend.laplacian_fft(f, grid)
    assert np.allclose(lap_fd, lap_fft, atol=5e-2)


def test_poisson_satisfies_equation(backend, grid):
    """nabla^2 V = -4*pi*(n - <n>) for the FFT Poisson solver."""
    rng = np.random.default_rng(1)
    # Smooth, band-limited periodic charge built from low-G plane waves.
    coords = grid.get_xyz()
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    L = 10.0
    k = 2 * np.pi / L
    n = (1.0
         + 0.3 * np.sin(k * x)
         + 0.2 * np.cos(2 * k * y)
         + 0.1 * np.sin(k * z))

    v = backend.poisson(n, grid)
    lap_v = backend.laplacian_fft(v, grid)
    expected = -4 * np.pi * (n - n.mean())
    assert np.allclose(lap_v, expected, atol=1e-6)


def test_poisson_zero_mean_potential(backend, grid):
    n = np.ones(grid.shape)  # uniform -> potential should be flat (zero mean)
    v = backend.poisson(n, grid)
    assert v.mean() == pytest.approx(0.0, abs=1e-10)


def test_laplacian_convergence_with_refinement(backend, cubic_cell):
    """Finite-difference Laplacian error decreases as the grid is refined."""
    L = 10.0
    k = 2 * np.pi / L

    def max_error(n):
        g = Grid((n, n, n), cubic_cell, pbc=True)
        x = g.get_xyz()[..., 0]
        f = np.sin(k * x)
        lap = backend.laplacian(f, g)
        return np.max(np.abs(lap + k**2 * f))

    err_coarse = max_error(16)
    err_fine = max_error(32)
    assert err_fine < err_coarse
