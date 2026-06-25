# -*- coding: utf-8 -*-
# file: test_xc_libxc.py
"""Tests for the libxc-backed GGA functionals (PBE / PBEsol)."""

import numpy as np
import pytest

from poraque.backends.numpy import NumpyBackend
from poraque.core import Density, Grid
from poraque.functionals import resolve_xc
from poraque.functionals.xc import _PYLIBXC_AVAILABLE

libxc_required = pytest.mark.skipif(
    not _PYLIBXC_AVAILABLE, reason="pylibxc (libxc) is not installed"
)


@pytest.fixture
def smooth_density():
    L, N = 10.0, 20
    grid = Grid((N, N, N), np.eye(3) * L, pbc=True)
    coords = grid.get_xyz()
    r2 = np.sum((coords - L / 2) ** 2, axis=-1)
    n = 0.5 * np.exp(-r2 / 4.0) + 1e-4
    return grid, Density(grid, n)


class TestResolveXC:
    def test_lda_always_available(self):
        assert resolve_xc("lda").name == "XC (LDA)"

    def test_unknown_functional_raises(self):
        with pytest.raises(NotImplementedError):
            resolve_xc("scan")

    def test_case_insensitive(self):
        assert resolve_xc("LDA").name == "XC (LDA)"


@libxc_required
class TestPBE:
    def test_pbe_and_pbesol_construct(self):
        assert resolve_xc("pbe").name == "XC (PBE)"
        assert resolve_xc("pbesol").name == "XC (PBEsol)"

    def test_energy_is_finite_and_negative(self, smooth_density):
        grid, rho = smooth_density
        backend = NumpyBackend()
        for name in ("pbe", "pbesol"):
            e = resolve_xc(name).energy(rho, None, grid, backend)
            assert np.isfinite(e)
            assert e < 0.0  # XC energy of a bound density is negative

    def test_potential_shape_and_finiteness(self, smooth_density):
        grid, rho = smooth_density
        v = resolve_xc("pbe").potential(rho, None, grid, NumpyBackend())
        assert v.shape == grid.shape
        assert np.all(np.isfinite(v))

    @pytest.mark.parametrize("name", ["pbe", "pbesol"])
    def test_potential_is_functional_derivative(self, smooth_density, name):
        """v_xc must equal dE_xc/dn: E[n+dn]-E[n] ≈ integral(v * dn)."""
        grid, rho = smooth_density
        backend = NumpyBackend()
        func = resolve_xc(name)

        e0 = func.energy(rho, None, grid, backend)
        v = func.potential(rho, None, grid, backend)

        coords = grid.get_xyz()
        dn = 1e-4 * np.exp(
            -np.sum((coords - np.array([4.0, 5.0, 5.0])) ** 2, axis=-1) / 2.0)
        e1 = func.energy(Density(grid, rho.data + dn), None, grid, backend)

        lhs = e1 - e0
        rhs = backend.integrate(v * dn, grid)
        assert lhs == pytest.approx(rhs, rel=2e-2)

    def test_pbe_differs_from_pbesol(self, smooth_density):
        grid, rho = smooth_density
        backend = NumpyBackend()
        e_pbe = resolve_xc("pbe").energy(rho, None, grid, backend)
        e_sol = resolve_xc("pbesol").energy(rho, None, grid, backend)
        assert not np.isclose(e_pbe, e_sol)
