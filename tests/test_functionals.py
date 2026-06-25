# -*- coding: utf-8 -*-
# file: test_functionals.py
"""Unit tests for energy functionals and their functional derivatives."""

import numpy as np
import pytest

from poraque.core import Density
from poraque.functionals import (
    DiracExchange,
    Hartree,
    LDA,
    PW92Correlation,
    TFvW,
    ThomasFermi,
    VonWeizsaecker,
)


def smooth_density(grid, n_electrons=4.0, seed=0):
    """A smooth, strictly positive periodic density that integrates to N."""
    coords = grid.get_xyz()
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    L = 10.0
    k = 2 * np.pi / L
    data = (1.0
            + 0.4 * np.sin(k * x)
            + 0.3 * np.cos(k * y)
            + 0.2 * np.sin(k * z + 0.5))
    data = np.maximum(data, 0.05)
    rho = Density(grid, data)
    rho.normalize(n_electrons)
    return rho


def smooth_perturbation(grid, amp=1e-3):
    coords = grid.get_xyz()
    x, y = coords[..., 0], coords[..., 1]
    k = 2 * np.pi / 10.0
    return amp * np.cos(2 * k * x) * np.sin(k * y)


def fd_derivative_check(func, grid, backend, n_electrons=4.0, tol=1e-5):
    """Compare analytic potential to a central finite-difference of the energy."""
    rho = smooth_density(grid, n_electrons)
    dn = smooth_perturbation(grid)
    eps = 1.0

    v = func.potential(rho, None, grid, backend)
    analytic = backend.integrate(v * dn, grid)

    e_plus = func.energy(Density(grid, rho.data + eps * dn), None, grid, backend)
    e_minus = func.energy(Density(grid, rho.data - eps * dn), None, grid, backend)
    numeric = (e_plus - e_minus) / (2 * eps)

    assert numeric == pytest.approx(analytic, rel=tol, abs=1e-9)


class TestFunctionalDerivatives:
    def test_thomas_fermi(self, grid, backend):
        fd_derivative_check(ThomasFermi(), grid, backend)

    def test_von_weizsaecker(self, grid, backend):
        fd_derivative_check(VonWeizsaecker(), grid, backend)

    def test_dirac_exchange(self, grid, backend):
        fd_derivative_check(DiracExchange(), grid, backend)

    def test_pw92_correlation(self, grid, backend):
        fd_derivative_check(PW92Correlation(), grid, backend, tol=1e-4)

    def test_hartree(self, grid, backend):
        fd_derivative_check(Hartree(), grid, backend)

    def test_lda(self, grid, backend):
        fd_derivative_check(LDA(), grid, backend, tol=1e-4)

    def test_tfvw(self, grid, backend):
        fd_derivative_check(TFvW(lambda_vw=0.2), grid, backend)


class TestThomasFermi:
    def test_uniform_density_limit(self, grid, backend):
        # For uniform n, T_TF = C_TF * n^(5/3) * V.
        n0 = 0.5
        rho = Density(grid, np.full(grid.shape, n0))
        tf = ThomasFermi()
        expected = tf.C_TF * n0 ** (5 / 3) * grid.volume
        assert tf.energy(rho, None, grid, backend) == pytest.approx(expected)

    def test_coeff_scaling(self, grid, backend):
        rho = smooth_density(grid)
        e1 = ThomasFermi(coeff=1.0).energy(rho, None, grid, backend)
        e2 = ThomasFermi(coeff=0.5).energy(rho, None, grid, backend)
        assert e2 == pytest.approx(0.5 * e1)


class TestTFvW:
    def test_reduces_to_tf_when_lambda_zero(self, grid, backend):
        rho = smooth_density(grid)
        tfvw = TFvW(lambda_vw=0.0)
        tf = ThomasFermi()
        assert tfvw.energy(rho, None, grid, backend) == pytest.approx(
            tf.energy(rho, None, grid, backend)
        )

    def test_vw_term_additive(self, grid, backend):
        rho = smooth_density(grid)
        tfvw = TFvW(lambda_vw=1.0, c_tf=1.0)
        tf = ThomasFermi().energy(rho, None, grid, backend)
        vw = VonWeizsaecker().energy(rho, None, grid, backend)
        assert tfvw.energy(rho, None, grid, backend) == pytest.approx(tf + vw)


class TestVonWeizsaecker:
    def test_constant_density_zero_energy(self, grid, backend):
        # vW energy vanishes for a uniform density (no gradients).
        rho = Density(grid, np.full(grid.shape, 0.3))
        vw = VonWeizsaecker()
        assert vw.energy(rho, None, grid, backend) == pytest.approx(0.0, abs=1e-10)

    def test_lambda_scaling(self, grid, backend):
        rho = smooth_density(grid)
        e1 = VonWeizsaecker(lambda_=1.0).energy(rho, None, grid, backend)
        e2 = VonWeizsaecker(lambda_=0.25).energy(rho, None, grid, backend)
        assert e2 == pytest.approx(0.25 * e1)


class TestXC:
    def test_lda_is_sum_of_x_and_c(self, grid, backend):
        rho = smooth_density(grid)
        ex = DiracExchange().energy(rho, None, grid, backend)
        ec = PW92Correlation().energy(rho, None, grid, backend)
        lda = LDA().energy(rho, None, grid, backend)
        assert lda == pytest.approx(ex + ec)

    def test_exchange_is_negative(self, grid, backend):
        rho = smooth_density(grid)
        assert DiracExchange().energy(rho, None, grid, backend) < 0.0

    def test_correlation_is_negative(self, grid, backend):
        rho = smooth_density(grid)
        assert PW92Correlation().energy(rho, None, grid, backend) < 0.0
