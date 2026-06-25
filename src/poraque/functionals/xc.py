# -*- coding: utf-8 -*-
# file: xc.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Exchange-correlation functionals.

The functionals here follow the shared :class:`~poraque.functionals.base.Functional`
interface (``energy`` and ``potential``) so they can be reused unchanged by both
the OF-DFT engine and the KS-DFT engine.

Two families are provided:

* native, dependency-free **LDA** functionals (Dirac exchange + PW92
  correlation), and
* libxc-backed **GGA** functionals (:class:`PBE`, :class:`PBEsol`) that wrap the
  `Libxc <https://www.tddft.org/programs/libxc/>`_ library through its
  ``pylibxc`` Python bindings.
"""

import numpy as np

from .base import Functional
from ..profiling import profiler

try:  # libxc is only required for the GGA functionals (PBE / PBEsol).
    import pylibxc
    _PYLIBXC_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without pylibxc
    pylibxc = None
    _PYLIBXC_AVAILABLE = False

# Small density floor to keep n^p and 1/n stable in vacuum regions.
_DENS_FLOOR = 1e-30


class DiracExchange(Functional):
    r"""
    Dirac (LDA / Slater) exchange functional.

    .. math::

        E_x[n] = -C_x \int n(\mathbf{r})^{4/3}\, d\mathbf{r}, \qquad
        C_x = \frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}

    with potential :math:`v_x = -\frac{4}{3} C_x\, n^{1/3}`.
    """

    def __init__(self):
        super().__init__("Exchange (Dirac)")
        self.C_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

    def energy(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        return -self.C_x * backend.integrate(n ** (4.0 / 3.0), grid)

    def potential(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        return -(4.0 / 3.0) * self.C_x * n ** (1.0 / 3.0)


class PW92Correlation(Functional):
    """
    Perdew-Wang (1992) LDA correlation functional (spin-unpolarized).

    Implements the parametrization of the correlation energy per electron
    ``eps_c(rs)`` and its corresponding potential via
    ``v_c = eps_c - (rs/3) d eps_c / d rs``.
    """

    # Parameters for the paramagnetic (unpolarized) correlation energy.
    _A = 0.031091
    _alpha1 = 0.21370
    _beta1 = 7.5957
    _beta2 = 3.5876
    _beta3 = 1.6382
    _beta4 = 0.49294

    def __init__(self):
        super().__init__("Correlation (PW92)")

    def _rs(self, n):
        n = np.maximum(n, _DENS_FLOOR)
        return (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)

    def _eps_and_deps(self, rs):
        """Return ``eps_c(rs)`` and ``d eps_c / d rs``."""
        A, a1 = self._A, self._alpha1
        b1, b2, b3, b4 = self._beta1, self._beta2, self._beta3, self._beta4

        sqrt_rs = np.sqrt(rs)
        # Q = 2A (b1 rs^1/2 + b2 rs + b3 rs^3/2 + b4 rs^2)
        Q = 2.0 * A * (b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs**2)
        # dQ/drs
        dQ = 2.0 * A * (0.5 * b1 / sqrt_rs + b2 + 1.5 * b3 * sqrt_rs + 2.0 * b4 * rs)

        log_term = np.log1p(1.0 / Q)
        eps = -2.0 * A * (1.0 + a1 * rs) * log_term

        # d eps / drs
        d_log = -(1.0 / (Q * (Q + 1.0))) * dQ  # d/drs log(1 + 1/Q)
        deps = -2.0 * A * (a1 * log_term + (1.0 + a1 * rs) * d_log)
        return eps, deps

    def energy(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        rs = self._rs(n)
        eps, _ = self._eps_and_deps(rs)
        return backend.integrate(n * eps, grid)

    def potential(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        rs = self._rs(n)
        eps, deps = self._eps_and_deps(rs)
        # v_c = eps - (rs/3) d eps/d rs
        v_c = eps - (rs / 3.0) * deps
        # Zero out the (numerically meaningless) vacuum contribution.
        return np.where(density.data > _DENS_FLOOR, v_c, 0.0)


class LDA(Functional):
    """
    Convenience exchange-correlation functional combining Dirac exchange and
    PW92 correlation.

    Parameters
    ----------
    correlation : bool, optional
        Include PW92 correlation (default ``True``). When ``False`` this is a
        pure exchange-only LDA.
    """

    def __init__(self, correlation=True):
        super().__init__("XC (LDA)")
        self._x = DiracExchange()
        self._c = PW92Correlation() if correlation else None

    def energy(self, density, system, grid, backend):
        e = self._x.energy(density, system, grid, backend)
        if self._c is not None:
            e += self._c.energy(density, system, grid, backend)
        return e

    def potential(self, density, system, grid, backend):
        v = self._x.potential(density, system, grid, backend)
        if self._c is not None:
            v = v + self._c.potential(density, system, grid, backend)
        return v


# --------------------------------------------------------------------------- #
# Spectral (plane-wave) differential operators for GGA functionals
# --------------------------------------------------------------------------- #
def _spectral_gradient(field, grid):
    r"""
    Cartesian gradient :math:`\nabla f` via the plane-wave basis.

    Each component is ``Re(IFFT(i G_a * FFT(f)))`` using the grid's reciprocal
    vectors (valid for any, including non-orthogonal, cell). Returns an array of
    shape ``(3, Nx, Ny, Nz)``.
    """
    f_g = np.fft.fftn(field)
    gx, gy, gz = grid.get_g_vectors()
    dx = np.real(np.fft.ifftn(1j * gx * f_g))
    dy = np.real(np.fft.ifftn(1j * gy * f_g))
    dz = np.real(np.fft.ifftn(1j * gz * f_g))
    return np.array([dx, dy, dz])


def _spectral_divergence(vector, grid):
    r"""
    Divergence :math:`\nabla\cdot\mathbf{F}` of a ``(3, ...)`` field via FFT.

    Computed as ``Re(IFFT(i (G_x F_x + G_y F_y + G_z F_z)))``, the spectral
    adjoint of :func:`_spectral_gradient`.
    """
    gx, gy, gz = grid.get_g_vectors()
    fx = np.fft.fftn(vector[0])
    fy = np.fft.fftn(vector[1])
    fz = np.fft.fftn(vector[2])
    div_g = 1j * (gx * fx + gy * fy + gz * fz)
    return np.real(np.fft.ifftn(div_g))


class LibXC(Functional):
    r"""
    Exchange-correlation functional evaluated through libxc (``pylibxc``).

    A functional is specified as a list of libxc component identifiers (e.g.
    exchange + correlation), whose energy densities and potentials are summed.
    Both LDA- and GGA-family components are supported. For GGA components the
    reduced density gradient :math:`\sigma = |\nabla n|^2` is built with the
    spectral gradient, and the potential includes the gradient-dependent term

    .. math::

        v_{xc} = \frac{\partial e}{\partial n}
                 - 2\,\nabla\cdot\!\left(\frac{\partial e}{\partial \sigma}\,
                   \nabla n\right),

    where ``e = n\,\varepsilon_{xc}`` is the XC energy density returned by libxc.

    Parameters
    ----------
    label : str
        Human-readable functional name (used in the energy decomposition).
    components : sequence of (str, bool)
        Each entry is ``(libxc_name, is_gga)`` — the libxc functional string
        (e.g. ``"gga_x_pbe"``) and whether it needs the density gradient.

    Notes
    -----
    This functional is spin-unpolarized. It requires ``pylibxc``; constructing
    it without libxc installed raises a clear :class:`ImportError`.
    """

    def __init__(self, label, components):
        super().__init__(label)
        if not _PYLIBXC_AVAILABLE:
            raise ImportError(
                f"The {label!r} functional requires libxc (the 'pylibxc' "
                "package), which is not installed. Install it, e.g. with "
                "`conda install -c conda-forge libxc pylibxc`."
            )
        self._components = list(components)
        self._funcs = [
            (pylibxc.LibXCFunctional(name, "unpolarized"), bool(is_gga))
            for name, is_gga in components
        ]
        self._is_gga = any(is_gga for _, is_gga in components)

    def _eval(self, density, grid):
        """Return ``(n, eps, vrho, vsigma, grad_n)`` on the grid."""
        n = np.maximum(density.data, _DENS_FLOOR)
        inp = {"rho": np.ascontiguousarray(n.ravel())}
        grad = None
        if self._is_gga:
            grad = _spectral_gradient(n, grid)
            sigma = grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2
            inp["sigma"] = np.ascontiguousarray(np.maximum(sigma.ravel(), 1e-40))

        eps = np.zeros(n.size)
        vrho = np.zeros(n.size)
        vsigma = np.zeros(n.size)
        for func, is_gga in self._funcs:
            out = func.compute(inp, do_exc=True, do_vxc=True)
            eps += np.asarray(out["zk"]).ravel()
            vrho += np.asarray(out["vrho"]).ravel()
            if is_gga and out.get("vsigma") is not None:
                vsigma += np.asarray(out["vsigma"]).ravel()

        shp = n.shape
        return (n, eps.reshape(shp), vrho.reshape(shp),
                vsigma.reshape(shp), grad)

    def energy(self, density, system, grid, backend):
        with profiler.timer(f"XC energy [{self.name}]"):
            n, eps, _, _, _ = self._eval(density, grid)
            return backend.integrate(n * eps, grid)

    def potential(self, density, system, grid, backend):
        with profiler.timer(f"XC potential [{self.name}]"):
            n, _, vrho, vsigma, grad = self._eval(density, grid)
            v = vrho
            if self._is_gga:
                flux = np.array([vsigma * grad[0],
                                 vsigma * grad[1],
                                 vsigma * grad[2]])
                v = v - 2.0 * _spectral_divergence(flux, grid)
            return np.where(density.data > _DENS_FLOOR, v, 0.0)


class PBE(LibXC):
    """Perdew-Burke-Ernzerhof (PBE) GGA exchange-correlation (libxc)."""

    def __init__(self):
        super().__init__("XC (PBE)",
                         [("gga_x_pbe", True), ("gga_c_pbe", True)])


class PBEsol(LibXC):
    """PBEsol GGA exchange-correlation, revised for solids (libxc)."""

    def __init__(self):
        super().__init__("XC (PBEsol)",
                         [("gga_x_pbe_sol", True), ("gga_c_pbe_sol", True)])


# Registry of string-addressable XC functionals for the ASE calculator.
_XC_REGISTRY = {
    "lda": LDA,
    "pbe": PBE,
    "pbesol": PBEsol,
}


def resolve_xc(name):
    """
    Map a functional name (case-insensitive) to a constructed functional.

    Parameters
    ----------
    name : str
        One of ``"lda"``, ``"pbe"``, ``"pbesol"``.

    Returns
    -------
    Functional
        A freshly constructed functional instance.

    Raises
    ------
    NotImplementedError
        If ``name`` is not a recognized functional.
    """
    key = name.lower()
    if key not in _XC_REGISTRY:
        supported = ", ".join(sorted(_XC_REGISTRY))
        raise NotImplementedError(
            f"Exchange-correlation functional {name!r} is not implemented. "
            f"Supported functionals: {supported}."
        )
    return _XC_REGISTRY[key]()
