# -*- coding: utf-8 -*-
# file: paw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The PAW total energy: the grid, plus a one-center correction per atom.

A ``CHGCAR`` holds the pseudo-density plus the compensation charge,
:math:`n_T = \tilde n + \hat n`, on the grid, and after it every atom's
augmentation occupancies :math:`R(cc';LM)`; a ``TAUCAR`` holds the pseudo
kinetic energy density :math:`\tilde\tau`. Those three, the ``POTCAR`` and the
structure determine VASP's total energy completely, in the direct form

.. math::

    E = \int\tilde\tau
      + E_H[n_T] + E_{\rm loc}[n_T] + E_{\alpha Z} + E_{\rm Ewald}
      + E_{xc}[n_T + \tilde n_c]
      + \sum_a \Bigl\{ \sum_{c\le c'} \sqrt{4\pi}\,R_a(cc';00)\,\bar D_{cc'}
        + E_H[n^1_a] - E_H[\tilde n^1_a + \hat n_a]
        + E_{xc}[n^1_a + n_c] - E_{xc}[\tilde n^1_a + \hat n_a + \tilde n_c]
        - E_{xc}[n_c] \Bigr\}
      + \sum_t N_t E^{\rm atom}_t ,

in which no potential appears --- the double-counting terms of VASP's
band-energy form cancel exactly --- and the one-center kinetic energy, the
interaction with the nucleus and the frozen core, and the local
pseudopotential inside the sphere enter only through the linear strength
:math:`\bar D`, built from the ``POTCAR`` matrix ``DION``. The derivation, with
the VASP routines each term comes from, is in
``claude_runs/paw_energy/VASP_PAW_ENERGY_SPEC.md`` of the run that wrote this
module; :class:`OneCenterEnergy` follows it, and :func:`paw_total_energy`
assembles the whole.

Two treatments of the one-center exchange-correlation, as VASP's ``LASPH``:

``lasph=False``
    The VASP default: LDA to second order in the non-spherical components, the
    PBE gradient correction on the spherical part only.
``lasph=True``
    LDA and PBE evaluated on a product angular grid (Gauss--Legendre in
    :math:`\cos\theta` times equidistant :math:`\phi`), with the full gradient;
    what the Materials Project runs used.

Constants are VASP's own (:data:`AUTOA`, :data:`RYTOEV`): energies of
thousands of eV are compared at the meV level, and the CODATA Coulomb
constant differs from VASP's by 8e-6 relative.
"""

import numpy as np

from .xc import lda_xc, pbe_xc

#: VASP's constants (constant.F), not CODATA: see the module docstring.
AUTOA = 0.529177249
RYTOEV = 13.605826
HARTREE = 2.0 * RYTOEV
FELECT = 2.0 * RYTOEV * AUTOA
#: Hartree/Bohr³ to eV/Å³.
ENERGY_DENSITY = HARTREE / AUTOA ** 3
SQRT4PI = np.sqrt(4.0 * np.pi)


# ---------------------------------------------------------------------- #
# Radial numerics, as VASP does them
# ---------------------------------------------------------------------- #
def simpson_weights(r):
    """VASP's ``SET_SIMP``: Simpson weights accumulated from the last point."""
    n = r.size
    h = np.log(r[-1] / r[0]) / (n - 1)
    weights = np.zeros(n)
    for k in range(n - 1, 1, -2):
        weights[k] += r[k] * h / 3.0
        weights[k - 1] = 4.0 * r[k - 1] * h / 3.0
        weights[k - 2] = r[k - 2] * h / 3.0
    return weights


def radial_gradient(f, r):
    """VASP's ``GRAD`` at ``RADIAL_FDORDER = 2``: d f / d r on the log grid."""
    n = r.size
    h = np.log(r[-1] / r[0]) / (n - 1)
    d = np.empty_like(f)
    d[..., 0] = f[..., 1] - f[..., 0]
    d[..., 1:-1] = 0.5 * (f[..., 2:] - f[..., :-2])
    d[..., -1] = f[..., -1] - f[..., -2]
    return d / (h * r)


def radial_hartree_potential(L, r, rho):
    """
    VASP's ``RAD_POT_HAR`` potential: the LM coefficient of :math:`V_H`, eV.

    The actual potential of that component is this times :math:`Y_{LM}`; for
    L = 0, ``potential / (2 sqrt(pi))``.
    """
    n = r.size
    h = np.log(r[-1] / r[0]) / (n - 1)
    reverse = r[::-1]
    rl = reverse ** L
    t1 = rl * reverse * rho[::-1]
    t2 = rho[::-1] / rl
    h3 = h / 3.0

    def cumulative(t):
        v = np.zeros(n)
        for index in range(2, n, 2):
            v[index] = v[index - 2] + h3 * (t[index - 2] + 4 * t[index - 1]
                                            + t[index])
            v[index - 1] = v[index - 2] + h3 * (1.25 * t[index - 2]
                                                + 2.0 * t[index - 1]
                                                - 0.25 * t[index])
        if n % 2 == 0:
            v[n - 1] = v[n - 3] + h3 * (t[n - 3] + 4 * t[n - 2] + t[n - 1])
        return v

    v1, v2 = cumulative(t1), cumulative(t2)
    total = v1[-1]
    rl_forward = r ** L
    return (v2[::-1] * rl_forward + (total - v1[::-1])
            / (r * rl_forward)) * (FELECT * 4.0 * np.pi / (2 * L + 1))


def radial_hartree(L, r, weights, rho):
    """
    VASP's ``RAD_POT_HAR``: the Hartree energy ``(1/2) int U rho`` of one LM.

    ``rho`` is :math:`r^2 n_{LM}(r)`; the energy is in eV.
    """
    n = r.size
    h = np.log(r[-1] / r[0]) / (n - 1)
    reverse = r[::-1]
    rl = reverse ** L
    t1 = rl * reverse * rho[::-1]
    t2 = rho[::-1] / rl
    h3 = h / 3.0

    def cumulative(t):
        v = np.zeros(n)
        for index in range(2, n, 2):
            v[index] = v[index - 2] + h3 * (t[index - 2] + 4 * t[index - 1]
                                            + t[index])
            v[index - 1] = v[index - 2] + h3 * (1.25 * t[index - 2]
                                                + 2.0 * t[index - 1]
                                                - 0.25 * t[index])
        if n % 2 == 0:
            v[n - 1] = v[n - 3] + h3 * (t[n - 3] + 4 * t[n - 2] + t[n - 1])
        return v

    v1, v2 = cumulative(t1), cumulative(t2)
    total = v1[-1]
    rl_forward = r ** L
    # Index i of the reversed arrays is point n-1-i of the forward grid.
    potential = (v2[::-1] * rl_forward + (total - v1[::-1])
                 / (r * rl_forward)) * (FELECT * 4.0 * np.pi / (2 * L + 1))
    return 0.5 * np.sum(potential * rho * weights)


def _bessel(L, x):
    from scipy.special import spherical_jn

    return spherical_jn(L, x)


def compensation_shapes(r, rmax, lmax):
    r"""
    VASP's ``AUG_SETQ``: :math:`g_L(r)\,r^2` on the radial grid, per L.

    :math:`g_L = A_1 j_L(q_1 r) + A_2 j_L(q_2 r)` inside ``rmax``, zero
    outside; :math:`q_i r_{\max}` the first two zeros of :math:`j_L`, and the
    :math:`A_i` set by :math:`g_L'(r_{\max}) = 0` and
    :math:`\int_0^{r_{\max}} g_L r^{L+2} dr = 1` (32-point Gauss).
    """
    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    nodes, weights = np.polynomial.legendre.leggauss(32)
    radius = 0.5 * rmax * (nodes + 1.0)
    weights = 0.5 * rmax * weights
    shapes = np.zeros((lmax + 1, r.size))
    for L in range(lmax + 1):
        zeros, x = [], 1e-3
        while len(zeros) < 2:
            if np.sign(spherical_jn(L, x)) != np.sign(spherical_jn(L, x + 0.05)):
                zeros.append(brentq(lambda t: spherical_jn(L, t), x, x + 0.05,
                                    xtol=1e-14))
            x += 0.05
        q = np.array(zeros) / rmax
        matrix = np.array([
            [qi * spherical_jn(L, qi * rmax, derivative=True) for qi in q],
            [np.sum(spherical_jn(L, qi * radius) * radius ** (L + 2) * weights)
             for qi in q]])
        a = np.linalg.solve(matrix, np.array([0.0, 1.0]))
        inside = r <= rmax
        shapes[L, inside] = sum(ai * spherical_jn(L, qi * r[inside])
                                for ai, qi in zip(a, q)) * r[inside] ** 2
    return shapes


def _vasp_ylm(directions, lmax):
    """Real Y_LM in VASP's (= the records') order, at unit ``directions``."""
    from ..ml.paw import solid_harmonics  # the one definition of the order

    import torch

    values = solid_harmonics(torch.as_tensor(directions, dtype=torch.float64),
                             min(lmax, 4))
    result = [v.numpy() for v in values]
    if lmax > 4:
        from .pseudopotentials import sph_harm_y
        directions = np.asarray(directions)
        theta = np.arccos(directions[:, 2].clip(-1, 1))
        phi = np.arctan2(directions[:, 1], directions[:, 0])
        for ell in range(5, lmax + 1):
            columns = []
            for m in range(-ell, ell + 1):
                harmonic = sph_harm_y(ell, abs(m), theta, phi)
                columns.append(harmonic.real if m == 0 else
                               np.sqrt(2)*(-1)**m*(harmonic.imag if m < 0 else harmonic.real))
            result.append(np.column_stack(columns))
    return result


# ---------------------------------------------------------------------- #
# One atom
# ---------------------------------------------------------------------- #
class OneCenterEnergy:
    r"""
    The one-center energy of one element's atoms, from their occupancies.

    Parameters
    ----------
    setup : PawSetup
        The element's PAW dataset (:mod:`poraque.fields.vasp.paw_setup`).
    lasph : bool, optional
        Non-spherical one-center xc on an angular grid (VASP ``LASPH``).
    relativistic : bool, optional
        Relativistic LDA exchange, as VASP's ``TREL`` in the ``LASPH`` path.
    """

    def __init__(self, setup, lasph=True, relativistic=False, functional='pbe'):
        from ..fields.vasp.augmentation import record_layout

        self.setup = setup
        self.lasph = bool(lasph)
        self.relativistic = bool(relativistic)
        if functional not in ('lda', 'pbe'):
            raise ValueError('PAW one-center XC must be lda or pbe')
        self.functional = functional
        r = np.asarray(setup.grid, dtype=float)
        self.r = r
        self.weights = simpson_weights(r)
        self.channels = list(setup.channels)
        self.lmax = 2 * max(self.channels)
        self.layout = record_layout(self.channels)
        waves = [(np.asarray(ps, float), np.asarray(ae, float))
                 for ps, ae in setup.partial_waves]
        self.ps = np.array([w[0] for w in waves])
        self.ae = np.array([w[1] for w in waves])

        # Compensation radius: the first grid point beyond PSDMAX - 1e-5.
        target = setup.compensation_radius - 1e-5
        self.rcomp = float(r[np.argmax(r > target)])
        self.aug = compensation_shapes(r, self.rcomp, self.lmax)

        nch = len(self.channels)
        self.qpaw = np.zeros((nch, nch, self.lmax + 1))
        for a in range(nch):
            for b in range(nch):
                product = self.ae[a] * self.ae[b] - self.ps[a] * self.ps[b]
                for L in range(self.lmax + 1):
                    self.qpaw[a, b, L] = np.sum(product * r ** L * self.weights)

        # Frozen cores, as densities (the file stores sqrt(4 pi) r^2 n).
        self.core_ae = setup.tables["core charge-density"] / (SQRT4PI * r * r)
        self.core_ps = (setup.tables["core charge-density (pseudized)"]
                        / (SQRT4PI * r * r))

        # The linear strength D-bar (spec 2.3). VASP does not use the POTCAR's
        # reference potentials: it zeroes them and recomputes them at
        # start-up from the reference-atom occupancies (paw.F,
        # SET_PAW_ATOM_POT), and so does this.
        p_ae, p_ps = self.reference_potentials()
        self.reference = (p_ae, p_ps)
        self.dbar = np.array(setup.d0, dtype=float)
        for a in range(nch):
            for b in range(nch):
                if self.channels[a] != self.channels[b]:
                    self.dbar[a, b] = 0.0
                    continue
                self.dbar[a, b] += (
                    np.sum(self.ae[a] * self.ae[b] * p_ae * self.weights)
                    - np.sum(self.ps[a] * self.ps[b] * p_ps * self.weights)
                    - self.qpaw[a, b, 0] * np.sum(self.aug[0] * p_ps
                                                 * self.weights))
        self.core_xc = self._spherical_xc(self.core_ae, np.zeros_like(r)) \
            + float(setup.parameters.get("DEXC", 0.0) or 0.0)
        if self.lasph:
            self._angular_grid()

    def atomic_record(self):
        """The reference atom's occupancies (``uccopancies in atom``) as a
        record: R(cc';00) = F (2l+1) QATO(c,c') / sqrt(4 pi) for same-l pairs."""
        qato = self.setup.atomic_occupancies
        record = np.zeros(len(self.layout))
        for index, (a, b, L, m) in enumerate(self.layout):
            if L == 0 and self.channels[a] == self.channels[b]:
                factor = 1.0 if a == b else 2.0
                record[index] = (factor * (2 * self.channels[a] + 1)
                                 * qato[a, b] / SQRT4PI)
        return record

    def _spherical_potential(self, rho00, core):
        """-(V_H + V_xc) of a spherical density, as VASP's RAD_POT (eV)."""
        r = self.r
        hartree = radial_hartree_potential(0, r, rho00) / SQRT4PI
        n = rho00 / (SQRT4PI * r * r) + core              # Å^-3, with core
        nb = np.maximum(n, 0.0) * AUTOA ** 3
        g = radial_gradient(n, r) * AUTOA ** 4             # signed

        def f(density, gradient):
            if self.functional == 'lda':
                return lda_xc(0.5*density, 0.5*density)
            lda, corr = pbe_xc(0.5 * density, 0.5 * density,
                               0.5 * np.abs(gradient), 0.5 * np.abs(gradient),
                               np.abs(gradient))
            return lda + corr

        hn = np.maximum(1e-6 * nb, 1e-14)
        dfdn = (f(nb + hn, g) - f(np.maximum(nb - hn, 0.0), g)) \
            / (nb + hn - np.maximum(nb - hn, 0.0))
        hg = np.maximum(1e-6 * np.abs(g), 1e-14)
        dfdg = (f(nb, g + hg) - f(nb, g - hg)) / (2 * hg)  # Hartree Bohr
        # v = df/dn - (1/r^2) d/dr (r^2 df/dn'), in Bohr then eV.
        rb = r / AUTOA
        divergence = radial_gradient(dfdg, rb) + 2.0 * dfdg / rb
        vxc = (dfdn - divergence) * HARTREE
        return -(hartree + vxc)

    def reference_potentials(self):
        """VASP's recomputed P^AE and P^PS of the reference atom (eV)."""
        ae, ps = self.multipoles(self.atomic_record())
        return (self._spherical_potential(ae[0], self.core_ae),
                self._spherical_potential(ps[0], self.core_ps))

    # -- densities ---------------------------------------------------------
    def multipoles(self, record):
        r"""
        :math:`r^2 n_{LM}(r)` of the AE and the pseudo (with :math:`\hat n`)
        one-center densities, from one record set.

        Returns
        -------
        tuple of numpy.ndarray
            ``(ae, ps)``, each ``((lmax+1)^2, nr)``.
        """
        lm = (self.lmax + 1) ** 2
        ae = np.zeros((lm, self.r.size))
        ps = np.zeros((lm, self.r.size))
        qhat = np.zeros(lm)
        for value, (a, b, L, m) in zip(record, self.layout):
            if L > self.lmax:
                continue
            index = L * L + m
            ae[index] += value * self.ae[a] * self.ae[b]
            ps[index] += value * self.ps[a] * self.ps[b]
            qhat[index] += value * self.qpaw[a, b, L]
        for L in range(self.lmax + 1):
            for m in range(2 * L + 1):
                index = L * L + m
                ps[index] += qhat[index] * self.aug[L]
        return ae, ps

    # -- xc ----------------------------------------------------------------
    def _spherical_xc(self, n_up_core, m, core=None):
        """Spherical PBE energy of a density (Å⁻³), for the frozen core."""
        n = n_up_core
        r, w = self.r, self.weights
        gradient = np.abs(radial_gradient(n, r))
        lda, corr = pbe_xc(0.5 * n * AUTOA ** 3, 0.5 * n * AUTOA ** 3,
                           0.5 * gradient * AUTOA ** 4,
                           0.5 * gradient * AUTOA ** 4,
                           gradient * AUTOA ** 4)
        if self.functional == 'lda':
            corr = 0.0
        return float(np.sum(4.0 * np.pi * r * r * (lda + corr) * w)
                     * ENERGY_DENSITY)

    def _angular_grid(self):
        lmax_calc = self.lmax
        llmax = min(6, lmax_calc) + 1
        phi_points = 3 * (llmax + 1)
        theta_points = 3 * (llmax // 2 + 1)
        cos_theta, theta_weights = np.polynomial.legendre.leggauss(theta_points)
        phi = 2.0 * np.pi * np.arange(phi_points) / phi_points
        ct, ph = np.meshgrid(cos_theta, phi, indexing="ij")
        st = np.sqrt(1.0 - ct * ct)
        directions = np.stack([st * np.cos(ph), st * np.sin(ph), ct],
                              -1).reshape(-1, 3)
        self.angular_weights = (np.repeat(theta_weights, phi_points)
                                * (2.0 * np.pi / phi_points))
        self.directions = directions
        ylm = _vasp_ylm(directions, lmax_calc)
        self.ylm = np.concatenate(ylm, axis=1)                  # (P, LM)
        # Tangential gradient of Y_LM on the unit sphere:
        # grad S(u) - L S(u) u, by central differences of the polynomial.
        step = 1e-5
        gradients = []
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            plus = np.concatenate(_vasp_ylm(directions + shift, lmax_calc), 1)
            minus = np.concatenate(_vasp_ylm(directions - shift, lmax_calc), 1)
            gradients.append((plus - minus) / (2 * step))
        gradient = np.stack(gradients, -1)                      # (P, LM, 3)
        degrees = np.concatenate([[L] * (2 * L + 1)
                                  for L in range(lmax_calc + 1)])
        radial = (degrees[None, :, None] * self.ylm[..., None]
                  * directions[:, None, :])
        self.ylm_gradient = gradient - radial

    def _xc_lasph(self, total, magnetization, core):
        """One-center xc on the angular grid; densities as r^2 n_LM."""
        r, w = self.r, self.weights
        r2 = r * r
        n_tot = self.ylm @ total / r2 + core[None, :]           # (P, nr)
        g_tot_ang = np.einsum("pls,lr->psr", self.ylm_gradient, total) / (r2 * r)
        dn_tot = radial_gradient(n_tot, r)
        if magnetization is None:
            grad = (dn_tot[:, None, :] * self.directions[:, :, None]
                    + g_tot_ang)
            g = np.sqrt((grad ** 2).sum(1))
            lda, corr = pbe_xc(0.5 * n_tot * AUTOA ** 3,
                               0.5 * n_tot * AUTOA ** 3,
                               0.5 * g * AUTOA ** 4, 0.5 * g * AUTOA ** 4,
                               g * AUTOA ** 4, relativistic=self.relativistic)
        else:
            n_mag = self.ylm @ magnetization / r2
            g_mag_ang = np.einsum("pls,lr->psr", self.ylm_gradient,
                                  magnetization) / (r2 * r)
            up = np.maximum(0.5 * (n_tot + n_mag), 1e-7)
            down = np.maximum(0.5 * (n_tot - n_mag), 1e-7)
            dn_mag = radial_gradient(n_mag, r)

            def gradient(radial_part, angular_part):
                vector = (radial_part[:, None, :] * self.directions[:, :, None]
                          + angular_part)
                return np.sqrt((vector ** 2).sum(1))

            g_up = gradient(0.5 * (dn_tot + dn_mag), 0.5 * (g_tot_ang + g_mag_ang))
            g_dn = gradient(0.5 * (dn_tot - dn_mag), 0.5 * (g_tot_ang - g_mag_ang))
            g = gradient(dn_tot, g_tot_ang)
            lda, corr = pbe_xc(up * AUTOA ** 3, down * AUTOA ** 3,
                               g_up * AUTOA ** 4, g_dn * AUTOA ** 4,
                               g * AUTOA ** 4, relativistic=self.relativistic)
        if self.functional == 'lda':
            corr = 0.0
        e = (lda + corr) * ENERGY_DENSITY                       # eV/Å³
        return float(np.einsum("p,pr,r->", self.angular_weights, e, r2 * w))

    def _xc_spherical(self, total, magnetization, core):
        """VASP's LASPH = .FALSE.: Taylor LDA plus spherical GGA."""
        r, w = self.r, self.weights
        r2 = r * r
        n0 = total[0] / (SQRT4PI * r2) + core
        m0 = (np.zeros_like(n0) if magnetization is None
              else magnetization[0] / (SQRT4PI * r2))
        up, down = 0.5 * (n0 + m0), 0.5 * (n0 - m0)

        def e_lda(u, d):
            return lda_xc(np.maximum(u, 0) * AUTOA ** 3,
                          np.maximum(d, 0) * AUTOA ** 3) * ENERGY_DENSITY

        energy = np.sum(4.0 * np.pi * r2 * e_lda(up, down) * w)
        # Second-order term in the non-spherical components.
        lm = total.shape[0]
        if lm > 1:
            rho_t = total[1:] / r2
            rho_m = (np.zeros_like(rho_t) if magnetization is None
                     else magnetization[1:] / r2)
            rho_u, rho_d = 0.5 * (rho_t + rho_m), 0.5 * (rho_t - rho_m)
            hu = np.maximum(1e-3 * up, 1e-12)
            hd = np.maximum(1e-3 * down, 1e-12)
            e00 = e_lda(up, down)
            euu = (e_lda(up + hu, down) - 2 * e00 + e_lda(up - hu, down)) / hu ** 2
            edd = (e_lda(up, down + hd) - 2 * e00 + e_lda(up, down - hd)) / hd ** 2
            eud = (e_lda(up + hu, down + hd) - e_lda(up + hu, down - hd)
                   - e_lda(up - hu, down + hd) + e_lda(up - hu, down - hd)) \
                / (4 * hu * hd)
            taylor = 0.5 * (euu * (rho_u ** 2).sum(0) + edd * (rho_d ** 2).sum(0)
                            + 2 * eud * (rho_u * rho_d).sum(0))
            energy += np.sum(r2 * taylor * w)
        if self.functional == 'lda':
            return float(energy)
        # Spherical gradient correction.
        dn = radial_gradient(n0, r)
        dm = radial_gradient(m0, r)
        u_c, d_c = np.maximum(up, 1e-7), np.maximum(down, 1e-7)
        _, corr = pbe_xc(u_c * AUTOA ** 3, d_c * AUTOA ** 3,
                         np.abs(0.5 * (dn + dm)) * AUTOA ** 4,
                         np.abs(0.5 * (dn - dm)) * AUTOA ** 4,
                         np.abs(dn) * AUTOA ** 4)
        energy += np.sum(4.0 * np.pi * r2 * corr * ENERGY_DENSITY * w)
        return float(energy)

    def xc(self, total, magnetization, core):
        if self.lasph:
            return self._xc_lasph(total, magnetization, core)
        return self._xc_spherical(total, magnetization, core)

    # -- the energy --------------------------------------------------------
    def energy(self, records):
        """
        One atom's one-center energy and its parts, eV.

        Parameters
        ----------
        records : sequence of numpy.ndarray
            The atom's record sets in file order: the total, and for a
            spin-polarized calculation the magnetization.

        Returns
        -------
        dict
            ``linear``, ``hartree``, ``xc`` and ``total``.
        """
        total_record = np.asarray(records[0], dtype=float)
        ae, ps = self.multipoles(total_record)
        magnetization = None
        if len(records) > 1 and np.any(records[1]):
            ae_m, ps_m = self.multipoles(np.asarray(records[1], dtype=float))
            magnetization = (ae_m, ps_m)

        linear = 0.0
        for value, (a, b, L, m) in zip(total_record, self.layout):
            if L == 0 and self.channels[a] == self.channels[b]:
                linear += SQRT4PI * value * self.dbar[a, b]

        hartree = 0.0
        for L in range(self.lmax + 1):
            for m in range(2 * L + 1):
                index = L * L + m
                hartree += (radial_hartree(L, self.r, self.weights, ae[index])
                            - radial_hartree(L, self.r, self.weights, ps[index]))

        xc = (self.xc(ae, None if magnetization is None else magnetization[0],
                      self.core_ae)
              - self.xc(ps, None if magnetization is None else magnetization[1],
                        self.core_ps)
              - self.core_xc)
        return {"linear": linear, "hartree": hartree, "xc": xc,
                "total": linear + hartree + xc}


# ---------------------------------------------------------------------- #
# The grid
# ---------------------------------------------------------------------- #
def _reciprocal_table(values, psgmax):
    """A POTCAR reciprocal table as a cubic spline on q = i PSGMAX/1000."""
    from scipy.interpolate import CubicSpline

    q = np.arange(values.size) * psgmax / 1000.0
    return CubicSpline(q, values, bc_type="natural")


def _reciprocal_vectors(grid):
    shape = grid.shape
    k = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in shape],
                             indexing="ij"), -1)
    cell = np.asarray(grid.cell, dtype=float)
    return k, k @ (2.0 * np.pi * np.linalg.inv(cell).T)


def grid_energy(structure, grid, density, magnetization, tau, setups):
    r"""
    The grid terms of the PAW total energy, eV.

    Parameters
    ----------
    structure : Structure
    grid : FieldGrid
    density, magnetization : numpy.ndarray
        :math:`n_T = \tilde n + \hat n` and its magnetization (or ``None``),
        e/Å³ --- the ``CHGCAR`` blocks divided by the volume.
    tau : numpy.ndarray
        The total pseudo kinetic energy density, eV/Å³.
    setups : dict
        ``{element: PawSetup}``.

    Returns
    -------
    dict
        ``kinetic``, ``hartree``, ``local``, ``alpha_z``, ``ewald``, ``xc``.
    """
    from ..fields.structure import element_of
    from ..fields.constants import COULOMB_CONSTANT_EV_ANGSTROM
    from .energy import ewald_energy

    omega = grid.volume
    points = int(np.prod(grid.shape))
    k, g = _reciprocal_vectors(grid)
    g2 = (g ** 2).sum(-1)
    gnorm = np.sqrt(g2)
    nonzero = g2 > 1e-12
    safe_g2 = np.where(nonzero, g2, 1.0)

    coefficients = np.fft.fftn(density * omega) / points     # C(G)
    hartree = (2.0 * np.pi * FELECT / omega
               * np.sum(np.abs(coefficients[nonzero]) ** 2 / g2[nonzero]))

    elements = [element_of(symbol) for symbol in structure.symbols]
    frac = np.asarray(structure.scaled_positions, dtype=float)
    local, core_g, zval_total, alpha = 0.0, np.zeros(grid.shape, complex), 0.0, 0.0
    start = 0
    counts = [int(c) for c in structure.counts]
    for element, count in zip(elements, counts):
        zval_total += setups[element].zval * count
    for element, count in zip(elements, counts):
        setup = setups[element]
        positions = frac[start:start + count]
        start += count
        phase = np.exp(-2j * np.pi * np.tensordot(k, positions.T, axes=1))
        factor = phase.sum(-1)                                # S_t(G)
        table = setup.reciprocal["local part"]
        psgmax, values = float(table[0]), table[1:]
        spline = _reciprocal_table(values, psgmax)
        cutoff = psgmax - psgmax / 1000.0
        mask = nonzero & (gnorm < cutoff)
        short = (spline(gnorm[mask]) - 4.0 * np.pi * setup.zval * FELECT
                 / safe_g2[mask])
        local += np.sum(np.real(factor[mask] * np.conj(coefficients[mask]))
                        * short) / omega
        alpha += count * values[0] * zval_total / omega
        partial = setup.reciprocal.get("core charge-density (partial)")
        if partial is not None and np.any(partial):
            pspline = _reciprocal_table(partial, psgmax)
            within = gnorm < psgmax - 3.0 * psgmax / 1000.0
            core_g += np.where(within, pspline(np.where(within, gnorm, 0.0)),
                               0.0) * factor / omega
    core = np.real(np.fft.ifftn(core_g) * points)

    charges = {element: setups[element].zval for element in set(elements)}
    ewald = (ewald_energy(structure, charges)
             * FELECT / COULOMB_CONSTANT_EV_ANGSTROM)

    m = np.zeros_like(density) if magnetization is None else magnetization
    up = np.maximum(0.5 * (density + m) + 0.5 * core, 1e-10)
    down = np.maximum(0.5 * (density - m) + 0.5 * core, 1e-10)

    def gradient_norm(field):
        spectrum = np.fft.fftn(field)
        parts = [np.real(np.fft.ifftn(1j * g[..., axis] * spectrum))
                 for axis in range(3)]
        return parts

    gu, gd = gradient_norm(up), gradient_norm(down)
    norm = lambda parts: np.sqrt(sum(p * p for p in parts))  # noqa: E731
    gt = [a + b for a, b in zip(gu, gd)]
    lda, corr = pbe_xc(up * AUTOA ** 3, down * AUTOA ** 3,
                       norm(gu) * AUTOA ** 4, norm(gd) * AUTOA ** 4,
                       norm(gt) * AUTOA ** 4)
    xc = float(np.sum((lda + corr) * ENERGY_DENSITY) * omega / points)
    kinetic = float(np.sum(tau) * omega / points)
    return {"kinetic": kinetic, "hartree": float(hartree),
            "local": float(local), "alpha_z": float(alpha),
            "ewald": float(ewald), "xc": xc}


def atomic_energy(setups, structure, encut):
    """``EATOM``: sum over atoms of the POTCAR EATOM minus its basis error."""
    from ..fields.structure import element_of
    from scipy.interpolate import CubicSpline

    total = 0.0
    for symbol, count in zip(structure.symbols, structure.counts):
        setup = setups[element_of(symbol)]
        value = float(setup.parameters.get("EATOM", 0.0))
        if setup.error_table is not None and encut:
            cutoffs, errors = setup.error_table
            value -= float(CubicSpline(cutoffs, errors)(encut))
        total += int(count) * value
    return total


def paw_total_energy(structure, grid, density, magnetization, tau, records,
                     setups, encut=None, lasph=True, relativistic=False,
                     entropy=0.0, one_center=None):
    r"""
    The PAW total energy from the grid fields and every atom's occupancies.

    Parameters
    ----------
    structure, grid : Structure, FieldGrid
    density, magnetization, tau : numpy.ndarray
        As :func:`grid_energy`.
    records : list
        Per atom, in file order, the list of its record sets (total, and the
        magnetization for a spin-polarized calculation).
    setups : dict
        ``{element: PawSetup}``.
    encut : float, optional
        Plane-wave cutoff (eV), for the basis-set error of ``EATOM``.
    lasph, relativistic : bool
        One-center xc treatment; see :class:`OneCenterEnergy`.
    entropy : float
        The smearing term ``EENTRO``, which no file holds.
    one_center : dict, optional
        ``{element: OneCenterEnergy}``, reused across calls.

    Returns
    -------
    dict
        Every term, ``one_center`` (summed over atoms) and ``total``; comparable
        to VASP's ``TOTEN``.
    """
    from ..fields.structure import element_of

    terms = grid_energy(structure, grid, density, magnetization, tau, setups)
    one_center = one_center or {}
    species = [element_of(symbol) for symbol, count
               in zip(structure.symbols, structure.counts)
               for _ in range(int(count))]
    parts = {"linear": 0.0, "hartree": 0.0, "xc": 0.0}
    for element, atom_records in zip(species, records):
        if element not in one_center:
            one_center[element] = OneCenterEnergy(setups[element], lasph=lasph,
                                                  relativistic=relativistic)
        result = one_center[element].energy(atom_records)
        for key in parts:
            parts[key] += result[key]
    terms["one_center_linear"] = parts["linear"]
    terms["one_center_hartree"] = parts["hartree"]
    terms["one_center_xc"] = parts["xc"]
    terms["atomic"] = atomic_energy(setups, structure, encut)
    terms["entropy"] = float(entropy)
    terms["total"] = sum(terms.values())
    return terms


def vasp_total_energy(directory, potcar=None, lasph=None, entropy=0.0,
                      encut=None):
    r"""
    VASP's ``TOTEN`` rebuilt from a calculation's files, eV.

    Reads ``CHGCAR`` (grid, magnetization, occupancy records), ``TAUCAR``,
    ``POTCAR`` and ``INCAR`` from ``directory`` and returns
    :func:`paw_total_energy`'s terms. The run must have written ``TAUCAR``
    (``LTAU = .TRUE.``, VASP >= 6.6) and, for an element with d channels,
    occupancies to L = 4 (``LMAXMIX = 4``).

    Parameters
    ----------
    directory : str
    potcar : str, optional
        A ``POTCAR`` to use when the directory has none.
    lasph : bool, optional
        Defaults to the INCAR's ``LASPH`` (VASP's default is false).
    entropy : float, optional
        ``EENTRO``, which no file but the ``OUTCAR`` holds.
    encut : float, optional
        Defaults to the INCAR's ``ENCUT``.
    """
    import os

    from ..fields import ChargeDensity, KineticEnergyDensity, SpinDensity
    from ..fields.vasp.augmentation import occupancy_arrays
    from ..fields.vasp.paw_setup import PawSetup
    from ..fields.vasp.volumetric import read_augmentation_blocks

    incar = {}
    path = os.path.join(directory, "INCAR")
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                key, value = line.split("=", 1)
                incar[key.strip().upper()] = value.split("!")[0].split("#")[0].strip()
    if lasph is None:
        lasph = incar.get("LASPH", ".FALSE.").upper().lstrip(".").startswith("T")
    if encut is None and "ENCUT" in incar:
        encut = float(incar["ENCUT"])
    spin = incar.get("ISPIN", "1").strip() == "2"
    chgcar = os.path.join(directory, "CHGCAR")
    if spin:
        density = SpinDensity.read(chgcar)
        n, m = np.asarray(density.data[0]), np.asarray(density.data[1])
    else:
        density = ChargeDensity.read(chgcar)
        n, m = np.asarray(density.data), None
    tau = np.asarray(KineticEnergyDensity.read(
        os.path.join(directory, "TAUCAR"), grid=density.grid).data)
    potcar = os.path.join(directory, "POTCAR") if potcar is None else potcar
    setups = {setup.element: setup for setup in PawSetup.from_file(potcar)}
    _, blocks = read_augmentation_blocks(chgcar)
    sets = 2 if spin else 1
    occupancies, lengths = occupancy_arrays(blocks, channels=sets)
    records = [[occupancies[a, s, :lengths[a]] for s in range(sets)]
               for a in range(len(lengths))]
    return paw_total_energy(density.structure, density.grid, n, m, tau,
                            records, setups, encut=encut, lasph=lasph,
                            entropy=entropy)
