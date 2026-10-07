# -*- coding: utf-8 -*-
# file: energy.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The PAW total energy in torch, and the forces as its exact gradient.

:mod:`poraque.physics.paw` evaluates VASP's total energy from the grid fields
and every atom's occupancies, in NumPy, and reproduces ``TOTEN`` to the meV.
This module evaluates the same expression in torch, so that it can be
differentiated, and closes the chain

.. math::

    \{\mathbf R\} \;\to\; v_{\rm ext},\ \rho_{\rm sup},\ \text{structure
    channels} \;\xrightarrow{\ \texttt{ext2paw}\ }\; n_T,\ m,\ R_a(cc';LM)
    \;\xrightarrow{\ \texttt{paw2tau}\ }\; \tilde\tau \;\to\; E ,

every step of which is a differentiable function of the atomic positions. The
forces are then :math:`\mathbf F_a = -\partial E/\partial\mathbf R_a` by
automatic differentiation: the exact gradient of the energy the operators
predict, conservative by construction. They are not VASP's Hellmann--Feynman
plus Pulay forces --- those need the wavefunctions --- and they are only as
good as the energy surface the operators learned.

Three pieces:

:class:`OneCenterTorch`
    One element's one-center energy, a function of its atoms' occupancies,
    built from a :class:`~poraque.physics.paw.OneCenterEnergy`. The Hartree
    term is a precomputed quadratic form per L; the exchange--correlation is
    evaluated as in NumPy.
:class:`GridEnergyTorch`
    The grid terms for a fixed cell and mesh, with the local potential and the
    partial core rebuilt from the atomic positions by structure factors.
:class:`PawEnergyChain`
    The whole chain for one structure: the operators, the energy, the forces.
"""

import numpy as np
import torch

from ..physics.paw import (
    AUTOA,
    ENERGY_DENSITY,
    FELECT,
    SQRT4PI,
    OneCenterEnergy,
    radial_hartree_potential,
)
from ..physics.xc import lda_xc, pbe_xc


def _radial_gradient(f, r):
    """VASP's ``GRAD`` (second order) along the last axis, in torch."""
    n = r.shape[-1]
    h = float(np.log(float(r[-1]) / float(r[0])) / (n - 1))
    d = torch.cat([f[..., 1:2] - f[..., 0:1],
                   0.5 * (f[..., 2:] - f[..., :-2]),
                   f[..., -1:] - f[..., -2:-1]], dim=-1)
    return d / (h * r)


class OneCenterTorch(torch.nn.Module):
    """
    One element's one-center energy in torch.

    Parameters
    ----------
    numpy_energy : OneCenterEnergy
        The NumPy object, whose precomputed radial tables are taken over.
    dtype : torch.dtype, optional
    """

    def __init__(self, numpy_energy, dtype=torch.float64):
        super().__init__()
        oc = numpy_energy
        self.lasph = oc.lasph
        self.relativistic = oc.relativistic
        self.lmax = oc.lmax
        lm_total = (oc.lmax + 1) ** 2
        rows = [(k, a, b, L, m) for k, (a, b, L, m) in enumerate(oc.layout)]
        self.record_length = len(oc.layout)
        p_ae = np.zeros((self.record_length, oc.r.size))
        p_ps = np.zeros_like(p_ae)
        onehot = np.zeros((self.record_length, lm_total))
        linear = np.zeros(self.record_length)
        for k, a, b, L, m in rows:
            if L > oc.lmax:
                continue
            p_ae[k] = oc.ae[a] * oc.ae[b]
            p_ps[k] = oc.ps[a] * oc.ps[b] + oc.qpaw[a, b, L] * oc.aug[L]
            onehot[k, L * L + m] = 1.0
            if L == 0 and oc.channels[a] == oc.channels[b]:
                linear[k] = SQRT4PI * oc.dbar[a, b]
        degrees = np.concatenate([[L] * (2 * L + 1) for L in range(oc.lmax + 1)])
        # E_H of one LM component is (1/2) rho^T S_L rho, S_L the symmetric
        # part of diag(w) A_L, A_L the (linear) RAD_POT_HAR potential map.
        forms = []
        for L in range(oc.lmax + 1):
            columns = np.stack([radial_hartree_potential(L, oc.r, unit)
                                for unit in np.eye(oc.r.size)], axis=1)
            b_matrix = oc.weights[:, None] * columns
            forms.append(0.5 * (b_matrix + b_matrix.T))
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype)  # noqa: E731
        self.register_buffer("r", tensor(oc.r))
        self.register_buffer("weights", tensor(oc.weights))
        self.register_buffer("p_ae", tensor(p_ae))
        self.register_buffer("p_ps", tensor(p_ps))
        self.register_buffer("onehot", tensor(onehot))
        self.register_buffer("linear", tensor(linear))
        self.register_buffer("forms", tensor(np.stack(forms)))
        self.register_buffer("degrees", torch.as_tensor(degrees))
        self.register_buffer("core_ae", tensor(oc.core_ae))
        self.register_buffer("core_ps", tensor(oc.core_ps))
        self.core_xc = float(oc.core_xc)
        if self.lasph:
            self.register_buffer("ylm", tensor(oc.ylm))
            self.register_buffer("ylm_gradient", tensor(oc.ylm_gradient))
            self.register_buffer("directions", tensor(oc.directions))
            self.register_buffer("angular_weights", tensor(oc.angular_weights))

    def multipoles(self, records):
        """``(A, K)`` records to ``(A, LM, nr)`` AE and pseudo multipoles."""
        ae = torch.einsum("ak,kl,kr->alr", records, self.onehot, self.p_ae)
        ps = torch.einsum("ak,kl,kr->alr", records, self.onehot, self.p_ps)
        return ae, ps

    def hartree(self, rho):
        """(1/2) sum_LM rho_LM^T S_L rho_LM, per atom."""
        forms = self.forms[self.degrees]                  # (LM, nr, nr)
        return 0.5 * torch.einsum("alr,lrs,als->a", rho, forms, rho)

    def _xc_lasph(self, total, magnetization, core):
        r = self.r
        r2 = r * r
        n_tot = torch.einsum("pl,alr->apr", self.ylm, total) / r2 + core
        g_ang = torch.einsum("pls,alr->apsr", self.ylm_gradient, total) / (r2 * r)
        dn = _radial_gradient(n_tot, r)
        directions = self.directions[None, :, :, None]

        def norm(radial_part, angular_part):
            vector = radial_part[:, :, None, :] * directions + angular_part
            return torch.sqrt((vector ** 2).sum(2) + 1e-300)

        if magnetization is None:
            g = norm(dn, g_ang)
            lda, corr = pbe_xc(0.5 * n_tot * AUTOA ** 3, 0.5 * n_tot * AUTOA ** 3,
                               0.5 * g * AUTOA ** 4, 0.5 * g * AUTOA ** 4,
                               g * AUTOA ** 4, relativistic=self.relativistic)
        else:
            n_mag = torch.einsum("pl,alr->apr", self.ylm, magnetization) / r2
            g_mag = torch.einsum("pls,alr->apsr", self.ylm_gradient,
                                 magnetization) / (r2 * r)
            up = torch.clamp(0.5 * (n_tot + n_mag), min=1e-7)
            down = torch.clamp(0.5 * (n_tot - n_mag), min=1e-7)
            dm = _radial_gradient(n_mag, r)
            g_up = norm(0.5 * (dn + dm), 0.5 * (g_ang + g_mag))
            g_dn = norm(0.5 * (dn - dm), 0.5 * (g_ang - g_mag))
            g = norm(dn, g_ang)
            lda, corr = pbe_xc(up * AUTOA ** 3, down * AUTOA ** 3,
                               g_up * AUTOA ** 4, g_dn * AUTOA ** 4,
                               g * AUTOA ** 4, relativistic=self.relativistic)
        e = (lda + corr) * ENERGY_DENSITY
        return torch.einsum("p,apr,r->a", self.angular_weights, e,
                            r2 * self.weights)

    def _xc_spherical(self, total, magnetization, core):
        r = self.r
        r2 = r * r
        w = self.weights
        n0 = total[:, 0] / (SQRT4PI * r2) + core
        m0 = (torch.zeros_like(n0) if magnetization is None
              else magnetization[:, 0] / (SQRT4PI * r2))
        up, down = 0.5 * (n0 + m0), 0.5 * (n0 - m0)

        def e_lda(u, d):
            return lda_xc(torch.clamp(u, min=0) * AUTOA ** 3,
                          torch.clamp(d, min=0) * AUTOA ** 3) * ENERGY_DENSITY

        energy = (4.0 * np.pi * r2 * e_lda(up, down) * w).sum(-1)
        if total.shape[1] > 1:
            rho_t = total[:, 1:] / r2
            rho_m = (torch.zeros_like(rho_t) if magnetization is None
                     else magnetization[:, 1:] / r2)
            rho_u, rho_d = 0.5 * (rho_t + rho_m), 0.5 * (rho_t - rho_m)
            hu = torch.clamp(1e-3 * up, min=1e-12)
            hd = torch.clamp(1e-3 * down, min=1e-12)
            e00 = e_lda(up, down)
            euu = (e_lda(up + hu, down) - 2 * e00 + e_lda(up - hu, down)) / hu ** 2
            edd = (e_lda(up, down + hd) - 2 * e00 + e_lda(up, down - hd)) / hd ** 2
            eud = (e_lda(up + hu, down + hd) - e_lda(up + hu, down - hd)
                   - e_lda(up - hu, down + hd) + e_lda(up - hu, down - hd)) \
                / (4 * hu * hd)
            taylor = 0.5 * (euu * (rho_u ** 2).sum(1) + edd * (rho_d ** 2).sum(1)
                            + 2 * eud * (rho_u * rho_d).sum(1))
            energy = energy + (r2 * taylor * w).sum(-1)
        dn = _radial_gradient(n0, r)
        dm = _radial_gradient(m0, r)
        _, corr = pbe_xc(torch.clamp(up, min=1e-7) * AUTOA ** 3,
                         torch.clamp(down, min=1e-7) * AUTOA ** 3,
                         torch.abs(0.5 * (dn + dm)) * AUTOA ** 4,
                         torch.abs(0.5 * (dn - dm)) * AUTOA ** 4,
                         torch.abs(dn) * AUTOA ** 4)
        return energy + (4.0 * np.pi * r2 * corr * ENERGY_DENSITY * w).sum(-1)

    def xc(self, total, magnetization, core):
        if self.lasph:
            return self._xc_lasph(total, magnetization, core)
        return self._xc_spherical(total, magnetization, core)

    def forward(self, records):
        """
        Per-atom one-center energy, eV.

        Parameters
        ----------
        records : torch.Tensor
            ``(A, S, K)``: each atom's record sets (total, magnetization), in
            physical units and file order, for atoms of this element.
        """
        records = records[..., :self.record_length]
        total = records[:, 0]
        ae, ps = self.multipoles(total)
        magnetization = None
        if records.shape[1] > 1:
            magnetization = self.multipoles(records[:, 1])
        linear = total @ self.linear
        hartree = self.hartree(ae) - self.hartree(ps)
        xc = (self.xc(ae, None if magnetization is None else magnetization[0],
                      self.core_ae)
              - self.xc(ps, None if magnetization is None else magnetization[1],
                        self.core_ps)
              - self.core_xc)
        return linear + hartree + xc


class GridEnergyTorch(torch.nn.Module):
    """
    The grid terms of the PAW energy for one cell and mesh, in torch.

    Parameters
    ----------
    grid : FieldGrid
    elements : list of str
        The element of every atom, in order.
    setups : dict
        ``{element: PawSetup}``.
    dtype : torch.dtype, optional
    """

    def __init__(self, grid, elements, setups, dtype=torch.float64):
        super().__init__()
        from ..physics.paw import _reciprocal_table, _reciprocal_vectors

        self.shape = tuple(int(n) for n in grid.shape)
        self.omega = float(grid.volume)
        self.points = int(np.prod(self.shape))
        k, g = _reciprocal_vectors(grid)
        g2 = (g ** 2).sum(-1)
        gnorm = np.sqrt(g2)
        nonzero = g2 > 1e-12
        safe = np.where(nonzero, g2, 1.0)
        self.types = sorted(set(elements))
        type_index = [self.types.index(e) for e in elements]
        local, core = [], []
        for element in self.types:
            setup = setups[element]
            table = setup.reciprocal["local part"]
            psgmax, values = float(table[0]), table[1:]
            spline = _reciprocal_table(values, psgmax)
            mask = nonzero & (gnorm < psgmax - psgmax / 1000.0)
            short = np.where(mask, spline(np.where(mask, gnorm, 0.0))
                             - 4.0 * np.pi * setup.zval * FELECT / safe, 0.0)
            local.append(short)
            partial = setup.reciprocal.get("core charge-density (partial)")
            if partial is not None and np.any(partial):
                within = gnorm < psgmax - 3.0 * psgmax / 1000.0
                pspline = _reciprocal_table(partial, psgmax)
                core.append(np.where(within, pspline(np.where(within, gnorm,
                                                              0.0)), 0.0))
            else:
                core.append(np.zeros_like(gnorm))
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype)  # noqa: E731
        self.register_buffer("k", tensor(k))
        self.register_buffer("g", tensor(g))
        self.register_buffer("inverse_g2", tensor(np.where(nonzero, 1.0 / safe,
                                                           0.0)))
        self.register_buffer("local_tables", tensor(np.stack(local)))
        self.register_buffer("core_tables", tensor(np.stack(core)))
        self.register_buffer("type_index", torch.as_tensor(type_index))

    def structure_factors(self, frac):
        """``(T, N1, N2, N3)`` complex S_t(G) of fractional positions ``(A, 3)``."""
        phase = -2.0 * np.pi * torch.einsum("xyzc,ac->axyz", self.k, frac)
        per_atom = torch.complex(torch.cos(phase), torch.sin(phase))
        factors = torch.zeros((len(self.types),) + self.shape,
                              dtype=per_atom.dtype, device=frac.device)
        return factors.index_add(0, self.type_index.to(frac.device), per_atom)

    def forward(self, frac, density, magnetization, tau):
        """
        Grid terms, eV: ``kinetic``, ``hartree``, ``local``, ``xc``.

        Parameters
        ----------
        frac : torch.Tensor
            ``(A, 3)`` fractional positions.
        density, magnetization, tau : torch.Tensor
            ``n_T``, ``m`` (or ``None``) in e/Å³, :math:`\\tilde\\tau` in eV/Å³.
        """
        factors = self.structure_factors(frac)
        coefficients = torch.fft.fftn(density * self.omega) / self.points
        hartree = (2.0 * np.pi * FELECT / self.omega
                   * (coefficients.abs() ** 2 * self.inverse_g2).sum())
        local = ((factors * torch.conj(coefficients)[None]).real
                 * self.local_tables).sum() / self.omega
        core_g = (factors * self.core_tables).sum(0) / self.omega
        core = torch.fft.ifftn(core_g).real * self.points
        m = torch.zeros_like(density) if magnetization is None else magnetization
        up = torch.clamp(0.5 * (density + m) + 0.5 * core, min=1e-10)
        down = torch.clamp(0.5 * (density - m) + 0.5 * core, min=1e-10)

        def gradient(field):
            spectrum = torch.fft.fftn(field)
            return torch.stack([torch.fft.ifftn(1j * self.g[..., axis]
                                                * spectrum).real
                                for axis in range(3)])

        gu, gd = gradient(up), gradient(down)
        norm = lambda v: torch.sqrt((v ** 2).sum(0) + 1e-300)  # noqa: E731
        lda, corr = pbe_xc(up * AUTOA ** 3, down * AUTOA ** 3,
                           norm(gu) * AUTOA ** 4, norm(gd) * AUTOA ** 4,
                           norm(gu + gd) * AUTOA ** 4)
        xc = ((lda + corr) * ENERGY_DENSITY).sum() * self.omega / self.points
        kinetic = tau.sum() * self.omega / self.points
        return {"kinetic": kinetic, "hartree": hartree, "local": local,
                "xc": xc}


class PawEnergyChain:
    r"""
    Total energy and forces of one structure, through the trained operators.

    The cell and the grid are fixed; the atomic positions are the variables.
    Every input field the density operator reads --- the external potential,
    the atomic superposition, the per-element structure channels --- is a sum
    over atoms of a per-species kernel times a phase, so it is rebuilt here as
    :math:`\sum_t K_t(\mathbf G)\,S_t(\mathbf G;\{\mathbf R\})`, with the
    kernels computed once by the very code that built the training inputs
    (and checked against it at construction).

    Parameters
    ----------
    density_operator : FieldOperator
        An ``ext2paw`` operator: the pseudo-density (plus compensation charge),
        its magnetization and every atom's occupancies.
    kinetic_operator : FieldOperator
        A ``paw2tau`` operator (density and occupancies to
        :math:`\tilde\tau`), or a ``chg2tau`` one.
    structure : Structure
        The geometry; its cell is kept, its positions are the starting point.
    grid : FieldGrid
        The mesh the operators are evaluated on.
    potcar_dir : str
        The POTCAR library (``<dir>/<element>/POTCAR``) the operators were
        trained with.
    lasph : bool, optional
        One-center xc treatment, as the training data's ``LASPH``.
    encut : float, optional
        Plane-wave cutoff of the training data, for ``EATOM``'s basis error.
    entropy : float, optional
        A smearing term to add, eV.
    precision : str, optional
        ``"float64"`` (default) runs the operators in double precision, which
        the forces need; ``"float32"`` keeps them as trained.
    device : str or torch.device, optional
    normalize_density : bool, optional
        Rescale the predicted :math:`\tilde n + \hat n` to the valence
        electron count, which it integrates to exactly in VASP. On by default:
        the electrostatic terms are thousands of eV, and on a 32-atom Pt cell
        a 0.07 % count error alone moved the energy by 8 eV.
    """

    def __init__(self, density_operator, kinetic_operator, structure, grid,
                 potcar_dir, lasph=True, encut=None, entropy=0.0,
                 precision="float64", device=None, normalize_density=True):
        import os

        from ..fields import element_of
        from ..fields.external import ExternalPotential
        from ..fields.vasp.paw_setup import PawSetup
        from ..fields.vasp.poscar import symbol_to_z
        from ..fields.vasp.potcar import Potcar
        from ..physics.energy import ewald_energy
        from ..physics.paw import atomic_energy

        self.density_operator = density_operator
        self.kinetic_operator = kinetic_operator
        self.device = torch.device(device) if device is not None \
            else density_operator.device
        for operator in (density_operator, kinetic_operator):
            operator.model.eval()
            if precision == "float64":
                operator.set_precision("float64")
            operator.model.to(self.device)
        self.structure = structure
        self.grid = grid
        self.elements = [element_of(symbol) for symbol, count
                         in zip(structure.symbols, structure.counts)
                         for _ in range(int(count))]
        types = sorted(set(self.elements))
        self.types = types
        self.setups = {element: PawSetup.from_file(
            os.path.join(potcar_dir, element, "POTCAR"))[0]
            for element in types}
        potcar = Potcar.from_library(potcar_dir, types, parse_tables=True)
        self.grid_energy = GridEnergyTorch(grid, self.elements,
                                           self.setups).to(self.device)
        self.one_center = {
            element: OneCenterTorch(OneCenterEnergy(self.setups[element],
                                                    lasph=lasph)).to(self.device)
            for element in types}

        cell = np.asarray(structure.cell, dtype=float)

        # Per-species spectra, before any real projection (see
        # fields.external.potcar_local_spectrum on why that matters).
        from ..fields.external import potcar_local_spectrum

        entries = {entry.element: entry for entry in potcar}
        complex_tensor = lambda x: torch.as_tensor(  # noqa: E731
            np.asarray(x, dtype=complex), dtype=torch.complex128,
            device=self.device)
        self.potential_kernels = torch.stack([
            complex_tensor(potcar_local_spectrum(grid, entries[element]))
            for element in types])
        library = density_operator.baseline
        spec = density_operator.structure_input or {}
        self.structure_input = spec
        magnitude = np.sqrt(grid.get_g2())
        self.density_kernels = torch.stack([
            complex_tensor(library.lookup(element).evaluate(magnitude)
                           / grid.volume) for element in types])

        self.species = torch.as_tensor(
            [[symbol_to_z(element) for element in self.elements]],
            device=self.device)
        self.cell = torch.as_tensor(cell, dtype=torch.float64,
                                    device=self.device)
        charges = {element: self.setups[element].zval for element in types}
        from ..fields.constants import COULOMB_CONSTANT_EV_ANGSTROM

        self._coulomb_ratio = FELECT / COULOMB_CONSTANT_EV_ANGSTROM
        self._charges = charges
        self.normalize_density = bool(normalize_density)
        self.electrons = float(sum(charges[element]
                                   for element in self.elements))
        zval_total = sum(charges[element] for element in self.elements)
        alpha = sum(float(self.setups[element].reciprocal["local part"][1])
                    for element in self.elements) * zval_total / grid.volume
        self._constant = (alpha + atomic_energy(self.setups, structure, encut)
                          + float(entropy))
        self._ewald = lambda s: ewald_energy(s, charges) * self._coulomb_ratio

        # The rebuilt inputs must be the training inputs: checked once.
        from ..fields.atomic import structure_channels

        frac = torch.as_tensor(np.asarray(structure.scaled_positions),
                               dtype=torch.float64, device=self.device)
        channels, superposition = self.input_fields(frac)
        checks = [(channels[0], ExternalPotential.from_potcar_tables(
                      structure, grid, potcar).data, "external potential"),
                  (superposition, structure_channels(
                      structure, grid, library, superposition=True,
                      elements=[])[0], "atomic superposition")]
        for rebuilt, direct, name in checks:
            rebuilt = rebuilt.detach().cpu().numpy()
            error = np.abs(rebuilt - direct).max() / max(np.abs(direct).max(),
                                                         1e-12)
            if error > 1e-8:
                raise RuntimeError(f"the rebuilt {name} differs from the "
                                   f"direct one by {error:.2e} (relative).")

    # ------------------------------------------------------------------ #
    def input_fields(self, frac):
        """The density operator's physical input channels, and rho_sup."""
        factors = self.grid_energy.structure_factors(frac)
        points = factors[0].numel()

        def real_space(spectrum):
            return torch.fft.ifftn(spectrum).real * points

        potential = real_space((self.potential_kernels * factors).sum(0))
        per_element = {element: real_space(self.density_kernels[index]
                                           * factors[index])
                       for index, element in enumerate(self.types)}
        superposition = sum(per_element.values())
        channels = [potential]
        if self.structure_input:
            if self.structure_input.get("superposition", True):
                channels.append(superposition)
            for element in self.structure_input.get("elements", []):
                channels.append(per_element.get(element,
                                                torch.zeros_like(potential)))
        return torch.stack(channels), superposition

    def fields(self, frac):
        """
        The predicted fields at ``frac``: ``(n_T, m or None, records, tau)``.

        ``records`` is ``(A, S, K)`` in physical units, file order.
        """
        dens = self.density_operator
        dtype = next(dens.model.parameters()).dtype
        inputs, superposition = self.input_fields(frac)
        normalized = dens.input_transform(inputs)
        sites = {"species": self.species,
                 "positions": frac[None].to(dtype),
                 "atom_mask": torch.ones_like(self.species, dtype=torch.bool)}
        field, occupancy = dens.model(normalized[None].to(dtype),
                                      self.cell[None].to(dtype), sites=sites)
        physical = dens.target_transform.inverse(field.double())[0]
        density = physical[0] + superposition
        if self.normalize_density:
            volume_element = self.grid.volume / density.numel()
            density = density * (self.electrons
                                 / (density.sum() * volume_element))
        magnetization = physical[1] if physical.shape[0] > 1 else None
        records = dens.site_transform.inverse(occupancy.double(),
                                              self.species)[0]

        kin = self.kinetic_operator
        kdtype = next(kin.model.parameters()).dtype
        stack = [density] + ([magnetization] if magnetization is not None
                             else [])
        source = torch.stack(stack[:kin.in_channels])
        knormalized = kin.input_transform(source)
        if kin.task.site_input is not None:
            sets = kin.in_channels
            ksites = {"records": kin.site_transform.normalize(
                          records[None, :, :sets], self.species).to(kdtype),
                      "species": self.species,
                      "positions": frac[None].to(kdtype),
                      "atom_mask": sites["atom_mask"]}
            output = kin.model(knormalized[None].to(kdtype),
                               self.cell[None].to(kdtype), sites=ksites)
        else:
            output = kin.model(knormalized[None].to(kdtype),
                               self.cell[None].to(kdtype))
        tau = kin.target_transform.inverse(output.double())[0, 0]
        return density, magnetization, records, tau

    def energy(self, frac=None, components=False):
        """
        The total energy at fractional positions ``frac`` (``(A, 3)``), eV.

        Comparable to VASP's ``TOTEN`` of the training data.
        """
        if frac is None:
            frac = torch.as_tensor(np.asarray(self.structure.scaled_positions),
                                   dtype=torch.float64, device=self.device)
        density, magnetization, records, tau = self.fields(frac)
        terms = self.grid_energy(frac, density, magnetization, tau)
        one_center = torch.zeros((), dtype=torch.float64, device=self.device)
        for element in self.types:
            index = [a for a, e in enumerate(self.elements) if e == element]
            one_center = one_center + self.one_center[element](
                records[index]).sum()
        moved = self._structure_at(frac)
        total = (sum(terms.values()) + one_center + self._ewald(moved)
                 + self._constant)
        if components:
            parts = {key: float(value.detach()) for key, value in terms.items()}
            parts.update(one_center=float(one_center.detach()),
                         ewald=float(self._ewald(moved)),
                         constant=self._constant, total=float(total.detach()))
            return total, parts
        return total

    def _structure_at(self, frac):
        from ..fields.structure import Structure

        s = self.structure
        return Structure(np.asarray(s.cell), list(s.symbols), list(s.counts),
                         frac.detach().cpu().numpy() % 1.0)

    def forces(self, frac=None):
        """
        :math:`-\\partial E/\\partial\\mathbf R`, eV/Å, ``(A, 3)``, with the
        energy: ``(energy, forces)``.
        """
        from ..physics.forces import ewald_forces

        if frac is None:
            frac = np.asarray(self.structure.scaled_positions, dtype=float)
        frac = torch.as_tensor(np.asarray(frac), dtype=torch.float64,
                               device=self.device).requires_grad_(True)
        energy = self.energy(frac)
        (gradient,) = torch.autograd.grad(energy, frac)
        inverse = np.linalg.inv(np.asarray(self.structure.cell, dtype=float))
        forces = -gradient.cpu().numpy() @ inverse.T
        forces += ewald_forces(self._structure_at(frac),
                               self._charges) * self._coulomb_ratio
        return float(energy.detach()), forces
