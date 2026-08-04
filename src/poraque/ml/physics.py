# -*- coding: utf-8 -*-
# file: physics.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Differentiable DFT operators and physics-informed loss terms.

This module is the executable half of the PI-FNO plan in ``docs/notes/pi_fno.md``.
Everything here is written in PyTorch and is differentiable with respect to the
predicted field, so each function can be dropped straight into a training
objective.

Conventions
-----------
Fields arrive in the units used by :mod:`poraque.fields` — lengths in Å,
:math:`\rho` in e/Å³, potentials in eV, :math:`\tau` in eV/Å³. Functionals with
a natural atomic-unit form (Thomas-Fermi, von Weizsäcker) convert internally
and return eV/Å³, so callers never mix unit systems.

Differential operators are **spectral**: derivatives are taken as
:math:`\nabla \to i\mathbf{G}` on the FFT mesh. On a plane-wave grid this is
exact for the band-limited fields DFT produces, whereas a finite-difference
stencil would leave a discretization error that the physics losses would then
try — wrongly — to attribute to the network.
"""

import numpy as np
import torch

from ..fields.constants import (
    BOHR_TO_ANGSTROM,
    C_TF,
    COULOMB_CONSTANT_EV_ANGSTROM,
    HARTREE_TO_EV,
)

#: eV/Å³ per Hartree/Bohr³.
_HA_BOHR3_TO_EV_ANG3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
#: eV per Hartree, for potentials (functional derivatives of energy densities).
_HA_TO_EV = HARTREE_TO_EV


# ---------------------------------------------------------------------- #
# Cell metric
#
# These two helpers are the only place the code inverts or takes the
# determinant of a lattice matrix. Both run on the **CPU in float64** and move
# only the 3x3 (or scalar) result to the target device. Two reasons:
#
#   * Accuracy. A skewed cell has a poorly conditioned inverse, and every G
#     vector on the grid inherits that error; float64 costs nothing here.
#   * Portability. Apple's MPS backend supports neither float64 nor
#     `linalg.det`, so doing this work on-device would make the whole package
#     unusable on Apple Silicon. See poraque.ml.device.
#
# The cost is O(1) per structure against an O(N log N) FFT, i.e. unmeasurable.
# ---------------------------------------------------------------------- #
def cell_reciprocal(cell, device=None, dtype=torch.float32):
    r"""
    Reciprocal lattice :math:`2\pi\,(\mathbf{A}^{-1})^{T}` for a batch of cells.

    Parameters
    ----------
    cell : torch.Tensor
        ``(B, 3, 3)`` or ``(3, 3)`` lattice vectors in Å.
    device : torch.device, optional
        Where to place the result; defaults to the input's device.
    dtype : torch.dtype, optional
        Result dtype.

    Returns
    -------
    torch.Tensor
        ``(B, 3, 3)`` reciprocal vectors in Å⁻¹, rows ``b1, b2, b3``.
    """
    cell = cell if cell.dim() == 3 else cell.unsqueeze(0)
    device = device if device is not None else cell.device
    # Move to host BEFORE widening: casting an MPS tensor to float64 in place
    # raises, because the dtype is unrepresentable on that backend.
    host = cell.detach().cpu().to(torch.float64)
    reciprocal = 2.0 * np.pi * torch.linalg.inv(host).transpose(-1, -2)
    return reciprocal.to(device=device, dtype=dtype)


def cell_volume(cell, device=None, dtype=torch.float32):
    """
    Absolute cell volume :math:`|\\det \\mathbf{A}|` for a batch of cells.

    Parameters
    ----------
    cell : torch.Tensor
        ``(B, 3, 3)`` or ``(3, 3)`` lattice vectors in Å.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor
        ``(B,)`` volumes in Å³.
    """
    cell = cell if cell.dim() == 3 else cell.unsqueeze(0)
    device = device if device is not None else cell.device
    # Host first, then widen -- see cell_reciprocal.
    host = cell.detach().cpu().to(torch.float64)
    return torch.linalg.det(host).abs().to(device=device, dtype=dtype)


# ---------------------------------------------------------------------- #
# Spectral differential operators
# ---------------------------------------------------------------------- #
def reciprocal_vectors(cell, shape, device=None, dtype=torch.float32):
    r"""
    Cartesian reciprocal-lattice vectors of the FFT mesh.

    Parameters
    ----------
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å (rows are ``a1, a2, a3``).
    shape : tuple of int
        Spatial shape ``(Nx, Ny, Nz)``.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor
        ``(B, 3, Nx, Ny, Nz)`` with the Cartesian components of
        :math:`\mathbf{G}`, in Å⁻¹.
    """
    cell = cell if cell.dim() == 3 else cell.unsqueeze(0)
    device = device or cell.device
    reciprocal = cell_reciprocal(cell, device=device, dtype=dtype)  # (B, 3, 3)

    frequencies = [
        torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype) for n in shape
    ]
    mesh = torch.meshgrid(*frequencies, indexing="ij")           # 3 x (Nx,Ny,Nz)
    integers = torch.stack(mesh, dim=0)                          # (3, Nx, Ny, Nz)

    # G_alpha = sum_j m_j b_{j alpha}
    return torch.einsum("bja,jxyz->baxyz", reciprocal, integers)


def spectral_gradient(field, cell):
    r"""
    Gradient :math:`\nabla f` via :math:`\nabla \to i\mathbf{G}`.

    Parameters
    ----------
    field : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` or ``(B, Nx, Ny, Nz)`` real field.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.

    Returns
    -------
    torch.Tensor
        ``(B, 3, Nx, Ny, Nz)`` gradient, in field-units per Å.
    """
    values = field.squeeze(1) if field.dim() == 5 else field
    shape = tuple(values.shape[-3:])
    g = reciprocal_vectors(cell, shape, values.device, values.dtype)

    transformed = torch.fft.fftn(values.to(torch.complex64), dim=(-3, -2, -1))
    return torch.fft.ifftn(1j * g * transformed.unsqueeze(1),
                           dim=(-3, -2, -1)).real


def spectral_laplacian(field, cell):
    r"""
    Laplacian :math:`\nabla^2 f` via :math:`\nabla^2 \to -|\mathbf{G}|^2`.

    Parameters
    ----------
    field : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` or ``(B, Nx, Ny, Nz)``.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.

    Returns
    -------
    torch.Tensor
        Same shape as ``field``, in field-units per Å².
    """
    squeezed = field.dim() == 5
    values = field.squeeze(1) if squeezed else field
    shape = tuple(values.shape[-3:])
    g2 = reciprocal_vectors(cell, shape, values.device, values.dtype).pow(2).sum(1)

    transformed = torch.fft.fftn(values.to(torch.complex64), dim=(-3, -2, -1))
    result = torch.fft.ifftn(-g2 * transformed, dim=(-3, -2, -1)).real
    return result.unsqueeze(1) if squeezed else result


def integrate(field, cell):
    r"""
    Integrate a field over the cell, :math:`\int f\,d^3r`.

    Parameters
    ----------
    field : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` or ``(B, Nx, Ny, Nz)``.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.

    Returns
    -------
    torch.Tensor
        ``(B,)`` integrals.
    """
    values = field.squeeze(1) if field.dim() == 5 else field
    volume = cell_volume(cell, device=values.device, dtype=values.dtype)
    n_points = float(np.prod(values.shape[-3:]))
    return values.sum(dim=(-3, -2, -1)) * volume / n_points


def volume_element(field, cell):
    r"""
    Volume per grid point, :math:`\Delta v = \Omega / (N_1 N_2 N_3)`.

    Returned shaped ``(B, 1, 1, 1, 1)`` so it broadcasts against a field.
    """
    values = field.squeeze(1) if field.dim() == 5 else field
    volume = cell_volume(cell, device=values.device, dtype=values.dtype)
    n_points = float(np.prod(values.shape[-3:]))
    return (volume / n_points).reshape(-1, *([1] * (field.dim() - 1)))


# ---------------------------------------------------------------------- #
# Functional derivatives
# ---------------------------------------------------------------------- #
def functional_derivative(energy, density, cell, create_graph=False,
                          retain_graph=None):
    r"""
    Functional derivative :math:`\delta F/\delta\rho` of a scalar functional.

    ``energy`` is any differentiable map from a density field to a per-structure
    scalar, so this covers an analytic functional, a neural network, or a
    composition of both. Autograd supplies the derivative in one backward pass:
    no finite differences, and no derivative to work out by hand — which is the
    traditional obstacle to proposing a new kinetic energy functional.

    Parameters
    ----------
    energy : callable
        ``rho -> (B,)`` tensor. Must be differentiable with respect to ``rho``.
    density : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` density in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.
    create_graph : bool, optional
        Keep the graph so the derivative can itself appear in a loss that is
        backpropagated. Required for physics-informed training; costs memory.
    retain_graph : bool, optional
        Passed through to :func:`torch.autograd.grad`; defaults to
        ``create_graph``.

    Returns
    -------
    torch.Tensor
        ``δF/δρ`` with the same shape as ``density``, in
        ``[F] / (e/Å³) / Å³`` — for :math:`T_s` in eV that is eV per electron,
        i.e. a potential.

    Notes
    -----
    **The discretisation factor is not optional.** With
    :math:`F = \sum_i f_i\,\Delta v` autograd returns
    :math:`\partial F/\partial\rho_i`, whereas the functional derivative is
    defined through :math:`\delta F = \int (\delta F/\delta\rho)\,\delta\rho\,
    \mathrm{d}^3 r`. Matching the two gives

    .. math:: \frac{\delta F}{\delta \rho}(\mathbf r_i)
              = \frac{1}{\Delta v}\,\frac{\partial F}{\partial \rho_i}.

    Omitting the :math:`1/\Delta v` rescales the whole potential by the number
    of grid points — a factor of :math:`3\times10^4` on a 32³ mesh — silently,
    with nothing raised. Every consumer here goes through this function so the
    factor is applied exactly once.
    """
    if not torch.is_tensor(density):
        raise TypeError("density must be a torch.Tensor")

    # A leaf that requires grad, without disturbing the caller's tensor.
    rho = density.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        value = energy(rho)
        if value.dim() == 0:
            value = value.reshape(1)
        gradient, = torch.autograd.grad(
            value.sum(), rho, create_graph=create_graph,
            retain_graph=create_graph if retain_graph is None else retain_graph,
        )

    return gradient / volume_element(density, cell)


def kinetic_potential(tau, density, cell, create_graph=False):
    r"""
    Kinetic potential :math:`\delta T_s/\delta\rho` from a KEDF.

    ``tau`` maps a density to a kinetic energy *density*; the integral over the
    cell gives :math:`T_s`, and its functional derivative is the quantity
    orbital-free DFT actually consumes.

    Parameters
    ----------
    tau : callable
        ``rho -> tau(rho)``, both ``(B, 1, Nx, Ny, Nz)``; :math:`\tau` in
        eV/Å³. Accepts the analytic functionals of this module and any learned
        operator.
    density : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` density in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.
    create_graph : bool, optional
        Keep the graph for physics-informed training.

    Returns
    -------
    torch.Tensor
        :math:`\delta T_s/\delta\rho` in eV, same shape as ``density``.

    Examples
    --------
    Recovering the analytic Thomas-Fermi potential from its energy density:

    >>> import torch
    >>> from poraque.ml.physics import kinetic_potential, thomas_fermi_tau
    >>> rho = torch.rand(1, 1, 8, 8, 8) * 0.3 + 0.05
    >>> cell = torch.eye(3).unsqueeze(0) * 6.0
    >>> potential = kinetic_potential(thomas_fermi_tau, rho, cell)
    >>> potential.shape
    torch.Size([1, 1, 8, 8, 8])

    Notes
    -----
    A model with small pointwise error in :math:`\tau` can still have a poor
    derivative, because differentiation amplifies high-frequency error: a
    ripple of amplitude :math:`\epsilon` at wavevector :math:`G` contributes
    :math:`\epsilon` to the value but :math:`\epsilon G` to the gradient.
    Validate the derivative directly rather than inferring it from the
    :math:`\tau` error.
    """
    return functional_derivative(
        lambda rho: integrate(tau(rho), cell), density, cell,
        create_graph=create_graph,
    )


def operator_kinetic_potential(operator, density, cell, create_graph=False):
    r"""
    :math:`\delta T_s/\delta\rho` from a trained ``chg2tau``
    :class:`~poraque.ml.training.FieldOperator`.

    Handles the normalisation round trip: the operator consumes and returns
    normalised fields, while the derivative must be taken with respect to the
    **physical** density, or its magnitude is meaningless.

    Parameters
    ----------
    operator : FieldOperator
        A trained ``chg2tau`` operator.
    density : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` physical density in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.
    create_graph : bool, optional
        Keep the graph so this can enter a differentiable loss.

    Returns
    -------
    torch.Tensor
        :math:`\delta T_s/\delta\rho` in eV.
    """
    if operator.task.name != "chg2tau":
        raise ValueError(
            f"kinetic potential requires a chg2tau operator, got "
            f"{operator.task.name!r}."
        )

    def tau(rho):
        normalized = operator.input_transform(rho)
        return operator.target_transform.inverse(operator.model(normalized, cell))

    return kinetic_potential(tau, density, cell, create_graph=create_graph)


# ---------------------------------------------------------------------- #
# Electrostatics
# ---------------------------------------------------------------------- #
def hartree_potential(density, cell):
    r"""
    Hartree potential of an electron density.

    Solves :math:`\nabla^2 v_H = -4\pi e^2 \rho` in reciprocal space,

    .. math:: v_H(\mathbf{G}) = \frac{4\pi e^2 \rho(\mathbf{G})}{G^2},
              \qquad v_H(\mathbf{G}=0) = 0,

    with the :math:`\mathbf{G}=0` term dropped — the same neutralizing
    background convention used by
    :class:`~poraque.fields.ExternalPotential`, so ``v_H`` and ``v_ext`` are
    directly addable.

    Parameters
    ----------
    density : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` electron density in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors in Å.

    Returns
    -------
    torch.Tensor
        Potential energy of an electron, in eV, same shape as ``density``.
        Positive (repulsive), as it must be.
    """
    squeezed = density.dim() == 5
    values = density.squeeze(1) if squeezed else density
    shape = tuple(values.shape[-3:])

    g2 = reciprocal_vectors(cell, shape, values.device, values.dtype).pow(2).sum(1)
    kernel = torch.zeros_like(g2)
    nonzero = g2 > 1e-12
    kernel[nonzero] = 1.0 / g2[nonzero]

    transformed = torch.fft.fftn(values.to(torch.complex64), dim=(-3, -2, -1))
    potential = torch.fft.ifftn(
        4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * kernel * transformed,
        dim=(-3, -2, -1),
    ).real
    return potential.unsqueeze(1) if squeezed else potential


def poisson_residual(potential, density, cell, background=True):
    r"""
    Residual of :math:`\nabla^2 v_{\rm ext} - 4\pi e^2 n_{\rm ion}`.

    Useful as a hard consistency check on a *predicted* external potential, and
    as the exact identity validated in ``tests/test_fields.py``.

    Parameters
    ----------
    potential : torch.Tensor
        Potential energy of an electron, eV.
    density : torch.Tensor
        Ionic number-charge density, e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    background : bool, optional
        Subtract the cell average of ``density``, matching the
        :math:`\mathbf{G}=0` convention.

    Returns
    -------
    torch.Tensor
        Residual field in eV/Å².
    """
    source = density
    if background:
        source = density - density.mean(dim=(-3, -2, -1), keepdim=True)
    return (spectral_laplacian(potential, cell)
            - 4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * source)


# ---------------------------------------------------------------------- #
# Orbital-free kinetic energy functionals
# ---------------------------------------------------------------------- #
def thomas_fermi_tau(density):
    r"""
    :math:`\tau_{\rm TF}[\rho] = C_{\rm TF}\,\rho^{5/3}` in eV/Å³.

    The exact kinetic energy density of the uniform electron gas, and the
    leading term of the gradient expansion. It is the correct *slowly-varying*
    limit any learned KEDF must reproduce.
    """
    rho_atomic = density.clamp_min(0.0) * BOHR_TO_ANGSTROM ** 3
    return C_TF * rho_atomic.pow(5.0 / 3.0) * _HA_BOHR3_TO_EV_ANG3


def thomas_fermi_potential(density):
    r"""
    :math:`\delta T_{\rm TF}/\delta\rho = \tfrac{5}{3}C_{\rm TF}\rho^{2/3}`, in eV.
    """
    rho_atomic = density.clamp_min(0.0) * BOHR_TO_ANGSTROM ** 3
    return (5.0 / 3.0) * C_TF * rho_atomic.pow(2.0 / 3.0) * _HA_TO_EV


def von_weizsacker_tau(density, cell, epsilon=1e-10):
    r"""
    :math:`\tau_{\rm vW}[\rho] = |\nabla\rho|^2/(8\rho)` in eV/Å³.

    Exact for any one-orbital (nodeless) system and, by the Hoffmann-Ostenhof
    inequality, a rigorous **lower bound** on the true kinetic energy density
    for any system. That inequality is the single most useful hard constraint
    available for the ``CHGCAR -> TAUCAR`` task; see
    :func:`von_weizsacker_bound_loss`.

    Parameters
    ----------
    density : torch.Tensor
        ``(B, 1, Nx, Ny, Nz)`` in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    epsilon : float, optional
        Floor on :math:`\rho` guarding the division.

    Returns
    -------
    torch.Tensor
        Same shape as ``density``, in eV/Å³.
    """
    squeezed = density.dim() == 5
    values = density.squeeze(1) if squeezed else density

    rho_atomic = values.clamp_min(0.0) * BOHR_TO_ANGSTROM ** 3
    # d/d(Bohr) = d/d(Ang) * (Ang per Bohr)
    gradient = spectral_gradient(rho_atomic, cell) * BOHR_TO_ANGSTROM
    tau = gradient.pow(2).sum(1) / (8.0 * rho_atomic.clamp_min(epsilon))
    tau = tau * _HA_BOHR3_TO_EV_ANG3
    return tau.unsqueeze(1) if squeezed else tau


def von_weizsacker_potential(density, cell, epsilon=1e-10):
    r"""
    :math:`\delta T_{\rm vW}/\delta\rho = -\tfrac{1}{2}\nabla^2\sqrt{\rho}/\sqrt{\rho}`, in eV.
    """
    squeezed = density.dim() == 5
    values = density.squeeze(1) if squeezed else density

    rho_atomic = values.clamp_min(epsilon) * BOHR_TO_ANGSTROM ** 3
    root = torch.sqrt(rho_atomic)
    laplacian = spectral_laplacian(root, cell) * BOHR_TO_ANGSTROM ** 2
    potential = -0.5 * laplacian / root * _HA_TO_EV
    return potential.unsqueeze(1) if squeezed else potential


# ---------------------------------------------------------------------- #
# Physics-informed loss terms
# ---------------------------------------------------------------------- #
def electron_count_loss(density, cell, n_electrons):
    r"""
    Penalize violation of :math:`\int\rho\,d^3r = N`.

    Particle-number conservation is exact and is the cheapest useful physical
    constraint: it costs one reduction and fixes the single global degree of
    freedom a pointwise regression loss controls worst.

    Parameters
    ----------
    density : torch.Tensor
        Predicted density in e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    n_electrons : torch.Tensor or float
        Target valence electron count per structure.

    Returns
    -------
    torch.Tensor
        Scalar mean squared *relative* error.
    """
    target = torch.as_tensor(n_electrons, dtype=density.dtype,
                             device=density.device).reshape(-1)
    predicted = integrate(density, cell)
    return ((predicted - target) / target.clamp_min(1e-8)).pow(2).mean()


def positivity_loss(field):
    r"""
    Penalize negative values, :math:`\langle \mathrm{ReLU}(-f)^2\rangle / \langle f^2\rangle`.

    Both :math:`\rho` and :math:`\tau` are non-negative by definition. Prefer
    a :class:`~poraque.ml.transforms.Log` output parameterization, which makes
    positivity structural rather than penalized; use this when the output head
    is unconstrained.

    The result is normalized by the field's own mean square, so it is
    dimensionless and comparable across materials whose densities differ by
    orders of magnitude — otherwise a single loss weight could not serve a
    whole dataset.
    """
    scale = field.pow(2).mean().clamp_min(1e-30)
    return torch.relu(-field).pow(2).mean() / scale


def von_weizsacker_bound_loss(tau, density, cell):
    r"""
    Penalize violation of the exact bound :math:`\tau \ge \tau_{\rm vW}[\rho]`.

    One-sided by construction: a prediction above the bound is free, one below
    it is quadratically penalized. This encodes a theorem rather than a
    heuristic, so it can be weighted aggressively.

    The violation is normalized by the mean square of :math:`\tau_{\rm vW}`
    itself, making the term dimensionless and material-independent — a raw
    ``(eV/Å³)²`` penalty would vary by orders of magnitude between a light
    semiconductor and a transition-metal oxide and no single weight could
    serve both.

    Parameters
    ----------
    tau : torch.Tensor
        Predicted kinetic energy density, eV/Å³.
    density : torch.Tensor
        Input density, e/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    bound = von_weizsacker_tau(density, cell)
    scale = bound.pow(2).mean().clamp_min(1e-30)
    return torch.relu(bound - tau).pow(2).mean() / scale


def kinetic_energy_loss(tau, cell, target_energy):
    r"""
    Match the integrated kinetic energy :math:`T_s = \int\tau\,d^3r`.

    A pointwise loss on :math:`\tau` does not pin down its integral, yet the
    integral is the quantity that actually enters the total energy. Constrain
    it explicitly when reference :math:`T_s` values are available.

    Parameters
    ----------
    tau : torch.Tensor
        Predicted kinetic energy density, eV/Å³.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    target_energy : torch.Tensor or float
        Reference :math:`T_s` per structure, eV.

    Returns
    -------
    torch.Tensor
        Scalar mean squared relative error.
    """
    target = torch.as_tensor(target_energy, dtype=tau.dtype,
                             device=tau.device).reshape(-1)
    return ((integrate(tau, cell) - target) / target.abs().clamp_min(1e-8)) \
        .pow(2).mean()


def euler_lagrange_residual(density, v_external, cell, lam=1.0 / 9.0,
                            v_xc=None, epsilon=1e-10, kinetic=None):
    r"""
    Residual of the orbital-free Euler-Lagrange equation.

    At the ground state the OF-DFT variational condition holds pointwise,

    .. math::

        \frac{\delta T_s}{\delta\rho}(\mathbf{r})
        + v_{\rm ext}(\mathbf{r})
        + v_{H}[\rho](\mathbf{r})
        + v_{xc}[\rho](\mathbf{r})
        \;=\; \mu ,

    with a *constant* chemical potential :math:`\mu`. Subtracting the cell
    average of the left-hand side removes :math:`\mu` — which is unknown and
    material-dependent — and leaves a residual that must vanish for the exact
    density. Because it involves :math:`v_{\rm ext}` and :math:`\rho` only, it
    is a self-contained physical constraint on the ``EXTCAR -> CHGCAR`` map
    that needs **no additional labels**.

    Parameters
    ----------
    density : torch.Tensor
        Predicted density, e/Å³.
    v_external : torch.Tensor
        External potential, eV (the network's input for this task).
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    lam : float, optional
        Weight of the von Weizsäcker term in the ``TF + lam*vW`` kinetic
        functional. ``1/9`` is the second-order gradient expansion; ``1`` is
        appropriate for strongly inhomogeneous, molecular-like densities.
    v_xc : torch.Tensor, optional
        Exchange-correlation potential in eV. Omitted terms simply weaken the
        constraint; they do not bias it, since what is enforced is the
        *constancy* of the sum.
    epsilon : float, optional
        Density floor.
    kinetic : callable or torch.Tensor, optional
        Replaces the analytic ``TF + lam*vW`` surrogate with a **learned**
        kinetic potential. Either

        * a tensor already holding :math:`\delta T_s/\delta\rho` in eV, or
        * a callable ``rho -> tau(rho)``, in which case the derivative is
          taken here by autograd (:func:`kinetic_potential`), with
          ``create_graph=True`` so the residual stays differentiable.

        Passing a trained ``chg2tau`` operator's :math:`\tau` closes the loop
        between the two models: the constraint on ``ext2chg`` is then built
        from the learned functional rather than from a fixed approximation.

    Returns
    -------
    torch.Tensor
        Residual field in eV, zero-mean by construction.

    Notes
    -----
    With the analytic surrogate the residual is only as good as
    ``TF + lam*vW``, which is an approximation — the term supplies a physically
    correct *inductive bias*, not ground truth, and should carry a modest
    weight. Supplying ``kinetic`` removes that limitation, at the cost of
    coupling the two models; see ``docs/notes/model2_architecture.md`` §5, including
    the warning that the residual alone has trivial solutions and the data
    terms must stay dominant.
    """
    if kinetic is None:
        kinetic_term = (thomas_fermi_potential(density)
                        + lam * von_weizsacker_potential(density, cell, epsilon))
    elif callable(kinetic):
        kinetic_term = kinetic_potential(kinetic, density, cell,
                                         create_graph=True)
    else:
        kinetic_term = kinetic

    total = kinetic_term + v_external + hartree_potential(density, cell)
    if v_xc is not None:
        total = total + v_xc
    return total - total.mean(dim=(-3, -2, -1), keepdim=True)


def euler_lagrange_loss(density, v_external, cell, **kwargs):
    """
    Mean squared :func:`euler_lagrange_residual`, normalized by its own scale.

    Normalization keeps the term dimensionless and comparable across materials
    whose potentials differ by an order of magnitude.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    residual = euler_lagrange_residual(density, v_external, cell, **kwargs)
    scale = v_external.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
    return (residual / scale).pow(2).mean()
