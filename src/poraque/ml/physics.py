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

from functools import lru_cache

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
#
#   Checked on CUDA, not assumed: `.cpu()` on a CUDA tensor synchronises, and
#   this runs several times per step, so keeping it on the device looked like a
#   free win. Measured on a V100 with 32^3 grids it is 2 % *slower* -- a 3x3
#   `linalg.det` costs 0.138 ms on the device against 0.098 ms via the host,
#   because the kernel launch for 27 numbers costs more than the copy it
#   avoids, and the synchronisation is cheap because the queue behind it is one
#   57 ms step. Worth re-measuring only if the grids grow enough for that queue
#   to get long -- 128^3, or a batch an order of magnitude larger.
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
@lru_cache(maxsize=64)
def _integer_mesh(shape, device, dtype):
    r"""
    The integer mesh :math:`(m_1, m_2, m_3)` of an FFT grid, ``(3, Nx, Ny, Nz)``.

    Memoised because it depends only on ``(shape, device, dtype)`` and **not on
    the cell**, while :func:`reciprocal_vectors` is called several times per
    batch --- once for the model's coordinate channels, again for each spectral
    gradient in an :math:`H^1` loss, again for the von Weizsäcker bound --- each
    time allocating and filling :math:`3 N_x N_y N_z` values that were identical
    to the last ones. Measured on a V100 with the in-RAM dataset cache on: 1.6 %
    of training time, consistent across repetitions, on every backend and
    probably most on MPS, where allocation is dearer.

    ``maxsize=64`` is sized against
    :class:`~poraque.ml.data.ShapeBucketSampler`, which batches by grid shape:
    the number of distinct shapes in a dataset is therefore small and stable
    --- 19 for the 115-structure set this was measured on --- and one entry per
    shape per dtype fits with room to spare.

    Parameters
    ----------
    shape : tuple of int
        Must be a tuple of plain ``int``. A list is unhashable and a
        :class:`torch.Size` of tensors would key the cache on objects it then
        keeps alive; :func:`reciprocal_vectors` normalises before calling.
    device : torch.device
    dtype : torch.dtype

    Returns
    -------
    torch.Tensor
        ``(3, Nx, Ny, Nz)``. Shared between callers, so **it must not be
        mutated in place** --- every use downstream is a read.
    """
    frequencies = [
        torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype) for n in shape
    ]
    return torch.stack(torch.meshgrid(*frequencies, indexing="ij"), dim=0)


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

    # Normalised to a plain tuple of ints before it becomes a cache key: a
    # `torch.Size` hashes correctly but a list does not, and an entry keyed on
    # tensors would hold them alive for the life of the process.
    integers = _integer_mesh(tuple(int(n) for n in shape),
                             torch.device(device), dtype)  # (3, Nx, Ny, Nz)

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
        ``(B, 3, Nx, Ny, Nz)`` for a single field, or
        ``(B, C, 3, Nx, Ny, Nz)`` when a channel axis of width :math:`C > 1`
        was given — one gradient per channel, in field-units per Å.

    Notes
    -----
    A multi-channel field used to be a ``RuntimeError``: the channel axis was
    removed with ``squeeze(1)``, which is a **silent no-op** on an axis of
    width 2, and the surviving axis then broadcast against the three Cartesian
    components. Every spin-polarised run reaches this through
    :class:`~poraque.ml.losses.SobolevLoss`, so ``loss: sobolev`` could not be
    used on ``ISPIN = 2`` data at all.
    """
    values = field[:, 0] if field.dim() == 5 and field.shape[1] == 1 else field
    shape = tuple(values.shape[-3:])
    g = reciprocal_vectors(cell, shape, values.device, values.dtype)

    # fftn on the real tensor keeps the input precision: float64 fields get
    # complex128 transforms, so a double-precision run stays double here.
    transformed = torch.fft.fftn(values, dim=(-3, -2, -1))
    if values.dim() == 5:
        # (B, C, 1, ...) * (B, 1, 3, ...) -> (B, C, 3, ...)
        product = 1j * g.unsqueeze(1) * transformed.unsqueeze(2)
    else:
        product = 1j * g * transformed.unsqueeze(1)
    return torch.fft.ifftn(product, dim=(-3, -2, -1)).real


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

    Notes
    -----
    A multi-channel field was worse here than a crash. ``squeeze(1)`` is a
    no-op on an axis of width 2, leaving ``(B, C, …)`` to broadcast against a
    ``(B, …)`` kernel: that raises when ``C != B`` and, when ``C == B``, lines
    the *channel* axis up with the *batch* axis and returns a finite, wrong
    answer. A two-channel field in a batch of two is not a contrived case.
    """
    single = field.dim() == 5 and field.shape[1] == 1
    values = field[:, 0] if single else field
    shape = tuple(values.shape[-3:])
    g2 = reciprocal_vectors(cell, shape, values.device, values.dtype).pow(2).sum(1)
    if values.dim() == 5:
        g2 = g2.unsqueeze(1)          # (B, 1, ...) against (B, C, ...)

    transformed = torch.fft.fftn(values, dim=(-3, -2, -1))
    result = torch.fft.ifftn(-g2 * transformed, dim=(-3, -2, -1)).real
    return result.unsqueeze(1) if single else result


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
    # Channel 0 is rho. The Hartree term is the classical repulsion of the
    # total charge and the magnetisation is not one of its arguments;
    # `squeeze(1)` was a silent no-op on a spin-polarised (rho, m) pair, which
    # then broadcast a (B, ...) kernel against a (B, 2, ...) field -- a crash
    # for most batch sizes and, at batch 2, a finite wrong answer.
    values = density[:, 0] if squeezed else density
    shape = tuple(values.shape[-3:])

    g2 = reciprocal_vectors(cell, shape, values.device, values.dtype).pow(2).sum(1)
    kernel = torch.zeros_like(g2)
    nonzero = g2 > 1e-12
    kernel[nonzero] = 1.0 / g2[nonzero]

    transformed = torch.fft.fftn(values, dim=(-3, -2, -1))
    potential = torch.fft.ifftn(
        4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * kernel * transformed,
        dim=(-3, -2, -1),
    ).real
    return potential.unsqueeze(1) if squeezed else potential


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
    # Channel 0 is rho; a spin-polarised (rho, m) pair must not reach the
    # gradient -- squeeze(1) was a silent no-op on the size-2 channel axis.
    values = density[:, 0] if squeezed else density

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
    values = density[:, 0] if squeezed else density  # channel 0 is rho

    rho_atomic = values.clamp_min(epsilon) * BOHR_TO_ANGSTROM ** 3
    root = torch.sqrt(rho_atomic)
    laplacian = spectral_laplacian(root, cell) * BOHR_TO_ANGSTROM ** 2
    potential = -0.5 * laplacian / root * _HA_TO_EV
    return potential.unsqueeze(1) if squeezed else potential


# ---------------------------------------------------------------------- #
# Exchange and correlation
#
# `euler_lagrange_residual` has always accepted a `v_xc` argument and there
# has never been anything in the package that could produce one, so every
# residual ever computed here silently omitted the term. That is not a small
# omission: v_xc is of order 10 eV in a valence region, which is the same
# order as the kinetic potential it is being weighed against.
#
# LDA only. PBE would need the gradient terms of its enhancement factor
# differentiated as well, and the residual is not yet accurate enough for
# that difference to be the limiting error.
# ---------------------------------------------------------------------- #
#: Dirac exchange coefficient, :math:`-\tfrac34(3/\pi)^{1/3}`.
_DIRAC_X = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

#: Perdew-Wang 1992 correlation parameters, unpolarised branch. Identical to
#: :data:`poraque.physics.energy._PW92`; the two must not drift apart.
_PW92 = dict(A=0.031091, alpha1=0.21370,
             beta1=7.5957, beta2=3.5876, beta3=1.6382, beta4=0.49294)

#: PBE parameters. None are fitted: kappa is the Lieb-Oxford bound and mu and
#: beta follow from the linear response of the uniform gas. Mirrors
#: :mod:`poraque.physics.energy`.
_PBE_KAPPA = 0.804
_PBE_BETA = 0.06672455060314922
_PBE_MU = _PBE_BETA * np.pi ** 2 / 3.0
_PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi ** 2

#: Functionals :func:`xc_potential` accepts, matching
#: :data:`poraque.physics.energy.XC_FUNCTIONALS` so one name means one thing
#: across the package.
XC_FUNCTIONALS = ("pbe", "lda", "pbe-x", "lda-x", "x-only", "none")

#: Those needing a density gradient, and therefore a ``cell``.
_GRADIENT_FUNCTIONALS = ("pbe", "pbe-x")


def lda_exchange_potential(density):
    r"""
    Dirac exchange potential :math:`v_{\rm x} = -(3\rho/\pi)^{1/3}`, in eV.

    Parameters
    ----------
    density : torch.Tensor
        Density in e/Å³, any shape.

    Returns
    -------
    torch.Tensor
        Same shape, in eV.
    """
    rho = density.clamp_min(0.0) * BOHR_TO_ANGSTROM ** 3
    return (4.0 / 3.0) * _DIRAC_X * rho.pow(1.0 / 3.0) * _HA_TO_EV


def pw92_correlation_potential(density, epsilon=1e-12):
    r"""
    Perdew-Wang 1992 correlation potential, unpolarised branch, in eV.

    Uses :math:`v_{\rm c} = \varepsilon_{\rm c}
    - \tfrac{r_s}{3}\,\mathrm{d}\varepsilon_{\rm c}/\mathrm{d}r_s`, with the
    derivative written in closed form rather than taken by autograd, so the
    function is usable on a tensor that does not require grad and costs
    nothing extra when it does.

    Parameters
    ----------
    density : torch.Tensor
        Density in e/Å³.
    epsilon : float, optional
        Density floor, in e/Bohr³ after conversion. In vacuum
        :math:`r_s \to \infty` and the expression is numerically dead; the
        floor keeps it finite without affecting any region that carries
        electrons.

    Returns
    -------
    torch.Tensor
        Same shape, in eV.
    """
    rho = (density.clamp_min(0.0) * BOHR_TO_ANGSTROM ** 3).clamp_min(epsilon)
    r_s = (3.0 / (4.0 * np.pi * rho)).pow(1.0 / 3.0)
    root = r_s.sqrt()

    p = _PW92
    q0 = -2.0 * p["A"] * (1.0 + p["alpha1"] * r_s)
    q1 = 2.0 * p["A"] * (p["beta1"] * root + p["beta2"] * r_s
                         + p["beta3"] * r_s * root
                         + p["beta4"] * r_s * r_s)
    # d/dr_s of both, for the chain rule below.
    dq0 = -2.0 * p["A"] * p["alpha1"]
    dq1 = 2.0 * p["A"] * (0.5 * p["beta1"] / root + p["beta2"]
                          + 1.5 * p["beta3"] * root
                          + 2.0 * p["beta4"] * r_s)

    log_term = torch.log1p(1.0 / q1)
    eps_c = q0 * log_term
    # d(eps_c)/dr_s; the second piece is q0 * d/dr_s log(1 + 1/q1).
    deps_c = dq0 * log_term - q0 * dq1 / (q1 * q1 + q1)

    return (eps_c - (r_s / 3.0) * deps_c) * _HA_TO_EV


def _pw92_epsilon(rho_bohr):
    """PW92 correlation energy per electron, Hartree. Torch mirror."""
    r_s = (3.0 / (4.0 * np.pi * rho_bohr)).pow(1.0 / 3.0)
    root = r_s.sqrt()
    p = _PW92
    denominator = 2.0 * p["A"] * (p["beta1"] * root + p["beta2"] * r_s
                                 + p["beta3"] * r_s * root
                                 + p["beta4"] * r_s * r_s)
    return (-2.0 * p["A"] * (1.0 + p["alpha1"] * r_s)
            * torch.log1p(1.0 / denominator))


def xc_energy_density(density, cell=None, functional="pbe", epsilon=1e-30):
    r"""
    Exchange-correlation energy density :math:`e_{\rm xc}[\rho]`, in eV/Å³.

    The torch counterpart of :func:`poraque.physics.energy.xc_energy`, written
    so that it is differentiable with respect to ``density``. That is what lets
    :func:`xc_potential` obtain the gradient-corrected potential by autograd
    instead of by hand.

    Parameters
    ----------
    density : torch.Tensor
        Density in e/Å³.
    cell : torch.Tensor, optional
        ``(B, 3, 3)`` lattice vectors, Å. Required for the gradient-corrected
        functionals and ignored by the local ones.
    functional : str, optional
        One of :data:`XC_FUNCTIONALS`.
    epsilon : float, optional
        Floor, in e/Bohr³, keeping the intermediate algebra finite in vacuum.

    Returns
    -------
    torch.Tensor
        Same shape as ``density``, in eV/Å³.
    """
    name = str(functional).lower()
    if name in ("none", "off"):
        return torch.zeros_like(density)
    if name not in XC_FUNCTIONALS:
        raise ValueError(
            f"Unknown xc functional {functional!r}; expected one of "
            f"{list(XC_FUNCTIONALS)}."
        )
    if name in _GRADIENT_FUNCTIONALS and cell is None:
        raise ValueError(
            f"functional={functional!r} needs the density gradient, so a "
            f"cell is required. Pass cell=, or use 'lda' for a local one."
        )

    squeezed = density.dim() == 5
    # Channel 0 is rho. `squeeze(1)` was used here and is a silent no-op on a
    # spin-polarised (rho, m) pair, which then reached spectral_gradient as a
    # two-channel field. This form of v_xc is the unpolarised one evaluated on
    # the total density -- see von_weizsacker_tau, which already did this.
    values = density[:, 0] if squeezed else density

    # The gradient is taken of the UNCLIPPED field: clipping first puts a kink
    # wherever a band-limited density rings below zero, and differentiating a
    # kink spectrally rings far worse than the undershoot it removed.
    raw = values * BOHR_TO_ANGSTROM ** 3
    rho = raw.clamp_min(0.0)
    safe = rho.clamp_min(epsilon)

    if name in _GRADIENT_FUNCTIONALS:
        gradient = spectral_gradient(raw, cell) * BOHR_TO_ANGSTROM
        gradient_squared = gradient.pow(2).sum(1)
    else:
        gradient_squared = None

    k_f = (3.0 * np.pi ** 2 * safe).pow(1.0 / 3.0)

    # ---- exchange ---------------------------------------------------- #
    enhancement = 1.0
    if name in ("pbe", "pbe-x"):
        s2 = gradient_squared / (2.0 * k_f * safe).pow(2)
        enhancement = (1.0 + _PBE_KAPPA
                       - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA))
    e_xc = _DIRAC_X * rho.pow(4.0 / 3.0) * enhancement

    # ---- correlation ------------------------------------------------- #
    if name in ("pbe", "lda"):
        epsilon_c = _pw92_epsilon(safe)
        if name == "pbe":
            k_s = torch.sqrt(4.0 * k_f / np.pi)
            t2 = gradient_squared / (2.0 * k_s * safe).pow(2)
            # eps_c < 0, so expm1 is positive; it stays accurate where the
            # difference of the two terms would otherwise cancel.
            a = ((_PBE_BETA / _PBE_GAMMA)
                 / torch.expm1(-epsilon_c / _PBE_GAMMA).clamp_min(1e-30))
            at2 = a * t2
            ratio = (1.0 + at2) / (1.0 + at2 + at2 * at2)
            h = _PBE_GAMMA * torch.log1p((_PBE_BETA / _PBE_GAMMA) * t2 * ratio)
            epsilon_c = epsilon_c + h
        e_xc = e_xc + rho * epsilon_c

    e_xc = e_xc * _HA_BOHR3_TO_EV_ANG3
    return e_xc.unsqueeze(1) if squeezed else e_xc


def xc_potential(density, functional="pbe", cell=None, epsilon=1e-12,
                 create_graph=False):
    r"""
    Exchange-correlation potential :math:`\delta E_{\rm xc}/\delta\rho`, in eV.

    Local functionals use their closed form. Gradient-corrected ones need

    .. math::

        v_{\rm xc} = \frac{\partial e}{\partial\rho}
          - \nabla\!\cdot\!\frac{\partial e}{\partial\nabla\rho} ,

    and that divergence term is taken by **autograd** through
    :func:`xc_energy_density` rather than derived by hand. The spectral
    gradient inside is itself differentiable and exact for a band-limited
    field, so the result carries no discretisation error, and there is no
    hand-derived expression to get wrong.

    .. important::

       The default is PBE, because the reference calculations this package is
       built around are PBE (``PAW_PBE`` potentials, ``LEXCH = PE``). Using an
       LDA potential on a PBE density does not approximate the right answer,
       it answers a different question, and the difference is of order 1 eV in
       a valence region. Set this to whatever generated the data.

    Parameters
    ----------
    density : torch.Tensor
        Density in e/Å³.
    functional : str, optional
        One of :data:`XC_FUNCTIONALS`. Default ``"pbe"``.
    cell : torch.Tensor, optional
        ``(B, 3, 3)`` lattice vectors, Å. Required for ``"pbe"`` and
        ``"pbe-x"``.
    epsilon : float, optional
        Density floor for the local correlation branch.
    create_graph : bool, optional
        Keep the autograd graph so the potential may appear in a loss that is
        itself backpropagated. Gradient-corrected functionals only.

    Returns
    -------
    torch.Tensor
        Same shape as ``density``, in eV.

    Raises
    ------
    ValueError
        If ``functional`` is unknown, or is gradient-corrected and no ``cell``
        was given.
    """
    name = str(functional).lower()
    # Channel 0 is rho, reduced once so every branch below agrees on what it
    # is a functional of. This is the unpolarised form evaluated on the total
    # density (see xc_energy_density), so m is not one of its arguments: the
    # LDA branch is elementwise and would otherwise return v_x(m) in the
    # second channel as though the magnetisation were a density, while the
    # gradient-corrected branch would return an exactly-zero one. Both are
    # answers to a question nobody asked, and both then broadcast into an
    # Euler-Lagrange residual built from single-channel potentials.
    if density.dim() == 5 and density.shape[1] > 1:
        density = density[:, :1]
    if name in ("none", "off"):
        return torch.zeros_like(density)
    if name in ("x", "exchange", "lda_x", "x-only"):
        name = "lda-x"
    if name == "lda-x":
        return lda_exchange_potential(density)
    if name in ("lda", "lda_xc", "pw92"):
        return (lda_exchange_potential(density)
                + pw92_correlation_potential(density, epsilon))
    if name not in _GRADIENT_FUNCTIONALS:
        raise ValueError(
            f"Unknown xc functional {functional!r}; expected one of "
            f"{list(XC_FUNCTIONALS)}."
        )
    if cell is None:
        raise ValueError(
            f"functional={functional!r} needs the density gradient, so a "
            f"cell is required. Pass cell=, or use 'lda' for a local one."
        )

    # Deliberately NOT via `functional_derivative`, which detaches its input.
    # Detaching is right there: for a learned functional the graph one wants to
    # keep runs to the model parameters, not to the density. Here there are no
    # parameters, and the density is typically an operator's OUTPUT, so the
    # connection back to it is the only thing worth preserving: v_xc[rho] has
    # to be differentiable for an Euler-Lagrange residual built from a
    # predicted density to train anything.
    attached = density.requires_grad
    rho = density if attached else density.detach().requires_grad_(True)

    with torch.enable_grad():
        energy = integrate(xc_energy_density(rho, cell, name), cell)
        gradient, = torch.autograd.grad(
            energy.sum(), rho,
            create_graph=create_graph and attached,
            retain_graph=create_graph and attached,
        )

    return gradient / volume_element(density, cell)


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
    # Every term below is a functional of rho, so the magnetisation is reduced
    # away once, here, rather than in each of them. The elementwise helpers --
    # thomas_fermi_potential and the LDA branch of xc_potential -- cannot
    # detect a channel they were handed by mistake: they would return
    # (5/3)C_TF m^(2/3) and v_x(m) in a second channel, finite numbers with no
    # meaning, and clamp_min(0) on a signed m on top of that.
    if density.dim() == 5 and density.shape[1] > 1:
        density = density[:, :1]

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
        # A string names a functional and is evaluated here; a tensor is used
        # as given. Before this the argument existed and nothing in the
        # package could build one, so every residual silently omitted it.
        if isinstance(v_xc, str):
            v_xc = xc_potential(density, v_xc, cell=cell)
        total = total + v_xc
    return total - total.mean(dim=(-3, -2, -1), keepdim=True)


def exact_kinetic_potential(density, v_external, cell, xc="pbe", mu=None):
    r"""
    The exact :math:`\delta T_s/\delta\rho` of a **ground-state** density.

    At the ground state the Euler-Lagrange equation holds pointwise, and every
    term in it except the kinetic potential is known exactly. So it can be read
    backwards:

    .. math::

        \frac{\delta T_s}{\delta\rho}(\rr)
          = \mu - v_{\rm ext}(\rr) - v_{H}[\rho](\rr) - v_{xc}[\rho](\rr) .

    This matters because it is the **only** route to a label for the functional
    derivative. A derivative needs a functional, and ``TAUCAR`` is a field: no
    amount of it yields :math:`\delta T_s/\delta\rho`. The Euler-Lagrange
    equation converts a ground-state density, which is what reference data
    consists of, into a pointwise target for the quantity orbital-free theory
    actually consumes.

    .. warning::

       Valid **only** where the input is a converged ground-state density. Fed
       an arbitrary or partly converged density it returns a well-defined
       field that means nothing, and nothing here can detect the difference.

       On pseudopotential data the result also absorbs whatever the local
       picture is missing, the nonlocal projector terms above all. It is the
       kinetic potential of an effective *local* problem reproducing that
       density, which is what an orbital-free calculation needs, but it is not
       the all-electron :math:`\delta T_s/\delta\rho`.

    Parameters
    ----------
    density : torch.Tensor
        Ground-state density, e/Å³.
    v_external : torch.Tensor
        External potential, eV.
    cell : torch.Tensor
        ``(B, 3, 3)`` lattice vectors, Å.
    xc : str, optional
        Passed to :func:`xc_potential`. Omitting it (``"none"``) leaves the
        result wrong by :math:`v_{xc}`, which is of order 10 eV.
    mu : float or torch.Tensor, optional
        Chemical potential. When ``None`` the cell average is removed instead,
        which is the same statement with the unknown constant eliminated.

    Returns
    -------
    torch.Tensor
        Kinetic potential in eV, zero-mean unless ``mu`` is given.
    """
    if density.dim() == 5 and density.shape[1] > 1:
        density = density[:, :1]      # a functional of rho; see the residual
    known = (v_external + hartree_potential(density, cell)
             + xc_potential(density, xc, cell=cell))
    if mu is None:
        return -(known - known.mean(dim=(-3, -2, -1), keepdim=True))
    return mu - known


def exact_pauli_potential(density, v_external, cell, xc="pbe", mu=None):
    r"""
    The Pauli potential :math:`v_{\rm P} = \delta T_{\rm P}/\delta\rho` of a
    ground-state density, from the Levy-Perdew-Sahni equation.

    LPS writes the exact density as a single effective orbital,
    :math:`-\tfrac12\nabla^2\sqrt\rho + (v_{\rm ext}+v_H+v_{xc}+v_{\rm P})
    \sqrt\rho = \mu\sqrt\rho`. Dividing by :math:`\sqrt\rho` and using
    :math:`\delta T_{\rm vW}/\delta\rho = -\tfrac12\nabla^2\sqrt\rho/\sqrt\rho`
    turns it into the Euler-Lagrange equation with the kinetic potential split
    into its bosonic and Pauli parts. The two are therefore the *same*
    condition, and this function is :func:`exact_kinetic_potential` minus the
    von Weizsäcker term.

    The reason to want it separately is Levy-Ou-Yang: :math:`v_{\rm P}\ge0`
    pointwise, an exact constraint on the derivative rather than on the value.

    Parameters
    ----------
    density, v_external, cell, xc, mu
        As :func:`exact_kinetic_potential`.

    Returns
    -------
    torch.Tensor
        :math:`v_{\rm P}` in eV, zero-mean unless ``mu`` is given.

    Notes
    -----
    With ``mu=None`` the result is shifted by an unknown constant, so its sign
    carries no information: any field can be made non-negative by raising
    :math:`\mu`. The testable statement is how large a :math:`\mu` is required,
    and where the binding points sit.
    """
    kinetic = exact_kinetic_potential(density, v_external, cell, xc, mu)
    vw = von_weizsacker_potential(density, cell)
    if mu is None:
        vw = vw - vw.mean(dim=(-3, -2, -1), keepdim=True)
    return kinetic - vw


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
