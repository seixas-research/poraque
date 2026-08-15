# -*- coding: utf-8 -*-
# file: kan.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Nonlinearities used inside one :class:`~poraque.ml.fno.FNOBlock`.

Every activation an :class:`~poraque.ml.fno.FNO3d` can use lives here, behind
one factory, :func:`build_activation`. Two families:

**Stateless** (``gelu``, ``relu``, ``silu``, ``tanh``) — the original,
parameter-free nonlinearities. ``gelu`` is the default and stays the default;
nothing about how these compute has changed. They were previously looked up
as bare functions (:data:`torch.nn.functional.gelu` and friends); they are
now thin :class:`torch.nn.Module` wrappers (:mod:`torch.nn`'s own ``GELU``,
``ReLU``, ``SiLU``, ``Tanh``) instead, purely so :class:`FNOBlock` can hold
*every* activation the same way — as a submodule — regardless of whether it
happens to carry parameters. A stateless wrapper's ``state_dict()`` is empty,
so this changes nothing about what a ``gelu`` checkpoint contains.

**KAN-style, learnable** (``kan_bspline``, ``kan_cheby``) — a Kolmogorov-
Arnold Network replaces a fixed nonlinearity with a *learned* univariate
function. Implemented here at the coarsest granularity that still means
something inside a spectral block: **channel-wise**, one learned function
:math:`\phi_c` per channel, applied elementwise to every voxel of that
channel —

.. math:: y_c(\mathbf r) \;\leftarrow\; \phi_c\big(y_c(\mathbf r)\big)
          \quad\text{for every voxel } \mathbf r,

not the *edge-wise* form of the original KAN paper (Liu et al., 2024), which
would give every ``(input channel, output channel)`` pair its own function
and turn each :math:`1\times1\times1` pointwise map into a KAN layer outright
— a real architecture change, out of scope here and left as a note in
``FUTURE.md``.

Both learnable variants share one stabilisation trick: at initialisation each
:math:`\phi_c` is (to within a small random perturbation) exactly
:func:`torch.nn.functional.gelu` —

.. math:: \phi_c(x) = w_c \cdot \mathrm{GELU}(x) + \varepsilon_c(x),
          \qquad w_c = 1,\ \varepsilon_c \text{ small at init},

so switching ``activation`` on a config that has always used ``gelu`` starts
training from behaviour the rest of the codebase has already been tuned
against, and only departs from it as :math:`\varepsilon_c` is learned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#: name -> stateless torch.nn.Module class. Every instance below is
#: parameter-free, so wrapping them costs nothing: their state_dict is `{}`.
SIMPLE_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}

#: The learnable, per-channel KAN variants.
KAN_ACTIVATIONS = frozenset({"kan_bspline", "kan_cheby"})

#: Every valid ``activation`` name, stateless and learnable together.
ACTIVATIONS = frozenset(SIMPLE_ACTIVATIONS) | KAN_ACTIVATIONS

#: Std. dev. of the learnable residual at initialisation, shared by both KAN
#: variants. Not exposed through the config: it sets how far from GELU
#: training starts, not what the operator is allowed to learn, and 0.1 is
#: small enough that the near-identity-at-init property holds by construction
#: (see each class's docstring) without needing to be tuned per run.
_RESIDUAL_INIT_SCALE = 0.1


def _channel_view(vector, ndim, channels):
    """
    Reshape a ``(channels,)`` tensor to broadcast against ``(B, C, ...)``.

    Parameters
    ----------
    vector : torch.Tensor
        ``(channels,)``.
    ndim : int
        Number of dimensions of the tensor it will be broadcast against.
    channels : int

    Returns
    -------
    torch.Tensor
        ``(1, channels, 1, ..., 1)`` with ``ndim - 2`` trailing singleton axes.
    """
    return vector.view(1, channels, *([1] * (ndim - 2)))


def _check_channels(x, channels, cls_name):
    if x.shape[1] != channels:
        raise ValueError(
            f"{cls_name} was built for {channels} channels, got input with "
            f"{x.shape[1]} on axis 1 (expected axis 1 to be the channel axis, "
            f"as it is throughout poraque.ml.fno)."
        )


# ---------------------------------------------------------------------- #
# Chebyshev-polynomial KAN activation
# ---------------------------------------------------------------------- #
class ChebyKANActivation(nn.Module):
    r"""
    Channel-wise learnable activation via a Chebyshev-polynomial residual.

    Each channel :math:`c` gets its own function

    .. math::
        \phi_c(x) = w_c\,\mathrm{GELU}(x)
                    + \sum_{k=0}^{K} a_{c,k}\, T_k\big(\tanh(x)\big),

    applied elementwise to every voxel of that channel. :math:`T_k` is the
    Chebyshev polynomial of the first kind, evaluated by the stable three-term
    recurrence :math:`T_k = 2xT_{k-1} - T_{k-2}` (:math:`T_0=1`,
    :math:`T_1=x`); :math:`\tanh` bounds the argument to :math:`(-1, 1)`, the
    interval on which :math:`|T_k|\le 1` and the recurrence stays well
    conditioned — outside it :math:`T_k` grows like :math:`x^k` and a wide
    pre-activation tail (routine for an untrained network) would make the
    residual blow up before the base term ever gets a chance to dominate.

    The recurrence is evaluated incrementally — keeping only :math:`T_{k-1}`
    and :math:`T_{k-2}` at each step and accumulating the weighted sum as it
    goes — rather than by building a ``(..., K+1)`` stack of every
    :math:`T_k` and contracting it against the coefficients afterwards, which
    would multiply the activation's memory footprint by :math:`K+1` for a
    tensor already shaped ``(batch, channels, N_x, N_y, N_z)``.

    Parameters
    ----------
    channels : int
        Number of channels; one function per channel.
    degree : int, optional
        Highest Chebyshev order :math:`K`. ``degree + 1`` coefficients per
        channel.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{degree}+2)` — one base weight
    :math:`w_c` and :math:`\text{degree}+1` Chebyshev coefficients per
    channel, all real. At the default ``width: 16`` and ``degree=6`` that is
    :math:`16 \times 8 = 128` extra parameters *per Fourier layer*.
    """

    def __init__(self, channels, degree=6):
        super().__init__()
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}.")
        self.channels = int(channels)
        self.degree = int(degree)

        self.base_weight = nn.Parameter(torch.ones(self.channels))
        # Small init: the Chebyshev term starts as a near-zero perturbation on
        # top of the base activation (see class docstring). |T_k| <= 1 on the
        # tanh-bounded domain, so this std. dev. bounds the residual's typical
        # magnitude directly, independent of `degree`.
        coeff = torch.randn(self.channels, self.degree + 1) * (
            _RESIDUAL_INIT_SCALE / (self.degree + 1) ** 0.5
        )
        self.cheby_coeff = nn.Parameter(coeff)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, ...)`` with ``C == channels`` on axis 1 — any number of
            trailing spatial axes, so this also accepts the small non-5D
            tensors the unit tests exercise it on.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        _check_channels(x, self.channels, "ChebyKANActivation")
        ndim = x.dim()
        base = F.gelu(x) * _channel_view(self.base_weight, ndim, self.channels)

        x_bounded = torch.tanh(x)
        coeff = self.cheby_coeff.to(dtype=x.dtype)                # (C, K+1)

        t_prev2 = torch.ones_like(x_bounded)                      # T_0
        residual = t_prev2 * _channel_view(coeff[:, 0], ndim, self.channels)
        if self.degree >= 1:
            t_prev1 = x_bounded                                    # T_1
            residual = residual + t_prev1 * _channel_view(
                coeff[:, 1], ndim, self.channels)
            for k in range(2, self.degree + 1):
                t_k = 2.0 * x_bounded * t_prev1 - t_prev2
                residual = residual + t_k * _channel_view(
                    coeff[:, k], ndim, self.channels)
                t_prev2, t_prev1 = t_prev1, t_k

        return base + residual

    def extra_repr(self):
        return f"channels={self.channels}, degree={self.degree}"


# ---------------------------------------------------------------------- #
# B-spline KAN activation
# ---------------------------------------------------------------------- #
class BSplineKANActivation(nn.Module):
    r"""
    Channel-wise learnable activation via a B-spline residual (KAN-style).

    Each channel :math:`c` gets its own function

    .. math::
        \phi_c(x) = w_c\,\mathrm{GELU}(x)
                    + \sum_{i=1}^{n} a_{c,i}\,
                      B_i\big(\mathrm{clamp}(x,\, x_{\min},\, x_{\max})\big),

    applied elementwise. :math:`\{B_i\}` are the ``spline_order``-degree
    B-spline basis functions (:math:`n = \text{grid\_size} +
    \text{spline\_order}` of them) on a uniform knot grid spanning
    ``grid_range``, built once at construction by the Cox–de Boor recursion
    and stored as a **buffer** — fixed, not learned, and moved to a device or
    a precision by the ordinary :meth:`~torch.nn.Module.to`/
    :func:`~poraque.ml.fno.set_precision` machinery like any other buffer.
    This is the representation the original KAN paper uses (Liu et al. 2024),
    restricted here to one function per channel rather than one per
    input/output edge — see the module docstring.

    Out-of-range handling
    ----------------------
    An input outside ``grid_range`` is **clamped**, not linearly extrapolated.
    A B-spline basis sums to a bounded value only *inside* its knot support;
    evaluated outside it the boundary basis functions are not normalised and
    the residual can grow without control exactly where training has the
    least information about what it should do — an FNO's pre-activations are
    unbounded, so a rare wide tail is the expected case, not a pathological
    one. Clamping trades a saturated *residual* gradient at the extreme
    voxels for a residual that never diverges; the GELU base term keeps
    responding to the true, unclamped value everywhere, so :math:`\phi_c`
    itself never actually saturates. Linear extrapolation was the other
    option the design allows and was not taken: it would need its own
    bookkeeping at the two boundary segments and can amplify rather than
    dampen a wide tail, which is the opposite of what an activation function
    sitting inside a residual stream should do at initialisation.

    Parameters
    ----------
    channels : int
        Number of channels; one function per channel.
    grid_size : int, optional
        Number of intervals the knot grid is divided into.
    spline_order : int, optional
        Degree of the B-spline (``3`` = cubic, the KAN paper's default).
    grid_range : sequence of float, optional
        ``(low, high)`` support of the un-extended grid.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{grid\_size}+\text{spline\_order}+1)`
    — one base weight and :math:`\text{grid\_size}+\text{spline\_order}`
    spline coefficients per channel, all real. At the defaults
    (``width: 16``, ``grid_size=8``, ``spline_order=3``) that is
    :math:`16\times 12 = 192` extra parameters *per Fourier layer*.
    """

    def __init__(self, channels, grid_size=8, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}.")
        if spline_order < 0:
            raise ValueError(f"spline_order must be >= 0, got {spline_order}.")
        low, high = float(grid_range[0]), float(grid_range[1])
        if not high > low:
            raise ValueError(f"grid_range must be increasing, got {grid_range!r}.")

        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (low, high)
        self.n_basis = self.grid_size + self.spline_order

        # Extended knot vector: `spline_order` extra knots on each side of
        # [low, high], so the Cox-de Boor recursion below has enough support
        # to build a degree-`spline_order` basis across the whole interval.
        # Standard construction (Liu et al. 2024 and the efficient-kan
        # reference implementation it popularised).
        step = (high - low) / self.grid_size
        knots = torch.arange(
            -self.spline_order, self.grid_size + self.spline_order + 1,
            dtype=torch.float32,
        ) * step + low
        self.register_buffer("knots", knots)

        self.base_weight = nn.Parameter(torch.ones(self.channels))
        coeff = torch.randn(self.channels, self.n_basis) * (
            _RESIDUAL_INIT_SCALE / max(self.n_basis, 1) ** 0.5
        )
        self.spline_coeff = nn.Parameter(coeff)

        # Clamped strictly below `high`, so `x < grid[-1]` always holds and
        # the half-open degree-0 indicator (see `_basis`) never has to
        # special-case the exact right edge. A fraction of the knot spacing,
        # so it never removes a whole interval's worth of support.
        self._clamp_eps = step * 1e-4

    def _basis(self, x):
        """
        Cox-de Boor B-spline basis, evaluated at every entry of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Already clamped to ``grid_range`` (minus :attr:`_clamp_eps` on
            the high side); any shape.

        Returns
        -------
        torch.Tensor
            ``x.shape + (n_basis,)``.
        """
        grid = self.knots.to(dtype=x.dtype)                       # (n_knots,)
        x = x.unsqueeze(-1)                                        # (..., 1)

        # Degree 0: indicator of each knot interval. No gradient w.r.t. x --
        # correctly so, a degree-0 B-spline is a step function almost
        # everywhere -- gradient flow starts at the order >= 1 step below.
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for order in range(1, self.spline_order + 1):
            left_num = x - grid[:-(order + 1)]
            left_den = (grid[order:-1] - grid[:-(order + 1)]).clamp_min(1e-12)
            right_num = grid[order + 1:] - x
            right_den = (grid[order + 1:] - grid[1:-order]).clamp_min(1e-12)
            bases = (left_num / left_den) * bases[..., :-1] \
                  + (right_num / right_den) * bases[..., 1:]

        return bases                                   # (..., n_basis)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, ...)`` with ``C == channels`` on axis 1.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        _check_channels(x, self.channels, "BSplineKANActivation")
        ndim = x.dim()
        base = F.gelu(x) * _channel_view(self.base_weight, ndim, self.channels)

        low, high = self.grid_range
        x_clamped = x.clamp(low, high - self._clamp_eps)
        basis = self._basis(x_clamped)                  # (B, C, ..., n_basis)
        coeff = self.spline_coeff.to(dtype=x.dtype)      # (C, n_basis)
        coeff_view = coeff.view(1, self.channels, *([1] * (ndim - 2)),
                                self.n_basis)
        residual = (basis * coeff_view).sum(-1)

        return base + residual

    def extra_repr(self):
        return (f"channels={self.channels}, grid_size={self.grid_size}, "
                f"spline_order={self.spline_order}, grid_range={self.grid_range}")


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
def build_activation(name, channels, *, kan_grid_size=8, kan_spline_order=3,
                     kan_grid_range=(-2.0, 2.0), kan_degree=6):
    """
    Build the nonlinearity used inside one :class:`~poraque.ml.fno.FNOBlock`.

    Parameters
    ----------
    name : str
        One of :data:`ACTIVATIONS`: ``'gelu'``, ``'relu'``, ``'silu'``,
        ``'tanh'`` (stateless), or ``'kan_bspline'`` / ``'kan_cheby'``
        (per-channel learnable).
    channels : int
        Width of the block this activation sits in. Unused by the stateless
        variants; required by the KAN ones, where each channel gets its own
        learned function.
    kan_grid_size, kan_spline_order, kan_grid_range : optional
        Forwarded to :class:`BSplineKANActivation`; ignored for every other
        ``name``.
    kan_degree : int, optional
        Forwarded to :class:`ChebyKANActivation`; ignored for every other
        ``name``.

    Returns
    -------
    torch.nn.Module
    """
    if name in SIMPLE_ACTIVATIONS:
        return SIMPLE_ACTIVATIONS[name]()
    if name == "kan_bspline":
        return BSplineKANActivation(channels, grid_size=kan_grid_size,
                                    spline_order=kan_spline_order,
                                    grid_range=kan_grid_range)
    if name == "kan_cheby":
        return ChebyKANActivation(channels, degree=kan_degree)
    raise ValueError(
        f"Unknown activation {name!r}; expected one of {sorted(ACTIVATIONS)}."
    )
