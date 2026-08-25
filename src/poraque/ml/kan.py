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
parameter-free nonlinearities; nothing about how any of them compute has
changed. ``silu`` is the default as of 2026-08-17 (``gelu`` was the default
before that, and every model trained before then used it — see FUTURE.md for
the measurements that motivated the switch). They were previously looked up
as bare functions (:data:`torch.nn.functional.gelu` and friends); they are
now thin :class:`torch.nn.Module` wrappers (:mod:`torch.nn`'s own ``GELU``,
``ReLU``, ``SiLU``, ``Tanh``) instead, purely so :class:`FNOBlock` can hold
*every* activation the same way — as a submodule — regardless of whether it
happens to carry parameters. A stateless wrapper's ``state_dict()`` is empty,
so this changes nothing about what a checkpoint using any of the four
contains.

**KAN-style, learnable** (``kan_bspline``, ``kan_cheby``, ``kan_rbf``,
``kan_rational``) — a Kolmogorov-Arnold Network replaces a fixed nonlinearity
with a *learned* univariate function. Implemented here at the coarsest
granularity that still means something inside a spectral block:
**channel-wise**, one learned function :math:`\phi_c` per channel, applied
elementwise to every voxel of that channel —

.. math:: y_c(\mathbf r) \;\leftarrow\; \phi_c\big(y_c(\mathbf r)\big)
          \quad\text{for every voxel } \mathbf r,

not the *edge-wise* form of the original KAN paper (Liu et al., 2024), which
would give every ``(input channel, output channel)`` pair its own function
and turn each :math:`1\times1\times1` pointwise map into a KAN layer outright
— a real architecture change, out of scope here and left as a note in
``FUTURE.md``.

Four bases are offered, differing only in what :math:`\phi_c` is built from:
``kan_bspline`` a B-spline on a fixed knot grid (the original KAN paper),
``kan_cheby`` a Chebyshev-polynomial expansion, ``kan_rbf`` a sum of fixed
Gaussian radial basis functions (the "FastKAN" simplification, Li 2024, which
trades the Cox-de Boor recursion for a closed-form basis), and ``kan_rational``
a learned rational (Padé-style) function of :math:`x` with a denominator
guarded to stay clear of zero (Molina et al., 2020). All four share the
"fixed basis / fixed functional form, learned coefficients" pattern, so they
differ in expressiveness and cost, not in how they plug into an
:class:`~poraque.ml.fno.FNOBlock`.

Every learnable variant shares one stabilisation trick, taken directly from
the original KAN paper's own construction — its edge activation is
:math:`\phi(x) = w_b\, b(x) + w_s\, \mathrm{spline}(x)` with base function
:math:`b = \mathrm{SiLU}`, a fixed nonlinearity plus a learned residual that
starts small — reproduced here at initialisation as

.. math:: \phi_c(x) = w_c \cdot \mathrm{SiLU}(x) + \varepsilon_c(x),
          \qquad w_c = 1,\ \varepsilon_c \text{ small at init},

so switching ``activation`` on a config starts training from a function
close to the well-understood ``silu`` nonlinearity and only departs from it
as :math:`\varepsilon_c` is learned. (An earlier version of this module used
``GELU`` as :math:`b`, matching this project's *own* default stateless
activation rather than the paper's; corrected 2026-08-17 — see
``FUTURE.md``.)

``use_base=False`` (``kan_use_base: false`` in a config) drops the
:math:`w_c\,\mathrm{SiLU}(x)` term entirely, so :math:`\phi_c` is *only* the
learned residual — no base weight, no fixed nonlinearity, nothing but the
spline/Chebyshev/RBF/rational function. This is the more minimal reading of
"KAN" some descriptions use — the original paper keeps a base term
deliberately, noting it helps optimisation, and it is the default here for
the same reason. Each residual was designed to sit *on top of* an unbounded
base, not to be the sole nonlinearity, so switching it off changes more than
removing one term: :class:`ChebyKANActivation` and
:class:`BSplineKANActivation` become bounded for every input, and
:class:`RBFKANActivation` and (with the default degrees)
:class:`RationalKANActivation` decay to zero for a large
:math:`|x|` — a real behavioural difference from a base-carrying channel,
not only a smaller parameter count. See each class's docstring.
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
KAN_ACTIVATIONS = frozenset({"kan_bspline", "kan_cheby", "kan_rbf", "kan_rational"})

#: Every valid ``activation`` name, stateless and learnable together.
ACTIVATIONS = frozenset(SIMPLE_ACTIVATIONS) | KAN_ACTIVATIONS

#: Std. dev. of the learnable residual at initialisation, shared by every KAN
#: variant. Not exposed through the config: it sets how far from the base
#: activation (or, under ``use_base=False``, from zero) training starts, not
#: what the operator is allowed to learn, and 0.1 is small enough that the
#: near-identity-at-init property holds by construction (see each class's
#: docstring) without needing to be tuned per run.
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
        \phi_c(x) = w_c\,\mathrm{SiLU}(x)
                    + \sum_{k=0}^{K} a_{c,k}\, T_k\big(\tanh(x)\big),

    applied elementwise to every voxel of that channel (``use_base=False``
    drops the first term; see the module docstring). :math:`T_k` is the
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
    use_base : bool, optional
        Include the :math:`w_c\,\mathrm{SiLU}(x)` base term (default,
        matching the original KAN paper). ``False`` gives a "pure" KAN
        channel — only the Chebyshev residual, no base weight, and the
        output is then bounded for every input (:math:`|T_k(\tanh x)|\le1`),
        unlike a base-carrying channel. See the module docstring.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{degree}+2)` (``use_base=True``) or
    :math:`\text{channels}\times(\text{degree}+1)` (``use_base=False``) —
    :math:`\text{degree}+1` Chebyshev coefficients per channel, plus one base
    weight :math:`w_c` when the base term is included, all real. At the
    default ``width: 16`` and ``degree=6`` that is
    :math:`16 \times 8 = 128` extra parameters *per Fourier layer* with the
    base term, :math:`16\times7=112` without it.
    """

    def __init__(self, channels, degree=6, use_base=True):
        super().__init__()
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}.")
        self.channels = int(channels)
        self.degree = int(degree)
        self.use_base = bool(use_base)

        self.base_weight = (nn.Parameter(torch.ones(self.channels))
                            if self.use_base else None)
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

        if not self.use_base:
            return residual
        base = F.silu(x) * _channel_view(self.base_weight, ndim, self.channels)
        return base + residual

    def extra_repr(self):
        return (f"channels={self.channels}, degree={self.degree}, "
                f"use_base={self.use_base}")


# ---------------------------------------------------------------------- #
# B-spline KAN activation
# ---------------------------------------------------------------------- #
class BSplineKANActivation(nn.Module):
    r"""
    Channel-wise learnable activation via a B-spline residual (KAN-style).

    Each channel :math:`c` gets its own function

    .. math::
        \phi_c(x) = w_c\,\mathrm{SiLU}(x)
                    + \sum_{i=1}^{n} a_{c,i}\,
                      B_i\big(\mathrm{clamp}(x,\, x_{\min},\, x_{\max})\big),

    applied elementwise (``use_base=False`` drops the first term; see the
    module docstring — note the residual then *saturates* rather than
    growing for a wide tail, since clamping is what bounds it, and there is
    no base term left to track the true unclamped value). :math:`\{B_i\}`
    are the ``spline_order``-degree
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
    voxels for a residual that never diverges; with ``use_base=True`` (the
    default) the SiLU base term keeps responding to the true, unclamped
    value everywhere, so :math:`\phi_c` itself never actually saturates —
    with ``use_base=False`` there is no such term, and :math:`\phi_c` does
    saturate for a wide tail, since the residual is then the whole output.
    Linear extrapolation was the other
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
    use_base : bool, optional
        Include the :math:`w_c\,\mathrm{SiLU}(x)` base term (default,
        matching the original KAN paper). ``False`` gives a "pure" KAN
        channel — only the B-spline residual, no base weight. See the
        module docstring and "Out-of-range handling" above.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{grid\_size}+\text{spline\_order}+1)`
    (``use_base=True``) or
    :math:`\text{channels}\times(\text{grid\_size}+\text{spline\_order})`
    (``use_base=False``) —
    :math:`\text{grid\_size}+\text{spline\_order}` spline coefficients per
    channel, plus one base weight when the base term is included, all real.
    At the defaults (``width: 16``, ``grid_size=8``, ``spline_order=3``)
    that is :math:`16\times 12 = 192` extra parameters *per Fourier layer*
    with the base term, :math:`16\times11=176` without it.
    """

    def __init__(self, channels, grid_size=8, spline_order=3,
                 grid_range=(-2.0, 2.0), use_base=True):
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
        self.use_base = bool(use_base)
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

        self.base_weight = (nn.Parameter(torch.ones(self.channels))
                            if self.use_base else None)
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

        low, high = self.grid_range
        x_clamped = x.clamp(low, high - self._clamp_eps)
        basis = self._basis(x_clamped)                  # (B, C, ..., n_basis)
        coeff = self.spline_coeff.to(dtype=x.dtype)      # (C, n_basis)
        coeff_view = coeff.view(1, self.channels, *([1] * (ndim - 2)),
                                self.n_basis)
        residual = (basis * coeff_view).sum(-1)

        if not self.use_base:
            return residual
        base = F.silu(x) * _channel_view(self.base_weight, ndim, self.channels)
        return base + residual

    def extra_repr(self):
        return (f"channels={self.channels}, grid_size={self.grid_size}, "
                f"spline_order={self.spline_order}, grid_range={self.grid_range}, "
                f"use_base={self.use_base}")


# ---------------------------------------------------------------------- #
# Radial-basis-function KAN activation
# ---------------------------------------------------------------------- #
class RBFKANActivation(nn.Module):
    r"""
    Channel-wise learnable activation via a radial-basis-function residual.

    Each channel :math:`c` gets its own function

    .. math::
        \phi_c(x) = w_c\,\mathrm{SiLU}(x)
                    + \sum_{i=1}^{n} a_{c,i}\,
                      \exp\!\Big(-\frac{(x - \mu_i)^2}{2\sigma^2}\Big),

    applied elementwise (``use_base=False`` drops the first term; see the
    module docstring — with no base term the residual, and so
    :math:`\phi_c` itself, decays to zero for a wide tail, since every
    Gaussian in the sum does). The centers :math:`\{\mu_i\}` are ``grid_size + 1``
    fixed points evenly spaced across ``grid_range``, built once at
    construction and stored as a **buffer** — fixed, not learned, exactly
    like :class:`BSplineKANActivation`'s knot vector, whose ``grid_size`` and
    ``grid_range`` this class reuses. Replacing the B-spline basis with a
    Gaussian one is the "FastKAN" simplification (Li, 2024): a Gaussian RBF
    is a smooth, closed-form stand-in for a cubic B-spline that needs no
    Cox-de Boor recursion, at the cost of a slightly less local basis.

    The shared width :math:`\sigma` is fixed to the spacing between adjacent
    centers, so it narrows automatically as ``grid_size`` grows finer rather
    than needing its own hyperparameter kept in sync by hand.

    Out-of-range handling
    ----------------------
    Unlike the B-spline basis, nothing here needs clamping: a Gaussian decays
    smoothly and unconditionally to zero as :math:`x` moves away from every
    center, so the residual vanishes on its own for a pre-activation far
    outside ``grid_range``. With ``use_base=True`` (the default) the ``SiLU``
    base term is what is left — the same "never actually saturates" property
    :class:`BSplineKANActivation` achieves by explicit clamping, here for
    free from the shape of the basis itself; with ``use_base=False`` there is
    nothing left, and :math:`\phi_c` itself decays to zero for a wide tail.

    Parameters
    ----------
    channels : int
        Number of channels; one function per channel.
    grid_size : int, optional
        Number of intervals the center grid is divided into; ``grid_size + 1``
        centers.
    grid_range : sequence of float, optional
        ``(low, high)`` span the centers are spread across.
    use_base : bool, optional
        Include the :math:`w_c\,\mathrm{SiLU}(x)` base term (default,
        matching the original KAN paper). ``False`` gives a "pure" KAN
        channel — only the RBF residual, no base weight. See "Out-of-range
        handling" above.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{grid\_size}+2)` (``use_base=True``) or
    :math:`\text{channels}\times(\text{grid\_size}+1)` (``use_base=False``) —
    :math:`\text{grid\_size}+1` RBF coefficients per channel, plus one base
    weight when the base term is included, all real.
    """

    def __init__(self, channels, grid_size=8, grid_range=(-2.0, 2.0),
                 use_base=True):
        super().__init__()
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}.")
        low, high = float(grid_range[0]), float(grid_range[1])
        if not high > low:
            raise ValueError(f"grid_range must be increasing, got {grid_range!r}.")

        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.grid_range = (low, high)
        self.use_base = bool(use_base)
        self.n_basis = self.grid_size + 1

        centers = torch.linspace(low, high, self.n_basis, dtype=torch.float32)
        self.register_buffer("centers", centers)
        # Fixed to the center spacing (not learned, not user-facing): keeps
        # neighbouring Gaussians overlapping by a constant amount regardless
        # of grid_size, the same role `step` plays in the B-spline knot
        # vector. A plain float, not a buffer -- nothing to move between
        # devices or precisions.
        self.sigma = (high - low) / self.grid_size

        self.base_weight = (nn.Parameter(torch.ones(self.channels))
                            if self.use_base else None)
        coeff = torch.randn(self.channels, self.n_basis) * (
            _RESIDUAL_INIT_SCALE / max(self.n_basis, 1) ** 0.5
        )
        self.rbf_coeff = nn.Parameter(coeff)

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
        _check_channels(x, self.channels, "RBFKANActivation")
        ndim = x.dim()

        centers = self.centers.to(dtype=x.dtype)                  # (n_basis,)
        diff = x.unsqueeze(-1) - centers                          # (..., n_basis)
        basis = torch.exp(-(diff * diff) / (2.0 * self.sigma ** 2))

        coeff = self.rbf_coeff.to(dtype=x.dtype)                  # (C, n_basis)
        coeff_view = coeff.view(1, self.channels, *([1] * (ndim - 2)),
                                self.n_basis)
        residual = (basis * coeff_view).sum(-1)

        if not self.use_base:
            return residual
        base = F.silu(x) * _channel_view(self.base_weight, ndim, self.channels)
        return base + residual

    def extra_repr(self):
        return (f"channels={self.channels}, grid_size={self.grid_size}, "
                f"grid_range={self.grid_range}, sigma={self.sigma:.4g}, "
                f"use_base={self.use_base}")


# ---------------------------------------------------------------------- #
# Rational (Padé-style) KAN activation
# ---------------------------------------------------------------------- #
class RationalKANActivation(nn.Module):
    r"""
    Channel-wise learnable activation via a rational-function residual.

    Each channel :math:`c` gets its own function

    .. math::
        \phi_c(x) = w_c\,\mathrm{SiLU}(x) + \frac{P_c(x)}{Q_c(x)},
        \qquad
        P_c(x) = \sum_{k=0}^{K_p} a_{c,k}\,x^k, \quad
        Q_c(x) = 1 + \sum_{k=1}^{K_q} |b_{c,k}|\,x^{2k},

    applied elementwise (``use_base=False`` drops the first term; see the
    module docstring) — the "safe" rational/Padé activation design of
    Molina et al. (2020), adapted to the per-channel KAN setting of this
    module. :math:`Q_c` sums :math:`1` with non-negative even-power terms, so
    :math:`Q_c(x) \ge 1` for every real :math:`x` and the residual can never
    divide by, or even approach, zero — an *unconstrained* rational function
    can develop a pole anywhere training happens to push a root of its
    denominator, which a spectral block cannot recover from (one infinite
    voxel poisons every mode the FFT touches). Learning :math:`b_{c,k}`
    unconstrained and taking :math:`|b_{c,k}|` only in the forward pass keeps
    the parameterisation free for the optimiser while the forward pass
    enforces positivity by construction.

    With the default degrees (:math:`K_p = 4`, :math:`K_q = 4`) the
    denominator's highest power, :math:`x^{8}`, outgrows the numerator's,
    :math:`x^4`, so :math:`P_c(x)/Q_c(x) \to 0` as :math:`|x| \to \infty` —
    with ``use_base=True`` (the default) the residual decays on its own for
    a wide pre-activation tail, the same "never actually saturates" outcome
    :class:`BSplineKANActivation` reaches by clamping and
    :class:`ChebyKANActivation` by a bounding ``tanh``; with
    ``use_base=False`` there is no base term to keep responding, so
    :math:`\phi_c` itself decays to zero for a wide tail instead.

    Parameters
    ----------
    channels : int
        Number of channels; one function per channel.
    num_degree : int, optional
        Highest power :math:`K_p` in the numerator.
    den_degree : int, optional
        Number of even powers (:math:`x^2, x^4, \dots, x^{2K_q}`) in the
        denominator.
    use_base : bool, optional
        Include the :math:`w_c\,\mathrm{SiLU}(x)` base term (default,
        matching the original KAN paper). ``False`` gives a "pure" KAN
        channel — only the rational residual, no base weight.

    Notes
    -----
    Parameters added over a stateless activation:
    :math:`\text{channels}\times(\text{num\_degree}+\text{den\_degree}+2)`
    (``use_base=True``) or
    :math:`\text{channels}\times(\text{num\_degree}+\text{den\_degree}+1)`
    (``use_base=False``) — :math:`\text{num\_degree}+1` numerator
    coefficients and :math:`\text{den\_degree}` denominator coefficients per
    channel, plus one base weight when the base term is included, all real.
    """

    def __init__(self, channels, num_degree=4, den_degree=4, use_base=True):
        super().__init__()
        if num_degree < 0:
            raise ValueError(f"num_degree must be >= 0, got {num_degree}.")
        if den_degree < 0:
            raise ValueError(f"den_degree must be >= 0, got {den_degree}.")
        self.channels = int(channels)
        self.num_degree = int(num_degree)
        self.den_degree = int(den_degree)
        self.use_base = bool(use_base)

        self.base_weight = (nn.Parameter(torch.ones(self.channels))
                            if self.use_base else None)
        num_coeff = torch.randn(self.channels, self.num_degree + 1) * (
            _RESIDUAL_INIT_SCALE / (self.num_degree + 1) ** 0.5
        )
        self.num_coeff = nn.Parameter(num_coeff)
        # Small-but-nonzero init, not zero: a zero denominator coefficient
        # has zero gradient at x=0 under |.|, which would leave every
        # den_coeff stuck exactly where it started for any input that never
        # strays from the origin. Same _RESIDUAL_INIT_SCALE as every other
        # variant's residual, so Q_c also starts close to its own "do
        # nothing" state (Q_c = 1) without being pinned to it.
        if self.den_degree > 0:
            den_coeff = torch.randn(self.channels, self.den_degree) * (
                _RESIDUAL_INIT_SCALE / self.den_degree ** 0.5
            )
            self.den_coeff = nn.Parameter(den_coeff)
        else:
            self.den_coeff = None

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
        _check_channels(x, self.channels, "RationalKANActivation")
        ndim = x.dim()

        # Incremental powers of x, not a (..., degree) stack -- same memory
        # discipline as ChebyKANActivation.
        num_coeff = self.num_coeff.to(dtype=x.dtype)               # (C, Kp+1)
        numerator = _channel_view(num_coeff[:, 0], ndim, self.channels)
        x_power = torch.ones_like(x)
        for k in range(1, self.num_degree + 1):
            x_power = x_power * x
            numerator = numerator + x_power * _channel_view(
                num_coeff[:, k], ndim, self.channels)

        denominator = torch.ones_like(x)
        if self.den_degree > 0:
            den_coeff = self.den_coeff.to(dtype=x.dtype).abs()     # (C, Kq)
            x_squared = x * x
            x_power = torch.ones_like(x)
            for k in range(self.den_degree):
                x_power = x_power * x_squared
                denominator = denominator + x_power * _channel_view(
                    den_coeff[:, k], ndim, self.channels)

        residual = numerator / denominator
        if not self.use_base:
            return residual
        base = F.silu(x) * _channel_view(self.base_weight, ndim, self.channels)
        return base + residual

    def extra_repr(self):
        return (f"channels={self.channels}, num_degree={self.num_degree}, "
                f"den_degree={self.den_degree}, use_base={self.use_base}")


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
def build_activation(name, channels, *, kan_grid_size=8, kan_spline_order=3,
                     kan_grid_range=(-2.0, 2.0), kan_degree=6,
                     kan_rational_num_degree=4, kan_rational_den_degree=4,
                     kan_use_base=True):
    """
    Build the nonlinearity used inside one :class:`~poraque.ml.fno.FNOBlock`.

    Parameters
    ----------
    name : str
        One of :data:`ACTIVATIONS`: ``'gelu'``, ``'relu'``, ``'silu'``,
        ``'tanh'`` (stateless), or ``'kan_bspline'`` / ``'kan_cheby'`` /
        ``'kan_rbf'`` / ``'kan_rational'`` (per-channel learnable).
    channels : int
        Width of the block this activation sits in. Unused by the stateless
        variants; required by the KAN ones, where each channel gets its own
        learned function.
    kan_grid_size, kan_grid_range : optional
        Forwarded to :class:`BSplineKANActivation` and, since they share the
        same fixed-grid design, to :class:`RBFKANActivation` too. Ignored for
        every other ``name``.
    kan_spline_order : int, optional
        Forwarded to :class:`BSplineKANActivation`; ignored for every other
        ``name``.
    kan_degree : int, optional
        Forwarded to :class:`ChebyKANActivation`; ignored for every other
        ``name``.
    kan_rational_num_degree, kan_rational_den_degree : int, optional
        Forwarded to :class:`RationalKANActivation`; ignored for every other
        ``name``.
    kan_use_base : bool, optional
        Forwarded to whichever of the four learnable classes ``name``
        selects; ignored for the stateless variants. ``True`` (default,
        matching the original KAN paper) includes each channel's
        :math:`w_c\\,\\mathrm{SiLU}(x)` base term; ``False`` gives a "pure"
        KAN activation with no base term at all — see the module docstring.

    Returns
    -------
    torch.nn.Module
    """
    if name in SIMPLE_ACTIVATIONS:
        return SIMPLE_ACTIVATIONS[name]()
    if name == "kan_bspline":
        return BSplineKANActivation(channels, grid_size=kan_grid_size,
                                    spline_order=kan_spline_order,
                                    grid_range=kan_grid_range,
                                    use_base=kan_use_base)
    if name == "kan_cheby":
        return ChebyKANActivation(channels, degree=kan_degree,
                                  use_base=kan_use_base)
    if name == "kan_rbf":
        return RBFKANActivation(channels, grid_size=kan_grid_size,
                                grid_range=kan_grid_range,
                                use_base=kan_use_base)
    if name == "kan_rational":
        return RationalKANActivation(channels, num_degree=kan_rational_num_degree,
                                     den_degree=kan_rational_den_degree,
                                     use_base=kan_use_base)
    raise ValueError(
        f"Unknown activation {name!r}; expected one of {sorted(ACTIVATIONS)}."
    )


# ---------------------------------------------------------------------- #
# Symbolic readout
# ---------------------------------------------------------------------- #
def _silu_expr(x):
    """
    The exact closed form of SiLU (Swish-1), as a SymPy expression.

    :math:`\\mathrm{SiLU}(x) = x\\,\\sigma(x) = \\dfrac{x}{1+e^{-x}}` --
    matches :func:`torch.nn.functional.silu` exactly (no special function
    needed, unlike the GELU base an earlier version of this module used --
    see the module docstring).
    """
    import sympy

    return x / (1 + sympy.exp(-x))


def _bspline_symbolic_basis(knots, spline_order, x):
    """
    The Cox-de Boor recursion, evaluated symbolically.

    Line for line the same algorithm as :meth:`BSplineKANActivation._basis`
    -- same indices, same recursion -- just written over a Python list of
    :class:`sympy.Piecewise` pieces instead of tensor slices along the last
    axis. The result is the *exact* piecewise-polynomial function the
    trained module computes, not an approximation of it.

    Parameters
    ----------
    knots : sequence of float
        The extended knot vector (``activation.knots``, as plain floats).
    spline_order : int
    x : sympy.Expr
        Typically the free symbol, or a clamped expression in it -- see
        :func:`symbolic_expression`.

    Returns
    -------
    list of sympy.Expr
        Length ``len(knots) - 1 - spline_order`` (``n_basis``).
    """
    import sympy

    bases = [
        sympy.Piecewise((sympy.Integer(1), (x >= knots[i]) & (x < knots[i + 1])),
                        (sympy.Integer(0), True))
        for i in range(len(knots) - 1)
    ]

    for order in range(1, spline_order + 1):
        new_bases = []
        for i in range(len(bases) - 1):
            left_den = knots[i + order] - knots[i]
            right_den = knots[i + order + 1] - knots[i + 1]
            left_den = left_den if left_den != 0 else 1e-12   # matches the
            right_den = right_den if right_den != 0 else 1e-12  # tensor path's clamp_min(1e-12)
            left = (x - knots[i]) / left_den * bases[i]
            right = (knots[i + order + 1] - x) / right_den * bases[i + 1]
            new_bases.append(left + right)
        bases = new_bases
    return bases


def symbolic_expression(activation, channel, decimals=4, simplify=False):
    r"""
    Read one channel's learned function out of a trained KAN activation as a
    closed-form SymPy expression -- straight from its stored coefficients,
    knots, or centers. No fitting, no search.

    This is a different sense of "symbolic" from
    :mod:`poraque.ml.symbolic`, which *searches* the space of short
    expressions for one that reproduces a trained operator's predictions.
    Every activation here already *is* a fixed functional form -- (with
    ``use_base=True``, the default) SiLU plus a residual built from a small,
    explicit set of learned numbers, or (``use_base=False``) the residual
    alone -- so recovering the symbolic function one channel computes is a
    **readout** of parameters that were symbolic all along, not a regression
    against black-box outputs. This is the literal content of the claim that
    a KAN is interpretable: the interpretation was never hidden inside the
    weights, it *is* the weights. A ``use_base=False`` channel's expression
    contains no ``exp``/``erf`` from a base function at all -- only whatever
    the residual itself is built from.

    Parameters
    ----------
    activation : ChebyKANActivation, BSplineKANActivation, RBFKANActivation, or RationalKANActivation
        A trained (or freshly constructed) KAN activation module -- e.g.
        ``operator.model.blocks[layer].activation`` off a loaded
        :class:`~poraque.ml.training.FieldOperator`.
    channel : int
        Which of ``activation.channels`` to read out; every channel has its
        own independently learned function.
    decimals : int or None, optional
        Round every coefficient to this many places before building the
        expression, for readability. ``None`` keeps full float precision
        (mainly useful for a numerical cross-check against the module's own
        :meth:`~torch.nn.Module.forward`).
    simplify : bool, optional
        Run :func:`sympy.simplify` on the result before returning it. Off
        by default: a Chebyshev/RBF/rational residual is already close to
        SymPy's simplified form, and simplification is slow -- sometimes
        very slow -- on the B-spline case's nested :class:`sympy.Piecewise`
        expression.

    Returns
    -------
    sympy.Expr
        A function of the single free symbol ``x``, the value entering this
        channel of this Fourier block. Requires SymPy -- an optional
        dependency, the same one :func:`poraque.ml.symbolic.expression_to_latex`
        uses.

    Examples
    --------
    >>> import sympy
    >>> activation = ChebyKANActivation(channels=1, degree=2)
    >>> expr = symbolic_expression(activation, channel=0)
    >>> x = sympy.Symbol("x")
    >>> callable(sympy.lambdify(x, expr))
    True
    """
    import sympy

    if not isinstance(activation, (ChebyKANActivation, BSplineKANActivation,
                                   RBFKANActivation, RationalKANActivation)):
        raise TypeError(
            f"symbolic_expression has nothing to read out of "
            f"{type(activation).__name__}: a stateless activation (gelu, "
            f"relu, silu, tanh) already *is* its own closed form, with no "
            f"learned parameters to recover. Pass one of the four learnable "
            f"KAN variants instead."
        )
    if not (0 <= channel < activation.channels):
        raise ValueError(
            f"channel must be in [0, {activation.channels}), got {channel}."
        )

    def _round(value):
        value = float(value)
        return value if decimals is None else round(value, decimals)

    x = sympy.Symbol("x")
    # use_base=False -> no base term in the returned expression at all, not
    # a zero-valued one: a "pure" channel's symbolic readout should contain
    # no exp/erf from a base function, only whatever the residual is built
    # from -- see the module and function docstrings.
    base = (sympy.Float(_round(activation.base_weight[channel].item()))
                * _silu_expr(x)
            if activation.use_base else sympy.Integer(0))

    if isinstance(activation, ChebyKANActivation):
        x_bounded = sympy.tanh(x)
        residual = sympy.Integer(0)
        for k in range(activation.degree + 1):
            c = sympy.Float(_round(activation.cheby_coeff[channel, k].item()))
            residual += c * sympy.chebyshevt(k, x_bounded)

    elif isinstance(activation, RBFKANActivation):
        residual = sympy.Integer(0)
        for i in range(activation.n_basis):
            mu = sympy.Float(_round(activation.centers[i].item()))
            c = sympy.Float(_round(activation.rbf_coeff[channel, i].item()))
            residual += c * sympy.exp(-(x - mu) ** 2
                                      / (2 * activation.sigma ** 2))

    elif isinstance(activation, RationalKANActivation):
        numerator = sympy.Integer(0)
        for k in range(activation.num_degree + 1):
            c = sympy.Float(_round(activation.num_coeff[channel, k].item()))
            numerator += c * x ** k
        denominator = sympy.Integer(1)
        if activation.den_degree > 0:
            for k in range(activation.den_degree):
                c = sympy.Float(_round(abs(activation.den_coeff[channel, k].item())))
                denominator += c * x ** (2 * (k + 1))
        residual = numerator / denominator

    else:  # BSplineKANActivation
        low, high = activation.grid_range
        eps = float(activation._clamp_eps)
        x_clamped = sympy.Piecewise((sympy.Float(low), x < low),
                                    (sympy.Float(high - eps), x > high - eps),
                                    (x, True))
        knots = [float(k) for k in activation.knots.tolist()]
        bases = _bspline_symbolic_basis(knots, activation.spline_order, x_clamped)
        residual = sympy.Integer(0)
        for i, basis_i in enumerate(bases):
            c = sympy.Float(_round(activation.spline_coeff[channel, i].item()))
            residual += c * basis_i

    expr = base + residual
    return sympy.simplify(expr) if simplify else expr
