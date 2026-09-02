# -*- coding: utf-8 -*-
# file: fno.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Fourier Neural Operator for 3D scalar fields on crystal lattices.

An FNO learns a mapping between *function spaces* rather than between fixed
vectors. One layer is

.. math::

    v_{\ell+1}(\mathbf{r}) = \sigma\!\Big(
        W v_{\ell}(\mathbf{r}) + b
        + \mathcal{F}^{-1}\big[R_{\ell}(\mathbf{G})\cdot
          \mathcal{F}[v_{\ell}](\mathbf{G})\big](\mathbf{r})
    \Big),

where :math:`W` is a pointwise (1×1×1) linear map and :math:`R_\ell` is a
learned complex multiplier acting on the lowest Fourier modes. Two properties
make this the natural architecture for the present problem:

* **The convolution is periodic by construction.** The FFT enforces exactly the
  Born-von Kármán boundary conditions a crystal unit cell already obeys — no
  padding, no edge artefacts, no wasted capacity learning periodicity.
* **The learned weights live in mode space, not on the grid.** Nothing in
  :math:`R_\ell` refers to :math:`(N_x, N_y, N_z)`, so a single set of
  parameters applies to every material regardless of its grid.

Handling grids that differ between materials
--------------------------------------------
Grid-shape invariance is a *design requirement* here, since ``ENCUT`` and cell
size fix ``NGXF, NGYF, NGZF`` per material. Three mechanisms deliver it:

1. **Dynamic mode truncation** (:class:`SpectralConv3d`). Weights are allocated
   for ``modes`` coefficients per axis; each forward pass uses only
   ``min(modes, modes_available_on_this_grid)`` of them and leaves the rest
   untouched. A 24³ sample and a 120³ sample share the same parameters.
2. **Resolution-invariant normalization.** Both transforms use
   ``norm="forward"``, so the coefficients approximate *continuous* Fourier
   series coefficients and their magnitude does not drift with :math:`N`. With
   the default (``"backward"``) convention a 120³ field would enter a layer
   with amplitudes ~125× those of a 24³ field and the shared weights would be
   meaningless.
3. **Physical mode selection** (``mode_selection="physical"``). Truncating at a
   fixed *index* keeps a different physical wavevector band for every cell
   size. Truncating at a fixed :math:`G_{\max}` instead — retaining
   :math:`n_i = \lfloor G_{\max} L_i / 2\pi \rfloor` modes — makes the operator
   act on the same band of physics in every material. Preferred when the
   dataset spans a wide range of cell sizes.

The cell itself is fed to the network through fractional-coordinate channels
and FiLM conditioning (:class:`CellEncoder`), so the operator can distinguish a
dense small cell from a sparse large one.
"""

import numpy as np
import torch
import torch.nn as nn

from .kan import ACTIVATIONS, build_activation

#: Contraction performed by the spectral convolution.
_SPECTRAL_EQUATION = "bixyz,ioxyz->boxyz"

#: Precisions an operator can compute in, as ``name -> (real, complex)``.
#:
#: The pair is the point. An FNO carries **complex** parameters — the mode
#: multipliers of :class:`SpectralConv3d` — alongside the real weights of its
#: pointwise layers, and the two must move together: a float64 activation
#: entering a complex64 multiplier fails outright, and the reverse silently
#: rounds.
PRECISIONS = {
    "float32": (torch.float32, torch.complex64),
    "float64": (torch.float64, torch.complex128),
}


def resolve_precision(precision):
    """
    Normalise a precision name to its ``(real, complex)`` torch dtypes.

    Parameters
    ----------
    precision : str or torch.dtype
        ``"float32"``/``"float64"``, or either torch dtype.

    Returns
    -------
    tuple of torch.dtype
    """
    if isinstance(precision, torch.dtype):
        precision = {torch.float32: "float32",
                     torch.float64: "float64"}.get(precision, str(precision))
    key = str(precision).strip().lower().replace("torch.", "")
    if key not in PRECISIONS:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of "
            f"{sorted(PRECISIONS)}.")
    return PRECISIONS[key]


def set_precision(module, precision):
    r"""
    Convert a model to another precision, complex parameters included.

    .. warning::

       Neither PyTorch idiom does the right thing to an FNO, and both fail
       quietly enough to matter:

       * ``model.double()`` converts only tensors for which
         ``is_floating_point()`` is true. A complex64 spectral weight is not
         one, so it is **left behind** — and the next forward pass dies with
         ``expected scalar type ComplexDouble but found ComplexFloat``.
       * ``model.to(torch.float64)`` is worse: it converts complex64 to
         *float64*, **discarding the imaginary part of every Fourier
         multiplier**. No error, and a model that has lost half of its
         spectral parameters.

       This function maps real dtypes to the real target and complex dtypes to
       the matching complex target, which is the only conversion that leaves
       the operator meaning what it meant.

    Parameters
    ----------
    module : torch.nn.Module
    precision : str or torch.dtype
        See :func:`resolve_precision`.

    Returns
    -------
    torch.nn.Module
        ``module``, converted in place, so this can be chained.

    Examples
    --------
    >>> model = FNO3d(width=8, modes=4, n_layers=1)
    >>> _ = set_precision(model, "float64")
    >>> {p.dtype for p in model.parameters()} == {torch.float64,
    ...                                           torch.complex128}
    True
    """
    real, complex_dtype = resolve_precision(precision)

    def convert(tensor):
        if tensor.is_complex():
            return tensor.to(complex_dtype)
        if tensor.is_floating_point():
            return tensor.to(real)
        return tensor

    # `_apply` rather than a loop over `parameters()`: it also reaches buffers
    # and any `.grad` already attached, and it rebinds parameters in place.
    return module._apply(convert)


def model_precision(module):
    """
    The precision a model currently computes in.

    Returns
    -------
    str
        ``"float32"``, ``"float64"``, or ``"mixed"`` when the real and complex
        halves disagree — which is a broken model rather than a configuration,
        and is reported rather than guessed at.
    """
    dtypes = {parameter.dtype for parameter in module.parameters()}
    for name, (real, complex_dtype) in PRECISIONS.items():
        if dtypes <= {real, complex_dtype}:
            return name
    return "mixed"


def complex_contract(equation, x, weight):
    r"""
    Contract two complex tensors, portably across accelerators.

    ``torch.einsum`` on **complex** operands is not implemented on Apple's MPS
    backend: it lowers to an ``mps.gather`` over ``complex<f32>`` values, which
    Metal rejects outright — the process aborts rather than raising, so it
    cannot even be caught. Since this contraction *is* the Fourier layer, the
    whole architecture would be unusable on Apple Silicon.

    Splitting the product into real arithmetic,

    .. math:: (a + ib)(c + id) = (ac - bd) + i(ad + bc),

    uses only real ``einsum`` calls, which every backend supports. It is
    numerically identical to the complex path — a complex multiply performs the
    same four real products internally — and costs four kernel launches
    instead of one. CUDA and CPU keep the native path, where it is fastest.

    .. warning::
       The ``.contiguous()`` calls below are **load-bearing, not tidiness**.
       The operands here are strided views: high-frequency corners such as
       ``x_ft[..., nx-m1:, ny-m2:, :m3]``, and ``weight[index]``. Taking
       ``.real``/``.imag`` of a non-contiguous complex tensor yields a real
       view with a stride of two elements, and MPS silently computes the wrong
       ``einsum`` over such a view — no error, no warning, results off by
       40-90 %. Materialising the operands first reduces the error to ~1e-7,
       i.e. ordinary float32 rounding. Verified in ``tests/test_device.py``.

    Parameters
    ----------
    equation : str
        ``einsum`` subscript string.
    x, weight : torch.Tensor
        Complex operands.

    Returns
    -------
    torch.Tensor
        Complex result of the contraction.
    """
    # The C kernel, when this is an inference call on CPU. It is ~7x faster
    # than `einsum` at batch 1 (see poraque.ml.backend), which is the shape
    # every prediction has.
    #
    # `torch.is_grad_enabled()` is the load-bearing condition, not a
    # micro-optimisation: the kernel writes into a plain tensor and records
    # nothing on the autograd tape, so using it while a graph is being built
    # would produce a model that trains to a constant with no error anywhere.
    # `predict()` is wrapped in `torch.no_grad()`, so inference takes this path
    # and training does not.
    if (equation == _SPECTRAL_EQUATION
            and x.device.type == "cpu"
            and not torch.is_grad_enabled()):
        from . import backend

        contracted = backend.contract(x, weight)
        if contracted is not None:
            return contracted

    if x.device.type != "mps":
        return torch.einsum(equation, x, weight)

    x = x.contiguous()
    weight = weight.contiguous()
    x_real, x_imag = x.real, x.imag
    w_real, w_imag = weight.real, weight.imag
    return torch.complex(
        torch.einsum(equation, x_real, w_real) - torch.einsum(equation, x_imag, w_imag),
        torch.einsum(equation, x_real, w_imag) + torch.einsum(equation, x_imag, w_real),
    )


# ---------------------------------------------------------------------- #
# Spectral convolution
# ---------------------------------------------------------------------- #
class SpectralConv3d(nn.Module):
    r"""
    Learned multiplication of the lowest Fourier modes.

    Because ``rfftn`` halves the last axis, the retained block of coefficients
    is the union of four "corners" in the first two axes — combinations of low
    positive and low negative frequencies — each carrying its own weight
    tensor.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts.
    modes : tuple of int
        Maximum number of retained modes ``(m1, m2, m3)`` per axis. These are
        *capacity* limits: on a grid too coarse to supply them, fewer are used.

    Notes
    -----
    The parameter count is ``4 * in * out * m1 * m2 * m3`` complex numbers and
    is independent of the grid, which is exactly what allows one model to serve
    materials of many different shapes.
    """

    def __init__(self, in_channels, out_channels, modes=(12, 12, 12)):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(m) for m in modes)
        if any(m < 1 for m in self.modes):
            raise ValueError(f"All mode counts must be >= 1, got {modes!r}.")

        # He-style scaling keeps activations O(1) through the stack.
        scale = 1.0 / (self.in_channels * self.out_channels)
        # `torch.cfloat` is why this project has no `torch.compile` option.
        # Inductor does not generate code for complex operators -- it says so,
        # every time -- so the one part of an FNO that compilation would be
        # invoked to fuse is the part it hands straight back to eager, while
        # the wrapper and the graph breaks are still paid for. Measured on a
        # V100 at 150 epochs x 2 repetitions: +7.9 % per epoch at
        # mode="default" and +9.4 % at "max-autotune", 200-314 s of cold
        # compilation on top, a validation error that moved reproducibly, and
        # an eighteen-minute first epoch with no output on four DDP ranks. The
        # flag was removed in 26.9.3; `RETIRED_KEYS` in ml/config.py refuses it
        # by name. See CUDA_4.md sections 7 and 8, and note what was NOT tried:
        # compiling only the real-valued submodules, where 44.4 % of the GPU
        # time actually is. Re-open whole-model compilation only if Inductor
        # gains complex-operator codegen.
        self.weight = nn.Parameter(
            scale * torch.randn(4, self.in_channels, self.out_channels,
                                *self.modes, dtype=torch.cfloat)
        )

    def effective_modes(self, shape, max_modes=None):
        """
        Modes actually usable on a grid of ``shape``.

        Parameters
        ----------
        shape : tuple of int
            Spatial shape ``(Nx, Ny, Nz)``.
        max_modes : tuple of int, optional
            Extra per-axis cap, e.g. from a physical ``G_max``.

        Returns
        -------
        tuple of int
            ``(m1, m2, m3)``, each at least 1.
        """
        nx, ny, nz = shape
        available = (nx // 2, ny // 2, nz // 2 + 1)
        caps = max_modes if max_modes is not None else self.modes
        return tuple(
            max(1, min(int(self.modes[i]), int(caps[i]), int(available[i])))
            for i in range(3)
        )

    def forward(self, x, max_modes=None):
        """
        Apply the spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C_in, Nx, Ny, Nz)`` real tensor.
        max_modes : tuple of int, optional
            Per-axis cap on the retained modes for this call.

        Returns
        -------
        torch.Tensor
            ``(B, C_out, Nx, Ny, Nz)``.
        """
        batch, _, nx, ny, nz = x.shape
        m1, m2, m3 = self.effective_modes((nx, ny, nz), max_modes)

        # norm="forward" -> resolution-invariant Fourier-series coefficients.
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")

        # `x_ft.dtype`, not a literal `torch.cfloat`: `rfftn` of a float64
        # input gives ComplexDouble, so a hard-coded ComplexFloat accumulator
        # made `model.double()` fail outright with "expected scalar type
        # ComplexDouble but found ComplexFloat" -- the whole double-precision
        # path was unreachable.
        out_ft = torch.zeros(
            batch, self.out_channels, nx, ny, nz // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )

        # The four low-frequency corners of the (axis-0, axis-1) plane.
        corners = (
            (slice(None, m1), slice(None, m2), 0),
            (slice(None, m1), slice(ny - m2, None), 1),
            (slice(nx - m1, None), slice(None, m2), 2),
            (slice(nx - m1, None), slice(ny - m2, None), 3),
        )
        for slice_1, slice_2, index in corners:
            out_ft[:, :, slice_1, slice_2, :m3] = complex_contract(
                _SPECTRAL_EQUATION,
                x_ft[:, :, slice_1, slice_2, :m3],
                self.weight[index, :, :, :m1, :m2, :m3],
            )

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1),
                                norm="forward")

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, modes={self.modes}")


# ---------------------------------------------------------------------- #
# Cell conditioning
# ---------------------------------------------------------------------- #
class CellEncoder(nn.Module):
    """
    Embed a unit cell into a feature vector for FiLM conditioning.

    Descriptors are invariant under rotation of the cell and under permutation-
    free relabelling: the three lattice lengths, the cosines of the three
    angles, and the cube root of the volume. Feeding these to the network is
    what lets one operator distinguish materials whose grids look alike but
    whose physical scales do not.

    Parameters
    ----------
    embedding_dim : int, optional
        Width of the produced embedding.
    length_scale : float, optional
        Ångström scale used to non-dimensionalize lengths.
    """

    def __init__(self, embedding_dim=32, length_scale=10.0):
        super().__init__()
        self.length_scale = float(length_scale)
        self.net = nn.Sequential(
            nn.Linear(7, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def descriptors(cell, length_scale=10.0):
        """
        Build the ``(B, 7)`` descriptor tensor from ``(B, 3, 3)`` cells.

        Returns
        -------
        torch.Tensor
            ``[a, b, c, cos(alpha), cos(beta), cos(gamma), V^(1/3)]`` with
            lengths divided by ``length_scale``.
        """
        from .physics import cell_volume

        lengths = torch.linalg.norm(cell, dim=-1)                    # (B, 3)
        normalized = cell / lengths.unsqueeze(-1).clamp_min(1e-12)
        cosines = torch.stack(
            [
                (normalized[:, 1] * normalized[:, 2]).sum(-1),
                (normalized[:, 0] * normalized[:, 2]).sum(-1),
                (normalized[:, 0] * normalized[:, 1]).sum(-1),
            ],
            dim=-1,
        )
        # cell_volume evaluates the determinant on the CPU in float64: MPS
        # implements neither, and the 3x3 cost is irrelevant next to the FFTs.
        volume = cell_volume(cell, device=cell.device,
                             dtype=cell.dtype).clamp_min(1e-12)
        return torch.cat(
            [lengths / length_scale, cosines,
             (volume ** (1.0 / 3.0) / length_scale).unsqueeze(-1)],
            dim=-1,
        )

    def forward(self, cell):
        """Embed ``(B, 3, 3)`` cells into ``(B, embedding_dim)``."""
        return self.net(self.descriptors(cell, self.length_scale).to(
            next(self.net.parameters()).dtype))


class FiLM(nn.Module):
    """
    Feature-wise linear modulation: ``x -> (1 + gamma(c)) * x + beta(c)``.

    Conditions every channel of a feature map on the cell embedding without
    touching the spatial dimensions, so it is oblivious to the grid shape.
    """

    def __init__(self, embedding_dim, channels):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, embedding):
        """Modulate ``(B, C, Nx, Ny, Nz)`` by ``(B, embedding_dim)``."""
        gamma, beta = self.projection(embedding).chunk(2, dim=-1)
        shape = (x.shape[0], -1, 1, 1, 1)
        return (1.0 + gamma.reshape(shape)) * x + beta.reshape(shape)


# ---------------------------------------------------------------------- #
# Blocks and model
# ---------------------------------------------------------------------- #
def _group_count(channels, preferred=8):
    """
    Largest divisor of ``channels`` not exceeding ``preferred``.

    ``GroupNorm`` requires the channel count to be divisible by the group
    count, so an arbitrary ``width`` (e.g. 12 with 8 groups) would otherwise
    fail at construction time.
    """
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FNOBlock(nn.Module):
    """
    One Fourier layer: spectral convolution + pointwise skip, normed and gated.

    Parameters
    ----------
    channels : int
        Width of the block.
    modes : tuple of int
        Mode capacity of the spectral convolution.
    embedding_dim : int, optional
        Cell-embedding width; ``0`` disables FiLM conditioning.
    activation : str, optional
        One of :data:`~poraque.ml.kan.ACTIVATIONS`: ``'silu'`` (default),
        ``'gelu'``, ``'relu'``, ``'tanh'`` (stateless), or ``'kan_bspline'``
        / ``'kan_cheby'`` / ``'kan_rbf'`` / ``'kan_rational'`` (per-channel
        learnable — see :mod:`poraque.ml.kan`).
    activation_kwargs : dict, optional
        Variant-specific hyperparameters, forwarded to
        :func:`~poraque.ml.kan.build_activation`. Ignored by the stateless
        variants.
    n_groups : int, optional
        Preferred number of ``GroupNorm`` groups; reduced to the largest
        divisor of ``channels`` that does not exceed it. Group normalization is
        used deliberately: batch statistics are meaningless when every sample
        has a different spatial extent, and group statistics are not.
    """

    def __init__(self, channels, modes, embedding_dim=0, activation="silu",
                 activation_kwargs=None, n_groups=8):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels, n_groups), channels)
        self.film = FiLM(embedding_dim, channels) if embedding_dim else None
        self.activation = build_activation(activation, channels,
                                           **(activation_kwargs or {}))

    def forward(self, x, embedding=None, max_modes=None):
        """Apply the layer to ``(B, C, Nx, Ny, Nz)``."""
        y = self.spectral(x, max_modes=max_modes) + self.pointwise(x)
        y = self.norm(y)
        if self.film is not None and embedding is not None:
            y = self.film(y, embedding)
        return x + self.activation(y)          # residual: stabilizes deep stacks


class FNO3d(nn.Module):
    r"""
    Fourier Neural Operator mapping one 3D scalar field to another.

    Parameters
    ----------
    in_channels, out_channels : int, optional
        Input/output field components (1 for a single scalar field).
    width : int, optional
        Channel width of the Fourier layers.
    modes : int or tuple of int, optional
        Mode capacity per axis.
    n_layers : int, optional
        Number of Fourier layers.
    projection_channels : int, optional
        Hidden width of the output projection.
    use_coordinates : bool, optional
        Append the three fractional coordinates as input channels. They are
        cheap, grid-shape agnostic, and give the operator a positional signal.
    cell_conditioning : bool, optional
        Enable :class:`CellEncoder` + :class:`FiLM`.
    embedding_dim : int, optional
        Width of the cell embedding.
    activation : str, optional
        Nonlinearity used throughout every :class:`FNOBlock`. One of
        :data:`~poraque.ml.kan.ACTIVATIONS`: the stateless ``'silu'``
        (default as of 2026-08-17; ``'gelu'`` was the default before then —
        see FUTURE.md for the measurements that motivated the switch),
        ``'gelu'``, ``'relu'``, ``'tanh'``, or the per-channel learnable
        Kolmogorov-Arnold variants ``'kan_bspline'`` / ``'kan_cheby'`` /
        ``'kan_rbf'`` / ``'kan_rational'`` (see :mod:`poraque.ml.kan`). Every
        learnable variant starts at initialisation close to ``silu`` (its own
        base term, matching the original KAN paper) and is opt-in: every
        existing config and checkpoint keeps using whatever ``activation`` it
        was trained with — recorded in the checkpoint, never inferred.
    kan_grid_size, kan_grid_range : optional
        Hyperparameters of ``'kan_bspline'`` and, since it reuses the same
        fixed-grid design, ``'kan_rbf'``; ignored otherwise. See
        :class:`~poraque.ml.kan.BSplineKANActivation` and
        :class:`~poraque.ml.kan.RBFKANActivation`.
    kan_spline_order : int, optional
        Hyperparameter of ``'kan_bspline'``; ignored otherwise.
    kan_degree : int, optional
        Hyperparameter of ``'kan_cheby'``; ignored otherwise. See
        :class:`~poraque.ml.kan.ChebyKANActivation`.
    kan_rational_num_degree, kan_rational_den_degree : int, optional
        Hyperparameters of ``'kan_rational'``; ignored otherwise. See
        :class:`~poraque.ml.kan.RationalKANActivation`.
    kan_use_base : bool, optional
        Whether each of the four learnable KAN variants includes its
        :math:`w_c\,\mathrm{SiLU}(x)` base term (``True``, the default,
        matching the original KAN paper) or is "pure" — residual only, no
        base weight, no fixed nonlinearity mixed in (``False``). Ignored by
        the stateless variants. See :mod:`poraque.ml.kan`.
    mode_selection : {"fixed", "physical"}, optional
        ``"fixed"`` truncates at a constant mode index (standard FNO).
        ``"physical"`` truncates at the constant wavevector :attr:`g_max`, so
        every material contributes the same band of physics. The latter
        requires ``cell`` at call time.
    g_max : float, optional
        Cutoff wavevector in Å⁻¹ for ``mode_selection="physical"``.

    Examples
    --------
    >>> import torch
    >>> model = FNO3d(width=8, modes=4, n_layers=2)
    >>> cell = torch.eye(3).unsqueeze(0) * 5.0
    >>> model(torch.randn(1, 1, 16, 16, 16), cell).shape
    torch.Size([1, 1, 16, 16, 16])
    >>> model(torch.randn(1, 1, 12, 20, 24), cell).shape   # different grid, same weights
    torch.Size([1, 1, 12, 20, 24])
    """

    def __init__(self, in_channels=1, out_channels=1, width=32, modes=12,
                 n_layers=4, projection_channels=128, use_coordinates=True,
                 cell_conditioning=True, embedding_dim=32, activation="silu",
                 kan_grid_size=8, kan_spline_order=3, kan_grid_range=(-2.0, 2.0),
                 kan_degree=6, kan_rational_num_degree=4,
                 kan_rational_den_degree=4, kan_use_base=True,
                 mode_selection="fixed", g_max=None,
                 projection_activation=None):
        super().__init__()
        if isinstance(modes, int):
            modes = (modes, modes, modes)
        if mode_selection not in ("fixed", "physical"):
            raise ValueError(f"Unknown mode_selection: {mode_selection!r}.")
        if mode_selection == "physical" and g_max is None:
            raise ValueError("mode_selection='physical' requires g_max (1/Ang).")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation!r}; expected one of "
                f"{sorted(ACTIVATIONS)}."
            )
        # `None` means "whatever the Fourier layers use", which is what a
        # caller means by `activation` unless they say otherwise. The read-out
        # used a hard-coded GELU until 2026-08-28; a checkpoint written before
        # then carries no record of this and `from_state` restores "gelu" for
        # it, so an old model still computes what it was trained to compute.
        projection_activation = str(projection_activation or activation)
        if projection_activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown projection_activation: {projection_activation!r}; "
                f"expected one of {sorted(ACTIVATIONS)}."
            )
        self.projection_activation = projection_activation

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes = tuple(int(m) for m in modes)
        self.use_coordinates = bool(use_coordinates)
        self.mode_selection = mode_selection
        self.g_max = None if g_max is None else float(g_max)
        # Recorded as plain attributes so FieldOperator.state() can persist
        # the parts of the architecture that live in no tensor shape: a
        # reloaded model must not silently fall back to the defaults.
        self.cell_conditioning = bool(cell_conditioning)
        self.embedding_dim = int(embedding_dim)
        self.activation = activation
        # KAN hyperparameters. Recorded unconditionally, the same way g_max is
        # recorded even under mode_selection="fixed": harmless when unused,
        # and it means a checkpoint's architecture record has one shape
        # regardless of which activation trained it.
        self.kan_grid_size = int(kan_grid_size)
        self.kan_spline_order = int(kan_spline_order)
        self.kan_grid_range = tuple(float(v) for v in kan_grid_range)
        self.kan_degree = int(kan_degree)
        self.kan_rational_num_degree = int(kan_rational_num_degree)
        self.kan_rational_den_degree = int(kan_rational_den_degree)
        self.kan_use_base = bool(kan_use_base)

        self.cell_encoder = CellEncoder(embedding_dim) if cell_conditioning else None
        conditioning = embedding_dim if cell_conditioning else 0

        activation_kwargs = dict(
            kan_grid_size=self.kan_grid_size,
            kan_spline_order=self.kan_spline_order,
            kan_grid_range=self.kan_grid_range,
            kan_degree=self.kan_degree,
            kan_rational_num_degree=self.kan_rational_num_degree,
            kan_rational_den_degree=self.kan_rational_den_degree,
            kan_use_base=self.kan_use_base,
        )
        lifting_channels = self.in_channels + (3 if use_coordinates else 0)
        self.lift = nn.Conv3d(lifting_channels, self.width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock(self.width, self.modes, embedding_dim=conditioning,
                     activation=activation, activation_kwargs=activation_kwargs)
            for _ in range(int(n_layers))
        ])
        self.project = nn.Sequential(
            nn.Conv3d(self.width, projection_channels, kernel_size=1),
            build_activation(self.projection_activation, projection_channels,
                             **activation_kwargs),
            nn.Conv3d(projection_channels, self.out_channels, kernel_size=1),
        )

    # ------------------------------------------------------------------ #
    def physical_modes(self, cell, shape):
        r"""
        Modes spanning ``|G| <= g_max`` for the given cell.

        The number of reciprocal-lattice points along axis :math:`i` inside a
        sphere of radius :math:`G_{\max}` is
        :math:`\lfloor G_{\max} L_i / 2\pi \rfloor`, with :math:`L_i` the
        lattice-vector length. This is what makes the truncation physical
        rather than merely index-based.

        Parameters
        ----------
        cell : torch.Tensor
            ``(B, 3, 3)`` lattice vectors in Å. The batch is reduced with a
            ``max`` so a shape-bucketed batch shares one truncation.
        shape : tuple of int
            Spatial shape of the sample.

        Returns
        -------
        tuple of int
        """
        lengths = torch.linalg.norm(cell, dim=-1).max(dim=0).values
        counts = torch.floor(self.g_max * lengths / (2.0 * np.pi)).long()
        return tuple(max(1, int(c)) for c in counts)

    @staticmethod
    def coordinate_channels(shape, device, dtype):
        """
        Fractional coordinates as ``(1, 3, Nx, Ny, Nz)``.

        Fractional (not Cartesian) coordinates are used so the channels stay in
        ``[0, 1)`` for every material regardless of cell size.
        """
        axes = [torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)[:-1]
                for n in shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=0).unsqueeze(0)

    def forward(self, x, cell=None):
        """
        Map an input field to an output field.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C_in, Nx, Ny, Nz)``.
        cell : torch.Tensor, optional
            ``(B, 3, 3)`` lattice vectors in Å. Required when cell
            conditioning or physical mode selection is enabled.

        Returns
        -------
        torch.Tensor
            ``(B, C_out, Nx, Ny, Nz)``.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a 5D (B,C,Nx,Ny,Nz) tensor, got {tuple(x.shape)}.")
        shape = tuple(x.shape[-3:])

        if self.use_coordinates:
            coordinates = self.coordinate_channels(shape, x.device, x.dtype)
            x = torch.cat([x, coordinates.expand(x.shape[0], -1, -1, -1, -1)], dim=1)

        embedding = None
        if self.cell_encoder is not None:
            if cell is None:
                raise ValueError("cell_conditioning=True requires `cell` at forward time.")
            embedding = self.cell_encoder(cell)

        max_modes = None
        if self.mode_selection == "physical":
            if cell is None:
                raise ValueError("mode_selection='physical' requires `cell` at forward time.")
            max_modes = self.physical_modes(cell, shape)

        v = self.lift(x)
        for block in self.blocks:
            v = block(v, embedding=embedding, max_modes=max_modes)
        return self.project(v)

    def n_parameters(self):
        """Total number of trainable parameters (complex weights count twice)."""
        return sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in self.parameters() if p.requires_grad
        )
