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

import warnings
from functools import lru_cache

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
def effective_modes(modes, shape, max_modes=None):
    """
    Modes actually usable on a grid of ``shape``, given a mode capacity.

    Shared by both spectral convolutions rather than written twice: the
    truncation rule is the same statement about the grid whether the kernel
    that follows it is dense or radial, and two copies of it would be two
    places for a physical cutoff to stop agreeing with an index one.

    Parameters
    ----------
    modes : tuple of int
        Per-axis mode *capacity* — how many weights exist.
    shape : tuple of int
        Spatial shape ``(Nx, Ny, Nz)``.
    max_modes : tuple of int, optional
        Extra per-axis cap, e.g. from a physical ``G_max``.

    Returns
    -------
    tuple of int
        ``(m1, m2, m3)``, each at least 1 and each within the grid.
    """
    nx, ny, nz = shape
    available = (nx // 2, ny // 2, nz // 2 + 1)
    caps = max_modes if max_modes is not None else modes
    return tuple(
        max(1, min(int(modes[i]), int(caps[i]), int(available[i])))
        for i in range(3)
    )


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
        return effective_modes(self.modes, shape, max_modes)

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
# Rotation-equivariant spectral convolution
# ---------------------------------------------------------------------- #
#: Radius in Å⁻¹ that the radial basis spans when nothing else fixes one.
#:
#: A run with ``mode_selection="physical"`` has already named the band it cares
#: about, and ``g_basis`` defaults to that ``g_max``; this value is reached by
#: every ``"fixed"`` run, **including one that states a** ``g_max``, since that
#: key truncates nothing there and has no business resizing the basis. 8 Å⁻¹
#: is :math:`|\mathbf{G}|` at mode 8 in a 6.3 Å cell, which covers this
#: project's own platinum cells (7–8 Å at ``modes=8``) with the outermost node
#: just past the corner of the retained box. It is a *scale*, not a cutoff:
#: modes beyond it are clamped onto the last basis function rather than
#: dropped, so the operator's high-frequency response goes flat instead of to
#: zero.
DEFAULT_G_BASIS = 8.0

#: Relative slack that lifts the spherical cutoff off the shell it sits on.
#:
#: The cutoff is a comparison between two floating-point routes to the same
#: quantity: ``radius`` comes from the reciprocal cell contracted against the
#: integer frequencies, ``inscribed`` from :math:`2\pi m_i/|a_i|`. They agree
#: analytically and differ in the last bits, so a mode sitting *exactly* on
#: :math:`|\mathbf{G}| = g_{\rm inscribed}` — the face-centre modes
#: :math:`(\pm m, 0, 0)` and their rotations — falls on whichever side of the
#: strict ``<`` the round-off puts it, and the three axes need not agree. One
#: kept where its own rotation image was dropped is a discrete change in the
#: retained set, and therefore in the operator: measured, **6 of 20 cubic
#: cells swept from 4 Å to 13.5 Å lost their equivariance in float32**, at
#: :math:`3\times10^{-4}` to :math:`1.7\times10^{-3}` rather than the
#: :math:`3\times10^{-7}` of the cells that happen not to tie. It is not a
#: float32 phenomenon either — two of the same twenty tie in float64.
#:
#: The defect is exactly two modes, and always the same two. A tying cell
#: keeps 150 of them where the other precision keeps 148, and the surplus is
#: the pair :math:`(-m, 0, 0)` and :math:`(0, -m, 0)`. The third face is the
#: one ``rfftn`` never stored — axis 3 carries :math:`[0, m_3)` and nothing
#: negative — which is what makes a retained face *asymmetric* rather than
#: merely extra, and is why the cutoff has to exclude the shell rather than
#: include it consistently. Over 200 cubic cells from 3 Å to 13 Å the float32
#: and float64 retained sets disagreed on 69 to 83 of them, at every mode
#: count from 4 to 16, and on none afterwards.
#:
#: Excluding the shell with a relative slack makes membership independent of
#: which route computed the radius. The value has to sit far above the
#: arithmetic disagreement (a few ulps) and far below the spacing between mode
#: shells, whose nearest relative gap below the cutoff is :math:`1/2m^2` —
#: :math:`8\times10^{-3}` at ``modes=8``, :math:`5\times10^{-4}` at 32. 1e-5
#: is two orders clear of both ends. Every cell that was already exact is
#: unchanged to every digit printed: the slack removes the tie shell and
#: touches nothing else.
CUTOFF_SLACK = 1e-5


def default_g_basis(g_max=None, mode_selection="fixed"):
    r"""
    The radial-basis radius a run gets when it states none.

    One function rather than a rule written twice: :class:`FNO3d` applies it,
    and ``poraque-train`` reports the band against it before the model exists.
    Two copies would be two places for the diagnostic to describe an
    architecture other than the one built --- which is the exact class of
    defect it was added to surface.

    Parameters
    ----------
    g_max : float or None
        ``model.g_max``.
    mode_selection : str
        ``"fixed"`` or ``"physical"``.

    Returns
    -------
    float
        ``g_max`` under ``"physical"``, where the retained support is
        :math:`\min_i 2\pi m_i/L_i` with :math:`m_i = \lfloor g_{\max}
        L_i/2\pi \rfloor` and therefore fits inside ``g_max`` by
        construction; :data:`DEFAULT_G_BASIS` otherwise.
    """
    if g_max is not None and mode_selection == "physical":
        return float(g_max)
    return float(DEFAULT_G_BASIS)


@lru_cache(maxsize=64)
def _mode_frequencies(modes, device, dtype):
    r"""
    Signed integer frequencies of a retained mode block, ``(3, 2m1, 2m2, m3)``.

    The block's axis order is the one :class:`RadialSpectralConv3d` gathers in:
    ``[0, 1, …, m-1, -m, …, -1]`` on the two full axes, and ``[0, …, m3-1]`` on
    the ``rfftn`` axis, which carries no negative half. Reading these as signed
    frequencies rather than as array indices is the whole point — the radius
    :math:`|\mathbf{G}|` of mode :math:`-1` is that of mode :math:`+1`, and an
    unsigned index would put it at the far edge of the basis.

    Memoised on the same argument as
    :func:`~poraque.ml.physics._integer_mesh` and for the same reason: it
    depends on the mode counts and not on the cell, while a training step asks
    for it once per Fourier layer per batch.

    Parameters
    ----------
    modes : tuple of int
        ``(m1, m2, m3)``, already reduced to what the grid supports. Must be a
        plain tuple: it is a cache key.
    device : torch.device
    dtype : torch.dtype
        A *real* dtype — these are the frequencies the radius is built from,
        not the coefficients it multiplies.

    Returns
    -------
    torch.Tensor
        ``(3, 2m1, 2m2, m3)``. Shared between callers, so it must not be
        mutated in place.
    """
    m1, m2, m3 = modes
    axis_1 = torch.cat([torch.arange(0, m1, device=device, dtype=dtype),
                        torch.arange(-m1, 0, device=device, dtype=dtype)])
    axis_2 = torch.cat([torch.arange(0, m2, device=device, dtype=dtype),
                        torch.arange(-m2, 0, device=device, dtype=dtype)])
    axis_3 = torch.arange(0, m3, device=device, dtype=dtype)
    return torch.stack(
        torch.meshgrid(axis_1, axis_2, axis_3, indexing="ij"), dim=0)


@lru_cache(maxsize=64)
def _mode_counts(modes, device, dtype):
    """
    ``(m1, m2, m3)`` as a device tensor, memoised.

    Three numbers, and building them cost a host allocation and a
    host-to-device copy on **every forward pass of every equivariant layer** --
    ``torch.tensor([...], device=...)`` is a transfer however small its
    payload, and on CUDA a small transfer is dominated by its launch rather
    than by its bytes. They are constant for the run, so they are built once.

    Memoised on the same key as :func:`_mode_frequencies`, and bounded for the
    same reason: the distinct mode counts in a run are the distinct grid shapes
    a :class:`~poraque.ml.data.ShapeBucketSampler` produces, which was 19 on
    the dataset this was measured against.
    """
    return torch.tensor([float(m) for m in modes], device=device, dtype=dtype)


def retained_radii(cell, modes, device=None, dtype=None):
    r"""
    :math:`|\mathbf{G}|` of every mode a spectral layer retains, and which of
    them survive the spherical cutoff.

    Written once and called twice: by
    :meth:`RadialSpectralConv3d.radial_basis`, which turns the radius into a
    kernel, and by :func:`retained_band`, which reports what band a dataset
    asks for so ``g_basis`` can be sized against it. Two copies of this
    geometry would be two places for the diagnostic to stop describing the
    operator --- which is the whole failure the diagnostic exists to catch.

    Parameters
    ----------
    cell : torch.Tensor or array_like
        ``(B, 3, 3)`` lattice vectors in Å; a bare ``(3, 3)`` is read as one
        sample.
    modes : tuple of int
        ``(m1, m2, m3)`` actually retained on this grid --- the output of
        :func:`effective_modes`, not the raw capacity.
    device : torch.device, optional
        Defaults to the cell's.
    dtype : torch.dtype, optional
        Real dtype to compute in. Defaults to the cell's.

    Returns
    -------
    radius : torch.Tensor
        ``(B, 2m1, 2m2, m3)``, in Å⁻¹.
    kept : torch.Tensor
        ``(B, 2m1, 2m2, m3)`` boolean: inside the largest sphere inscribed in
        the retained box.
    """
    # Local, like `CellEncoder.descriptors`: `ml.physics` imports the
    # transforms, so a module-level import here closes a cycle.
    from .physics import cell_reciprocal

    cell = torch.as_tensor(cell)
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    device = cell.device if device is None else torch.device(device)
    dtype = cell.dtype if dtype is None else dtype
    modes = tuple(int(m) for m in modes)

    frequencies = _mode_frequencies(modes, device, dtype)
    reciprocal = cell_reciprocal(cell, device=device, dtype=dtype)
    # G_alpha = sum_j n_j b_{j alpha}; the same contraction
    # `reciprocal_vectors` performs, over the mode block rather than the whole
    # grid.
    vectors = torch.einsum("bja,jxyz->baxyz", reciprocal, frequencies)
    radius = torch.linalg.vector_norm(vectors, dim=1)   # (B, 2m1, 2m2, m3)

    # The plane n_i = m_i lies at 2*pi*m_i / |a_i| from the origin, so the
    # largest sphere inside the retained parallelepiped has that radius,
    # minimised over the three axes. Cell lengths are all it takes, for any
    # cell -- no inverse, no per-mode geometry.
    lengths = torch.linalg.vector_norm(cell.to(device=device, dtype=dtype),
                                       dim=-1)                    # (B, 3)
    counts = _mode_counts(modes, lengths.device, lengths.dtype)
    inscribed = (2.0 * np.pi * counts / lengths).amin(dim=-1)
    # Strictly inside, not on the sphere, and the strictness is what makes the
    # retained set symmetric. `rfftn` gives axis i the frequencies [0, m_i) and
    # [-m_i, 0): the mode -m_i is kept and +m_i is not, so the box is lopsided
    # by exactly one face per axis. |G| < 2*pi*m_i/|a_i| forces |n_i| < m_i on
    # every axis at once, which excludes that face -- and the remaining ball is
    # then both rotation-invariant and closed under inversion, without which
    # the equivariance is good to 2e-3 rather than to the float.
    #
    # `CUTOFF_SLACK` is what makes that exclusion survive round-off: the shell
    # modes are exactly *on* the boundary, and the two sides of the comparison
    # are computed by different routes. See the constant.
    inscribed = inscribed * (1.0 - CUTOFF_SLACK)
    return radius, radius < inscribed.view(-1, 1, 1, 1)


class RadialSpectralConv3d(nn.Module):
    r"""
    Spectral convolution whose kernel depends on :math:`|\mathbf{G}|` alone.

    :class:`SpectralConv3d` learns one complex number per retained mode, so it
    can — and does — treat :math:`\mathbf{G}` and :math:`R\mathbf{G}`
    differently for a rotation :math:`R`. That freedom is what stops the
    surrounding network from being equivariant, and it is the *only* thing that
    does: every other operation in :class:`FNOBlock` is pointwise in the voxel
    index (the 1×1×1 convolution, the activation), reduces over statistics a
    permutation of voxels leaves alone (``GroupNorm``), or is conditioned on
    quantities that are already rotation-invariant (:class:`CellEncoder`'s
    lengths, angle cosines and volume).

    Constraining the multiplier to

    .. math::

        R_{io}(\mathbf{G}) = \sum_{r} c_{ior}\,\varphi_r(|\mathbf{G}|),
        \qquad c_{ior} \in \mathbb{R},

    makes the layer a convolution with a **real radial kernel**, and a
    convolution with a radial kernel commutes with every rotation. Two
    consequences follow that are worth stating rather than deriving twice:

    * the coefficients are **real**, not complex, because
      :math:`R(-\mathbf{G}) = R(\mathbf{G})` for a radial multiplier and a real
      output field needs :math:`R(-\mathbf{G}) = \overline{R(\mathbf{G})}`.
      Both hold only for a real :math:`R`. Nothing is lost by it: a real-space
      kernel that is a function of :math:`|\mathbf{r}|` is real and
      centrosymmetric already, so the constrained class is exactly the radial
      one and not a subset of it. The layer is therefore equivariant under the
      full :math:`O(3)`, inversion included, rather than only the
      :math:`SE(3)` the construction was asked for;
    * the arithmetic needs **no complex** ``einsum``. The real and imaginary
      parts of the spectrum are contracted separately against the same real
      coefficients, which is why this class does not call
      :func:`complex_contract` and pays none of its MPS workaround.

    The cost is capacity, and it is large. The dense layer holds
    :math:`4\,C_{\rm in}C_{\rm out}m_1m_2m_3` complex numbers; this one holds
    :math:`C_{\rm in}C_{\rm out}R` real ones — at ``width=16``, ``modes=8`` and
    ``n_radial=16`` that is 4 096 against 1 048 576, a factor of 256. An
    equivariant model of the same width is a much smaller model, and the width
    is where that is bought back.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts.
    modes : tuple of int
        Mode *capacity* per axis, as for :class:`SpectralConv3d`.
    n_radial : int
        Number of radial basis functions. Sets the resolution of the kernel in
        :math:`|\mathbf{G}|`, and is the layer's entire angular-to-radial
        trade: it replaces :math:`m_1m_2m_3` per channel pair. Also its entire
        overfitting surface --- see :class:`FNO3d` on why raising it first is
        the wrong reflex on a small set.
    g_basis : float
        Radius in Å⁻¹ that the basis spans; see :data:`DEFAULT_G_BASIS` for the
        constant and :func:`resolve_g_basis` for reading it off the data, which
        is what a set of mixed cell sizes needs.
    spherical_cutoff : bool
        Discard retained modes outside the largest sphere inscribed in the
        retained box. **A radial multiplier is only half of equivariance**: the
        set it is applied over has to be rotation-invariant too, and the
        retained set is a box. On a cubic cell with :math:`m_1 = m_2 = m_3` the
        box happens to be invariant under the octahedral group, which is why
        the lattice-symmetry test passes either way; for any other cell, or
        whenever the grid caps one axis and not another, the corners of the box
        are modes whose rotations were never retained, and the operator
        distinguishes :math:`\mathbf{G}` from :math:`R\mathbf{G}` again.
        Masking to the inscribed sphere costs about half the retained
        coefficients — the ball is 52 % of the cube — and buys equivariance
        under *every* rotation rather than under the twenty-four the box
        tolerates. Off is available and is a deliberate weakening.

    Notes
    -----
    The basis is Gaussian on a uniform grid of :math:`u = |\mathbf{G}|/g_{\rm
    basis} \in [0, 1]`, with the width set to the node spacing so neighbours
    cross at :math:`e^{-1/4}` and the partition is smooth without being flat.
    The radius is computed from the sample's **own cell**, so the same
    coefficient means the same physical wavevector in every material — which
    is the property that makes a radial kernel a statement about space rather
    than about an index box.
    """

    def __init__(self, in_channels, out_channels, modes=(12, 12, 12),
                 n_radial=16, g_basis=DEFAULT_G_BASIS, spherical_cutoff=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(m) for m in modes)
        if any(m < 1 for m in self.modes):
            raise ValueError(f"All mode counts must be >= 1, got {modes!r}.")
        self.n_radial = int(n_radial)
        if self.n_radial < 2:
            raise ValueError(
                f"n_radial must be >= 2, got {n_radial!r}: a single basis "
                f"function is a kernel with no dependence on |G| at all, "
                f"which is a scalar multiple of the identity.")
        self.g_basis = float(g_basis)
        if not self.g_basis > 0.0:
            raise ValueError(f"g_basis must be positive, got {g_basis!r}.")
        self.spherical_cutoff = bool(spherical_cutoff)

        # He-style scaling, as in the dense layer, but over the radial index:
        # that is what the channel sum now runs against.
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels,
                                self.n_radial)
        )
        # Buffers rather than constants so `.to(device)` and `.double()` carry
        # them with the parameters; a basis left on the CPU in float32 would
        # fail the first forward pass of a float64 CUDA model.
        self.register_buffer("centres",
                             torch.linspace(0.0, 1.0, self.n_radial))
        self.register_buffer(
            "inverse_width",
            torch.tensor(float(self.n_radial - 1)))

    def effective_modes(self, shape, max_modes=None):
        """Modes usable on a grid of ``shape``; see :func:`effective_modes`."""
        return effective_modes(self.modes, shape, max_modes)

    def radial_basis(self, cell, modes, device, dtype):
        r"""
        Evaluate the basis at every retained mode: ``(B, R, 2m1, 2m2, m3)``.

        Parameters
        ----------
        cell : torch.Tensor
            ``(B, 3, 3)`` lattice vectors in Å.
        modes : tuple of int
            ``(m1, m2, m3)`` actually retained on this grid.
        device : torch.device
        dtype : torch.dtype
            Real dtype of the network's activations.

        Returns
        -------
        torch.Tensor
            ``(B, R, 2m1, 2m2, m3)`` real basis values.
        """
        radius, kept = retained_radii(cell, modes, device=device, dtype=dtype)

        # Clamped rather than extrapolated: beyond `g_basis` every mode shares
        # the outermost basis function, so the response goes flat. Letting the
        # Gaussians decay instead would silently zero the high modes of any
        # cell larger than the basis was sized for. Free in the measurements
        # LNCC ran -- 71 % of modes clamped cost nothing -- while a `g_basis`
        # wider than the retained band leaves basis functions where no mode
        # exists and costs 14-43 %. See :func:`resolve_g_basis`.
        reduced = (radius / self.g_basis).clamp(max=1.0).unsqueeze(1)
        centres = self.centres.to(dtype).view(1, -1, 1, 1, 1)
        basis = torch.exp(
            -((reduced - centres) * self.inverse_width.to(dtype)) ** 2)

        if self.spherical_cutoff:
            basis = basis * kept.unsqueeze(1).to(basis.dtype)
        return basis

    def forward(self, x, cell, max_modes=None, basis=None):
        """
        Apply the equivariant spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C_in, Nx, Ny, Nz)`` real tensor.
        cell : torch.Tensor
            ``(B, 3, 3)`` lattice vectors in Å. Required, unlike the dense
            layer: the kernel is a function of a physical radius, and there is
            no radius without a cell.
        max_modes : tuple of int, optional
            Per-axis cap on the retained modes for this call.
        basis : torch.Tensor, optional
            A basis from :meth:`radial_basis` for this batch and these modes,
            computed once by :meth:`FNO3d.forward` and shared by every layer in
            the stack. Omitted, the layer builds its own --- which is what a
            standalone call does, and what every call did before the sharing
            was added.

        Returns
        -------
        torch.Tensor
            ``(B, C_out, Nx, Ny, Nz)``.
        """
        if cell is None:
            raise ValueError(
                "RadialSpectralConv3d requires `cell`: its kernel is a "
                "function of |G| in 1/Ang, which the mode index alone does "
                "not determine.")
        batch, _, nx, ny, nz = x.shape
        m1, m2, m3 = self.effective_modes((nx, ny, nz), max_modes)

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
        head_1, tail_1 = slice(None, m1), slice(nx - m1, None)
        head_2, tail_2 = slice(None, m2), slice(ny - m2, None)

        # One contiguous block of the retained modes, ordered to match
        # `_mode_frequencies`. `effective_modes` caps m1 at nx//2 and m2 at
        # ny//2, so the head and tail slices never overlap and no coefficient
        # is gathered twice.
        block = torch.cat([
            torch.cat([x_ft[:, :, head_1, head_2, :m3],
                       x_ft[:, :, head_1, tail_2, :m3]], dim=-2),
            torch.cat([x_ft[:, :, tail_1, head_2, :m3],
                       x_ft[:, :, tail_1, tail_2, :m3]], dim=-2),
        ], dim=-3)

        if basis is None:
            basis = self.radial_basis(cell, (m1, m2, m3), x.device, x.dtype)
        elif tuple(basis.shape) != (batch, self.n_radial, 2 * m1, 2 * m2, m3):
            # A shared basis is an optimisation, and an optimisation that can
            # be handed the wrong tensor has to say so. Every axis but the
            # radial one would broadcast against the mode block, so a basis
            # built for another grid or another sample would return a finite
            # field computed from the wrong geometry.
            raise ValueError(
                f"basis has shape {tuple(basis.shape)}, expected "
                f"{(batch, self.n_radial, 2 * m1, 2 * m2, m3)} for a "
                f"{(m1, m2, m3)} mode block of {batch} sample(s).")

        # Real arithmetic throughout. `torch.cat` above returns a contiguous
        # tensor, so `.real`/`.imag` are dense views rather than the strided
        # ones `complex_contract` documents MPS getting wrong.
        #
        # Two contractions rather than one, and that was measured. Stacking the
        # real and imaginary halves into a single `(2B, ...)` einsum halves the
        # kernel launches, but the `torch.cat` and the `basis.repeat` it needs
        # copy more than the launches cost: 3.83 ms against 5.26 ms on MPS at
        # batch 8, width 32, n_radial 32. The launches are not what this layer
        # is spending its time on -- the FFTs are.
        real, imaginary = block.real, block.imag
        weight = self.weight.to(real.dtype)
        out_real = torch.einsum(
            "borxyz,brxyz->boxyz",
            torch.einsum("bixyz,ior->borxyz", real, weight), basis)
        out_imaginary = torch.einsum(
            "borxyz,brxyz->boxyz",
            torch.einsum("bixyz,ior->borxyz", imaginary, weight), basis)
        out_block = torch.complex(out_real, out_imaginary)

        out_ft = torch.zeros(
            batch, self.out_channels, nx, ny, nz // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )
        out_ft[:, :, head_1, head_2, :m3] = out_block[:, :, :m1, :m2, :]
        out_ft[:, :, head_1, tail_2, :m3] = out_block[:, :, :m1, m2:, :]
        out_ft[:, :, tail_1, head_2, :m3] = out_block[:, :, m1:, :m2, :]
        out_ft[:, :, tail_1, tail_2, :m3] = out_block[:, :, m1:, m2:, :]

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1),
                                norm="forward")

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, modes={self.modes}, "
                f"n_radial={self.n_radial}, g_basis={self.g_basis}, "
                f"spherical_cutoff={self.spherical_cutoff}")


# ---------------------------------------------------------------------- #
# Sizing the radial basis against the data
# ---------------------------------------------------------------------- #
def retained_band(cells, shapes, modes, mode_selection="fixed", g_max=None,
                  spherical_cutoff=True, g_basis=None):
    r"""
    What band of :math:`|\mathbf{G}|` an operator retains over a set of samples.

    The radial basis spans ``[0, g_basis]`` while the *retained* band is a
    property of each material's own cell, and the two are unrelated until
    somebody checks. Over the 92 training materials of LNCC's six-metal set the
    retained band edge ran from 1.94 to 20.96 Å⁻¹ at ``modes=8`` --- cell
    lengths span 2.29 to 25.87 Å --- so no single constant fits the spread, and
    the run said nothing about which end of it a config had landed on.

    The two ways to miss are **not** symmetric, which is the finding this
    function exists to make visible:

    * **Too narrow.** Modes beyond ``g_basis`` are clamped onto the outermost
      basis function and share one response. Measured cost: none. 71.3 % of
      modes clamped moved the validation error by less than the seed spread.
    * **Too wide.** Basis functions sit at radii no mode reaches, and are dead
      capacity in a layer whose capacity *is* the radial bank. Measured cost:
      14 % to 43 %, with six of sixteen functions orphaned.

    Parameters
    ----------
    cells : sequence of array_like
        One ``(3, 3)`` cell per sample, in Å.
    shapes : sequence of tuple of int
        The matching grid shapes.
    modes : int or tuple of int
        Mode capacity, as :class:`FNO3d` takes it.
    mode_selection : str, optional
        ``"fixed"`` or ``"physical"``; the latter needs ``g_max``.
    g_max : float, optional
        Physical cutoff in Å⁻¹.
    spherical_cutoff : bool, optional
        Count only the modes inside the inscribed sphere, which is what an
        equivariant layer keeps.
    g_basis : float, optional
        When given, the clamped fractions are measured against it.

    Returns
    -------
    dict
        ``edges`` (per-sample band edge, Å⁻¹), ``retained`` (per-sample mode
        counts), ``n_samples``, and --- when ``g_basis`` was given ---
        ``clamped_fraction`` over all retained modes, ``clamped_samples``
        (materials with any mode beyond the basis) and ``inside_samples``
        (materials wholly within it).
    """
    if isinstance(modes, int):
        modes = (modes, modes, modes)
    modes = tuple(int(m) for m in modes)
    if mode_selection not in ("fixed", "physical"):
        raise ValueError(f"Unknown mode_selection: {mode_selection!r}.")
    if mode_selection == "physical" and g_max is None:
        raise ValueError("mode_selection='physical' requires g_max (1/Ang).")

    edges, retained, beyond = [], [], []
    for cell, shape in zip(cells, shapes):
        # float64 throughout: this is a measurement of the geometry, and the
        # cutoff comparison is the one place in this file where the last bits
        # decide a discrete outcome (see CUTOFF_SLACK).
        cell = torch.as_tensor(np.asarray(cell, dtype=float)).reshape(1, 3, 3)
        max_modes = None
        if mode_selection == "physical":
            lengths = torch.linalg.vector_norm(cell[0], dim=-1)
            max_modes = tuple(
                max(1, int(np.floor(float(g_max) * float(length)
                                    / (2.0 * np.pi))))
                for length in lengths)
        block = effective_modes(modes, tuple(int(n) for n in shape), max_modes)
        radius, kept = retained_radii(cell, block, dtype=torch.float64)
        radius = radius[kept] if spherical_cutoff else radius.reshape(-1)

        retained.append(int(radius.numel()))
        edges.append(float(radius.max()) if radius.numel() else 0.0)
        if g_basis is not None:
            beyond.append(int((radius > float(g_basis)).sum()))

    report = {"n_samples": len(edges),
              "edges": np.asarray(edges, dtype=float),
              "retained": np.asarray(retained, dtype=int)}
    if g_basis is None or not edges:
        return report

    beyond = np.asarray(beyond, dtype=int)
    total = int(report["retained"].sum())
    report["clamped_fraction"] = float(beyond.sum()) / total if total else 0.0
    report["clamped_samples"] = int((beyond > 0).sum())
    report["inside_samples"] = int((beyond == 0).sum())
    return report


def resolve_g_basis(cells, shapes, modes, **kwargs):
    r"""
    Size the radial basis from the data: the **median** retained band edge.

    ``g_basis: auto`` resolves through here, before the model is built, so the
    number a checkpoint records is a number and not a word.

    The median is what fits the measurements rather than what sounds tidy. Of
    the seven configurations LNCC ran where the basis and the band could be
    compared, the four that performed well had ``g_basis`` within a few percent
    of the median band edge (24 against 22.8, 32 against 30.0, 32 against 31.2,
    and at ``modes=16`` a basis of 20 beating one of 6 in both seeds), and the
    one that performed badly --- 14 % worse at one resolution and 43 % at
    another --- had a basis of 48 against a median edge of 31.2, which left six
    of its sixteen radial functions at radii no mode reaches.

    Half the set is then clamped, and that is the deliberate half: clamping is
    free and orphaning is not (see :func:`retained_band`). The maximum would
    orphan nothing on the largest cell and everything on the smallest.

    Parameters
    ----------
    cells, shapes, modes
        As :func:`retained_band`.
    **kwargs
        ``mode_selection``, ``g_max``, ``spherical_cutoff``; forwarded.

    Returns
    -------
    float
        Å⁻¹. Falls back to :data:`DEFAULT_G_BASIS` for an empty set, or for one
        whose band edges are all zero --- neither is a dataset this can be
        asked about, and raising would stop a run over a diagnostic.
    """
    report = retained_band(cells, shapes, modes, g_basis=None, **kwargs)
    edges = report["edges"]
    if not edges.size or not float(np.median(edges)) > 0.0:
        return float(DEFAULT_G_BASIS)
    return float(np.median(edges))


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
        / ``'kan_chebyshev'`` / ``'kan_rbf'`` / ``'kan_rational'`` (per-channel
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
    equivariant : bool, optional
        Use :class:`RadialSpectralConv3d` in place of :class:`SpectralConv3d`,
        making the layer — and, given a lifting stage with no coordinate
        channels, the whole network — equivariant under rotation. Everything
        else in this block already is.
    radial_kwargs : dict, optional
        ``n_radial``, ``g_basis`` and ``spherical_cutoff``, forwarded to the
        radial layer. Ignored when ``equivariant`` is false.
    """

    def __init__(self, channels, modes, embedding_dim=0, activation="silu",
                 activation_kwargs=None, n_groups=8, equivariant=False,
                 radial_kwargs=None):
        super().__init__()
        self.equivariant = bool(equivariant)
        if self.equivariant:
            self.spectral = RadialSpectralConv3d(channels, channels, modes,
                                                 **(radial_kwargs or {}))
        else:
            self.spectral = SpectralConv3d(channels, channels, modes)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels, n_groups), channels)
        self.film = FiLM(embedding_dim, channels) if embedding_dim else None
        self.activation = build_activation(activation, channels,
                                           **(activation_kwargs or {}))

    def forward(self, x, embedding=None, max_modes=None, cell=None,
                basis=None):
        """Apply the layer to ``(B, C, Nx, Ny, Nz)``."""
        # The cell is passed only to the layer that needs it: the dense
        # spectral convolution's weights are indexed by mode and know nothing
        # about a lattice, and giving it an argument it ignores would make the
        # two layers look interchangeable in a way they are not. The same goes
        # for `basis`, which only exists for the radial kernel.
        if self.equivariant:
            y = (self.spectral(x, cell, max_modes=max_modes, basis=basis)
                 + self.pointwise(x))
        else:
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
        Kolmogorov-Arnold variants ``'kan_bspline'`` / ``'kan_chebyshev'`` /
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
        Hyperparameter of ``'kan_chebyshev'``; ignored otherwise. See
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
    equivariant : bool, optional
        Build every Fourier layer from :class:`RadialSpectralConv3d` instead of
        :class:`SpectralConv3d`, making the whole operator equivariant under
        rotation: rotate the input field and the predicted field rotates with
        it, to the precision of the float. Requires ``use_coordinates=False``
        and raises otherwise — the fractional-coordinate channels are a
        :math:`\ell = 1` object being fed to a network that treats every
        channel as a scalar, and they break translation equivariance as well.
        Off by default, so every existing config and checkpoint is unaffected.
    n_radial : int, optional
        Radial basis size of the equivariant layers; ignored otherwise. This is
        the whole capacity of a kernel with no angular freedom left --- and
        ``modes`` adds none of it, the coefficient count being independent of
        the retained band.

        Which does not make it the knob to raise *first*, though this docstring
        said so until 2026-09-04. On 92 training materials the error rose
        monotonically with ``n_radial`` across 8, 16, 32 and 64, while raising
        ``modes`` from 4 to 16 improved it 2.7× at *identical* parameter count
        for 36 % more time per epoch. On a small training set the radial bank
        is where the excess capacity accumulates and the bandwidth is free.
        Expect the ordering to invert as the set grows; that is untested.
    g_basis : float, optional
        Radius in Å⁻¹ the radial basis spans; ignored otherwise. Defaults to
        ``g_max`` under ``mode_selection="physical"``, where the retained band
        provably fits inside it, and to :data:`DEFAULT_G_BASIS` otherwise ---
        including under ``"fixed"`` with a ``g_max`` stated, where that key
        truncates nothing and must not resize the basis either.

        Prefer :func:`resolve_g_basis` over a hand-picked constant: the useful
        value tracks the band the modes actually retain, which varies by an
        order of magnitude across a set of mixed cell sizes.
    spherical_cutoff : bool, optional
        Mask the retained modes to the sphere inscribed in the retained box.
        On by default, and it is not decoration: a radial multiplier over a
        *box* of modes is still equivariant only under the box's own symmetry
        group. Measured on a tetragonal cell, switching it off takes the
        rotation error from :math:`3\times10^{-7}` — float32 round-off — to
        :math:`5\times10^{-2}`.

    Examples
    --------
    >>> import torch
    >>> model = FNO3d(width=8, modes=4, n_layers=2)
    >>> cell = torch.eye(3).unsqueeze(0) * 5.0
    >>> model(torch.randn(1, 1, 16, 16, 16), cell).shape
    torch.Size([1, 1, 16, 16, 16])
    >>> model(torch.randn(1, 1, 12, 20, 24), cell).shape   # different grid, same weights
    torch.Size([1, 1, 12, 20, 24])

    The equivariant variant, whose kernel is a function of :math:`|\mathbf{G}|`
    alone:

    >>> rotating = FNO3d(width=8, modes=4, n_layers=2, equivariant=True,
    ...                  use_coordinates=False)
    >>> rotating.blocks[0].spectral.weight.is_complex()   # real coefficients
    False
    """

    def __init__(self, in_channels=1, out_channels=1, width=32, modes=12,
                 n_layers=4, projection_channels=128, use_coordinates=True,
                 cell_conditioning=True, embedding_dim=32, activation="silu",
                 kan_grid_size=8, kan_spline_order=3, kan_grid_range=(-2.0, 2.0),
                 kan_degree=6, kan_rational_num_degree=4,
                 kan_rational_den_degree=4, kan_use_base=True,
                 mode_selection="fixed", g_max=None,
                 projection_activation=None,
                 equivariant=False, n_radial=16, g_basis=None,
                 spherical_cutoff=True):
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
        if equivariant and self.use_coordinates:
            raise ValueError(
                "equivariant=True is incompatible with use_coordinates=True. "
                "The three fractional-coordinate channels are not three "
                "scalar fields: under a rotation of the cell they transform "
                "into each other, so feeding them to a network built to treat "
                "every channel as a scalar breaks the equivariance the "
                "spectral layer was constrained to provide -- and they break "
                "translation equivariance outright, being absolute positions. "
                "Set use_coordinates: false.")
        self.mode_selection = mode_selection
        self.g_max = None if g_max is None else float(g_max)
        # Recorded as plain attributes so FieldOperator.state() can persist
        # the parts of the architecture that live in no tensor shape: a
        # reloaded model must not silently fall back to the defaults.
        self.cell_conditioning = bool(cell_conditioning)
        self.embedding_dim = int(embedding_dim)
        self.activation = activation
        # Rotational equivariance. Recorded whether or not it is on, for the
        # same reason `g_max` is: a checkpoint's architecture record has one
        # shape regardless of which variant trained it.
        self.equivariant = bool(equivariant)
        self.n_radial = int(n_radial)
        self.spherical_cutoff = bool(spherical_cutoff)
        # `g_max` when the run has already named a physical band, and only then
        # the bare default. Resolving it here rather than at forward time is
        # what puts the number in the architecture record, where a reader can
        # see which of the two a model was trained with.
        #
        # `mode_selection` is part of that condition, and it was not until
        # 2026-09-04. Under "physical" the coupling is sound: the retained
        # support is `min_i 2*pi*m_i/L_i` with `m_i = floor(g_max*L_i/2*pi)`,
        # which is <= g_max by construction, so a basis spanning g_max spans
        # the band. Under "fixed" g_max truncates nothing -- the docstring says
        # so, and a reader concludes the key is inert -- and it was silently
        # resizing the basis anyway. LNCC ran an entire 90-run architecture
        # study through a leftover `g_max: 6`, so every equivariant model in it
        # was built with a 6.0 basis rather than the 8.0 default, with nothing
        # in the log, the resolved config or the run record saying so.
        if isinstance(g_basis, str):
            # "auto" is resolved from the training split before the model is
            # built (`resolve_g_basis`), because the band a sample retains
            # depends on its cell and a constructor has seen no samples.
            raise ValueError(
                f"g_basis={g_basis!r} is not a number. `g_basis: auto` is "
                f"resolved from the training split by "
                f"`poraque.ml.fno.resolve_g_basis` before the model is built, "
                f"so that the number reaching the architecture record is a "
                f"number: the band a sample retains depends on its cell, and "
                f"a constructor has seen no cells.")
        if g_basis is not None:
            self.g_basis = float(g_basis)
        else:
            self.g_basis = default_g_basis(self.g_max, self.mode_selection)
            if self.equivariant and self.g_max is not None \
                    and self.mode_selection != "physical":
                warnings.warn(
                    f"model.g_max = {self.g_max:g} does not size the radial "
                    f"basis under mode_selection: {self.mode_selection!r}, "
                    f"where it truncates nothing either. g_basis is "
                    f"{self.g_basis:g} 1/Ang, the default. State "
                    f"model.equivariant.g_basis to choose one, or 'auto' to "
                    f"size it from the training split.",
                    RuntimeWarning, stacklevel=2)
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

        activation_kwargs = {
            "kan_grid_size": self.kan_grid_size,
            "kan_spline_order": self.kan_spline_order,
            "kan_grid_range": self.kan_grid_range,
            "kan_degree": self.kan_degree,
            "kan_rational_num_degree": self.kan_rational_num_degree,
            "kan_rational_den_degree": self.kan_rational_den_degree,
            "kan_use_base": self.kan_use_base,
        }
        lifting_channels = self.in_channels + (3 if use_coordinates else 0)
        self.lift = nn.Conv3d(lifting_channels, self.width, kernel_size=1)
        radial_kwargs = {"n_radial": self.n_radial, "g_basis": self.g_basis,
                         "spherical_cutoff": self.spherical_cutoff}
        self.blocks = nn.ModuleList([
            FNOBlock(self.width, self.modes, embedding_dim=conditioning,
                     activation=activation, activation_kwargs=activation_kwargs,
                     equivariant=self.equivariant, radial_kwargs=radial_kwargs)
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

        if self.equivariant and cell is None:
            raise ValueError("equivariant=True requires `cell` at forward "
                             "time: the spectral kernel is a function of |G|.")

        # The radial basis is a property of the batch geometry, not of a layer,
        # so it is built here once and handed down -- exactly as `embedding`
        # and `max_modes` above it already are, and for the same reason. Every
        # block in the stack is constructed from the same `self.modes` and the
        # same `radial_kwargs`, and they all see this call's `cell`, `shape`
        # and `max_modes`, so the tensor each one used to build for itself was
        # bit-identical to its neighbours'.
        #
        # It is not a micro-optimisation. Building it calls `cell_reciprocal`,
        # which moves the cell to the host to widen it to float64 (MPS cannot
        # represent one) and copies the result back -- so an `n_layers=3` model
        # made three device-to-host-to-device round trips per forward pass,
        # where the dense backbone makes none at all. On CUDA a device-to-host
        # copy also drains the queue, which is the part that costs more than
        # the arithmetic it was synchronising for.
        basis = None
        if self.equivariant:
            spectral = self.blocks[0].spectral
            basis = spectral.radial_basis(
                cell, spectral.effective_modes(shape, max_modes),
                x.device, x.dtype)

        v = self.lift(x)
        for block in self.blocks:
            v = block(v, embedding=embedding, max_modes=max_modes, cell=cell,
                      basis=basis)
        return self.project(v)

    def n_parameters(self):
        """Total number of trainable parameters (complex weights count twice)."""
        return sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in self.parameters() if p.requires_grad
        )
