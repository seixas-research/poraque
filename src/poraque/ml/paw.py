# -*- coding: utf-8 -*-
# file: paw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The ``ext2paw`` operator: the pseudo-density and every atom's PAW occupancies.

A ``CHGCAR`` is a grid plus, per atom, the one-center occupancies
:math:`\rho(\ell\ell'LM)` VASP writes after it. The grid half is a field and an
FNO predicts it, as ``ext2chg`` does. The per-atom half is not a field. It is a
set of spherical tensors, up to L = 4 for a d-element, that no grid model sees,
and ``experiments/paw_occupancies`` measured what predicts it:

* the **predicted density itself**, read at each atom, which carries the
  occupancies linearly (0.27 % on L = 0 from the DFT pseudo-density);
* read over **one band of** :math:`|\mathbf G|` **for every cell**, because a
  readout whose quadrature follows the grid failed at 184 % on held-out
  nanoparticles, whose grid spacing differs;
* mapped **per L, shared across M**, so the prediction rotates with the atom;
* with a nonlinear head where the linear one runs out, in the L > 0 blocks,
  which hold 98 % of what a linear map cannot explain.

The architecture follows directly:

:class:`SiteReadout`
    A grid-to-atom kernel integral --- the decoder half of a graph neural
    operator --- with kernels :math:`r^L Y_{LM}(\hat r)\,e^{-r^2/2s_n^2}`. The
    fields are first restricted to :math:`|\mathbf G| \le G_{\max}`. Low bands
    use Fourier-space kernel integrals, so even the narrowest Gaussian is
    resolved. Higher bands use the existing dense spatial quadrature.
:class:`NeighborExpansion`
    :math:`\sum_j R_n(r_{aj})\,Y_{LM}(\hat r_{aj})`: the geometry, first order.
:class:`EquivariantHead`
    Per element: invariants (L = 0 features and the norms of the others) through
    an MLP, which predicts L = 0 and **gates** a linear map of each L > 0 block.
    Linear skip paths on every block, so the head starts at least as expressive
    as the linear readout that was measured.
:class:`OccupancyTransform`
    The invertible training form: center L = 0, scale each (set, L, pair), fitted
    on the training split and stored in the checkpoint. Regrouping into L blocks
    happens in the head, through :func:`~poraque.fields.vasp.augmentation.irreps_blocks`.

The whole operator is rotation-equivariant exactly when its FNO backbone is
(``model.equivariant.enable`` with ``use_coordinates: false``); the readout,
the neighbor expansion and the head are equivariant by construction.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

#: Latent channels read at each atom, beside the predicted density itself.
READOUT_CHANNELS = 4

#: Radius of the readout sphere in Å. The augmentation sphere of PAW_PBE Pt is
#: 1.46 Å; the kernels decay well inside this.
READOUT_RADIUS = 2.5

#: Widths s_n in Å of the Gaussian radial factors, geometric.
READOUT_WIDTHS = tuple(float(s) for s in np.geomspace(0.2, 0.8, 8))

#: Neighbor expansion: Gaussian radial functions under a cosine cutoff.
NEIGHBOR_CUTOFF = 6.0
NEIGHBOR_CENTERS = tuple(float(c) for c in np.linspace(2.2, 5.8, 10))
NEIGHBOR_WIDTH = 0.3

#: Head sizes.
HEAD_HIDDEN = 32
GATED_CHANNELS = 16

#: Highest L the solid harmonics below are written for. d-channel elements need
#: 4 (d x d); an f-channel element needs 6 and is refused rather than truncated.
MAX_L = 4


# ---------------------------------------------------------------------- #
# Solid harmonics
# ---------------------------------------------------------------------- #
def solid_harmonics(d, lmax=MAX_L):
    r"""
    :math:`r^L Y_{LM}(\hat{\mathbf d})` for L = 0..``lmax``, as polynomials.

    Standard orthonormal real spherical harmonics, ``m = -L..L``: the
    convention VASP's augmentation records are written in, checked on this
    project's platinum data. As polynomials in the components of ``d`` rather
    than functions of its direction they are smooth at the origin, which is what
    lets a kernel built from them be sampled on a grid without aliasing.

    Parameters
    ----------
    d : torch.Tensor
        ``(..., 3)`` vectors in Å; pass unit vectors for :math:`Y_{LM}` itself.
    lmax : int

    Returns
    -------
    list of torch.Tensor
        ``(..., 2L+1)`` per L.
    """
    if lmax > MAX_L:
        raise ValueError(f"solid harmonics are written up to L = {MAX_L}, "
                         f"not {lmax}; an f-channel element needs L = 6.")
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    r2 = x * x + y * y + z * z
    pi = math.pi
    out = [torch.stack([torch.full_like(x, 0.5 / math.sqrt(pi))], -1)]
    if lmax >= 1:
        c = math.sqrt(3 / (4 * pi))
        out.append(torch.stack([c * y, c * z, c * x], -1))
    if lmax >= 2:
        a = 0.5 * math.sqrt(15 / pi)
        out.append(torch.stack([
            a * x * y, a * y * z, 0.25 * math.sqrt(5 / pi) * (3 * z * z - r2),
            a * x * z, 0.25 * math.sqrt(15 / pi) * (x * x - y * y)], -1))
    if lmax >= 3:
        out.append(torch.stack([
            0.25 * math.sqrt(35 / (2 * pi)) * y * (3 * x * x - y * y),
            0.5 * math.sqrt(105 / pi) * x * y * z,
            0.25 * math.sqrt(21 / (2 * pi)) * y * (5 * z * z - r2),
            0.25 * math.sqrt(7 / pi) * z * (5 * z * z - 3 * r2),
            0.25 * math.sqrt(21 / (2 * pi)) * x * (5 * z * z - r2),
            0.25 * math.sqrt(105 / pi) * z * (x * x - y * y),
            0.25 * math.sqrt(35 / (2 * pi)) * x * (x * x - 3 * y * y)], -1))
    if lmax >= 4:
        out.append(torch.stack([
            0.75 * math.sqrt(35 / pi) * x * y * (x * x - y * y),
            0.75 * math.sqrt(35 / (2 * pi)) * y * z * (3 * x * x - y * y),
            0.75 * math.sqrt(5 / pi) * x * y * (7 * z * z - r2),
            0.75 * math.sqrt(5 / (2 * pi)) * y * z * (7 * z * z - 3 * r2),
            (3 / 16) * math.sqrt(1 / pi)
            * (35 * z ** 4 - 30 * z * z * r2 + 3 * r2 * r2),
            0.75 * math.sqrt(5 / (2 * pi)) * x * z * (7 * z * z - 3 * r2),
            (3 / 8) * math.sqrt(5 / pi) * (x * x - y * y) * (7 * z * z - r2),
            0.75 * math.sqrt(35 / (2 * pi)) * x * z * (x * x - 3 * y * y),
            (3 / 16) * math.sqrt(35 / pi)
            * (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y))],
            -1))
    return out


def layout_lmax(channels):
    """Highest L a record of these projector channels carries."""
    return 2 * max(int(degree) for degree in channels)


# ---------------------------------------------------------------------- #
# Readout and geometry
# ---------------------------------------------------------------------- #
class SiteReadout(nn.Module):
    r"""
    Project fields onto atom-centered solid-harmonic Gaussians, over one band.

    .. math::

        c_{nLM}(\mathbf R_a) = \int f_{G_{\max}}(\mathbf R_a + \mathbf d)\;
            N_{nL}\, r^L Y_{LM}(\hat{\mathbf d})\, e^{-r^2/2s_n^2}\, w(r)\,
            d^3 d

    with :math:`f_{G_{\max}}` the field restricted to
    :math:`|\mathbf G| \le G_{\max}` and :math:`w` a cosine window at
    :data:`READOUT_RADIUS`. :math:`N_{nL}` makes each kernel unit-norm, so a
    projection's size does not depend on L or on the width.

    The restriction is what makes the readout a property of the field rather
    than of its grid. At low bands, spacing :math:`h = \pi / 2G_{\max}` would
    undersample the narrow Gaussian kernels even though it resolves the field.
    There we integrate each Fourier wave analytically over angles and use
    Gauss-Legendre radial quadrature. At higher bands, where that spacing
    resolves the kernels, we retain spatial quadrature around each atom.
    Cell geometry is fixed; gradients propagate to fields and atom positions.

    Parameters
    ----------
    g_max : float
        The band in Å⁻¹. Every field read must supply it: a grid whose Nyquist
        frequency is below ``g_max`` along any axis **raises**, since reading
        it would silently use a narrower band than training did.
    lmax : int
    """

    #: Bytes of spectral kernels kept per readout, on the model's device. The
    #: kernels are one per geometry (cell and grid), computed on the CPU with
    #: scipy's Bessel functions; once a training split's kernels exceed the
    #: budget the least recent are evicted and recomputed every epoch. 64 MiB
    #: was far below the ~360 MB the 785 CHONPS geometries need at a 6 1/Ang
    #: band, and recomputing them dominated the epoch; 1 GiB holds them all.
    #: The default for :attr:`cache_bytes`; ``model.paw.readout_cache_mib``
    #: sets it per run.
    SPECTRAL_CACHE_BYTES = 1024 ** 3

    def __init__(self, g_max, lmax=MAX_L):
        super().__init__()
        self.g_max = float(g_max)
        self.lmax = int(lmax)
        self.spacing = math.pi / (2.0 * self.g_max)
        widths = torch.tensor(READOUT_WIDTHS, dtype=torch.float64)
        # int_0^inf r^(2L+2) exp(-r^2/s^2) dr = Gamma(L + 3/2) s^(2L+3) / 2
        norms = torch.stack([
            1.0 / torch.sqrt(math.gamma(L + 1.5) * widths ** (2 * L + 3) / 2.0)
            for L in range(self.lmax + 1)])
        self.register_buffer("widths", widths.float())
        self.register_buffer("norms", norms.float())
        # Geometry only; not checkpoint state. Bound memory across diverse cells.
        self._spectral_cache = OrderedDict()
        self._spectral_cache_bytes = 0
        #: The kernel budget in bytes. A runtime setting, not architecture: it
        #: changes how often kernels are recomputed, never what they are, so
        #: it is not in the checkpoint.
        self.cache_bytes = int(self.SPECTRAL_CACHE_BYTES)

    @property
    def size(self):
        """Radial functions per field channel."""
        return len(READOUT_WIDTHS)

    def spectral_integral(self, fields, cell, frac):
        """Integrate band-limited Fourier waves against each radial kernel.

        The angular integral is 4π i^L j_L(Gr) Y_LM(Ghat). Radial integrals
        are computed in float64 once per geometry; all field operations and
        atom phases remain differentiable on the model's device.
        """
        from scipy.special import spherical_jn

        shape = tuple(fields.shape[1:])
        geometry = cell.detach().cpu().double().numpy()
        widths = self.widths.detach().cpu().double().numpy()
        norms = self.norms.detach().cpu().double().numpy()
        key = (shape, geometry.tobytes(), str(fields.device), fields.dtype,
               widths.tobytes(), norms.tobytes())
        cached = self._spectral_cache.get(key)
        if cached is None:
            nyquist = np.pi * np.array(shape) / np.linalg.norm(geometry, axis=1)
            if nyquist.min() < self.g_max * (1.0 - 1e-6):
                raise ValueError(
                    f"a grid of shape {shape} resolves {nyquist.min():.2f} "
                    f"1/Ang along its coarsest axis, below the "
                    f"{self.g_max:.2f} 1/Ang band. Use a finer grid.")
            integers = np.stack(np.meshgrid(*[
                np.fft.fftfreq(n, 1.0 / n) for n in shape], indexing="ij"), -1)
            vectors = integers @ (2 * math.pi * np.linalg.inv(geometry).T)
            magnitude = np.linalg.norm(vectors, axis=-1)
            keep = magnitude <= self.g_max
            for axis, n in enumerate(shape):
                keep &= np.abs(integers[..., axis]) < n / 2.0
            indices = np.flatnonzero(keep)
            modes = integers[keep]
            vectors, magnitude = vectors[keep], magnitude[keep]
            unit = vectors / np.maximum(magnitude[:, None], 1e-30)
            harmonics = solid_harmonics(torch.from_numpy(unit), self.lmax)
            nodes, weights = np.polynomial.legendre.leggauss(128)
            radius = (nodes + 1) * READOUT_RADIUS / 2
            weights = weights * READOUT_RADIUS / 2
            radial = np.exp(-radius[:, None] ** 2 / (2 * widths ** 2))
            radial *= (0.5 * (1 + np.cos(math.pi * radius / READOUT_RADIUS)))[:, None]
            kernels = []
            for L in range(self.lmax + 1):
                bessel = spherical_jn(L, magnitude[:, None] * radius[None])
                integral = (bessel * (weights * radius ** (L + 2))[None]) @ radial
                kernel = (4 * math.pi * integral * norms[L])[..., None] \
                    * harmonics[L].numpy()[:, None, :]
                kernels.append(torch.as_tensor(kernel, dtype=fields.dtype,
                                                device=fields.device))
            cached = (torch.as_tensor(indices, device=fields.device),
                      torch.as_tensor(modes, dtype=fields.dtype, device=fields.device),
                      kernels)
            size = sum(t.numel() * t.element_size() for t in
                       (cached[0], cached[1], *kernels))
            limit = int(self.cache_bytes)
            while self._spectral_cache and self._spectral_cache_bytes + size > limit:
                _, old = self._spectral_cache.popitem(last=False)
                self._spectral_cache_bytes -= sum(t.numel() * t.element_size()
                    for t in (old[0], old[1], *old[2]))
            if size <= limit:
                self._spectral_cache[key] = cached
                self._spectral_cache_bytes += size
        elif key in self._spectral_cache:
            self._spectral_cache.move_to_end(key)
        indices, modes, kernels = cached
        spectrum = torch.fft.fftn(fields, dim=(-3, -2, -1), norm="forward")
        coefficients = torch.view_as_real(spectrum).reshape(fields.shape[0], -1, 2)
        coefficients = coefficients[:, indices]
        phase = 2 * math.pi * ((frac % 1.0) @ modes.T)
        cosine, sine = torch.cos(phase)[:, None], torch.sin(phase)[:, None]
        real = coefficients[None, ..., 0] * cosine - coefficients[None, ..., 1] * sine
        imaginary = coefficients[None, ..., 0] * sine + coefficients[None, ..., 1] * cosine
        components = (real, -imaginary, -real, imaginary)
        out = {}
        for L, kernel in enumerate(kernels):
            value = torch.einsum("acg,gnm->acnm", components[L % 4], kernel)
            out[L] = value.reshape(frac.shape[0], -1, 2 * L + 1)
        return out

    def band_limited(self, fields, cell):
        """``fields`` restricted to the band and resampled: ``(C, S1, S2, S3)``."""
        channels, *shape = fields.shape
        lengths = torch.linalg.vector_norm(cell, dim=-1)
        nyquist = (math.pi * torch.tensor(shape, dtype=lengths.dtype,
                                          device=lengths.device) / lengths)
        if float(nyquist.min()) < self.g_max * (1.0 - 1e-6):
            raise ValueError(
                f"a grid of shape {tuple(shape)} on this cell resolves |G| up "
                f"to {float(nyquist.min()):.2f} 1/Ang along its coarsest axis, "
                f"below the {self.g_max:.2f} 1/Ang band the site readout was "
                f"trained on. Reading it would use a narrower band than "
                f"training did. Use a finer grid.")
        target = [int(math.ceil(float(length) / self.spacing))
                  for length in lengths]
        target = [n + (n % 2) for n in target]

        spectrum = torch.fft.fftn(fields, dim=(-3, -2, -1))
        k = [torch.fft.fftfreq(n, 1.0 / n, device=fields.device) for n in shape]
        kx, ky, kz = torch.meshgrid(*k, indexing="ij")
        integers = torch.stack([kx, ky, kz], -1)
        # G = 2 pi k . inv(cell)^T, rows as vectors: the convention
        # `frac @ cell = cart` implies.
        reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
        vectors = integers.to(reciprocal.dtype) @ reciprocal
        keep = torch.linalg.vector_norm(vectors, dim=-1) <= self.g_max
        # The Nyquist row is ambiguous in sign; it is below the band anyway.
        for axis, n in enumerate(shape):
            keep &= integers[..., axis].abs() < n / 2.0

        source = keep.nonzero(as_tuple=True)
        destination = [(integers[..., axis][source].long() % target[axis])
                       for axis in range(3)]
        # Real-valued indexing also keeps the backward scatter portable to
        # MPS, whose complex advanced-index backward is unsupported.
        out = torch.zeros(channels, *target, 2, dtype=fields.dtype,
                          device=fields.device)
        scale = float(np.prod(target)) / float(np.prod(shape))
        out[(slice(None), *destination, slice(None))] = torch.view_as_real(
            spectrum)[(slice(None), *source, slice(None))] * scale
        return torch.fft.ifftn(torch.view_as_complex(out), dim=(-3, -2, -1)).real

    def forward(self, fields, cell, frac):
        """
        Parameters
        ----------
        fields : torch.Tensor
            ``(C, Nx, Ny, Nz)`` for one sample.
        cell : torch.Tensor
            ``(3, 3)`` in Å.
        frac : torch.Tensor
            ``(A, 3)`` fractional positions.

        Returns
        -------
        dict
            ``{L: (A, C * n, 2L+1)}``.
        """
        if self.spacing > 1.25 * min(READOUT_WIDTHS):
            return self.spectral_integral(fields, cell, frac)
        dense = self.band_limited(fields, cell)
        channels, *shape = dense.shape
        shape_t = torch.tensor(shape, device=fields.device)
        reciprocal = torch.linalg.inv(cell).T
        half = [int(math.ceil(READOUT_RADIUS
                              * float(torch.linalg.vector_norm(reciprocal[i]))
                              * shape[i])) for i in range(3)]
        offsets = [torch.arange(-h, h + 1, device=fields.device) for h in half]
        ox, oy, oz = torch.meshgrid(*offsets, indexing="ij")
        stencil = torch.stack([ox, oy, oz], -1).reshape(-1, 3)      # (P, 3)

        frac = frac % 1.0
        center = torch.round(frac * shape_t.to(frac.dtype)).long()  # (A, 3)
        index = center[:, None, :] + stencil[None]                  # (A, P, 3)
        displacement = (index.to(frac.dtype) / shape_t.to(frac.dtype)
                        - frac[:, None, :]) @ cell                  # (A, P, 3)
        wrapped = index % shape_t
        values = dense[:, wrapped[..., 0], wrapped[..., 1],
                       wrapped[..., 2]]                             # (C, A, P)

        r2 = (displacement ** 2).sum(-1)
        r = torch.sqrt(r2)
        window = torch.where(
            r < READOUT_RADIUS,
            0.5 * (1.0 + torch.cos(math.pi * r / READOUT_RADIUS)),
            torch.zeros_like(r))
        widths = self.widths.to(r.dtype)
        radial = torch.exp(-r2[..., None] / (2.0 * widths ** 2)) \
            * window[..., None]                                     # (A, P, n)
        harmonics = solid_harmonics(displacement, self.lmax)
        from .physics import cell_volume

        volume = cell_volume(cell, dtype=fields.dtype)[0] / float(np.prod(shape))

        weighted = torch.einsum("cap,apn->acnp", values, radial) * volume
        out = {}
        for L in range(self.lmax + 1):
            c = torch.einsum("acnp,apm->acnm", weighted, harmonics[L])
            c = c * self.norms[L].to(c.dtype)[None, None, :, None]
            out[L] = c.reshape(c.shape[0], -1, 2 * L + 1)
        return out


class NeighborExpansion(nn.Module):
    r"""
    :math:`\sum_j R_n(r_{aj})\,Y_{LM}(\hat{\mathbf r}_{aj})` over periodic
    images within :data:`NEIGHBOR_CUTOFF`: ``{L: (A, n, 2L+1)}``.
    """

    def __init__(self, lmax=MAX_L, minimum=None):
        super().__init__()
        self.lmax = int(lmax)
        centers = torch.tensor(NEIGHBOR_CENTERS)
        if minimum is not None and minimum != NEIGHBOR_CENTERS[0]:
            if isinstance(minimum, bool) or not 0 <= float(minimum) < NEIGHBOR_CENTERS[-1]:
                raise ValueError("neighbor minimum must be non-negative and below 5.8 Ang")
            centers = torch.linspace(float(minimum), float(centers[-1]),
                                     len(NEIGHBOR_CENTERS), dtype=centers.dtype)
        self.register_buffer("centers", centers)

    @property
    def size(self):
        return len(NEIGHBOR_CENTERS)

    def forward(self, cell, frac):
        reciprocal = torch.linalg.inv(cell).T
        reach = [int(math.ceil(NEIGHBOR_CUTOFF
                               * float(torch.linalg.vector_norm(reciprocal[i]))))
                 for i in range(3)]
        ranges = [torch.arange(-m, m + 1, device=cell.device) for m in reach]
        ix, iy, iz = torch.meshgrid(*ranges, indexing="ij")
        images = torch.stack([ix, iy, iz], -1).reshape(-1, 3).to(cell.dtype)
        cart = frac @ cell
        shifts = images @ cell                                        # (T, 3)
        d = (cart[None, :, None, :] + shifts[None, None, :, :]
             - cart[:, None, None, :]).reshape(len(frac), -1, 3)      # (A, AT, 3)
        r = torch.linalg.vector_norm(d, dim=-1)
        active = (r > 1e-3) & (r < NEIGHBOR_CUTOFF)
        safe = torch.where(active, r, torch.ones_like(r))
        unit = d / safe[..., None]
        cutoff = torch.where(active, 0.5 * (torch.cos(
            math.pi * r / NEIGHBOR_CUTOFF) + 1.0), torch.zeros_like(r))
        centers = self.centers.to(r.dtype)
        radial = torch.exp(-(r[..., None] - centers) ** 2
                           / (2 * NEIGHBOR_WIDTH ** 2)) * cutoff[..., None]
        harmonics = solid_harmonics(unit, self.lmax)
        return {L: torch.einsum("apn,apm->anm", radial, harmonics[L])
                for L in range(self.lmax + 1)}


# ---------------------------------------------------------------------- #
# Head
# ---------------------------------------------------------------------- #
class EquivariantHead(nn.Module):
    r"""
    Features ``{L: (A, C_L, 2L+1)}`` to record blocks ``{L: (A, S, P_L, 2L+1)}``.

    Every map acts on the channel index and is shared across M, and L > 0 maps
    carry no bias, so a rotation of the inputs by the Wigner matrix
    :math:`D^L` rotates the outputs by the same :math:`D^L`. The nonlinearity
    lives in the invariants: an MLP over the L = 0 features and the norms of the
    rest predicts L = 0 and emits one gate per L > 0 channel.
    """

    def __init__(self, inputs, pairs, sets):
        super().__init__()
        self.inputs = {int(L): int(c) for L, c in inputs.items()}
        self.pairs = {int(L): int(p) for L, p in pairs.items()}
        self.sets = int(sets)
        scalars = sum(self.inputs.values())
        self.norm = nn.LayerNorm(scalars)
        self.mlp = nn.Sequential(nn.Linear(scalars, HEAD_HIDDEN), nn.SiLU(),
                                 nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.SiLU())
        self.scalar_out = nn.Linear(HEAD_HIDDEN, self.sets * self.pairs.get(0, 0))
        self.scalar_skip = nn.Linear(self.inputs[0],
                                     self.sets * self.pairs.get(0, 0))
        self.up = nn.ModuleDict()
        self.gate = nn.ModuleDict()
        self.down = nn.ModuleDict()
        self.skip = nn.ModuleDict()
        for L, count in self.pairs.items():
            if L == 0:
                continue
            key = str(L)
            self.up[key] = nn.Linear(self.inputs[L], GATED_CHANNELS, bias=False)
            self.gate[key] = nn.Linear(HEAD_HIDDEN, GATED_CHANNELS)
            self.down[key] = nn.Linear(GATED_CHANNELS, self.sets * count,
                                       bias=False)
            self.skip[key] = nn.Linear(self.inputs[L], self.sets * count,
                                       bias=False)

    def forward(self, features):
        atoms = features[0].shape[0]
        scalars = [features[0][..., 0]]
        scalars += [torch.sqrt((features[L] ** 2).sum(-1) + 1e-12)
                    for L in sorted(self.inputs) if L > 0]
        hidden = self.mlp(self.norm(torch.cat(scalars, dim=-1)))

        out = {}
        if 0 in self.pairs:
            value = self.scalar_out(hidden) + self.scalar_skip(features[0][..., 0])
            out[0] = value.reshape(atoms, self.sets, self.pairs[0], 1)
        for L, count in self.pairs.items():
            if L == 0:
                continue
            key = str(L)
            x = features[L].transpose(1, 2)                    # (A, 2L+1, C)
            gates = torch.sigmoid(self.gate[key](hidden))[:, None, :]
            value = self.down[key](self.up[key](x) * gates) + self.skip[key](x)
            out[L] = value.reshape(atoms, 2 * L + 1, self.sets,
                                   count).permute(0, 2, 3, 1)
        return out


# ---------------------------------------------------------------------- #
# The operator
# ---------------------------------------------------------------------- #
class PAWOperatorModel(nn.Module):
    """
    FNO backbone, density head, and a per-atom occupancy head reading both.

    Called as ``model(x, cell)`` it is the field operator and returns the
    density, so everything that treats a model as field-to-field --- prediction,
    evaluation, the architecture record --- works unchanged. Called with
    ``sites`` it also returns the occupancies, normalized as the
    :class:`OccupancyTransform` defines, flattened into file order:
    ``(B, A, sets, record_length)``.

    Parameters
    ----------
    backbone : FNO3d
    layouts : dict
        ``{atomic number: projector channels}`` for every element it predicts.
    g_max : float
        The readout band, Å⁻¹.
    """

    def __init__(self, backbone, layouts, g_max, neighbor_min=None):
        super().__init__()
        from ..fields.vasp.augmentation import irreps_blocks

        self.backbone = backbone
        self.layouts = {int(z): tuple(int(degree) for degree in channels)
                        for z, channels in layouts.items()}
        if not self.layouts:
            raise ValueError("an ext2paw operator needs the projector channels "
                             "of at least one element.")
        self.g_max = float(g_max)
        self.sets = int(backbone.out_channels)
        self.lmax = max(layout_lmax(channels)
                        for channels in self.layouts.values())
        self.readout = SiteReadout(self.g_max, self.lmax)
        self.neighbors = NeighborExpansion(self.lmax, minimum=neighbor_min)
        self.reduce = nn.Conv3d(backbone.width, READOUT_CHANNELS, kernel_size=1)

        read = (READOUT_CHANNELS + backbone.out_channels) * self.readout.size
        inputs = {L: read + self.neighbors.size
                  for L in range(self.lmax + 1)}
        self.heads = nn.ModuleDict()
        self._blocks = {}
        for z, channels in self.layouts.items():
            blocks = irreps_blocks(channels)
            self._blocks[z] = {L: torch.as_tensor(index)
                               for L, index in blocks.items()}
            self.heads[str(z)] = EquivariantHead(
                inputs, {L: index.shape[0] for L, index in blocks.items()},
                self.sets)
        self.record_length = max(sum(index.size for index in
                                     irreps_blocks(channels).values())
                                 for channels in self.layouts.values())

    # The field operator's architecture, read through by FieldOperator.
    @property
    def in_channels(self):
        return self.backbone.in_channels

    @property
    def out_channels(self):
        return self.backbone.out_channels

    @property
    def use_coordinates(self):
        return self.backbone.use_coordinates

    def n_parameters(self):
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in self.parameters() if p.requires_grad)

    def forward(self, x, cell=None, sites=None):
        latent = self.backbone.encode(x, cell)
        field = self.backbone.project(latent)
        if sites is None:
            return field
        return field, self.occupancies(latent, field, cell, sites)

    def occupancies(self, latent, field, cell, sites):
        reads = torch.cat([self.reduce(latent), field], dim=1)
        species, positions = sites["species"], sites["positions"]
        mask = sites["atom_mask"]
        batch, atoms = species.shape
        out = reads.new_zeros(batch, atoms, self.sets, self.record_length)
        for b in range(batch):
            count = int(mask[b].sum())
            if not count:
                continue
            frac = positions[b, :count].to(reads.dtype)
            features = self.readout(reads[b], cell[b].to(reads.dtype), frac)
            geometry = self.neighbors(cell[b].to(reads.dtype), frac)
            features = {L: torch.cat([features[L], geometry[L]], dim=1)
                        for L in features}
            for z in torch.unique(species[b, :count]).tolist():
                if int(z) not in self.layouts:
                    raise KeyError(
                        f"atomic number {int(z)} has no occupancy head; this "
                        f"operator predicts {sorted(self.layouts)}.")
                chosen = (species[b, :count] == z).nonzero(as_tuple=True)[0]
                blocks = self.heads[str(int(z))](
                    {L: f[chosen] for L, f in features.items()})
                flat = reads.new_zeros(len(chosen), self.sets,
                                       self.record_length)
                for L, value in blocks.items():
                    index = self._blocks[int(z)][L].to(reads.device)
                    flat[:, :, index.reshape(-1)] = value.reshape(
                        len(chosen), self.sets, -1)
                out[b, chosen] = flat
        return out


# ---------------------------------------------------------------------- #
# Occupancies as input: the paw2tau operator
# ---------------------------------------------------------------------- #
#: Widths s_n in Å of the Gaussians the occupancies are written with.
WRITER_WIDTHS = (0.25, 0.45, 0.8)

#: Field channels the occupancies are written into, by default.
WRITER_CHANNELS = 8


class SiteWriter(nn.Module):
    r"""
    Per-atom occupancy records to scalar fields: the transpose of the readout.

    Each atom :math:`a` of element :math:`Z` contributes to channel :math:`k`

    .. math::

        f_k(\mathbf r) = \sum_a \sum_{nLM} c^a_{knLM}\,
            N_{nL}\, S_{LM}(\mathbf r - \mathbf R_a)\,
            e^{-|\mathbf r - \mathbf R_a|^2 / 2 s_n^2},
        \qquad
        c^a_{knLM} = \sum_q W^{Z,L}_{kn,q}\, z^a_{q,M},

    where :math:`S_{LM}(\mathbf d) = d^L Y_{LM}(\hat{\mathbf d})` is a solid
    harmonic (:func:`solid_harmonics`), :math:`z^a` the atom's normalized record
    regrouped into L blocks (:func:`~poraque.fields.vasp.augmentation.irreps_blocks`;
    ``q`` runs over the record sets and the projector pairs of the block) and
    :math:`W^{Z,L}` one learned matrix per element and L, **shared across M**.
    A rotation of the atom's occupancies by the Wigner matrix :math:`D^L`
    therefore rotates the kernel it writes, and the field is a scalar field:
    the operator reading it stays equivariant.

    The field is built in reciprocal space, where it is exact. By Hobson's
    theorem the Fourier transform of a solid harmonic times a Gaussian is the
    same solid harmonic of :math:`\mathbf G`,

    .. math::

        \int S_{LM}(\mathbf d)\, e^{-d^2/2s^2}\, e^{-i\mathbf G\cdot\mathbf d}\,
            d^3d = (2\pi)^{3/2} s^{2L+3} (-i)^L S_{LM}(\mathbf G)\,
            e^{-s^2 G^2/2},

    so each atom is a phase :math:`e^{-i\mathbf G\cdot\mathbf R_a}` and the
    periodic sum over images comes free with the inverse FFT. Nothing depends on
    the grid but the band it can hold.

    Parameters
    ----------
    layouts : dict
        ``{atomic number: projector channels}``, as for the readout.
    sets : int
        Record sets per atom: one per density channel of the input.
    channels : int, optional
        Field channels written.
    widths : sequence of float, optional
        Gaussian widths in Å.
    """

    def __init__(self, layouts, sets, channels=WRITER_CHANNELS,
                 widths=WRITER_WIDTHS):
        super().__init__()
        from ..fields.vasp.augmentation import irreps_blocks

        self.layouts = {int(z): tuple(int(degree) for degree in degrees)
                        for z, degrees in layouts.items()}
        self.sets = int(sets)
        self.channels = int(channels)
        self.widths = tuple(float(s) for s in widths)
        self.lmax = max(layout_lmax(degrees)
                        for degrees in self.layouts.values())
        self._blocks = {}
        self.maps = nn.ModuleDict()
        out = self.channels * len(self.widths)
        for z, degrees in self.layouts.items():
            blocks = irreps_blocks(degrees)
            self._blocks[z] = {L: torch.as_tensor(index)
                               for L, index in blocks.items()}
            self.maps[str(z)] = nn.ModuleDict({
                str(L): nn.Linear(self.sets * index.shape[0], out, bias=False)
                for L, index in blocks.items()})
        # Unit-norm kernels, so the coefficients are of order one whatever the
        # width: the same normalization the readout's kernels carry.
        norms = [[1.0 / math.sqrt(math.gamma(L + 1.5) * s ** (2 * L + 3) / 2.0)
                  for s in self.widths] for L in range(self.lmax + 1)]
        self.register_buffer("norms", torch.tensor(norms))

    def coefficients(self, records, species, count):
        r"""
        ``{L: (A, channels, widths, 2L+1)}`` for the first ``count`` atoms.

        Parameters
        ----------
        records : torch.Tensor
            ``(A, S, R)`` normalized records in file order, padded.
        species : torch.Tensor
            ``(A,)`` atomic numbers.
        """
        out = {L: records.new_zeros(count, self.channels, len(self.widths),
                                    2 * L + 1)
               for L in range(self.lmax + 1)}
        for z in torch.unique(species[:count]).tolist():
            z = int(z)
            if z not in self.layouts:
                raise KeyError(
                    f"atomic number {z} has no occupancy layout; this operator "
                    f"writes {sorted(self.layouts)}.")
            chosen = (species[:count] == z).nonzero(as_tuple=True)[0]
            for L, index in self._blocks[z].items():
                index = index.to(records.device)
                block = records[chosen][:, :, index]       # (n, S, P, 2L+1)
                n, sets, pairs, components = block.shape
                block = block.permute(0, 3, 1, 2).reshape(n, components,
                                                          sets * pairs)
                value = self.maps[str(z)][str(L)](block)   # (n, 2L+1, C*W)
                out[L][chosen] = value.reshape(
                    n, components, self.channels, len(self.widths)
                ).permute(0, 2, 3, 1)
        return out

    def forward(self, records, species, positions, atom_mask, cell, shape):
        """
        The written fields, ``(B, channels, Nx, Ny, Nz)``.

        Parameters
        ----------
        records : torch.Tensor
            ``(B, A, S, R)`` normalized records.
        species, positions, atom_mask : torch.Tensor
            ``(B, A)``, ``(B, A, 3)`` fractional, ``(B, A)``.
        cell : torch.Tensor
            ``(B, 3, 3)`` in Å.
        shape : tuple of int
            The grid.
        """
        nx, ny, nz = (int(n) for n in shape)
        dtype = records.dtype
        device = records.device
        k = [torch.fft.fftfreq(nx, 1.0 / nx, device=device, dtype=dtype),
             torch.fft.fftfreq(ny, 1.0 / ny, device=device, dtype=dtype),
             torch.fft.rfftfreq(nz, 1.0 / nz, device=device, dtype=dtype)]
        integers = torch.stack(torch.meshgrid(*k, indexing="ij"), -1)
        integers = integers.reshape(-1, 3)                  # (G, 3)
        fields = []
        for b in range(records.shape[0]):
            count = int(atom_mask[b].sum())
            geometry = cell[b].to(dtype)
            # Rows as vectors, `frac @ cell = cart`: G = 2 pi k . inv(cell)^T.
            reciprocal = 2.0 * math.pi * torch.linalg.inv(geometry).T
            vectors = integers @ reciprocal                 # (G, 3)
            g2 = (vectors ** 2).sum(-1)
            harmonics = solid_harmonics(vectors, self.lmax)
            spectrum_re = records.new_zeros(self.channels, integers.shape[0])
            spectrum_im = torch.zeros_like(spectrum_re)
            if count:
                coefficients = self.coefficients(records[b], species[b], count)
                phase = 2.0 * math.pi * (positions[b, :count].to(dtype)
                                         @ integers.T)      # (A, G)
                cosine, sine = torch.cos(phase), torch.sin(phase)
                widths = torch.tensor(self.widths, dtype=dtype, device=device)
                for L in range(self.lmax + 1):
                    # sum over atoms first: (C, W, M, G), real and imaginary
                    # parts of sum_a c_a exp(-i G.R_a)
                    c = coefficients[L]
                    real = torch.einsum("acwm,ag->cwmg", c, cosine)
                    imag = -torch.einsum("acwm,ag->cwmg", c, sine)
                    radial = ((2.0 * math.pi) ** 1.5
                              * widths[:, None] ** (2 * L + 3)
                              * torch.exp(-0.5 * widths[:, None] ** 2
                                          * g2[None])
                              * self.norms[L].to(dtype)[:, None])   # (W, G)
                    kernel = radial[:, None, :] * harmonics[L].T[None]  # (W,M,G)
                    real = (real * kernel[None]).sum((1, 2))
                    imag = (imag * kernel[None]).sum((1, 2))
                    # times (-i)^L
                    turn = L % 4
                    if turn == 1:
                        real, imag = imag, -real
                    elif turn == 2:
                        real, imag = -real, -imag
                    elif turn == 3:
                        real, imag = -imag, real
                    spectrum_re = spectrum_re + real
                    spectrum_im = spectrum_im + imag
            from .physics import cell_volume

            volume = cell_volume(geometry, dtype=dtype).reshape(-1)[0]
            spectrum = torch.complex(spectrum_re, spectrum_im) / volume
            fields.append(torch.fft.irfftn(
                spectrum.reshape(self.channels, nx, ny, nz // 2 + 1),
                s=(nx, ny, nz), dim=(-3, -2, -1), norm="forward"))
        return torch.stack(fields)


class OccupancyInputModel(nn.Module):
    """
    An FNO whose input is a field **and** every atom's PAW occupancies.

    The occupancies are written into :attr:`writer` channels by
    :class:`SiteWriter` and concatenated after the input field, so the backbone
    is an ordinary :class:`~poraque.ml.fno.FNO3d` over ``in_channels +
    writer.channels`` inputs. Called as ``model(x, cell, sites=...)``; ``sites``
    carries ``records`` (normalized, ``(B, A, S, R)``), ``species``,
    ``positions`` and ``atom_mask``, as collated.

    Parameters
    ----------
    backbone : FNO3d
        Built for the field channels plus the written ones.
    layouts : dict
        ``{atomic number: projector channels}``.
    sets : int
        Record sets per atom.
    channels : int, optional
        Written channels.
    """

    def __init__(self, backbone, layouts, sets, channels=WRITER_CHANNELS):
        super().__init__()
        self.backbone = backbone
        self.writer = SiteWriter(layouts, sets, channels)
        self.layouts = self.writer.layouts
        if backbone.in_channels <= self.writer.channels:
            raise ValueError(
                f"the backbone takes {backbone.in_channels} input channels, "
                f"which leaves none for the field beside the "
                f"{self.writer.channels} written ones.")

    @property
    def in_channels(self):
        """The field channels, without the written ones."""
        return self.backbone.in_channels - self.writer.channels

    @property
    def out_channels(self):
        return self.backbone.out_channels

    @property
    def use_coordinates(self):
        return self.backbone.use_coordinates

    def n_parameters(self):
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in self.parameters() if p.requires_grad)

    def forward(self, x, cell=None, sites=None):
        if sites is None or "records" not in sites:
            raise ValueError(
                "this operator reads every atom's PAW occupancies beside the "
                "field; call it with sites={'records', 'species', "
                "'positions', 'atom_mask'}.")
        written = self.writer(sites["records"].to(x.dtype), sites["species"],
                              sites["positions"], sites["atom_mask"], cell,
                              tuple(x.shape[-3:]))
        return self.backbone(torch.cat([x, written.to(x.dtype)], dim=1), cell)


# ---------------------------------------------------------------------- #
# The target
# ---------------------------------------------------------------------- #
class OccupancyTransform:
    r"""
    The invertible training form of the occupancies, per element.

    :math:`z = (\rho - \mu) / \sigma`, in file order, where :math:`\mu` is the
    training mean of the **L = 0** values only and :math:`\sigma` the RMS of each
    (set, L, pair) block over atoms and M. Both are shared across M, so the form
    commutes with rotations, and both are exactly invertible. A loss on the raw
    records would be 99 % an L = 0 loss; on :math:`z` every block is order one.

    The principal-axes rotation ``experiments/paw_occupancies`` also measured is
    left out. The head's last linear map absorbs any fixed rotation of the pair
    space, and whitening would inflate the 1e-4 tail of components to the size
    of the rest.

    Parameters
    ----------
    layouts : dict
        ``{atomic number: projector channels}``.
    sets : int
    mean, scale : dict
        ``{atomic number: (sets, record_length) array}``.
    set_weights : sequence of float, optional
        Weight of each record set in :meth:`loss`, ``1`` for every set by
        default. The second set of a spin-polarized density is the
        magnetization's, and its L > 0 blocks hold 0.005 % of the occupancy
        norm on the CHONPS set while being as unpredictable from the potential
        as the field magnetization is (98 % held-out error at L = 1): with
        every block weighted alike, those blocks were most of the occupancy
        loss and of the score the best epoch was chosen on. A weight below one
        on that set lets the charge blocks drive both. The transform itself
        --- normalization and inverse --- does not depend on it.
    """

    #: A block whose RMS is below this fraction of its element's largest is
    #: scaled by the floor instead --- a symmetry-zero block must not be
    #: divided by its own round-off.
    FLOOR = 1e-6

    def __init__(self, layouts, sets, mean, scale, set_weights=None):
        self.layouts = {int(z): tuple(int(degree) for degree in channels)
                        for z, channels in layouts.items()}
        self.sets = int(sets)
        weights = [1.0] * self.sets if set_weights is None \
            else [float(w) for w in set_weights]
        if len(weights) != self.sets or any(w < 0 for w in weights) \
                or not any(w > 0 for w in weights):
            raise ValueError(
                f"set_weights {set_weights!r}: expected {self.sets} "
                f"non-negative weight(s), not all zero.")
        self.set_weights = weights
        self.mean = {int(z): np.asarray(v, dtype=float) for z, v in mean.items()}
        self.scale = {int(z): np.asarray(v, dtype=float)
                      for z, v in scale.items()}
        self._tables = {}

    @classmethod
    def fit(cls, dataset, layouts, sets, set_weights=None):
        """
        Fit on a dataset's ``site_records``, which should be the training split.

        Raises
        ------
        ValueError
            When an element in the data has no layout, or its records are not
            the length its layout gives.
        """
        from ..fields.vasp.augmentation import irreps_blocks

        records = {}
        for index in range(len(dataset)):
            sites = dataset.site_records(index)
            for atom, z in enumerate(sites["species"]):
                length = int(sites["lengths"][atom])
                records.setdefault(int(z), []).append(
                    sites["occupancies"][atom, :, :length])

        mean, scale = {}, {}
        for z, values in records.items():
            if z not in layouts:
                raise ValueError(
                    f"the training data contains atomic number {z}, and no "
                    f"projector channels were found for it; they come from "
                    f"the POTCAR the potential was built with.")
            blocks = irreps_blocks(layouts[z])
            length = sum(index.size for index in blocks.values())
            stack = np.stack(values)                        # (N, S, length)
            if stack.shape[-1] != length:
                raise ValueError(
                    f"atomic number {z}: the records hold {stack.shape[-1]} "
                    f"values and projector channels {tuple(layouts[z])} give "
                    f"{length}; the POTCAR is not the one the data was "
                    f"computed with.")
            mu = np.zeros(stack.shape[1:])
            sigma = np.ones(stack.shape[1:])
            if 0 in blocks:
                index = blocks[0][:, 0]
                mu[:, index] = stack[:, :, index].mean(axis=0)
            centered = stack - mu
            rms = {}
            for L, index in blocks.items():
                block = centered[:, :, index]                # (N, S, P, M)
                rms[L] = np.sqrt((block ** 2).mean(axis=(0, 3)))    # (S, P)
            largest = max(float(v.max()) for v in rms.values()) or 1.0
            for L, index in blocks.items():
                value = np.maximum(rms[L], cls.FLOOR * largest)
                sigma[:, index] = value[:, :, None]
            mean[z], scale[z] = mu, sigma
        return cls({z: layouts[z] for z in records}, sets, mean, scale,
                   set_weights=set_weights)

    # -- tensor tables, keyed by atomic number ---------------------------- #
    def _table(self, device, dtype):
        key = (str(device), dtype)
        if key in self._tables:
            return self._tables[key]
        from ..fields.vasp.augmentation import irreps_blocks

        length = max(v.shape[-1] for v in self.mean.values())
        rows = torch.zeros(119, dtype=torch.long)
        mean = torch.zeros(len(self.mean) + 1, self.sets, length, dtype=dtype)
        scale = torch.ones(len(self.mean) + 1, self.sets, length, dtype=dtype)
        weight = torch.zeros(len(self.mean) + 1, 1, length, dtype=dtype)
        for row, z in enumerate(sorted(self.mean), start=1):
            rows[z] = row
            n = self.mean[z].shape[-1]
            mean[row, :, :n] = torch.as_tensor(self.mean[z], dtype=dtype)
            scale[row, :, :n] = torch.as_tensor(self.scale[z], dtype=dtype)
            blocks = irreps_blocks(self.layouts[z])
            for index in blocks.values():
                weight[row, 0, torch.as_tensor(index.reshape(-1))] = \
                    1.0 / (len(blocks) * index.size)
        table = tuple(t.to(device) for t in (rows, mean, scale, weight))
        self._tables[key] = table
        return table

    def _lookup(self, species, device, dtype):
        rows, mean, scale, weight = self._table(device, dtype)
        index = rows[species.clamp(0, 118)]
        return mean[index], scale[index], weight[index]

    def normalize(self, records, species):
        mean, scale, _ = self._lookup(species, records.device, records.dtype)
        n = records.shape[-1]
        return (records - mean[..., :n]) / scale[..., :n]

    def inverse(self, z, species):
        mean, scale, _ = self._lookup(species, z.device, z.dtype)
        n = z.shape[-1]
        return z * scale[..., :n] + mean[..., :n]

    def loss(self, predicted, target, species, atom_mask):
        """Mean squared error in the normalized form, each L block weighted
        equally, averaged over real atoms and sets."""
        # The model pads to its longest element; a shape bucket containing
        # only shorter records (e.g. H in a CHON model) pads to less.
        if predicted.shape[-1] < target.shape[-1]:
            raise ValueError("the prediction is shorter than the PAW target")
        predicted = predicted[..., :target.shape[-1]]
        _, _, weight = self._lookup(species, predicted.device, predicted.dtype)
        n = predicted.shape[-1]
        error = ((predicted - target) ** 2) * weight[..., :n]
        # Per-set weights, normalized to their mean so the default (all ones)
        # is the unweighted loss to the bit and a weighted one keeps its scale.
        sets = torch.as_tensor(self.set_weights, dtype=predicted.dtype,
                               device=predicted.device)
        sets = (sets / sets.mean())[:predicted.shape[2]]
        error = error * sets[:, None]
        atoms = atom_mask.sum().clamp(min=1) * predicted.shape[2]
        return (error * atom_mask[..., None, None]).sum() / atoms

    def state_dict(self):
        return {"layouts": {str(z): list(v) for z, v in self.layouts.items()},
                "sets": self.sets,
                "mean": {str(z): v.tolist() for z, v in self.mean.items()},
                "scale": {str(z): v.tolist() for z, v in self.scale.items()},
                "set_weights": list(self.set_weights)}

    @classmethod
    def from_state_dict(cls, state):
        return cls({int(z): v for z, v in state["layouts"].items()},
                   state["sets"],
                   {int(z): v for z, v in state["mean"].items()},
                   {int(z): v for z, v in state["scale"].items()},
                   set_weights=state.get("set_weights"))


def occupancy_error(predicted, target, paw_mask):
    r"""
    Relative RMS in physical units, pooled over atoms, sets and components:
    :math:`\sqrt{\sum|\hat\rho - \rho|^2 / \sum|\rho|^2}`.

    Parameters
    ----------
    predicted, target : torch.Tensor
        ``(B, A, S, L)`` in physical units.
    paw_mask : torch.Tensor
        ``(B, A, L)``, as :func:`~poraque.ml.data.collate_fields` builds it.

    Returns
    -------
    tuple of torch.Tensor
        ``(squared error, squared norm)`` sums, so a caller can pool batches
        before taking the ratio.
    """
    if predicted.shape[-1] < target.shape[-1]:
        raise ValueError("the prediction is shorter than the PAW target")
    predicted = predicted[..., :target.shape[-1]]
    valid = paw_mask[:, :, None, :].to(predicted.dtype)
    return (((predicted - target) ** 2) * valid).sum(), \
        ((target ** 2) * valid).sum()
