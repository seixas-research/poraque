# -*- coding: utf-8 -*-
# file: paw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The ``ext2paw`` operator: the pseudo-density and every atom's PAW occupancies.

A ``CHGCAR`` is a grid plus, per atom, the one-centre occupancies
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
    fields are first restricted to :math:`|\mathbf G| \le G_{\max}` and resampled
    to a spacing :math:`\pi / 2G_{\max}`, so the quadrature is the same wherever
    the atom sits and whatever grid the cell came on. The solid harmonics are
    polynomials, smooth at the nucleus, so their aliasing at that spacing is
    below 1e-4.
:class:`NeighbourExpansion`
    :math:`\sum_j R_n(r_{aj})\,Y_{LM}(\hat r_{aj})`: the geometry, first order.
:class:`EquivariantHead`
    Per element: invariants (L = 0 features and the norms of the others) through
    an MLP, which predicts L = 0 and **gates** a linear map of each L > 0 block.
    Linear skip paths on every block, so the head starts at least as expressive
    as the linear readout that was measured.
:class:`OccupancyTransform`
    The invertible training form: centre L = 0, scale each (set, L, pair), fitted
    on the training split and stored in the checkpoint. Regrouping into L blocks
    happens in the head, through :func:`~poraque.fields.vasp.augmentation.irreps_blocks`.

The whole operator is rotation-equivariant exactly when its FNO backbone is
(``model.equivariant.enable`` with ``use_coordinates: false``); the readout,
the neighbour expansion and the head are equivariant by construction.
"""

import math

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

#: Neighbour expansion: Gaussian radial functions under a cosine cutoff.
NEIGHBOUR_CUTOFF = 6.0
NEIGHBOUR_CENTRES = tuple(float(c) for c in np.linspace(2.2, 5.8, 10))
NEIGHBOUR_WIDTH = 0.3

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
    Project fields onto atom-centred solid-harmonic Gaussians, over one band.

    .. math::

        c_{nLM}(\mathbf R_a) = \int f_{G_{\max}}(\mathbf R_a + \mathbf d)\;
            N_{nL}\, r^L Y_{LM}(\hat{\mathbf d})\, e^{-r^2/2s_n^2}\, w(r)\,
            d^3 d

    with :math:`f_{G_{\max}}` the field restricted to
    :math:`|\mathbf G| \le G_{\max}` and :math:`w` a cosine window at
    :data:`READOUT_RADIUS`. :math:`N_{nL}` makes each kernel unit-norm, so a
    projection's size does not depend on L or on the width.

    The restriction is what makes the readout a property of the field rather
    than of its grid. The band-limited field is resampled to spacing
    :math:`h = \pi / 2G_{\max}`, and the integral is a sum over that fine grid
    around each atom. Every cell, coarse or fine, is integrated with the same
    quadrature, whose aliasing error is below 1e-4 for these kernels.

    Parameters
    ----------
    g_max : float
        The band in Å⁻¹. Every field read must supply it: a grid whose Nyquist
        frequency is below ``g_max`` along any axis **raises**, since reading
        it would silently use a narrower band than training did.
    lmax : int
    """

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

    @property
    def size(self):
        """Radial functions per field channel."""
        return len(READOUT_WIDTHS)

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
        out = torch.zeros(channels, *target, dtype=spectrum.dtype,
                          device=fields.device)
        scale = float(np.prod(target)) / float(np.prod(shape))
        out[(slice(None), *destination)] = spectrum[(slice(None), *source)] \
            * scale
        return torch.fft.ifftn(out, dim=(-3, -2, -1)).real

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
        centre = torch.round(frac * shape_t.to(frac.dtype)).long()  # (A, 3)
        index = centre[:, None, :] + stencil[None]                  # (A, P, 3)
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
        volume = torch.abs(torch.linalg.det(cell)) / float(np.prod(shape))

        weighted = torch.einsum("cap,apn->acnp", values, radial) * volume
        out = {}
        for L in range(self.lmax + 1):
            c = torch.einsum("acnp,apm->acnm", weighted, harmonics[L])
            c = c * self.norms[L].to(c.dtype)[None, None, :, None]
            out[L] = c.reshape(c.shape[0], -1, 2 * L + 1)
        return out


class NeighbourExpansion(nn.Module):
    r"""
    :math:`\sum_j R_n(r_{aj})\,Y_{LM}(\hat{\mathbf r}_{aj})` over periodic
    images within :data:`NEIGHBOUR_CUTOFF`: ``{L: (A, n, 2L+1)}``.
    """

    def __init__(self, lmax=MAX_L):
        super().__init__()
        self.lmax = int(lmax)
        self.register_buffer("centres", torch.tensor(NEIGHBOUR_CENTRES))

    @property
    def size(self):
        return len(NEIGHBOUR_CENTRES)

    def forward(self, cell, frac):
        reciprocal = torch.linalg.inv(cell).T
        reach = [int(math.ceil(NEIGHBOUR_CUTOFF
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
        active = (r > 1e-3) & (r < NEIGHBOUR_CUTOFF)
        safe = torch.where(active, r, torch.ones_like(r))
        unit = d / safe[..., None]
        cutoff = torch.where(active, 0.5 * (torch.cos(
            math.pi * r / NEIGHBOUR_CUTOFF) + 1.0), torch.zeros_like(r))
        centres = self.centres.to(r.dtype)
        radial = torch.exp(-(r[..., None] - centres) ** 2
                           / (2 * NEIGHBOUR_WIDTH ** 2)) * cutoff[..., None]
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
    ``sites`` it also returns the occupancies, normalised as the
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

    def __init__(self, backbone, layouts, g_max):
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
        self.neighbours = NeighbourExpansion(self.lmax)
        self.reduce = nn.Conv3d(backbone.width, READOUT_CHANNELS, kernel_size=1)

        read = (READOUT_CHANNELS + backbone.out_channels) * self.readout.size
        inputs = {L: read + self.neighbours.size
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
            geometry = self.neighbours(cell[b].to(reads.dtype), frac)
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
    """

    #: A block whose RMS is below this fraction of its element's largest is
    #: scaled by the floor instead --- a symmetry-zero block must not be
    #: divided by its own round-off.
    FLOOR = 1e-6

    def __init__(self, layouts, sets, mean, scale):
        self.layouts = {int(z): tuple(int(degree) for degree in channels)
                        for z, channels in layouts.items()}
        self.sets = int(sets)
        self.mean = {int(z): np.asarray(v, dtype=float) for z, v in mean.items()}
        self.scale = {int(z): np.asarray(v, dtype=float)
                      for z, v in scale.items()}
        self._tables = {}

    @classmethod
    def fit(cls, dataset, layouts, sets):
        """
        Fit on a dataset's ``site_targets``, which should be the training split.

        Raises
        ------
        ValueError
            When an element in the data has no layout, or its records are not
            the length its layout gives.
        """
        from ..fields.vasp.augmentation import irreps_blocks

        records = {}
        for index in range(len(dataset)):
            sites = dataset.site_targets(index)
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
            centred = stack - mu
            rms = {}
            for L, index in blocks.items():
                block = centred[:, :, index]                # (N, S, P, M)
                rms[L] = np.sqrt((block ** 2).mean(axis=(0, 3)))    # (S, P)
            largest = max(float(v.max()) for v in rms.values()) or 1.0
            for L, index in blocks.items():
                value = np.maximum(rms[L], cls.FLOOR * largest)
                sigma[:, index] = value[:, :, None]
            mean[z], scale[z] = mu, sigma
        return cls({z: layouts[z] for z in records}, sets, mean, scale)

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
        """Mean squared error in the normalised form, each L block weighted
        equally, averaged over real atoms and sets."""
        _, _, weight = self._lookup(species, predicted.device, predicted.dtype)
        n = predicted.shape[-1]
        error = ((predicted - target) ** 2) * weight[..., :n]
        atoms = atom_mask.sum().clamp(min=1) * predicted.shape[2]
        return (error * atom_mask[..., None, None]).sum() / atoms

    def state_dict(self):
        return {"layouts": {str(z): list(v) for z, v in self.layouts.items()},
                "sets": self.sets,
                "mean": {str(z): v.tolist() for z, v in self.mean.items()},
                "scale": {str(z): v.tolist() for z, v in self.scale.items()}}

    @classmethod
    def from_state_dict(cls, state):
        return cls({int(z): v for z, v in state["layouts"].items()},
                   state["sets"],
                   {int(z): v for z, v in state["mean"].items()},
                   {int(z): v for z, v in state["scale"].items()})


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
    valid = paw_mask[:, :, None, :].to(predicted.dtype)
    return (((predicted - target) ** 2) * valid).sum(), \
        ((target ** 2) * valid).sum()
