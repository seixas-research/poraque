# -*- coding: utf-8 -*-
# file: occupancies.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
What can predict a Pt atom's PAW augmentation occupancies, and in what form?

The target of ``ext2paw`` beside the pseudo-density: per atom, the 138 numbers
:math:`\rho(\ell\ell'LM)` VASP writes after the grid, one set for the total and
one for the magnetisation. Two conventions are pinned from the data before
anything is fitted, because every model below depends on them:

* the **channel order** is the POTCAR's projector order, d d s s p p (the
  ``l = 3`` line of the Description is the local potential, not a projector).
  In a bulk cell the non-zero components then sit exactly at L = 0 and L = 4,
  as cubic site symmetry requires; the order s s p p d d puts them at every L;
* the **M index** follows the standard real spherical harmonics, m = -L..L:
  the bulk L = 4 block is Y_40 + sqrt(5/7) Y_44 to six digits.

So a record is 9 L = 0 scalars, 8 L = 1 vectors, 10 L = 2, 4 L = 3 and 3 L = 4
spherical tensors, and a model can be exactly equivariant: one linear map per L,
shared across M. Everything is measured with whole structures held out.

Usage::

    python experiments/paw_occupancies/occupancies.py            # extract + evaluate
    python experiments/paw_occupancies/occupancies.py --refresh  # re-extract

The extraction (records from the native CHGCARs, projections from the cached
fields) is written once to ``data/cache/paw_occupancies/<cache tag>.npz``.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.special import spherical_jn, sph_harm_y

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Projector channels of PAW_PBE Pt, in POTCAR order: their l.
CHANNELS = (2, 2, 0, 0, 1, 1)
LMAX = 4

#: Neighbour expansion: Gaussian radial functions (Ang) under a cosine cutoff.
NEIGHBOUR_CUTOFF = 6.0
NEIGHBOUR_CENTRES = np.linspace(2.2, 5.8, 10)
NEIGHBOUR_WIDTH = 0.3

#: Grid projection around each nucleus: the augmentation sphere and a little.
PROJECTION_RADIUS = 2.0
PROJECTION_CENTRES = np.linspace(0.0, 2.0, 9)
PROJECTION_WIDTH = 0.2

#: The same projection computed from the Fourier coefficients inside one sphere
#: |G| <= G_max, common to every cell. Below the smallest per-axis Nyquist
#: frequency of the res64 cache (9.33 1/Ang, a 21.55 Ang nanoparticle box at 64
#: points), so every structure supplies exactly the same band.
SPECTRAL_GMAX = 9.0


# ---------------------------------------------------------------------- #
# Record layout
# ---------------------------------------------------------------------- #
def record_layout(channels=CHANNELS):
    """``[(a, b, L, M)]`` in file order: channel pairs a <= b, L by parity."""
    rows = []
    for a in range(len(channels)):
        for b in range(a, len(channels)):
            la, lb = channels[a], channels[b]
            for L in range(abs(la - lb), la + lb + 1, 2):
                for M in range(2 * L + 1):
                    rows.append((a, b, L, M))
    return rows


def irreps_index(channels=CHANNELS):
    """``{L: (P_L, 2L+1) int array}`` of positions in the 138-vector."""
    rows = record_layout(channels)
    blocks = {}
    for L in range(LMAX + 1):
        pairs = sorted({(a, b) for a, b, degree, _ in rows if degree == L})
        if pairs:
            blocks[L] = np.array([[rows.index((a, b, L, M))
                                   for M in range(2 * L + 1)]
                                  for a, b in pairs])
    return blocks


def to_irreps(records, blocks):
    """``(atoms, 138)`` -> ``{L: (atoms, P_L, 2L+1)}``; a permutation, exactly."""
    return {L: records[:, index] for L, index in blocks.items()}


def from_irreps(irreps, blocks, length):
    """The inverse of :func:`to_irreps`."""
    atoms = next(iter(irreps.values())).shape[0]
    out = np.zeros((atoms, length))
    for L, index in blocks.items():
        out[:, index] = irreps[L]
    return out


# ---------------------------------------------------------------------- #
# Geometry
# ---------------------------------------------------------------------- #
def real_sh(L, vectors):
    """Real spherical harmonics, m = -L..L, of unit ``vectors``: ``(n, 2L+1)``."""
    theta = np.arccos(np.clip(vectors[:, 2], -1.0, 1.0))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    Y = np.empty((len(vectors), 2 * L + 1))
    for m in range(-L, L + 1):
        c = sph_harm_y(L, abs(m), theta, phi)
        Y[:, L + m] = (c.real if m == 0 else
                       np.sqrt(2.0) * (-1) ** m * (c.real if m > 0 else c.imag))
    return Y


def family(natoms, lengths):
    if natoms == 32 and max(lengths) < 9.0:
        return "bulk"
    if min(lengths) > 18.0:
        return "nanoparticle"
    return "slab"


def neighbour_expansion(cell, frac):
    """
    ``{L: (atoms, n, 2L+1)}`` = sum_j R_n(r_aj) Y_LM(r_aj): the first-order
    atomic density expansion (ACE body order 2), equivariant by construction.
    """
    cart = frac @ cell
    images = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=float) @ cell
    out = {L: np.zeros((len(frac), len(NEIGHBOUR_CENTRES), 2 * L + 1))
           for L in range(LMAX + 1)}
    for a, origin in enumerate(cart):
        d = (cart[None, :, :] + images[:, None, :] - origin).reshape(-1, 3)
        r = np.linalg.norm(d, axis=1)
        keep = (r > 0.1) & (r < NEIGHBOUR_CUTOFF)
        d, r = d[keep], r[keep]
        cutoff = 0.5 * (np.cos(np.pi * r / NEIGHBOUR_CUTOFF) + 1.0)
        radial = np.exp(-(r[:, None] - NEIGHBOUR_CENTRES) ** 2
                        / (2 * NEIGHBOUR_WIDTH ** 2)) * cutoff[:, None]
        unit = d / r[:, None]
        for L in range(LMAX + 1):
            out[L][a] = radial.T @ real_sh(L, unit)
    return out


def grid_projection(values, cell, frac):
    """
    ``{L: (atoms, n, 2L+1)}`` = dV sum_r f(r) g_n(|r - R_a|) Y_LM: the field
    projected onto radial x angular functions around each nucleus. The same
    object a grid-to-atom kernel integral (a GNO readout) computes, with the
    kernel fixed rather than learned.
    """
    shape = np.array(values.shape)
    volume = abs(np.linalg.det(cell))
    dV = volume / values.size
    reciprocal = np.linalg.inv(cell).T
    half = np.ceil(PROJECTION_RADIUS * np.linalg.norm(reciprocal, axis=1)
                   * shape).astype(int)
    out = {L: np.zeros((len(frac), len(PROJECTION_CENTRES), 2 * L + 1))
           for L in range(LMAX + 1)}
    for a, f in enumerate(frac % 1.0):
        centre = np.rint(f * shape).astype(int)
        axes = [np.arange(c - h, c + h + 1) for c, h in zip(centre, half)]
        offsets = [ax / n - fi for ax, n, fi in zip(axes, shape, f)]
        du, dv, dw = np.meshgrid(*offsets, indexing="ij")
        d = np.stack([du, dv, dw], axis=-1).reshape(-1, 3) @ cell
        r = np.linalg.norm(d, axis=1)
        local = values[np.ix_(axes[0] % shape[0], axes[1] % shape[1],
                              axes[2] % shape[2])].reshape(-1)
        keep = r < PROJECTION_RADIUS
        d, r, local = d[keep], r[keep], local[keep]
        radial = np.exp(-(r[:, None] - PROJECTION_CENTRES) ** 2
                        / (2 * PROJECTION_WIDTH ** 2))
        radial *= 0.5 * (np.cos(np.pi * r / PROJECTION_RADIUS) + 1.0)[:, None]
        weighted = radial * local[:, None] * dV
        safe = r > 1e-8
        unit = np.zeros_like(d)
        unit[safe] = d[safe] / r[safe, None]
        unit[~safe] = (0.0, 0.0, 1.0)
        for L in range(LMAX + 1):
            Y = real_sh(L, unit)
            if L:
                Y[~safe] = 0.0
            out[L][a] = weighted.T @ Y
    return out


def _radial_transforms(gmax=SPECTRAL_GMAX, points=1500):
    r"""
    ``(G, integrals)`` with ``integrals[n, L, k]`` = int_0^R g_n(r) cut(r) j_L(G_k r) r^2 dr.
    """
    G = np.linspace(0.0, gmax * 1.001, points)
    r = np.linspace(0.0, PROJECTION_RADIUS, 801)
    radial = np.exp(-(r[:, None] - PROJECTION_CENTRES) ** 2
                    / (2 * PROJECTION_WIDTH ** 2))
    radial *= 0.5 * (np.cos(np.pi * r / PROJECTION_RADIUS) + 1.0)[:, None]
    integrals = np.empty((len(PROJECTION_CENTRES), LMAX + 1, points))
    for L in range(LMAX + 1):
        bessel = spherical_jn(L, np.outer(G, r))                 # (k, r)
        integrand = radial.T[:, None, :] * bessel[None] * r ** 2  # (n, k, r)
        integrals[:, L, :] = np.trapezoid(integrand, r, axis=-1)
    return G, integrals


_TRANSFORMS = None


def spectral_projection(values, cell, frac, weight=None, gmax=SPECTRAL_GMAX):
    r"""
    :func:`grid_projection` from the Fourier coefficients inside ``|G| <= gmax``.

    With :math:`f(\mathbf r) = \sum_G f_G e^{i\mathbf G\cdot\mathbf r}` and
    the plane-wave expansion,

    .. math:: c_{nLM}(\mathbf R) = \sum_{|G|\le G_{\max}} f_G
              e^{i\mathbf G\cdot\mathbf R}\, 4\pi i^L\, I_{nL}(|G|)\,
              Y_{LM}(\hat{\mathbf G}),

    which depends on the field's band inside the sphere and on nothing about the
    grid it was stored on. ``weight`` multiplies :math:`f_G` first:
    ``"laplacian"`` is :math:`-|G|^2`, which turns :math:`V_{\rm ext}` into
    (a multiple of) the pseudo-ion charge and removes its arbitrary constant.
    """
    global _TRANSFORMS
    if _TRANSFORMS is None:
        _TRANSFORMS = _radial_transforms(gmax)
    grid_G, integrals = _TRANSFORMS

    shape = values.shape
    coefficients = np.fft.fftn(values) / values.size
    k = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in shape],
                             indexing="ij"), axis=-1).reshape(-1, 3)
    vectors = 2.0 * np.pi * k @ np.linalg.inv(cell).T
    size = np.linalg.norm(vectors, axis=1)
    nyquist = np.all(np.abs(k) < np.array(shape) / 2.0, axis=1)
    keep = (size <= gmax) & nyquist
    vectors, size, f = vectors[keep], size[keep], coefficients.reshape(-1)[keep]
    if weight == "laplacian":
        f = -f * size ** 2

    unit = np.zeros_like(vectors)
    nonzero = size > 1e-12
    unit[nonzero] = vectors[nonzero] / size[nonzero, None]
    unit[~nonzero] = (0.0, 0.0, 1.0)
    phases = np.exp(1j * (frac @ cell) @ vectors.T) * f           # (atoms, G)
    out = {}
    for L in range(LMAX + 1):
        Y = real_sh(L, unit)
        if L:
            Y[~nonzero] = 0.0
        radial = np.stack([np.interp(size, grid_G, integrals[n, L])
                           for n in range(integrals.shape[0])], axis=1)   # (G, n)
        kernel = (radial[:, :, None] * Y[:, None, :]).reshape(len(size), -1)
        c = 4.0 * np.pi * (1j ** L) * (phases @ kernel)
        out[L] = c.real.reshape(len(frac), integrals.shape[0], 2 * L + 1)
    return out


# ---------------------------------------------------------------------- #
# Extraction
# ---------------------------------------------------------------------- #
def extract(cache, work):
    from poraque.fields import ChargeDensity, ExternalPotential
    from poraque.fields.vasp.augmentation import occupancy_arrays
    from poraque.fields.vasp.volumetric import read_augmentation_blocks

    natives = sorted(glob.glob(os.path.join(REPO, "data", "vasp", "structures",
                                            "structure_*")))
    arrays = {"records": [], "structure": [], "family": [], "atom": []}
    features = {name: {L: [] for L in range(LMAX + 1)}
                for name in ("positions", "density", "potential",
                             "density_spectral", "ion_spectral")}
    started = time.time()
    for s, native in enumerate(natives):
        name = os.path.basename(native)
        _, blocks = read_augmentation_blocks(os.path.join(native, "CHGCAR"))
        records, _ = occupancy_arrays(blocks, channels=2)
        density = ChargeDensity.read(os.path.join(cache, name, "CHGCAR"))
        potential = ExternalPotential.read(os.path.join(cache, name, "EXTCAR"))
        cell = np.asarray(density.grid.cell, float)
        frac = np.asarray(density.structure.scaled_positions, float)
        assert len(frac) == len(records), name

        kind = family(len(frac), np.linalg.norm(cell, axis=1))
        arrays["records"].append(records)
        arrays["structure"] += [s] * len(frac)
        arrays["family"] += [kind] * len(frac)
        arrays["atom"] += list(range(len(frac)))
        for key, value in (
                ("positions", neighbour_expansion(cell, frac)),
                ("density", grid_projection(density.data, cell, frac)),
                ("potential", grid_projection(potential.data, cell, frac)),
                ("density_spectral", spectral_projection(density.data, cell,
                                                         frac)),
                ("ion_spectral", spectral_projection(
                    potential.data, cell, frac, weight="laplacian"))):
            for L in range(LMAX + 1):
                features[key][L].append(value[L])
        print(f"  {name}  {kind:12s} {len(frac):3d} atoms  "
              f"{time.time() - started:6.1f} s", flush=True)

    payload = {"records": np.concatenate(arrays["records"]),
               "structure": np.asarray(arrays["structure"]),
               "family": np.asarray(arrays["family"]),
               "atom": np.asarray(arrays["atom"])}
    for key, per_L in features.items():
        for L, chunks in per_L.items():
            payload[f"{key}_L{L}"] = np.concatenate(chunks)
    os.makedirs(os.path.dirname(work), exist_ok=True)
    np.savez_compressed(work, **payload)
    return dict(np.load(work))


# ---------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------- #
def invariants(data, sources):
    """L = 0 features: the n00 projections and the power spectrum per L."""
    columns = []
    for source in sources:
        columns.append(data[f"{source}_L0"][:, :, 0])
        for L in range(LMAX + 1):
            F = data[f"{source}_L{L}"]
            n = F.shape[1]
            upper = np.triu_indices(n)
            power = np.einsum("anm,akm->ank", F, F)[:, upper[0], upper[1]]
            columns.append(power)
    return np.concatenate(columns, axis=1)


def equivariants(data, sources, L):
    return np.concatenate([data[f"{source}_L{L}"] for source in sources], axis=1)


class EquivariantRidge:
    """
    One ridge map per L from features ``(atoms, n, 2L+1)`` to targets
    ``(atoms, P_L, 2L+1)``, **shared across M**: exactly rotation-equivariant
    for L > 0, which have no intercept and are scaled without centring, and an
    ordinary ridge on invariants for L = 0.
    """

    def __init__(self, sources, alphas=np.logspace(-6, 4, 21), l0=None):
        self.sources, self.alphas, self.l0 = sources, alphas, l0

    def fit(self, data, index, targets):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.models = {}
        X0 = invariants(data, self.sources)[index]
        self.models[0] = (self.l0() if self.l0 else make_pipeline(
            StandardScaler(), RidgeCV(alphas=self.alphas)))
        self.models[0].fit(X0, targets[0][index][:, :, 0])
        for L in range(1, LMAX + 1):
            F = equivariants(data, self.sources, L)[index]
            X = F.transpose(0, 2, 1).reshape(-1, F.shape[1])
            Y = targets[L][index].transpose(0, 2, 1).reshape(
                -1, targets[L].shape[1])
            self.models[L] = make_pipeline(StandardScaler(with_mean=False),
                                           RidgeCV(alphas=self.alphas,
                                                   fit_intercept=False))
            self.models[L].fit(X, Y)
        return self

    def predict(self, data, index, targets):
        out = {0: self.models[0].predict(
            invariants(data, self.sources)[index])[:, :, None]}
        for L in range(1, LMAX + 1):
            F = equivariants(data, self.sources, L)[index]
            Y = self.models[L].predict(F.transpose(0, 2, 1).reshape(
                -1, F.shape[1]))
            out[L] = Y.reshape(len(F), 2 * L + 1, -1).transpose(0, 2, 1)
        return out


class ElementMean:
    """The training-set average record, in the file's frame (the current table)."""

    def fit(self, data, index, targets):
        self.mean = {L: targets[L][index].mean(axis=0) for L in targets}
        return self

    def predict(self, data, index, targets):
        return {L: np.broadcast_to(self.mean[L], targets[L][index].shape)
                for L in targets}


def kernel_l0():
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return GridSearchCV(make_pipeline(StandardScaler(),
                                      KernelRidge(kernel="rbf")),
                        {"kernelridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                         "kernelridge__gamma": [1e-4, 1e-3, 1e-2]}, cv=3)


# ---------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------- #
def errors(pred, targets, index):
    """Relative RMS over all components, L = 0 and L > 0, pooled over atoms."""
    def rel(Ls):
        e = sum(((pred[L] - targets[L][index]) ** 2).sum() for L in Ls)
        t = sum((targets[L][index] ** 2).sum() for L in Ls)
        return float(np.sqrt(e / t))
    return {"all": rel(list(targets)), "L0": rel([0]),
            "L>0": rel([L for L in targets if L])}


def splits(data, seed=0):
    from sklearn.model_selection import StratifiedKFold

    structures, first = np.unique(data["structure"], return_index=True)
    families = data["family"][first]
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for number, (train, test) in enumerate(folds.split(structures, families)):
        yield (f"fold {number}", np.isin(data["structure"], structures[train]),
               np.isin(data["structure"], structures[test]))
    for held in ("bulk", "slab", "nanoparticle"):
        yield (f"unseen {held}", data["family"] != held,
               data["family"] == held)


def evaluate(data, spin_set=0):
    blocks = irreps_index()
    targets = to_irreps(data["records"][:, spin_set, :], blocks)
    models = {
        "element mean (current table)": lambda: ElementMean(),
        "positions, equivariant ridge": lambda: EquivariantRidge(["positions"]),
        "V_ext projection": lambda: EquivariantRidge(["potential"]),
        "rho projection": lambda: EquivariantRidge(["density"]),
        "rho projection + positions": lambda: EquivariantRidge(
            ["density", "positions"]),
        "rho projection, kernel L=0": lambda: EquivariantRidge(
            ["density"], l0=kernel_l0),
        "rho spectral, |G|<=9": lambda: EquivariantRidge(["density_spectral"]),
        "rho spectral + positions": lambda: EquivariantRidge(
            ["density_spectral", "positions"]),
        "lap V_ext spectral (ion charge)": lambda: EquivariantRidge(
            ["ion_spectral"]),
    }
    results = {}
    for name, build in models.items():
        results[name] = {}
        for split, train, test in splits(data):
            model = build().fit(data, train, targets)
            pred = model.predict(data, test, targets)
            entry = errors(pred, targets, test)
            for kind in ("bulk", "slab", "nanoparticle"):
                mask = data["family"][test] == kind
                if mask.any():
                    sub = {L: p[mask] for L, p in pred.items()}
                    tmask = np.where(test)[0][mask]
                    entry[kind] = errors(sub, targets, tmask)["all"]
            results[name][split] = entry
        cv = [results[name][f"fold {i}"] for i in range(5)]
        summary = {key: float(np.mean([c[key] for c in cv if key in c]))
                   for key in ("all", "L0", "L>0", "bulk", "slab",
                               "nanoparticle")}
        summary.update({f"unseen {k}": results[name][f"unseen {k}"]["all"]
                        for k in ("bulk", "slab", "nanoparticle")})
        results[name]["summary"] = summary
        print(f"  {name:32s} " + "  ".join(f"{k} {v:.4f}"
                                           for k, v in summary.items()),
              flush=True)
    return results


def representation(data):
    """What the 138 numbers are made of, and what an invertible form costs."""
    from poraque.fields.vasp.augmentation import format_augmentation

    blocks = irreps_index()
    report = {}
    for spin_set, label in ((0, "total"), (1, "magnetisation")):
        records = data["records"][:, spin_set, :]
        irreps = to_irreps(records, blocks)
        energy = {L: float((v ** 2).sum()) for L, v in irreps.items()}
        whole = sum(energy.values())
        rank = {}
        for L, v in irreps.items():
            X = v.transpose(0, 2, 1).reshape(-1, v.shape[1])
            if L == 0:
                X = X - X.mean(axis=0)
            s = np.linalg.svd(X, compute_uv=False) ** 2
            share = np.cumsum(s) / s.sum()
            rank[L] = {f"{q}": int(np.searchsorted(share, q) + 1)
                       for q in (0.99, 0.999, 0.9999)}
            rank[L]["pairs"] = int(v.shape[1])
        report[label] = {
            "energy_by_L": {L: e / whole for L, e in energy.items()},
            "rank_by_L": rank,
            "norm_by_family": {
                kind: float(np.linalg.norm(records[data["family"] == kind])
                            / np.sqrt((data["family"] == kind).sum()))
                for kind in ("bulk", "slab", "nanoparticle")},
        }

    # The invertible training form: irreps blocks, per-(L, pair) scale, and an
    # orthogonal rotation of the pair space shared over M. Round trip:
    records = data["records"][:, 0, :]
    irreps = to_irreps(records, blocks)
    forward, inverse = {}, {}
    for L, v in irreps.items():
        mean = v.mean(axis=0) if L == 0 else np.zeros(v.shape[1:])
        scale = np.sqrt(((v - mean) ** 2).mean(axis=(0, 2)))[:, None]
        X = ((v - mean) / scale).transpose(0, 2, 1).reshape(-1, v.shape[1])
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        z = np.einsum("pq,aqm->apm", vt, (v - mean) / scale)
        forward[L] = z
        inverse[L] = np.einsum("pq,apm->aqm", vt, z) * scale + mean
    back = from_irreps(inverse, blocks, records.shape[1])

    def identical(a, b):
        return all(format_augmentation(list(a[i:i + 40]))
                   == format_augmentation(list(b[i:i + 40]))
                   for i in range(0, len(a), 40))

    regrouped = from_irreps(to_irreps(records, blocks), blocks,
                            records.shape[1])
    scaled = {}
    for L, v in irreps.items():
        scale = np.sqrt((v ** 2).mean(axis=(0, 2)))[:, None]
        scaled[L] = (v / scale) * scale
    rescaled = from_irreps(scaled, blocks, records.shape[1])
    tiny = np.abs(records) < 1e-12
    report["round_trip"] = {
        "regroup_bit_exact": bool(np.array_equal(regrouped, records)),
        "regroup_scale_text_identical": bool(identical(rescaled, records)),
        "centre_rotate_max_abs": float(np.abs(back - records).max()),
        "centre_rotate_text_identical": bool(identical(back, records)),
        "centre_rotate_text_identical_above_1e-12": bool(identical(
            np.where(tiny, 0.0, back), np.where(tiny, 0.0, records))),
        "symmetry_zero_fraction": float(tiny.mean()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default=os.path.join(
        REPO, "data", "cache", "res64_potcar"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    tag = os.path.basename(os.path.normpath(args.cache))
    work = os.path.join(REPO, "data", "cache", "paw_occupancies", f"{tag}.npz")
    if args.refresh or not os.path.exists(work):
        print(f"extracting from {args.cache}")
        data = extract(args.cache, work)
    else:
        data = dict(np.load(work))
    print(f"{len(data['records'])} atoms in "
          f"{len(np.unique(data['structure']))} structures")

    print("\nrepresentation")
    rep = representation(data)
    print(json.dumps(rep, indent=1))

    results = {"representation": rep}
    for spin_set, label in ((0, "total"), (1, "magnetisation")):
        print(f"\nmodels, {label} set (relative RMS error, held-out structures)")
        results[label] = evaluate(data, spin_set)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"results_{tag}.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1, default=str)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
