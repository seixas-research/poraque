# -*- coding: utf-8 -*-
# file: resample.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Spectral (Fourier) resampling of periodic fields.

Changing the resolution of a field produced by a plane-wave code is not an
interpolation problem — it is a **basis-truncation** problem. The field is a
finite Fourier series on the unit cell,

.. math:: f(\mathbf r) = \sum_{\mathbf G} c_{\mathbf G}\, e^{i\mathbf G\cdot\mathbf r},

so restricting it to a coarser grid means keeping the coefficients
:math:`c_{\mathbf G}` that the coarser grid can represent and discarding the
rest. Nothing is approximated: the coarse field is the *exact* band-limited
projection of the fine one, it is still exactly periodic, and its cell average
— hence :math:`\int\rho\,d^3r`, the electron count — is preserved to machine
precision.

Trilinear or spline interpolation would do none of these things. It would
alias high-frequency content down onto low frequencies, break periodicity at
the cell boundary, and shift the integral.

Downsampling is what makes the 128³ VASP fields tractable for prototyping a
neural operator on a CPU; upsampling is the same operation with the coefficient
block zero-padded instead of truncated, and is how an FNO trained at one
resolution is evaluated at another.

.. note::
   The Nyquist row of each axis is dropped rather than split between
   :math:`+G_{\max}` and :math:`-G_{\max}`. Keeping only one of a
   conjugate pair would break the Hermitian symmetry of a real field and leak
   a spurious imaginary part into the result. The cost is a single mode per
   axis, which is negligible at any useful truncation ratio.
"""

import numpy as np

from .grid import FieldGrid


def spectral_resample(data, shape):
    """
    Resample a periodic field onto a different grid by Fourier truncation.

    Parameters
    ----------
    data : array_like
        Real ``(N1, N2, N3)`` field.
    shape : tuple of int
        Target ``(M1, M2, M3)``. Each axis may be coarser or finer.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``shape``.

    Notes
    -----
    ``norm="forward"`` is used in both directions so the stored values are
    Fourier-series coefficients. The :math:`\\mathbf G = 0` coefficient is then
    the cell average, and copying it unchanged is what preserves the mean — and
    therefore any integral — exactly.
    """
    data = np.asarray(data, dtype=float)
    source_shape = data.shape
    target_shape = tuple(int(n) for n in shape)

    if len(target_shape) != 3 or any(n < 2 for n in target_shape):
        raise ValueError(f"Target shape must be three integers >= 2, got {shape!r}.")
    if target_shape == source_shape:
        return data.copy()

    coefficients = np.fft.fftn(data, norm="forward")
    resampled = np.zeros(target_shape, dtype=complex)

    # Highest frequency safely representable on *both* grids, per axis. The
    # -1 drops the Nyquist row, keeping the retained block conjugate-symmetric.
    keep = [min(source_shape[i], target_shape[i]) // 2 - 1 for i in range(3)]
    if any(k < 1 for k in keep):
        raise ValueError(
            f"Cannot resample {source_shape} -> {target_shape}: at least one "
            f"axis has too few modes."
        )

    # Positive and negative frequency blocks, copied by absolute frequency so
    # each coefficient lands on the same physical G in the target.
    slices = []
    for axis in range(3):
        n_keep = keep[axis]
        slices.append((
            (slice(0, n_keep + 1), slice(0, n_keep + 1)),                 # 0..+k
            (slice(source_shape[axis] - n_keep, None),
             slice(target_shape[axis] - n_keep, None)),                   # -k..-1
        ))

    for source_x, target_x in slices[0]:
        for source_y, target_y in slices[1]:
            for source_z, target_z in slices[2]:
                resampled[target_x, target_y, target_z] = \
                    coefficients[source_x, source_y, source_z]

    return np.real(np.fft.ifftn(resampled, norm="forward"))


def resample_grid(grid, shape):
    """
    A :class:`~poraque.fields.FieldGrid` with the same cell and a new shape.

    Parameters
    ----------
    grid : FieldGrid
        Source grid.
    shape : tuple of int
        Target shape.

    Returns
    -------
    FieldGrid
    """
    return FieldGrid(shape, grid.cell, encut=grid.encut, prec=grid.prec)


def resample_field(field, shape, grid=None):
    """
    Resample a field onto a new grid, spin pair included.

    Parameters
    ----------
    field : ScalarField or SpinDensity
        Field to resample. A :class:`~poraque.fields.SpinDensity` has its two
        channels truncated independently: each is band-limited in its own
        right, so this is the same operation spelled for the two-argument
        constructor.
    shape : tuple of int
        Target grid shape.
    grid : FieldGrid, optional
        Pre-built target grid to attach — pass the *same* object for every
        field of a material so they keep sharing one mesh after resampling.

    Returns
    -------
    ScalarField or SpinDensity
        A new instance of the same class.
    """
    target = grid if grid is not None else resample_grid(field.grid, shape)
    if tuple(target.shape) != tuple(shape):
        raise ValueError(
            f"Supplied grid has shape {target.shape}, expected {tuple(shape)}."
        )

    metadata = dict(field.metadata)
    metadata["resampled_from"] = tuple(field.grid.shape)

    from .spin import SpinDensity

    if isinstance(field, SpinDensity):
        return SpinDensity(spectral_resample(field.total, shape),
                           spectral_resample(field.magnetization, shape),
                           target, field.structure, metadata=metadata)
    return type(field)(spectral_resample(field.data, shape), target,
                       field.structure, metadata=metadata)


def downsampled_grid(grid, resolution):
    """
    The grid a material is cached and trained on, given its native one.

    Parameters
    ----------
    grid : FieldGrid
        The native grid.
    resolution : int or None
        Longest axis after downsampling, as :func:`downsample_shape` reads
        ``target_max``. ``None`` or ``0`` keeps the native grid, so a caller
        can pass the setting straight through.

    Returns
    -------
    FieldGrid
        ``grid`` itself when nothing is to be done, else a new grid on the same
        cell — one object, to be shared by every field of the material.
    """
    if not resolution:
        return grid
    return resample_grid(grid, downsample_shape(grid.shape,
                                               target_max=int(resolution)))


def downsample_shape(shape, factor=None, target_max=None):
    """
    Choose a coarser, FFT-friendly shape.

    Parameters
    ----------
    shape : tuple of int
        Source shape.
    factor : int, optional
        Divide every axis by this integer.
    target_max : int, optional
        Scale so the longest axis is at most this many points, preserving the
        aspect ratio — the right choice when materials have different grids,
        since it keeps the *physical* resolution comparable rather than forcing
        a common shape.

    Returns
    -------
    tuple of int
    """
    from .grid import fft_friendly_size

    shape = tuple(int(n) for n in shape)
    if factor is not None:
        scaled = [n / float(factor) for n in shape]
    elif target_max is not None:
        ratio = float(target_max) / max(shape)
        scaled = [n * ratio for n in shape]
    else:
        raise ValueError("Pass either `factor` or `target_max`.")

    return tuple(min(source, fft_friendly_size(value))
                 for source, value in zip(shape, scaled))
