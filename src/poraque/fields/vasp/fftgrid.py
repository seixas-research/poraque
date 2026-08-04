# -*- coding: utf-8 -*-
# file: fftgrid.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
VASP's own FFT grid rule, reproduced exactly.

:meth:`~poraque.fields.FieldGrid.from_encut` sizes a grid the way a plane-wave
code *should*, which is close to what VASP does and not identical to it. When a
predicted ``CHGCAR`` has to be read back by VASP the difference matters: the
file declares its own ``NGXF NGYF NGZF``, and a restart wants those to be the
grid the run itself would build.

The algorithm below is transcribed from ``main.F`` (VASP 6.2.0), where the
density grid is derived in two stages rather than one:

.. code-block:: fortran

    XCUTOF = SQRT(ENMAX/RYTOEV)/(2*PI/(ANORM(1)/AUTOA))
    WFACT  = 4   ! PREC = high | accurate | single
    WFACT  = 3   ! otherwise
    GRID%NGPTAR(1) = XCUTOF*WFACT + 0.5
    CALL FFTCHK(GRID%NGPTAR)               ! <- rounded here
    ...
    GRIDC%NGPTAR(1) = GRID%NGPTAR(1)*2     ! <- then doubled
    CALL FFTCHK(GRIDC%NGPTAR)

**The order is the whole point.** Rounding the coarse grid to an FFT-friendly
size and then doubling is not the same as computing the fine size directly and
rounding once: for a 27-atom gold cell at 450 eV the coarse grid is 61 → 64, so
the density grid is 128 — where rounding ``4 x 15.25 = 61`` in one step gives
64, a factor of two too small. That was a real discrepancy against every
reference calculation in this project's dataset.
"""

import numpy as np

from ..constants import HBAR2_OVER_2M_EV_ANGSTROM2
from ..grid import fft_friendly_size

#: ``PREC`` values taking the wrap-around-free coarse multiplier (``WFACT=4``).
_COARSE_WFACT_4 = ("h", "a", "s")

#: ``PREC`` values whose density grid is simply twice the coarse grid.
_DOUBLED = ("a", "n")


def get_valid_fft_grid_size(minimum):
    r"""
    Smallest admissible FFT length :math:`\ge` ``minimum``.

    VASP's ``FFTCH1`` accepts a length only when dividing out 2, 3, 5 and 7
    leaves 1 **and** the factor 2 appears at least once — so the value must be
    even as well as 7-smooth. ``FFTCHK`` then increments until that holds.

    Parameters
    ----------
    minimum : float or int
        Lower bound; a fractional value is raised to the next integer first.

    Returns
    -------
    int

    Examples
    --------
    >>> get_valid_fft_grid_size(61)
    64
    >>> get_valid_fft_grid_size(109)
    112
    """
    return int(fft_friendly_size(minimum, factors=(2, 3, 5, 7),
                                 force_even=True))


def cutoff_indices(cell, energy):
    r"""
    :math:`|a_i| G_{\rm cut} / 2\pi` per axis — VASP's ``XCUTOF``.

    Since :math:`\mathbf G\cdot\mathbf a_i = 2\pi m_i`, this bounds the plane
    wave index along each direction, for any cell shape.

    Parameters
    ----------
    cell : array_like
        ``(3, 3)`` lattice vectors as rows, Å.
    energy : float
        Cutoff in eV.

    Returns
    -------
    numpy.ndarray
        Three floats, unrounded.
    """
    lengths = np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3),
                             axis=1)
    g_cut = np.sqrt(float(energy) / HBAR2_OVER_2M_EV_ANGSTROM2)
    return g_cut * lengths / (2.0 * np.pi)


def vasp_grid_shapes(cell, encut, prec="normal", enaug=None):
    r"""
    The coarse and density FFT grids VASP would build for this cell.

    Parameters
    ----------
    cell : array_like
        ``(3, 3)`` lattice vectors as rows, Å.
    encut : float
        ``ENCUT`` in eV.
    prec : str, optional
        ``PREC``. Only the first letter is significant, as in VASP.
    enaug : float, optional
        ``ENAUG`` in eV, used only by ``PREC = High`` and ``PREC = Low``,
        whose density grids are derived from it rather than by doubling.
        Defaults to VASP's own fallback of ``1.5 * ENCUT``.

    Returns
    -------
    tuple of (tuple, tuple)
        ``(NGX, NGY, NGZ)`` and ``(NGXF, NGYF, NGZF)``.
    """
    key = str(prec).strip().lower()[:1] or "n"

    # ---- coarse grid: round(XCUTOF * WFACT), then FFTCHK ---------------- #
    wfact = 4 if key in _COARSE_WFACT_4 else 3
    # Fortran's `X + 0.5` assigned to an integer truncates, which is
    # round-half-up for the positive values this always produces.
    coarse = tuple(get_valid_fft_grid_size(int(value * wfact + 0.5))
                   for value in cutoff_indices(cell, encut))

    # ---- density grid --------------------------------------------------- #
    if key == "s":
        fine = coarse                                   # single: one grid
    elif key in _DOUBLED:
        fine = tuple(size * 2 for size in coarse)       # accurate, normal
    else:
        # high and low derive theirs from ENAUG instead.
        augmentation = float(enaug) if enaug else 1.5 * float(encut)
        aug_wfact = 16.0 / 3.0 if key == "h" else 3.0
        fine = tuple(int(value * aug_wfact)
                     for value in cutoff_indices(cell, augmentation))

    return coarse, tuple(get_valid_fft_grid_size(size) for size in fine)


def vasp_density_grid(cell, encut, prec="normal", enaug=None):
    """The density grid alone — what a ``CHGCAR`` declares."""
    return vasp_grid_shapes(cell, encut, prec=prec, enaug=enaug)[1]
