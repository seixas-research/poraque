# -*- coding: utf-8 -*-
# file: volumetric.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Reader/writer for VASP volumetric files (``CHGCAR``, ``CHG``, ``LOCPOT``, ...).

All of these share one layout::

    <POSCAR block>            # comment, scale, cell, species, counts, coords
    <blank line>
    NGXF NGYF NGZF
    v(1,1,1) v(2,1,1) v(3,1,1) ...    # Fortran order: x fastest, z slowest

Poraquê writes ``EXTCAR`` (external potential) and reads ``CHGCAR`` (charge
density) and ``TAUCAR`` (kinetic energy density) through this single code path,
which is what guarantees that the three files of one material are byte-for-byte
compatible on the same grid.

Anything after the first data block (spin channels, PAW augmentation
occupancies) is skipped by default and can be requested explicitly.
"""

import numpy as np

from .poscar import Poscar


def read_volumetric(path, read_all=False):
    """
    Read a VASP volumetric file.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read.
    read_all : bool, optional
        When true, also return any additional data blocks of the same grid
        size (e.g. the spin channel of a spin-polarised ``CHGCAR``).

    Returns
    -------
    structure : Poscar
        The structure header.
    data : numpy.ndarray
        ``(NGXF, NGYF, NGZF)`` array in C order (i.e. already un-transposed
        from the file's Fortran ordering).
    extra : list of numpy.ndarray
        Additional blocks; empty unless ``read_all`` is true.
    """
    with open(path, "r") as handle:
        lines = handle.read().splitlines()

    # --- Structure header: everything up to the first blank line ---
    blank = _first_blank_line(lines)
    structure = Poscar.from_string("\n".join(lines[:blank]))

    # --- Grid dimensions ---
    cursor = blank + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    shape = tuple(int(token) for token in lines[cursor].split()[:3])
    cursor += 1

    n_points = int(np.prod(shape))
    data, cursor = _read_block(lines, cursor, n_points, shape)

    extra = []
    if read_all:
        while cursor < len(lines):
            # Skip augmentation blocks / separators until another grid header
            # with the same dimensions shows up.
            tokens = lines[cursor].split()
            if len(tokens) == 3 and all(_is_int(t) for t in tokens) \
                    and tuple(int(t) for t in tokens) == shape:
                block, cursor = _read_block(lines, cursor + 1, n_points, shape)
                extra.append(block)
            else:
                cursor += 1

    return structure, data, extra


def read_augmentation(path):
    r"""
    Return the PAW augmentation section of a VASP volumetric file.

    A ``CHGCAR`` carries, after the grid block, one ``augmentation
    occupancies`` record per atom. These are the **one-centre** PAW terms: the
    part of the density that lives inside the augmentation spheres and is not
    representable on the plane-wave grid at all. ``ICHARG=1`` expects them, and
    a ``CHGCAR`` without them is not a restartable charge density.

    They are *on-site* quantities, so they do not depend on the FFT grid — the
    same record is valid whatever ``NGXF`` the file is written on.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``CHGCAR``-format file that has augmentation records.

    Returns
    -------
    tuple of (tuple, list of str)
        The grid shape the reference was written on, and the augmentation
        lines verbatim. The list is empty when the file has none — ``CHG``
        never does, and a ``CHGCAR`` from a norm-conserving run will not
        either.

    Notes
    -----
    A spin-polarised ``CHGCAR`` continues past the augmentation with a second
    grid block. That belongs to the reference's magnetisation, not to a
    prediction, so extraction stops at the next grid header rather than
    dragging it along.
    """
    with open(path, "r") as handle:
        lines = handle.read().splitlines()

    blank = _first_blank_line(lines)
    cursor = blank + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    shape = tuple(int(token) for token in lines[cursor].split()[:3])
    cursor += 1

    # Walk the grid block by value count, not by line count: the number of
    # columns is a formatting choice (5 in CHGCAR, 10 in CHG) and counting
    # lines would land in the wrong place on either.
    remaining = int(np.prod(shape))
    while remaining > 0 and cursor < len(lines):
        remaining -= len(lines[cursor].split())
        cursor += 1

    block = []
    while cursor < len(lines):
        tokens = lines[cursor].split()
        if (len(tokens) == 3 and all(_is_int(token) for token in tokens)
                and tuple(int(token) for token in tokens) == shape):
            break                       # a spin channel starts here
        block.append(lines[cursor])
        cursor += 1

    while block and not block[-1].strip():
        block.pop()
    return shape, block


def count_augmentation_records(block):
    """Number of ``augmentation occupancies`` records in an extracted block."""
    return sum(1 for line in block if "augmentation occupancies" in line)


def fortran_exponential(value, decimals=7, width=15):
    r"""
    Render one number exactly as Fortran's ``Ew.d`` edit descriptor does.

    Two things separate this from ``"%*.*E" % (width, decimals, value)``:

    **Mantissa normalisation.** Fortran puts the mantissa in :math:`[0.1, 1)`
    and C puts it in :math:`[1, 10)`, so ``6.424378`` is ``0.6424378E+01``
    here and ``6.4243780E+00`` there.

    **Leading-zero omission.** When the sign would push the field past
    ``width``, Fortran drops the optional zero before the decimal point rather
    than overflowing: ``E17.11`` writes ``-.23625518271E+04``, never
    ``-0.23625518271E+04``. This is not cosmetic. VASP reads its density block
    with ``(1X,E17.11)``, so a value written one column too wide loses its
    minus sign to the ``1X`` and is read back positive.

    Parameters
    ----------
    value : float
        The number to render. Non-finite input is written as zero.
    decimals : int, optional
        The ``d`` of ``Ew.d`` — digits after the decimal point.
    width : int, optional
        The ``w`` of ``Ew.d`` — the field width, *excluding* any ``1X`` the
        surrounding format supplies.

    Returns
    -------
    str
        Exactly ``width`` characters, right-justified; all asterisks if the
        value cannot be represented at all, as Fortran does.

    Examples
    --------
    >>> fortran_exponential(2362.5518271, decimals=11, width=17)
    '0.23625518271E+04'
    >>> fortran_exponential(-2362.5518271, decimals=11, width=17)
    '-.23625518271E+04'
    """
    if not np.isfinite(value) or value == 0.0:
        mantissa, exponent = 0.0, 0
    else:
        exponent = int(np.floor(np.log10(abs(value)))) + 1
        mantissa = value / (10.0 ** exponent)
        # Rounding can push the mantissa back to 1.0 (0.99999995 -> 1.0000000),
        # which is not a legal Fortran mantissa; renormalise when it does.
        if abs(round(mantissa, decimals)) >= 1.0:
            mantissa /= 10.0
            exponent += 1

    # Without an explicit `Ee`, Fortran writes a three-digit exponent by
    # dropping the exponent letter rather than widening the field, so both
    # forms occupy four characters: 0.5E-300 prints as `0.50000000000-299`.
    marker = f"E{exponent:+03d}" if abs(exponent) <= 99 else f"{exponent:+04d}"
    body = f"{mantissa:.{decimals}f}{marker}"
    if len(body) > width:
        if body.startswith("-0."):
            body = "-" + body[2:]
        elif body.startswith("0."):
            body = body[1:]
    if len(body) > width:
        return "*" * width
    return body.rjust(width)


def write_volumetric(path, structure, data, comment=None, columns=5,
                     width=17, decimals=11, direct=True, augmentation=None):
    """
    Write a VASP volumetric file (``CHGCAR`` layout).

    .. warning::
       The density block is **column-positional**, not whitespace-delimited.
       VASP reads it with a non-advancing ``(1X,E17.11)`` — 18 columns per
       value, five values per line, no separators — so a file that merely
       *looks* right fails with ``WARNING: chargedensity file is incomplete``
       if any field lands one column off. That is why the values are
       concatenated here rather than joined, and why they go through
       :func:`fortran_exponential` rather than ``printf``.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.
    structure : Poscar
        Structure header to write.
    data : array_like
        ``(NGX, NGY, NGZ)`` field in C order; it is transposed to the Fortran
        ordering VASP expects on write.
    comment : str, optional
        Overrides the structure's comment line. Use it to record what the file
        holds (units, field type) — VASP ignores this line entirely.
    columns : int, optional
        Values per line (``5`` matches ``CHGCAR``, ``10`` matches ``CHG``).
    width, decimals : int, optional
        The ``w`` and ``d`` of the Fortran ``Ew.d`` descriptor used per value.
        The defaults reproduce VASP's own ``(1X,E17.11)`` density block
        byte-for-byte; ``(11, 5)`` matches the ``1X,G11.5`` short format of
        ``CHG``. See the warning above before changing them.
    direct : bool, optional
        Write fractional (``Direct``) coordinates.
    augmentation : sequence of str, optional
        PAW augmentation lines from :func:`read_augmentation`, appended after
        the grid block. Required by ``ICHARG=1``; omitted the file is a plain
        density with no one-centre terms.

    Returns
    -------
    str
        The path written.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {data.shape}.")

    header = structure.to_string(direct=direct)
    flat = data.ravel(order="F")

    with open(path, "w") as handle:
        if comment is not None:
            body = header.split("\n", 1)[1]
            handle.write(f"{comment}\n{body}")
        else:
            handle.write(header)
        handle.write("\n")
        handle.write("  {:d}  {:d}  {:d}\n".format(*data.shape))

        # The leading space is the format's own `1X`, not a separator: it is
        # what keeps every field exactly `width + 1` columns wide whether or
        # not the value is negative.
        for start in range(0, flat.size, columns):
            chunk = flat[start:start + columns]
            handle.write("".join(
                " " + fortran_exponential(value, decimals=decimals,
                                          width=width)
                for value in chunk))
            handle.write("\n")

        # Appended verbatim. These records are a fixed-format Fortran read on
        # VASP's side, so reformatting them -- even reflowing the columns --
        # risks a file VASP declines to parse. Copying the text is the only
        # transformation that cannot introduce one.
        if augmentation:
            handle.write("\n".join(augmentation))
            handle.write("\n")

    return str(path)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _first_blank_line(lines):
    """Index of the first blank line, which terminates the POSCAR header."""
    for index, line in enumerate(lines):
        if not line.strip():
            return index
    raise ValueError(
        "Malformed volumetric file: no blank line separating the structure "
        "header from the grid data."
    )


def _read_block(lines, cursor, n_points, shape):
    """Read ``n_points`` floats starting at ``cursor`` and reshape them."""
    values = np.empty(n_points, dtype=float)
    filled = 0
    while filled < n_points and cursor < len(lines):
        tokens = lines[cursor].split()
        cursor += 1
        if not tokens:
            continue
        chunk = np.fromiter((float(t) for t in tokens), dtype=float,
                            count=len(tokens))
        take = min(chunk.size, n_points - filled)
        values[filled:filled + take] = chunk[:take]
        filled += take

    if filled < n_points:
        raise ValueError(
            f"Volumetric file truncated: expected {n_points} values, got {filled}."
        )
    return values.reshape(shape, order="F"), cursor


def _is_int(token):
    try:
        int(token)
    except ValueError:
        return False
    return True
