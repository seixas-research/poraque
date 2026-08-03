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


def write_volumetric(path, structure, data, comment=None, columns=5,
                     fmt="%18.11E", direct=True):
    """
    Write a VASP volumetric file (``CHGCAR`` layout).

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
    fmt : str, optional
        ``printf`` format for each value.
    direct : bool, optional
        Write fractional (``Direct``) coordinates.

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

        for start in range(0, flat.size, columns):
            chunk = flat[start:start + columns]
            handle.write(" ".join(fmt % value for value in chunk))
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
