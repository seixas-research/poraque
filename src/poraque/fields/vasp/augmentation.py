# -*- coding: utf-8 -*-
# file: augmentation.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
A transferable per-element table of PAW augmentation occupancies.

VASP's ``ICHARG=1`` wants more than a density on a grid. After the grid block a
``CHGCAR`` carries one record per atom — the **one-centre** PAW terms, the part
of the density inside the augmentation spheres. Those are contractions over the
converged wavefunctions,

.. math::

    \rho^a_{ij} = \sum_{n\mathbf k} f_{n\mathbf k}
        \langle\tilde\Psi_{n\mathbf k}|\tilde p^a_i\rangle
        \langle\tilde p^a_j|\tilde\Psi_{n\mathbf k}\rangle,

transformed to :math:`\rho(ll',L,M)` by Clebsch-Gordan coefficients before
being written (``TRANS_RHOLM`` in VASP's ``paw_base.F``). A grid-based model
predicts nothing of this: the pseudo-density is smooth inside the core radius
*by construction*, so the information is not there to recover.

What this module does instead is **borrow and average**. The records are read
off the training calculations, averaged per chemical element, and stored in the
model bundle; at inference they are written out again for a structure that has
no reference of its own.

.. warning::
   This is an approximation with a measured size. On the Au dataset a single
   averaged reference reproduces the true occupancies to about **9 % RMS**,
   and the dominant component varies by a factor of two across sites. It is a
   defensible starting guess for ``ICHARG=1``, not a converged on-site
   density, and it says nothing about elements or environments absent from the
   training set.
"""

import os

import numpy as np

from .poscar import Poscar
from .volumetric import fortran_exponential, read_augmentation

#: Header VASP writes before each record, and the field layout of the values.
_HEADER = "augmentation occupancies"
_PER_LINE = 5
_WIDTH = 15
_DECIMALS = 7


def parse_augmentation(block):
    """
    Split an extracted augmentation block into per-atom value arrays.

    Parameters
    ----------
    block : sequence of str
        Lines from :func:`~poraque.fields.vasp.volumetric.read_augmentation`.

    Returns
    -------
    list of numpy.ndarray
        One array per atom, in file order.
    """
    records, current = [], None
    for line in block:
        if _HEADER in line:
            current = []
            records.append(current)
        elif current is not None:
            current.extend(float(token) for token in line.split())
    return [np.asarray(values, dtype=float) for values in records]


def format_augmentation(records):
    """
    Render per-atom value arrays as VASP writes them.

    Mirrors ``WRT_RHO_PAW``: ``("augmentation occupancies",2I4)`` for the
    header, then ``(5E15.7)`` for the values.

    Parameters
    ----------
    records : sequence of array_like
        One array per atom, in the order the atoms appear in the structure.

    Returns
    -------
    list of str
    """
    lines = []
    for index, values in enumerate(records, start=1):
        values = np.asarray(values, dtype=float).ravel()
        lines.append(f"{_HEADER}{index:4d}{values.size:4d}")
        for start in range(0, values.size, _PER_LINE):
            chunk = values[start:start + _PER_LINE]
            lines.append("".join(
                fortran_exponential(v, decimals=_DECIMALS, width=_WIDTH)
                for v in chunk))
    return lines


def species_of_each_atom(structure):
    """Chemical symbol per atom, in the order the file lists them."""
    symbols = []
    for symbol, count in zip(structure.symbols, structure.counts):
        symbols.extend([str(symbol)] * int(count))
    return symbols


def reference_from_calculation(source, filename="CHGCAR"):
    """
    Per-element occupancies from one reference calculation.

    Parameters
    ----------
    source : str
        A calculation directory, in which case ``filename`` names the file to
        read; or the path to that file directly. The second form is what lets
        a flat archive — a Materials Project download, where the densities sit
        side by side as ``CHGCAR_<id>.gz`` rather than one per directory —
        contribute a reference too. Compressed files are read in place.
    filename : str, optional
        File to read inside ``source`` when it is a directory.

    Returns
    -------
    dict
        ``{element: {"sum": ndarray, "count": int}}``, an accumulator rather
        than a mean so several calculations can be combined without weighting
        a two-atom cell like a two-hundred-atom one.
    """
    path = source if os.path.isfile(source) else os.path.join(source, filename)
    if not os.path.exists(path):
        return {}

    _, block = read_augmentation(path)
    records = parse_augmentation(block)
    if not records:
        return {}

    structure = Poscar.from_file(path)
    species = species_of_each_atom(structure)
    if len(species) != len(records):
        # A record count that disagrees with the structure means the file and
        # the geometry are not the same system; averaging them would be worse
        # than having no reference at all.
        return {}

    totals = {}
    for element, values in zip(species, records):
        entry = totals.setdefault(
            element, {"sum": np.zeros_like(values), "count": 0})
        if entry["sum"].shape != values.shape:
            return {}                      # inconsistent channel count
        entry["sum"] += values
        entry["count"] += 1
    return totals


def build_reference(sources, filename="CHGCAR", log=None):
    r"""
    Average the augmentation records of several calculations, per element.

    Parameters
    ----------
    sources : iterable of str
        Reference calculations: directories, or paths to the density files
        themselves. See :func:`reference_from_calculation`.
    filename : str, optional
        Which file in each directory carries the records; ignored for entries
        that already name a file.
    log : callable, optional
        Progress sink.

    Returns
    -------
    dict
        ``{element: {"values": list, "atoms": int, "structures": int}}``,
        JSON-serialisable so it can travel inside a model bundle.
    """
    emit = log if log is not None else (lambda *_: None)
    totals, structures = {}, {}

    for directory in sources:
        contribution = reference_from_calculation(directory, filename)
        if not contribution:
            continue
        for element, entry in contribution.items():
            running = totals.setdefault(
                element, {"sum": np.zeros_like(entry["sum"]), "count": 0})
            if running["sum"].shape != entry["sum"].shape:
                emit(f"      PAW: {element} channel count differs between "
                     f"calculations; skipping {directory}")
                continue
            running["sum"] += entry["sum"]
            running["count"] += entry["count"]
            structures[element] = structures.get(element, 0) + 1

    reference = {}
    for element, entry in totals.items():
        if entry["count"]:
            reference[element] = {
                "values": (entry["sum"] / entry["count"]).tolist(),
                "atoms": int(entry["count"]),
                "structures": int(structures.get(element, 0)),
            }
            emit(f"      PAW reference: {element}  "
                 f"{len(reference[element]['values'])} values, averaged over "
                 f"{entry['count']} atoms in {structures.get(element, 0)} "
                 f"structure(s)")
    return reference


def records_for_structure(structure, reference):
    """
    Build the augmentation block for a structure from a stored reference.

    Parameters
    ----------
    structure : Poscar
        Supplies the species order; VASP expects one record per atom in
        exactly that order.
    reference : dict
        As produced by :func:`build_reference`.

    Returns
    -------
    tuple of (list of str, list of str)
        The lines, and the elements that were missing from the reference. A
        partial block is never returned: if any element is absent the lines
        come back empty, because a file with records for some atoms and not
        others is worse than one with none.
    """
    species = species_of_each_atom(structure)
    missing = sorted({element for element in species if element not in reference})
    if missing:
        return [], missing

    records = [np.asarray(reference[element]["values"], dtype=float)
               for element in species]
    return format_augmentation(records), []
