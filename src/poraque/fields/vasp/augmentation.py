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
   This is an approximation with a measured size. On the Pt dataset a single
   averaged reference reproduces the true occupancies to about **9 % RMS**,
   and the dominant component varies by a factor of two across sites. It is a
   defensible starting guess for ``ICHARG=1``, not a converged on-site
   density, and it says nothing about elements or environments absent from the
   training set.
"""

import os
import re

import numpy as np

from .poscar import Poscar
from .volumetric import fortran_exponential, read_augmentation

#: Header VASP writes before each record, and the field layout of the values.
_HEADER = "augmentation occupancies"
_PER_LINE = 5
_WIDTH = 15
_DECIMALS = 7

#: Version of the per-element tables built from parsed records, stamped into
#: every entry as ``"schema"``. 2 is the first built from records read to their
#: **declared** length; a cached table without the stamp was built by a parser
#: that let a spin-polarised file's MAGMOM line into the last record of the
#: first set (139 values for a free Pt atom, 170 for the last atom of a 32-atom
#: cell) and is rebuilt rather than reused.
RECORD_SCHEMA = 2


def parse_augmentation(block):
    """
    Split an extracted augmentation block into per-atom value arrays.

    Each record is read to the length its own header declares ---
    ``augmentation occupancies   1 138`` is atom 1, 138 values --- and not to
    the next header. The difference is not cosmetic. A spin-polarised
    ``CHGCAR`` writes a line of per-ion ``MAGMOM`` values after the last
    record of the first set (``NIONS`` numbers in a wider ``E20.12`` field), and
    reading to the next header folded them into that record: the last atom of
    every ``ISPIN = 2`` file came out ``NIONS`` values too long. The per-element
    average then refused the file for an inconsistent channel count and
    returned nothing for all 97 platinum cells, and the isolated atom's single
    record gained its one ``MAGMOM`` value --- 139 numbers for a 138-value
    ``PAW_PBE Pt`` record, stored in every cache and written back out by
    ``--add-paw``.

    Parameters
    ----------
    block : sequence of str
        Lines from :func:`~poraque.fields.vasp.volumetric.read_augmentation`.

    Returns
    -------
    list of numpy.ndarray
        One array per atom, in file order. A header whose length cannot be read
        (the ``2I4`` field overflows past 9999 atoms) falls back to reading up
        to the next header.
    """
    records, current, declared = [], None, None
    for line in block:
        if _HEADER in line:
            current = []
            records.append(current)
            declared = _declared_length(line)
        elif current is not None:
            values = [float(token) for token in line.split()]
            if declared is not None:
                values = values[:max(0, declared - len(current))]
            current.extend(values)
    return [np.asarray(values, dtype=float) for values in records]


def _declared_length(header):
    """
    The value count a record header states, or ``None``.

    VASP writes ``("augmentation occupancies",2I4)``: the atom index and the
    count, four columns each. Two tokens are the normal case; one token of
    eight digits is the two fields run together once the index reaches 1000.
    """
    tail = header.split(_HEADER, 1)[1].strip()
    tokens = tail.split()
    if len(tokens) == 2 and tokens[1].isdigit():
        return int(tokens[1])
    if len(tokens) == 1 and re.fullmatch(r"\d{8}", tokens[0]):
        return int(tokens[0][4:])
    return None


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


def record_layout(channels):
    r"""
    What each value of one atom's augmentation record is: ``[(a, b, L, M)]``.

    VASP's ``TRANS_RHOLM`` writes the one-centre occupancies of every projector
    channel pair ``a <= b`` in the POTCAR's channel order, for every ``L`` from
    ``|l_a - l_b|`` to ``l_a + l_b`` **in steps of two** (the Gaunt parity rule:
    no other L contributes to a density), and every ``M`` of that ``L`` as a
    real spherical harmonic, ``m = -L..L``. Both halves of that were pinned on
    this project's platinum data rather than assumed: with the order
    ``(2, 2, 0, 0, 1, 1)`` a bulk site's non-zero values sit exactly at L = 0
    and L = 4, as cubic symmetry requires, and its L = 4 block is
    :math:`Y_{40} + \sqrt{5/7}\,Y_{44}` to seven digits.

    Parameters
    ----------
    channels : sequence of int
        The l of each projector channel, as
        :attr:`~poraque.fields.vasp.potcar.PotcarSingle.projector_channels`.

    Returns
    -------
    list of tuple
        One ``(a, b, L, M)`` per value, in file order; its length is the
        record length (138 for ``PAW_PBE Pt``).
    """
    channels = [int(degree) for degree in channels]
    rows = []
    for a in range(len(channels)):
        for b in range(a, len(channels)):
            low, high = abs(channels[a] - channels[b]), channels[a] + channels[b]
            for L in range(low, high + 1, 2):
                for M in range(2 * L + 1):
                    rows.append((a, b, L, M))
    return rows


def irreps_blocks(channels):
    """
    The record grouped by ``L``: ``{L: (pairs, 2L+1) int array}`` of positions.

    A permutation of the record, and exactly invertible: every value belongs to
    one ``(pair, M)`` slot of one ``L``. Within a block the pairs are in record
    order and ``M`` runs ``-L..L``, so a map acting on the pair index alone and
    shared across ``M`` is rotation-equivariant.
    """
    import numpy as np

    rows = record_layout(channels)
    position = {row: index for index, row in enumerate(rows)}
    blocks = {}
    for L in sorted({row[2] for row in rows}):
        pairs = sorted({(a, b) for a, b, degree, _ in rows if degree == L})
        blocks[L] = np.array([[position[(a, b, L, M)]
                               for M in range(2 * L + 1)]
                              for a, b in pairs], dtype=int)
    return blocks


def occupancy_arrays(blocks, channels=None):
    r"""
    Per-atom occupancy records as one padded array: ``(atoms, sets, values)``.

    The shape a regression target needs, where the file has a flat list of
    records per set. Three things about the records decide it:

    * **one set per density channel.** A spin-polarised ``CHGCAR`` carries a
      second set after the magnetisation block, and a target that kept only
      the first would teach a model to write files VASP reads back wrongly
      (see :func:`~poraque.fields.vasp.volumetric.read_augmentation_blocks`);
    * **the length is a property of the element**, not of the file: it counts
      :math:`\rho(ll'LM)` over that species' projector channels --- 138 for
      ``PAW_PBE Pt``, whose six channels (s, s, p, p, d, d) give 138 by
      counting --- so a structure with two elements has two lengths and the
      shorter records are padded with zeros, which ``lengths`` tells apart
      from a zero occupancy;
    * **the sets agree atom by atom.** The magnetisation record of atom *i* is
      the same :math:`(ll'LM)` list as its total record, so a length that
      differs between sets means the file is not what it claims.

    Parameters
    ----------
    blocks : sequence of sequence of str
        Record lines per set, as
        :func:`~poraque.fields.vasp.volumetric.read_augmentation_blocks`
        returns them.
    channels : int, optional
        Sets the caller wants. Extra sets are dropped --- a density read as one
        channel from a spin-polarised file keeps the total's records --- and
        missing ones are **zeros**: an unpolarised member of a spin-polarised
        dataset carries :math:`m \equiv 0` on the grid, and its magnetisation
        occupancies vanish identically for the same reason. Defaults to the
        number of sets present.

    Returns
    -------
    values : numpy.ndarray
        ``(atoms, sets, max_length)``, float64, zero-padded.
    lengths : numpy.ndarray
        ``(atoms,)`` int, each atom's record length.

    Raises
    ------
    ValueError
        When there are no records, when the sets disagree in their atom count
        or in any atom's record length, or when ``channels`` is below one.
    """
    sets = [parse_augmentation(block) for block in blocks]
    sets = [records for records in sets if records]
    if not sets:
        raise ValueError("No augmentation occupancies to arrange.")

    atoms = len(sets[0])
    lengths = np.asarray([record.size for record in sets[0]], dtype=int)
    for number, records in enumerate(sets[1:], start=2):
        if len(records) != atoms:
            raise ValueError(
                f"Record set {number} has {len(records)} atoms where set 1 "
                f"has {atoms}; the sets of one file describe the same atoms.")
        mismatched = [index for index, record in enumerate(records)
                      if record.size != lengths[index]]
        if mismatched:
            raise ValueError(
                f"Record set {number} gives atom {mismatched[0] + 1} "
                f"{records[mismatched[0]].size} values where set 1 gives "
                f"{lengths[mismatched[0]]}; an atom's (ll'LM) list does not "
                f"change between spin channels.")

    channels = len(sets) if channels is None else int(channels)
    if channels < 1:
        raise ValueError(f"channels must be at least 1, got {channels}.")

    values = np.zeros((atoms, channels, int(lengths.max())), dtype=float)
    for set_index, records in enumerate(sets[:channels]):
        for atom, record in enumerate(records):
            values[atom, set_index, :record.size] = record
    return values, lengths


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
                "schema": RECORD_SCHEMA,
            }
            emit(f"      PAW reference: {element}  "
                 f"{len(reference[element]['values'])} values, averaged over "
                 f"{entry['count']} atoms in {structures.get(element, 0)} "
                 f"structure(s)")
    return reference


def reference_from_profiles(paw_profiles):
    """
    The per-element augmentation table inside a checkpoint's ``paw_profiles``.

    A checkpoint keys its PAW data by atomic number and keeps the augmentation
    occupancies under each element's ``"augmentation"``;
    :func:`records_for_structure` wants them keyed by symbol. This is the
    one translation between the two.

    Parameters
    ----------
    paw_profiles : dict
        ``{atomic_number: {"element": ..., "augmentation": {...}, ...}}``.

    Returns
    -------
    dict
        ``{element: augmentation entry}``, only for elements that carry one.
    """
    return {entry["element"]: entry["augmentation"]
            for entry in (paw_profiles or {}).values()
            if isinstance(entry, dict) and entry.get("augmentation")}


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
