# -*- coding: utf-8 -*-
# file: hdf5.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The same volumetric fields, stored as HDF5 instead of as text.

A ``CHGCAR`` is a **text** file: eleven significant digits per value, five
values to a line, roughly 18 bytes for what is 8 bytes of double. A 140³
density is 2.7 million values and about 49 MB of ASCII, and every read of it
parses 2.7 million strings. Gzip hides most of the size and none of the parse.

This module stores the identical content as an HDF5 dataset — binary, chunked,
optionally compressed by the library on the way in and out. Nothing about the
*physics* changes, and deliberately nothing about the *convention* changes
either:

**Values are written in the file convention, not the physical one.** A density
goes in as :math:`\rho\Omega`, exactly what a ``CHGCAR`` holds, because that is
what :meth:`~poraque.fields.base.ScalarField.to_file_values` produces and what
:meth:`~poraque.fields.base.ScalarField.from_file_values` expects back. The
consequence is the point of the whole design: :func:`read_volumetric` returns
``(structure, data, extra)`` whether it read text or HDF5, so **every** reader
above it — ``ScalarField.read``, ``FieldGrid.from_file``, ``SpinDensity.read``,
the dataset, the calculator — works unchanged and there is exactly one loader
in the codebase rather than two that must agree.

Layout
------
One file per material, one dataset per field::

    mp-124/fields.h5
        /                        attrs: format, version, created, comment,
                                        cell (3,3), symbols, counts,
                                        scaled_positions, selective_dynamics
        /CHGCAR                  (Nx, Ny, Nz) float64  attrs: units,
                                                              volume_scaled
        /CHGCAR_extra0           the magnetisation block of a spin-polarised
                                 density
        /CHGCAR_augmentation0    the PAW records that follow the block above it
        /CHGCAR_augmentation1    ... and the ones that follow ``_extra0``
        /EXTCAR                  (Nx, Ny, Nz) float64
        /TAUCAR                  (Nx, Ny, Nz) float64

The ``_extraN`` and ``_augmentationN`` datasets belong to the field they are
named after rather than standing beside it, so :func:`field_names` does not
offer them as fields and nothing downstream can mistake one for an input.

The PAW records
---------------
A ``CHGCAR`` is more than a grid: after each density block come the **PAW
augmentation occupancies**, the one-centre terms living inside the core radius
that no plane-wave grid represents. ``ICHARG = 1`` will not restart without
them.

They are stored here as the **text** they are, one variable-length string per
line, under ``<field>_augmentationN`` --- one block per grid block, so a
spin-polarised density has two. Text rather than parsed numbers for the same
reason :func:`~poraque.fields.vasp.volumetric.write_volumetric` appends them
verbatim: they are a fixed-format Fortran read on VASP's side, and copying the
characters is the only transformation that cannot produce a file VASP declines
to parse.

What a store still cannot do is be opened by VASP. That is what
:func:`~poraque.fields.vasp.templates.write_chgcar` is for, and it is why
keeping the records matters: a store is a complete container for a density, so
writing one back out gives VASP everything it had before.

Addressing
----------
A single field inside such a file is named ``<path>::<dataset>``::

    cache/mp-124/fields.h5::CHGCAR

:func:`split_target` parses that spelling and every reader accepts it, which is
what lets :class:`~poraque.ml.data.MaterialRecord` keep holding one path per
field with no idea that three of them now point into the same file.

Compression
-----------
``none``, ``gzip`` (levels 0-9) or ``lzf``, each with HDF5's byte-shuffle
filter on by default. Both codecs ship with h5py, so a file written here opens
anywhere h5py does — no plugin to install on the machine that reads it, which
is the reason neither ``blosc`` nor ``zstd`` is offered however much faster
they are.

Chunks are chosen by :func:`chunk_shape`: near-cubic blocks of about a
megabyte. Near-cubic rather than whole z-slabs because a compressed chunk must
be decompressed *whole*, so the chunk shape decides what a partial read costs —
and the partial reads this project makes are sub-boxes and downsampled strides,
not single planes. A whole-slab chunk would make a stride along x read the
entire field.

.. note::
   ``h5py`` is imported inside the functions that need it, never at module
   scope. Importing :mod:`poraque.fields` must not require it: a user with no
   HDF5 data should not need the library, and the error when one *is* needed
   should name what to install rather than fail at import time.
"""

import os
import re
from datetime import datetime, timezone

import numpy as np

#: Suffixes recognised as HDF5, with or without a ``::dataset`` selector.
HDF5_SUFFIXES = (".h5", ".hdf5", ".he5")

#: Separator between the file and the dataset inside it.
TARGET_SEPARATOR = "::"

#: Default filename for a material's field store.
DEFAULT_FILENAME = "fields.h5"

#: What this module writes into the root ``format`` attribute.
FORMAT_TAG = "poraque-fields"

#: Bumped when the layout changes in a way a reader must know about.
FORMAT_VERSION = 1

#: Compression codecs offered. ``None`` and ``"none"`` both mean uncompressed.
CODECS = ("none", "gzip", "lzf")

#: Target size of one chunk, in bytes, before compression.
CHUNK_TARGET_BYTES = 1 << 20

#: Names of the additional grid blocks a field may carry (a spin channel).
_EXTRA_PATTERN = re.compile(r".+_extra\d+$")

#: Names of the PAW augmentation blocks a field may carry, one per grid block.
_AUGMENTATION_PATTERN = re.compile(r".+_augmentation\d+$")

#: Datasets that belong to a field without being one.
_ANCILLARY_PATTERNS = (_EXTRA_PATTERN, _AUGMENTATION_PATTERN)


def _h5py():
    """:mod:`h5py`, imported on use with an actionable error when absent."""
    try:
        import h5py
    except ImportError as error:                                # pragma: no cover
        raise ImportError(
            "Reading or writing Poraquê fields as HDF5 needs h5py, which is "
            "not installed in this environment. `pip install h5py` (or "
            "`conda install h5py`) and try again; nothing else in Poraquê "
            "requires it."
        ) from error
    return h5py


# ---------------------------------------------------------------------- #
# Paths
# ---------------------------------------------------------------------- #
def split_target(path):
    """
    Split ``file.h5::DATASET`` into its two halves.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    tuple
        ``(file, dataset)``; ``dataset`` is ``None`` when the path names no
        one field.

    Examples
    --------
    >>> split_target("cache/mp-1/fields.h5::CHGCAR")
    ('cache/mp-1/fields.h5', 'CHGCAR')
    >>> split_target("cache/mp-1/fields.h5")
    ('cache/mp-1/fields.h5', None)
    """
    text = str(path)
    if TARGET_SEPARATOR not in text:
        return text, None
    filename, _, dataset = text.rpartition(TARGET_SEPARATOR)
    return filename, (dataset or None)


def join_target(path, dataset):
    """The ``file.h5::DATASET`` spelling of one field."""
    return f"{path}{TARGET_SEPARATOR}{dataset}"


def is_hdf5_path(path):
    """
    Whether ``path`` names an HDF5 store, by suffix.

    By suffix and not by content: this is asked before the file exists (a
    writer choosing a format) as often as after, and a signature check cannot
    answer the first case at all.
    """
    filename, _ = split_target(path)
    return str(filename).lower().endswith(HDF5_SUFFIXES)


# ---------------------------------------------------------------------- #
# Compression
# ---------------------------------------------------------------------- #
def compression_options(compression=None, level=4, shuffle=True):
    """
    Normalise a compression request into h5py dataset keywords.

    Parameters
    ----------
    compression : str or bool or dict or None, optional
        ``"none"``/``None``/``False`` for uncompressed, ``"gzip"``, ``"lzf"``,
        or a mapping already in this shape. ``True`` means ``"gzip"``.
    level : int, optional
        Gzip level 0-9. Ignored by ``lzf``, which has none — a level passed
        with ``lzf`` is dropped rather than raising, because a config that sets
        one level for a whole run should not become an error the moment the
        codec changes.
    shuffle : bool, optional
        HDF5's byte-shuffle filter, on by default whenever a codec is active.
        It regroups the bytes of a chunk by significance before compressing,
        which matters here because the thing being compressed is an array of
        doubles: the exponent bytes of neighbouring grid points are nearly
        identical and the low mantissa bytes are nearly random, and shuffling
        puts each kind together instead of interleaving them.

        Measured on the shipped 140³ platinum density (21.95 MB of float64):

        ==========  ==========  ==========
        codec       plain       shuffled
        ==========  ==========  ==========
        lzf         1.06x       **1.21x**
        gzip-1      1.19x       **1.36x**
        gzip-4      1.19x       **1.38x**
        gzip-9      1.19x       **1.39x**
        ==========  ==========  ==========

        It is also *faster* both ways — gzip-4 wrote in 0.30 s shuffled against
        0.42 s plain, and read in 0.04 s against 0.07 s — because the codec is
        handed less entropy to chew through. There is no case in that table for
        turning it off, which is why the flag exists only for reproducing a
        file written without it.

    Returns
    -------
    dict
        Keywords for ``create_dataset``; empty for no compression.

    Raises
    ------
    ValueError
        On an unknown codec or an out-of-range gzip level, naming what is
        accepted.
    """
    if isinstance(compression, dict):
        level = compression.get("level", level)
        shuffle = compression.get("shuffle", shuffle)
        compression = compression.get("codec", compression.get("compression"))

    if compression is True:
        compression = "gzip"
    if compression is None or compression is False:
        compression = "none"

    codec = str(compression).strip().lower()
    if codec in ("", "none", "off", "raw"):
        return {}
    if codec not in CODECS:
        raise ValueError(
            f"Unknown compression {compression!r}. Available: "
            f"{', '.join(CODECS)}. Both gzip and lzf ship with h5py, so a "
            f"file written with either opens wherever h5py does.")

    if codec == "lzf":
        return {"compression": "lzf", "shuffle": bool(shuffle)}

    level = int(level)
    if not 0 <= level <= 9:
        raise ValueError(
            f"gzip level must be between 0 and 9, got {level}. 0 is the "
            f"filter with no compression, which is slower than "
            f"compression='none' and no smaller.")
    return {"compression": "gzip", "compression_opts": level,
            "shuffle": bool(shuffle)}


def chunk_shape(shape, itemsize=8, target_bytes=CHUNK_TARGET_BYTES):
    """
    A near-cubic chunk of roughly ``target_bytes``.

    Halve the longest axis until the block fits, which lands on a shape whose
    sides are within a factor of two of each other whatever the grid's aspect
    ratio. A slab-shaped chunk would be cheaper to write and would make a
    strided read along the wrong axis touch the whole field; a compressed chunk
    has to be decompressed in full, so the chunk shape *is* the partial-read
    cost.

    Parameters
    ----------
    shape : sequence of int
    itemsize : int, optional
        Bytes per value.
    target_bytes : int, optional

    Returns
    -------
    tuple of int
        Never larger than ``shape`` on any axis, never smaller than 1.

    Examples
    --------
    >>> chunk_shape((140, 140, 140))
    (35, 35, 70)
    >>> chunk_shape((16, 16, 16))
    (16, 16, 16)
    """
    chunks = [max(1, int(n)) for n in shape]
    while np.prod(chunks) * itemsize > target_bytes:
        axis = int(np.argmax(chunks))
        if chunks[axis] <= 1:
            break
        chunks[axis] = max(1, chunks[axis] // 2)
    return tuple(chunks)


# ---------------------------------------------------------------------- #
# Writing
# ---------------------------------------------------------------------- #
def _structure_attributes(structure):
    """The header, flattened onto the root group."""
    attributes = {
        "cell": np.asarray(structure.cell, dtype=float),
        "symbols": np.array([str(s) for s in structure.symbols],
                            dtype=object),
        "counts": np.asarray([int(c) for c in structure.counts], dtype=np.int64),
        "scaled_positions": np.asarray(structure.scaled_positions, dtype=float),
    }
    comment = getattr(structure, "comment", None)
    if comment:
        attributes["comment"] = str(comment)
    selective = getattr(structure, "selective_dynamics", None)
    if selective is not None:
        attributes["selective_dynamics"] = np.asarray(selective, dtype=bool)
    return attributes


def _structure_from_attributes(attributes, path):
    """Rebuild the :class:`~poraque.fields.vasp.poscar.Poscar` header."""
    from .vasp.poscar import Poscar

    missing = [key for key in ("cell", "symbols", "counts", "scaled_positions")
               if key not in attributes]
    if missing:
        raise ValueError(
            f"{path}: this HDF5 file carries no structure header (missing "
            f"{', '.join(missing)}). A Poraquê field store keeps the geometry "
            f"on its root group, exactly as a CHGCAR keeps it in its first "
            f"lines; a file without one cannot say what cell its grid is in.")

    symbols = [s.decode() if isinstance(s, bytes) else str(s)
               for s in attributes["symbols"]]
    selective = attributes.get("selective_dynamics")
    return Poscar(
        np.asarray(attributes["cell"], dtype=float),
        symbols,
        [int(c) for c in attributes["counts"]],
        np.asarray(attributes["scaled_positions"], dtype=float),
        comment=str(attributes.get("comment", "")) or "poraque",
        selective_dynamics=(np.asarray(selective, dtype=bool)
                            if selective is not None else None),
    )


def write_field(path, name, field, compression=None, level=4, chunks=True,
                extra=(), attributes=None, values=None, shuffle=True,
                augmentation=None):
    """
    Write one field into a store, creating or updating the file.

    Repeated calls with different ``name`` build up a material's three fields
    in one file. The structure header is written the first time and **checked**
    on every subsequent call: a store whose ``CHGCAR`` and ``TAUCAR`` came from
    different cells is precisely the silent corruption the shared-grid rule
    exists to prevent, and it would be invisible on disk.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.h5`` file, with or without a ``::dataset`` selector — an
        explicit ``name`` wins over one in the path.
    name : str
        Dataset name, e.g. ``"CHGCAR"``.
    field : ScalarField
        Written via :meth:`~poraque.fields.base.ScalarField.to_file_values`, so
        the stored numbers are the ones a ``CHGCAR`` would hold.
    compression : str or None, optional
        ``"none"``, ``"gzip"`` or ``"lzf"``; see :func:`compression_options`.
    level : int, optional
        Gzip level.
    chunks : bool or tuple, optional
        ``True`` for :func:`chunk_shape`, an explicit tuple, or ``False`` for a
        contiguous dataset (which HDF5 requires when there is no filter).
    extra : sequence of numpy.ndarray, optional
        Additional blocks on the same grid — the magnetisation channel of a
        spin-polarised density — stored as ``<name>_extra0``, ``_extra1``, ...
    attributes : dict, optional
        Extra dataset attributes, recorded verbatim.
    values : numpy.ndarray, optional
        Written instead of ``field.to_file_values()``. Used by
        :func:`write_fields` to store the two channels of a spin density as a
        main block and an extra one, which is how a spin-polarised ``CHGCAR``
        carries them too.
    augmentation : sequence, optional
        PAW augmentation records, as the verbatim text lines
        :func:`~poraque.fields.vasp.volumetric.read_augmentation` returns.
        Either one block --- a flat sequence of ``str`` --- or one block per
        grid block, which is what a spin-polarised ``CHGCAR`` carries: a set
        of records after the total and another after the magnetisation. They
        are stored as ``<name>_augmentation0``, ``_augmentation1``, ...,
        indexed to match the grid block each one follows.

    Returns
    -------
    str
        ``path::name``, the address of what was written.

    Notes
    -----
    The records go in as **text**, not as parsed numbers, for the same reason
    :func:`~poraque.fields.vasp.volumetric.write_volumetric` appends them
    verbatim: they are a fixed-format Fortran read on VASP's side, and copying
    the characters is the only transformation that cannot introduce a file
    VASP declines to parse. A store is a container for them, not a second
    opinion about their layout.
    """
    h5py = _h5py()
    filename, in_path = split_target(path)
    name = name or in_path
    if not name:
        raise ValueError(
            f"{path!r} names a file but no dataset. Pass name='CHGCAR', or "
            f"spell the target as '{filename}{TARGET_SEPARATOR}CHGCAR'.")

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if values is None:
        values = field.to_file_values()
    values = np.ascontiguousarray(values, dtype=np.float64)
    options = compression_options(compression, level, shuffle)
    if chunks is True:
        chunks = chunk_shape(values.shape, values.dtype.itemsize)
    if not chunks and options:
        # HDF5 cannot filter a contiguous dataset. Silently writing it
        # uncompressed would make --compression a no-op that reports success.
        chunks = chunk_shape(values.shape, values.dtype.itemsize)

    blocks = augmentation_blocks(augmentation)

    with h5py.File(filename, "a") as handle:
        _write_header(handle, field.structure, filename)
        # Every ancillary dataset of this field goes, not just the ones about
        # to be rewritten: a store updated from two blocks to one would
        # otherwise keep the orphan and hand it back on the next read.
        for dataset_name in [name, *(key for key in handle
                                     if key.startswith(f"{name}_"))]:
            if dataset_name in handle:
                del handle[dataset_name]

        dataset = handle.create_dataset(
            name, data=values, chunks=chunks or None, **options)
        dataset.attrs["units"] = str(getattr(field, "unit", ""))
        dataset.attrs["volume_scaled"] = bool(
            getattr(type(field), "volume_scaled", False))
        dataset.attrs["field"] = str(getattr(field, "name", name))
        for key, value in (attributes or {}).items():
            dataset.attrs[key] = value

        for index, block in enumerate(extra):
            block = np.ascontiguousarray(block, dtype=np.float64)
            handle.create_dataset(f"{name}_extra{index}", data=block,
                                  chunks=chunks or None, **options)

        for index, lines in enumerate(blocks):
            _write_augmentation(handle, f"{name}_augmentation{index}", lines,
                                compression, level)

    return join_target(filename, name)


def augmentation_blocks(augmentation):
    """
    Normalise augmentation records to a list of blocks.

    A caller may hand over one block --- a flat sequence of lines, which is
    what a spin-unpolarised ``CHGCAR`` has and what
    :func:`~poraque.fields.vasp.volumetric.read_augmentation` returns --- or
    a sequence of such blocks, one per grid block. Both spellings are
    accepted, because requiring the nested one for the overwhelmingly common
    single-block case would be a wrapper every caller had to remember.

    Parameters
    ----------
    augmentation : sequence or None

    Returns
    -------
    list of list of str
        Empty when there is nothing to store. Empty blocks are dropped: a
        block of no records is not a record of no occupancies, it is the
        absence of one, and storing it would make ``ICHARG=1`` read zeros.
    """
    if not augmentation:
        return []
    first = next(iter(augmentation))
    if isinstance(first, str):
        augmentation = [augmentation]
    blocks = [[str(line) for line in block] for block in augmentation]
    return [block for block in blocks if block]


def _write_augmentation(handle, dataset_name, lines, compression, level):
    """One block of augmentation records, as variable-length UTF-8 strings."""
    h5py = _h5py()
    if not lines:
        return
    options = compression_options(compression, level, shuffle=False)
    # The shuffle filter reorders the bytes of fixed-width items and a
    # variable-length string is not one, so it is dropped here rather than
    # passed to h5py, which would refuse the combination.
    data = np.array([str(line) for line in lines],
                    dtype=h5py.string_dtype("utf-8"))
    handle.create_dataset(dataset_name, data=data,
                          chunks=(len(data),) if options else None, **options)


def _write_header(handle, structure, path):
    """Write the root attributes, or verify they already agree."""
    if "format" not in handle.attrs:
        handle.attrs["format"] = FORMAT_TAG
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["created"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        try:
            from ..version import __version__

            handle.attrs["poraque_version"] = __version__
        except ImportError:                                     # pragma: no cover
            pass
        for key, value in _structure_attributes(structure).items():
            handle.attrs[key] = value
        return

    existing = np.asarray(handle.attrs["cell"], dtype=float)
    if not np.allclose(existing, np.asarray(structure.cell, dtype=float),
                       atol=1e-5):
        raise ValueError(
            f"{path}: this store already holds fields in a different cell "
            f"(max difference "
            f"{np.abs(existing - np.asarray(structure.cell)).max():.2e} Å). "
            f"All fields of one material must share one mesh in one cell; "
            f"write the new material to its own file.")


def write_fields(path, fields, compression=None, level=4, chunks=True,
                 shuffle=True, augmentation=None):
    """
    Write a whole material in one call.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.h5`` file.
    fields : dict
        ``{name: ScalarField}``. A :class:`~poraque.fields.spin.SpinDensity`
        contributes its magnetisation channel as an ``_extra0`` block, exactly
        as a spin-polarised ``CHGCAR`` carries it as a second grid block.
    compression, level, chunks, shuffle
        See :func:`write_field`.
    augmentation : dict, optional
        ``{name: blocks}``, the PAW records belonging to each field. Only a
        density has any --- there is no augmentation of a kinetic energy
        density or of an external potential --- so this is keyed by field
        rather than given once for the store.

    Returns
    -------
    dict
        ``{name: address}``.
    """
    augmentation = augmentation or {}
    written = {}
    for name, field in fields.items():
        values, extra, attributes = _file_blocks(field)
        written[name] = write_field(path, name, field, values=values,
                                    compression=compression, level=level,
                                    chunks=chunks, extra=extra,
                                    attributes=attributes, shuffle=shuffle,
                                    augmentation=augmentation.get(name))
    return written


def _file_blocks(field):
    """
    ``(main block, extra blocks, attribute overrides)`` in the file convention.

    A :class:`~poraque.fields.spin.SpinDensity` is the one field with two
    blocks, and it has no ``to_file_values`` of its own — its writer scales the
    two channels by the volume inline. Doing the same here keeps the stored
    numbers identical to what its text writer produces, which is what lets
    ``SpinDensity.read`` come back through the shared reader unchanged.
    """
    magnetization = getattr(field, "magnetization", None)
    if magnetization is not None:
        volume = field.grid.volume
        return (np.asarray(field.total) * volume,
                (np.asarray(magnetization) * volume,),
                {"volume_scaled": True, "ispin": 2})
    return np.asarray(field.to_file_values()), (), {}


# ---------------------------------------------------------------------- #
# Reading
# ---------------------------------------------------------------------- #
def read_volumetric(path, read_all=False):
    """
    Read one field out of a store, in :func:`read_volumetric`'s own contract.

    This is the function :func:`poraque.fields.vasp.volumetric.read_volumetric`
    hands off to when it is given an HDF5 path, which is why its return shape
    is the text reader's and not a more natural one.

    Parameters
    ----------
    path : str or pathlib.Path
        ``file.h5::DATASET``. Without a selector the store's single field is
        used, and a store holding several is an error naming them — guessing
        which of ``CHGCAR`` and ``TAUCAR`` was meant is exactly the kind of
        helpfulness that trains a model on the wrong target.
    read_all : bool, optional
        Also return the ``_extraN`` blocks.

    Returns
    -------
    structure : Poscar
    data : numpy.ndarray
    extra : list of numpy.ndarray
    """
    h5py = _h5py()
    filename, dataset_name = split_target(path)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    with h5py.File(filename, "r") as handle:
        dataset_name = _resolve_dataset(handle, dataset_name, filename)
        structure = _structure_from_attributes(dict(handle.attrs), filename)
        data = np.asarray(handle[dataset_name][...], dtype=np.float64)

        extra = []
        if read_all:
            index = 0
            while f"{dataset_name}_extra{index}" in handle:
                extra.append(np.asarray(
                    handle[f"{dataset_name}_extra{index}"][...],
                    dtype=np.float64))
                index += 1

    return structure, data, extra


def read_augmentation(path):
    """
    The PAW augmentation records stored beside a field.

    The counterpart of
    :func:`poraque.fields.vasp.volumetric.read_augmentation`, and the function
    that one hands off to when it is given an HDF5 path.

    Parameters
    ----------
    path : str or pathlib.Path
        ``file.h5::DATASET``, or a store holding one field.

    Returns
    -------
    shape : tuple of int
        The grid the field is stored on, so the return value matches the text
        reader's contract exactly.
    blocks : list of list of str
        One block of verbatim record lines per grid block --- one for a
        spin-unpolarised density, two for a spin-polarised one. Empty when the
        store carries none, which is what a ``TAUCAR``, an ``EXTCAR`` and any
        density written before this became storable all have.
    """
    h5py = _h5py()
    filename, dataset_name = split_target(path)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    with h5py.File(filename, "r") as handle:
        dataset_name = _resolve_dataset(handle, dataset_name, filename)
        shape = tuple(int(n) for n in handle[dataset_name].shape)

        blocks = []
        index = 0
        while f"{dataset_name}_augmentation{index}" in handle:
            raw = handle[f"{dataset_name}_augmentation{index}"][...]
            blocks.append([_as_text(line) for line in raw])
            index += 1

    return shape, blocks


def _as_text(value):
    """One stored line as ``str``, whether h5py handed back bytes or not."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def has_augmentation(path):
    """Whether the addressed field carries PAW augmentation records."""
    h5py = _h5py()
    filename, dataset_name = split_target(path)
    if not os.path.exists(filename):
        return False
    with h5py.File(filename, "r") as handle:
        try:
            dataset_name = _resolve_dataset(handle, dataset_name, filename)
        except (KeyError, ValueError):
            return False
        return f"{dataset_name}_augmentation0" in handle


def _resolve_dataset(handle, dataset_name, path):
    """The dataset to read, checked against what the file holds."""
    if dataset_name is not None:
        if dataset_name not in handle:
            raise KeyError(
                f"{path} holds no dataset {dataset_name!r}. It has: "
                f"{', '.join(field_names(handle)) or '(none)'}.")
        return dataset_name

    names = field_names(handle)
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(f"{path}: this HDF5 file holds no field datasets.")
    raise ValueError(
        f"{path} holds {len(names)} fields ({', '.join(names)}) and the path "
        f"names none of them. Address one as "
        f"'{path}{TARGET_SEPARATOR}{names[0]}'.")


def field_names(handle_or_path):
    """
    The field datasets in a store, excluding the ancillary blocks.

    A field's ``_extraN`` grid blocks and ``_augmentationN`` PAW records
    belong to it rather than standing beside it, so neither is a field in its
    own right and neither may be offered as one to a reader asking what the
    store holds.

    Parameters
    ----------
    handle_or_path : h5py.File or str

    Returns
    -------
    list of str
        Sorted.
    """
    def _names(handle):
        return sorted(key for key in handle
                      if not any(p.match(key) for p in _ANCILLARY_PATTERNS))

    if hasattr(handle_or_path, "keys"):
        return _names(handle_or_path)

    h5py = _h5py()
    filename, _ = split_target(handle_or_path)
    with h5py.File(filename, "r") as handle:
        return _names(handle)


def peek_shape(path):
    """
    The grid shape of one field, reading no values.

    HDF5 keeps the shape in the object header, so this costs one seek whatever
    the grid — the same property that makes the text reader's own header peek
    cheap, and the reason shape bucketing never has to decode a dataset.

    Parameters
    ----------
    path : str
        ``file.h5::DATASET``.

    Returns
    -------
    tuple of int
    """
    h5py = _h5py()
    filename, dataset_name = split_target(path)
    with h5py.File(filename, "r") as handle:
        dataset_name = _resolve_dataset(handle, dataset_name, filename)
        return tuple(int(n) for n in handle[dataset_name].shape)


def is_spin_polarized(path):
    """Whether the addressed field carries a magnetisation block."""
    h5py = _h5py()
    filename, dataset_name = split_target(path)
    if not os.path.exists(filename):
        return False
    with h5py.File(filename, "r") as handle:
        try:
            dataset_name = _resolve_dataset(handle, dataset_name, filename)
        except (KeyError, ValueError):
            return False
        return f"{dataset_name}_extra0" in handle


def describe(path):
    """
    What a store holds, for a log line or a report.

    Returns
    -------
    dict
        ``fields`` (per-dataset shape, dtype, codec, chunks, stored bytes and
        compression ratio), ``file_bytes`` and the root attributes worth
        showing.
    """
    h5py = _h5py()
    filename, _ = split_target(path)
    report = {"path": filename, "file_bytes": os.path.getsize(filename),
              "fields": {}}
    with h5py.File(filename, "r") as handle:
        report["format"] = str(handle.attrs.get("format", "?"))
        report["created"] = str(handle.attrs.get("created", ""))
        report["poraque_version"] = str(handle.attrs.get("poraque_version", ""))
        for name in handle:
            dataset = handle[name]
            raw = int(np.prod(dataset.shape)) * dataset.dtype.itemsize
            stored = int(dataset.id.get_storage_size())
            report["fields"][name] = {
                "shape": tuple(int(n) for n in dataset.shape),
                "dtype": str(dataset.dtype),
                "compression": dataset.compression or "none",
                "compression_opts": dataset.compression_opts,
                "shuffle": bool(dataset.shuffle),
                "chunks": (tuple(int(n) for n in dataset.chunks)
                           if dataset.chunks else None),
                "raw_bytes": raw,
                "stored_bytes": stored,
                "ratio": (raw / stored) if stored else float("nan"),
            }
    return report
