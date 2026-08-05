# -*- coding: utf-8 -*-
# file: compressed.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Transparent reading of compressed calculation files.

Volumetric files are the largest artefacts a DFT run produces and they are
almost always shipped compressed: the Materials Project serves charge densities
gzipped, archives of published calculations are usually ``.zip`` or ``.tar``
members, and a local run is routinely gzipped to reclaim the roughly threefold
expansion an unzipped ``CHGCAR`` costs.

Every reader in :mod:`poraque.fields` therefore goes through :func:`open_text`
rather than the builtin :func:`open`, so a path ending in ``.gz``, ``.bz2``,
``.xz`` or ``.zip`` is decompressed **as it is read** and never has to exist on
disk in expanded form. The compression is chosen by suffix, which is what makes
this transparent: ``CHGCAR_mp-126.gz`` and ``CHGCAR`` reach the parser as the
same stream of lines.

.. note::

   Decompression is *streamed*, not staged. A 200 MB ``CHGCAR`` is decoded a
   buffer at a time into the array being filled, so peak memory tracks the
   parsed grid rather than the file — which matters, because materialising the
   text of one such file as a list of Python strings costs several times the
   grid it describes.
"""

import bz2
import gzip
import io
import lzma
import os
import zipfile
from contextlib import contextmanager

#: Suffix -> opener taking ``(path, mode)`` and returning a text handle.
#: ``.zip`` is absent because an archive is a container rather than a stream and
#: is handled separately by :func:`open_text`.
_OPENERS = {
    ".gz": gzip.open,
    ".bz2": bz2.open,
    ".xz": lzma.open,
    ".lzma": lzma.open,
}

#: Every suffix :func:`open_text` decompresses, ``.zip`` included.
COMPRESSION_SUFFIXES = tuple(_OPENERS) + (".zip",)


def is_compressed(path):
    """
    Whether ``path`` names a file :func:`open_text` will decompress.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    bool
    """
    return str(path).lower().endswith(COMPRESSION_SUFFIXES)


def strip_compression_suffix(path):
    """
    ``path`` with its compression suffix removed, if it has one.

    Useful for naming: ``CHGCAR_mp-126.gz`` describes a ``CHGCAR``, and a cache
    entry or a log line should say so.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    str

    Examples
    --------
    >>> strip_compression_suffix("CHGCAR_mp-126.gz")
    'CHGCAR_mp-126'
    >>> strip_compression_suffix("CHGCAR")
    'CHGCAR'
    """
    text = str(path)
    root, suffix = os.path.splitext(text)
    return root if suffix.lower() in COMPRESSION_SUFFIXES else text


@contextmanager
def open_text(path, encoding="utf-8", errors="replace"):
    """
    Open ``path`` for text reading, decompressing on the fly when needed.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read. A ``.gz``, ``.bz2``, ``.xz`` or ``.zip`` suffix selects
        the codec; anything else is opened as plain text.
    encoding : str, optional
        Text encoding. Volumetric files are ASCII in practice.
    errors : str, optional
        Decoding error policy. ``"replace"`` rather than ``"strict"`` because
        a stray byte in a comment line must not take down the read of a
        200 MB density that is otherwise perfectly well formed.

    Yields
    ------
    io.TextIOBase
        A text handle. Iterate it — the whole point is that the file is never
        held in memory at once.

    Raises
    ------
    ValueError
        If a ``.zip`` archive is empty, or holds more than one candidate member
        and none of them matches the archive's own name. Guessing which member
        was meant is exactly the kind of silent choice that produces a
        confusing error thousands of lines later.

    Examples
    --------
    >>> with open_text("data/MP/chgcar/CHGCAR_mp-126.gz") as handle:  # doctest: +SKIP
    ...     header = [next(handle) for _ in range(8)]
    """
    text = str(path)
    suffix = os.path.splitext(text)[1].lower()

    if suffix == ".zip":
        with zipfile.ZipFile(text) as archive:
            name = _zip_member(archive, text)
            # ZipFile.open yields bytes; TextIOWrapper adapts it without
            # extracting, so the member is decompressed as it is consumed.
            with archive.open(name) as raw:
                yield io.TextIOWrapper(raw, encoding=encoding, errors=errors)
        return

    opener = _OPENERS.get(suffix, open)
    with opener(text, mode="rt", encoding=encoding, errors=errors) as handle:
        yield handle


def _zip_member(archive, path):
    """
    Pick the single data member of a zip archive.

    Precedence: the only file in the archive; otherwise the member whose name
    matches the archive's own stem (``CHGCAR_mp-126.zip`` -> ``CHGCAR_mp-126``);
    otherwise it is ambiguous and the caller is told so.
    """
    members = [info.filename for info in archive.infolist()
               if not info.is_dir()
               and not info.filename.startswith("__MACOSX/")]
    if not members:
        raise ValueError(f"{path}: the archive is empty.")
    if len(members) == 1:
        return members[0]

    stem = os.path.basename(strip_compression_suffix(path))
    matches = [name for name in members if os.path.basename(name) == stem]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"{path}: the archive holds {len(members)} files ({sorted(members)!r}) "
        f"and none is named {stem!r}, so which one to read is ambiguous. "
        f"Extract the member you want, or repackage one file per archive."
    )
