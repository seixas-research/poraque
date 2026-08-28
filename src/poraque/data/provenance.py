# -*- coding: utf-8 -*-
# file: provenance.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Who computed a field, and with what.

Two readings, both cheap and both about a calculation *directory* rather than
about the numbers inside it: the DFT code's version, and a digest of the input
file that asked for the run. Neither says a field is right; together they say
which run it came from, which is what a stored reference needs to remain
attributable once it has been reduced to a form factor and copied into a
checkpoint.

Used by :func:`poraque.fields.atomic.augmentation_reference`, where an isolated
atom's PAW record is a *per-element, transferable* quantity that outlives the
directory it was read from — so the record has to carry its own origin.
"""

import hashlib
import os
import re

#: How VASP writes its own version on the first line of an ``OUTCAR``.
_VERSION_PATTERN = re.compile(r"vasp\.(\d+(?:\.\d+)*)", re.IGNORECASE)


def file_hash(path):
    """
    SHA-256 of a file, or ``None`` when it is not there.

    Parameters
    ----------
    path : str

    Returns
    -------
    str or None
        Hex digest. A missing file is ``None`` rather than an error: a
        provenance reading is a best effort by construction, and a record that
        cannot be taken is not a failure of the thing being recorded.
    """
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version(directory):
    """
    The DFT code's version string, read from the run's own output.

    Parameters
    ----------
    directory : str
        A calculation directory.

    Returns
    -------
    str or None
        E.g. ``"6.6.1"``. ``None`` when no output file records it — the normal
        case for an archived run stripped of everything but its densities.

    Notes
    -----
    Only the first line of ``OUTCAR`` is read; the version is the first token
    on it. ``vasprun.xml`` is consulted as a fallback because a run archived
    for size often keeps the XML and drops the 300 MB ``OUTCAR``.
    """
    outcar = os.path.join(directory, "OUTCAR")
    if os.path.exists(outcar):
        try:
            with open(outcar, "r", errors="replace") as handle:
                match = _VERSION_PATTERN.search(handle.readline())
        except OSError:
            match = None
        if match:
            return match.group(1)

    xml = os.path.join(directory, "vasprun.xml")
    if os.path.exists(xml):
        try:
            with open(xml, "r", errors="replace") as handle:
                for _ in range(64):
                    line = handle.readline()
                    if not line:
                        break
                    if 'name="version"' in line:
                        text = re.sub(r"<[^>]+>", " ", line).strip()
                        if text:
                            return text.split()[0]
        except OSError:
            pass
    return None
