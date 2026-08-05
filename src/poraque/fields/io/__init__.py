# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Pluggable ingestion layer for plane-wave DFT codes.

One registry maps a code name to a
:class:`~poraque.fields.io.base.CalculationReader`. Everything upstream of a
reader is code-specific; everything downstream — grids, fields, datasets,
models — sees only the neutral types.

Adding a code::

    from poraque.fields.io import CalculationReader, register_reader

    class MyCodeReader(CalculationReader):
        code = "mycode"
        structure_files = ("mycode.in",)
        field_files = {"external": "...", "density": "...", "kinetic": "..."}
        ...

    register_reader(MyCodeReader)

After that, ``ExternalPotential.from_calculation(directory, code="mycode")``
and :class:`~poraque.ml.data.FieldPairDataset` work unchanged.
"""

from .base import (
    FIELD_KINDS,
    CalculationParameters,
    CalculationReader,
    PseudopotentialInfo,
)
from .compressed import (
    COMPRESSION_SUFFIXES,
    is_compressed,
    open_text,
    strip_compression_suffix,
)
from .espresso import EspressoReader
from .gpaw import GpawReader
from .vasp import VaspReader

#: Registered readers, keyed by :attr:`CalculationReader.code`.
_READERS = {}


def register_reader(reader_class):
    """
    Register a reader class so it can be resolved by name and auto-detected.

    Parameters
    ----------
    reader_class : type
        A :class:`CalculationReader` subclass.

    Returns
    -------
    type
        ``reader_class``, so this can be used as a decorator.
    """
    if not issubclass(reader_class, CalculationReader):
        raise TypeError(f"{reader_class!r} is not a CalculationReader subclass.")
    _READERS[reader_class.code] = reader_class
    return reader_class


def available_codes():
    """Names of the registered codes."""
    return sorted(_READERS)


def get_reader(code):
    """
    Instantiate the reader for ``code``.

    Parameters
    ----------
    code : str or CalculationReader
        Registered code name, or an already-built reader (returned unchanged).

    Returns
    -------
    CalculationReader
    """
    if isinstance(code, CalculationReader):
        return code
    if isinstance(code, type) and issubclass(code, CalculationReader):
        return code()
    try:
        return _READERS[str(code).lower()]()
    except KeyError:
        raise KeyError(
            f"Unknown code {code!r}; registered: {available_codes()}."
        ) from None


def detect_reader(directory):
    """
    Identify which code produced ``directory``.

    Parameters
    ----------
    directory : str
        Calculation directory.

    Returns
    -------
    CalculationReader

    Raises
    ------
    ValueError
        If no registered reader recognises the directory.
    """
    for reader_class in _READERS.values():
        if reader_class.detect(directory):
            return reader_class()
    raise ValueError(
        f"Could not identify the DFT code that produced {directory!r}. "
        f"Registered codes: {available_codes()}. Pass code= explicitly."
    )


def resolve_reader(directory=None, code="auto"):
    """
    Resolve a reader from an explicit code or by sniffing ``directory``.

    Parameters
    ----------
    directory : str, optional
        Required when ``code="auto"``.
    code : str, optional
        Code name, or ``"auto"`` to detect.

    Returns
    -------
    CalculationReader
    """
    if code is None or (isinstance(code, str) and code.lower() == "auto"):
        if directory is None:
            raise ValueError("code='auto' needs a directory to inspect.")
        return detect_reader(directory)
    return get_reader(code)


register_reader(VaspReader)
register_reader(EspressoReader)
register_reader(GpawReader)

__all__ = [
    "COMPRESSION_SUFFIXES",
    "FIELD_KINDS",
    "CalculationParameters",
    "CalculationReader",
    "EspressoReader",
    "GpawReader",
    "PseudopotentialInfo",
    "VaspReader",
    "available_codes",
    "detect_reader",
    "get_reader",
    "is_compressed",
    "open_text",
    "register_reader",
    "resolve_reader",
    "strip_compression_suffix",
]
