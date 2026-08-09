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
from .aims import AimsReader
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


#: Code-specific exchange-correlation tags mapped onto the functionals Poraquê
#: can evaluate. Keys are upper-cased VASP ``GGA``/``LEXCH`` values and the
#: labels other codes write.
#:
#: ``None`` marks a functional that is recognised but not implemented: the
#: caller is warned and told what was substituted, rather than being handed a
#: silent mislabel.
XC_TAGS = {
    "PE": "pbe", "PBE": "pbe",
    "CA": "lda", "PZ": "lda", "LDA": "lda", "VWN": "lda", "PW": "lda",
    "91": None, "PW91": None,
    "PS": None, "PBESOL": None,
    "RP": None, "RPBE": None,
    "AM": None, "B3LYP": None, "HSE": None, "SCAN": None,
}

#: Substituted for a recognised but unimplemented functional.
XC_FALLBACK = "pbe"


def resolve_xc(directory=None, declared="auto", code="auto", warn=True):
    r"""
    Which exchange-correlation functional produced a calculation.

    This is a statement about the *data*. It is needed wherever the
    Euler-Lagrange equation is evaluated, because :math:`v_{\rm xc}` is one of
    its terms, and an LDA potential on a PBE density does not approximate the
    PBE one: the two differ by of order 1 eV in a valence region, and that
    error lands in the residual where it reads as the error of the kinetic
    functional.

    Resolution follows the order VASP itself uses:

    1. an explicit ``declared`` value, which always wins;
    2. the ``INCAR`` ``GGA`` tag, when the run set one;
    3. the ``LEXCH`` tag of the pseudopotentials, which is what VASP falls back
       to. A ``PAW_PBE`` library therefore resolves to ``"pbe"`` with nothing
       for the user to declare.

    Parameters
    ----------
    directory : str, optional
        Calculation directory. Required unless ``declared`` is explicit.
    declared : str, optional
        ``"auto"`` to detect, or a functional name to use as given.
    code : str, optional
        Passed to :func:`resolve_reader`.
    warn : bool, optional
        Emit a warning when a recognised functional is not implemented, or
        when detection fails and the default is used.

    Returns
    -------
    str
        A functional name accepted by
        :func:`poraque.ml.physics.xc_potential`.

    Examples
    --------
    >>> resolve_xc("run/", declared="auto")           # doctest: +SKIP
    'pbe'
    """
    import warnings

    if declared is not None and str(declared).lower() not in ("auto", ""):
        return str(declared).lower()

    if directory is None:
        raise ValueError("resolve_xc needs a directory when declared='auto'.")

    tag, source = None, None
    try:
        reader = resolve_reader(directory, code)
        parameters = reader.read_parameters(directory)
        if parameters.xc:
            tag, source = str(parameters.xc), "the INCAR GGA tag"
        if tag is None:
            for info in (reader.read_pseudopotentials(directory) or {}).values():
                if getattr(info, "functional", None):
                    tag, source = str(info.functional), "the POTCAR LEXCH tag"
                    break
    except (ValueError, KeyError, OSError):
        tag = None

    if tag is None:
        if warn:
            warnings.warn(
                f"Could not determine the exchange-correlation functional of "
                f"{directory!r}: no INCAR GGA tag and no pseudopotential "
                f"LEXCH. Assuming {XC_FALLBACK!r}. Set data.xc explicitly if "
                f"that is wrong.", RuntimeWarning, stacklevel=2)
        return XC_FALLBACK

    key = tag.strip().upper()
    if key not in XC_TAGS:
        if warn:
            warnings.warn(
                f"Unrecognised exchange-correlation tag {tag!r} from {source} "
                f"in {directory!r}; assuming {XC_FALLBACK!r}. Set data.xc "
                f"explicitly.", RuntimeWarning, stacklevel=2)
        return XC_FALLBACK

    resolved = XC_TAGS[key]
    if resolved is None:
        if warn:
            warnings.warn(
                f"{tag!r} (from {source}) is not implemented; using "
                f"{XC_FALLBACK!r} instead. The substitution is a real "
                f"approximation, not a relabelling.",
                RuntimeWarning, stacklevel=2)
        return XC_FALLBACK
    return resolved


register_reader(VaspReader)
register_reader(AimsReader)
register_reader(EspressoReader)
register_reader(GpawReader)

__all__ = [
    "AimsReader",
    "XC_FALLBACK",
    "XC_TAGS",
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
    "resolve_xc",
    "strip_compression_suffix",
]
