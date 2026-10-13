# -*- coding: utf-8 -*-
# file: codec.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The C text codec of volumetric grid blocks, with the Python path as fallback.

Reading and writing the ``E17.11`` text of a ``CHGCAR`` were **86 %** of the
cache build per material (measured on the Pt set at res 48), in pure Python:
a ``split()`` and a float conversion per value on the way in, a
:func:`~poraque.fields.vasp.volumetric.fortran_exponential` per value on the
way out. ``_codec.c`` does both, compiled on first use by
:mod:`poraque.utils.cbuild` and loaded through :mod:`ctypes`, and agrees with
the Python path exactly: reading goes through ``strtod`` (correctly rounded,
so bit-identical to ``float()``), writing reproduces ``fortran_exponential``
byte for byte.

When the library cannot be built or loaded -- no compiler, or
``PORAQUE_C_BACKEND=0`` -- :func:`load` returns ``None`` and the callers in
:mod:`poraque.fields.vasp.volumetric` keep the Python path. Torch-free.

Usage::

    python -m poraque.fields.vasp.codec     # build, then report what loaded
"""

import ctypes
import os
import threading

import numpy as np

from ...utils import cbuild

#: Source compiled by :func:`load`.
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_codec.c")

#: Must match ``PORAQUE_CODEC_ABI`` in the C.
ABI = 3

#: Bounds :func:`format_block` accepts; the C refuses anything outside them.
MAX_WIDTH = 64
MAX_DECIMALS = 40

_LOCK = threading.Lock()
_HANDLE = None
_STATUS = "not yet loaded"


def _flag_sets():
    """
    Portable flags only: no ``-march=native`` / ``-mcpu=native``.

    The library is cached under a name that records the platform string, not
    the CPU model, so on a cluster with a shared ``~/.cache`` a build tuned to
    the login node's CPU could be loaded on a compute node that lacks its
    instructions and die with *illegal instruction*. The codec has nothing to
    gain from the tuning anyway: its time is in ``strtod`` and ``snprintf``,
    which live in libc and are not recompiled.
    """
    return [["-O3", "-fPIC", "-shared"]]


def load(log=None):
    """
    The codec library, built on first use; ``None`` when unavailable.

    Returns
    -------
    ctypes.CDLL or None
    """
    global _HANDLE, _STATUS

    if _HANDLE is not None:
        return _HANDLE or None
    with _LOCK:
        if _HANDLE is not None:
            return _HANDLE or None
        if not cbuild.enabled():
            _HANDLE, _STATUS = False, "disabled by PORAQUE_C_BACKEND"
            return None
        path = cbuild.build(SOURCE, "poraque_codec", ABI, _flag_sets(),
                            log=log)
        if path is None:
            _HANDLE, _STATUS = False, "unavailable (could not build)"
            return None
        try:
            library = ctypes.CDLL(path)
            library.poraque_codec_abi.restype = ctypes.c_int
            if library.poraque_codec_abi() != ABI:
                raise OSError(f"ABI {library.poraque_codec_abi()} != {ABI}")
            library.poraque_parse_doubles.restype = ctypes.c_int64
            library.poraque_parse_doubles.argtypes = [
                ctypes.c_void_p, ctypes.c_int64,
                ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int)]
            library.poraque_skip_tokens.restype = ctypes.c_int64
            library.poraque_skip_tokens.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64)]
            library.poraque_format_block.restype = ctypes.c_int64
            library.poraque_format_block.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        except (OSError, AttributeError) as error:
            _HANDLE, _STATUS = False, f"unavailable ({error})"
            return None
        _HANDLE, _STATUS = library, f"loaded from {path}"
        return library


def library_path():
    """Where the compiled codec is (or would be) cached."""
    return cbuild.library_path(SOURCE, "poraque_codec", ABI, _flag_sets())


def build(force=False, log=None):
    """Compile the codec now (it is otherwise compiled on first use)."""
    return cbuild.build(SOURCE, "poraque_codec", ABI, _flag_sets(),
                        force=force, log=log)


def available():
    """Whether the codec loaded."""
    return load() is not None


def describe():
    """One line: what was loaded, or why not."""
    load()
    return f"C volumetric codec: {_STATUS}"


def selftest(seed=0):
    """
    Check the codec against the Python path on random values.

    Returns
    -------
    dict
        ``available``, and when it is, ``values`` checked and whether
        ``read`` (bit-for-bit) and ``write`` (byte-for-byte) agree.

    Raises
    ------
    AssertionError
        On any disagreement: a codec that is fast and wrong would change a
        cache silently.
    """
    from .volumetric import fortran_exponential

    if not available():
        return {"available": False, "status": _STATUS}
    rng = np.random.default_rng(seed)
    values = rng.normal(size=5000) * 10.0 ** rng.integers(-12, 12, 5000)
    values = np.concatenate([values, [0.0, -1.0, 0.1, -2362.5518271]])
    for width, decimals in ((17, 11), (11, 5)):
        python = "\n".join(
            "".join(" " + fortran_exponential(v, decimals=decimals,
                                              width=width)
                    for v in values[start:start + 5])
            for start in range(0, values.size, 5)) + "\n"
        written = format_block(values, 5, width, decimals)
        assert written.decode() == python, (
            f"C codec writes differently from fortran_exponential "
            f"(E{width}.{decimals})")
    text = format_block(values, 5, 17, 11)
    parsed = np.empty(values.size)
    count, _, _ = parse_into(text, 0, len(text), parsed, 0)
    expected = np.array(text.split(), dtype=float)
    assert count == values.size and np.array_equal(
        parsed.view(np.uint64), expected.view(np.uint64)), (
        "C codec reads differently from float()")
    return {"available": True, "status": _STATUS, "values": int(values.size),
            "read": True, "write": True}


def reset():
    """Forget the loaded library, so the next call loads again (for tests)."""
    global _HANDLE, _STATUS
    with _LOCK:
        _HANDLE, _STATUS = None, "not yet loaded"


def _address(buffer):
    """The address of a bytes object's data (no copy)."""
    return ctypes.cast(ctypes.c_char_p(buffer), ctypes.c_void_p).value


def parse_into(buffer, start, stop, out, filled):
    """
    Parse numbers from ``buffer[start:stop]`` into ``out[filled:]``.

    The window need not end on a line or a token boundary the C relies on:
    each token is copied into a bounded local buffer before conversion, so no
    NUL terminator is needed, and the caller keeps the unconsumed tail for the
    next window.

    Returns
    -------
    parsed : int
        Values written.
    end : int
        Offset in ``buffer`` just past the last value converted.
    bad : bool
        Whether parsing stopped at a token that is not a number.
    """
    library = load()
    end = ctypes.c_int64(0)
    bad = ctypes.c_int(0)
    target = out[filled:]
    parsed = library.poraque_parse_doubles(
        _address(buffer) + start, stop - start,
        target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), target.size,
        ctypes.byref(end), ctypes.byref(bad))
    return int(parsed), start + end.value, bool(bad.value)


def skip_tokens(buffer, start, stop, count):
    """
    Step over up to ``count`` tokens of ``buffer[start:stop]``, unconverted.

    Returns ``(skipped, end)``, ``end`` being the offset just past the last.
    """
    library = load()
    end = ctypes.c_int64(0)
    skipped = library.poraque_skip_tokens(_address(buffer) + start,
                                          stop - start, count,
                                          ctypes.byref(end))
    return int(skipped), start + end.value


def format_block(values, columns, width, decimals):
    """
    A grid block as bytes, or ``None`` when the codec is unavailable.

    ``values`` are written in the order given (the caller flattens in Fortran
    order), ``columns`` per line, each as ``" " + Ew.d``.
    """
    library = load()
    if library is None:
        return None
    if int(columns) < 1:
        raise ValueError(f"columns must be at least 1, got {columns!r}.")
    if not 1 <= int(width) <= MAX_WIDTH:
        raise ValueError(f"width must be 1..{MAX_WIDTH}, got {width!r}.")
    if not 0 <= int(decimals) <= MAX_DECIMALS:
        raise ValueError(
            f"decimals must be 0..{MAX_DECIMALS}, got {decimals!r}.")
    flat = np.ascontiguousarray(values, dtype=np.float64).ravel()
    n = flat.size
    if n == 0:
        return b""
    size = n * (int(width) + 1) + -(-n // int(columns))
    out = ctypes.create_string_buffer(size)
    written = library.poraque_format_block(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        int(columns), int(width), int(decimals), out)
    if written < 0:
        raise ValueError("the C codec refused the block's format arguments.")
    return out.raw if written == size else out.raw[:written]


if __name__ == "__main__":
    print(describe())
