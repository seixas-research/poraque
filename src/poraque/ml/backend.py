# -*- coding: utf-8 -*-
# file: backend.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Optional C backend for the spectral contraction on CPU.

What it accelerates and why
---------------------------
:class:`~poraque.ml.fno.SpectralConv3d` is three things in sequence: a real
FFT, a learned multiplication of the retained Fourier modes, and an inverse
FFT. Only the middle one is worth writing in C.

The FFTs are PyTorch's, which is pocketfft or MKL underneath — not something
to beat with hand-written C. The contraction is different: its cost is fixed by
the *mode count*, a model hyper-parameter, while the FFTs around it grow as
:math:`N^3 \log N`. Measured on one layer at width 32, modes 12:

.. code-block:: text

    grid        rfftn     zeros  contract    irfftn     total   contract %
    32x32x32  0.00212   0.00013   0.01673   0.00278   0.02257        74 %
    64x64x64  0.01665   0.00101   0.01673   0.02250   0.05785        29 %
    96x96x96  0.06198   0.00332   0.01671   0.08084   0.16419        10 %

At the 32³ resolution a training cache is normally built at, the contraction
*is* the layer. And ``torch.einsum`` runs it at roughly 3.4 GFLOP/s, well below
what the arithmetic deserves, because the pattern ``bixyz,ioxyz->boxyz``
reduces to a batched product over a layout whose reduction axis is strided by
the whole mode block. Reordering the loops so the flattened mode index is
innermost — contiguous in all three operands — is the whole optimisation, and
it is not expressible through ``einsum``.

How it is loaded
----------------
The kernel is plain C with no Python C API and no numpy headers, called through
:mod:`ctypes`. That keeps it out of the build system: ``hatchling`` never has to
compile anything, a wheel is pure Python, and the shared library is built **on
first use** into a cache directory and reused thereafter.

Every failure path falls back to :func:`torch.einsum`:

* no compiler on the machine,
* a compiler that rejects the flags,
* a cache directory that is not writable,
* an unsupported dtype, device, or non-contiguous operand,
* ``PORAQUE_C_BACKEND=0`` in the environment.

So the backend is an optimisation and never a dependency. The one thing it must
never do is *silently* change a number, which is what :func:`selftest` and
``tests/test_backend.py`` exist to prevent.

Controlling it
--------------
``PORAQUE_C_BACKEND``
    ``0``/``off``/``false`` disables it entirely; anything else (or unset)
    leaves it enabled.
``PORAQUE_CACHE_DIR``
    Where the compiled library is cached. Defaults to
    ``~/.cache/poraque``.

.. code-block:: python

    from poraque.ml.backend import describe, selftest

    print(describe())     # what was loaded, or why it was not
    selftest()            # agreement with torch.einsum, on random data
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading

#: Source file compiled by :func:`build`.
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_spectral.c")

#: ABI the Python side expects; must match ``PORAQUE_SPECTRAL_ABI`` in the C.
#: A cached library built by an older Poraquê is rebuilt rather than called.
ABI = 3

#: Guards the one-time build/load. Two threads reaching the backend at once
#: must not race to write the same file.
_LOCK = threading.Lock()

#: ``None`` until the first :func:`load`; then the handle or ``False``.
_HANDLE = None
_STATUS = "not yet loaded"


# ===================================================================== #
# Where the compiled library lives
# ===================================================================== #
def cache_dir():
    """Directory holding the compiled library."""
    override = os.environ.get("PORAQUE_CACHE_DIR")
    if override:
        return os.path.abspath(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "poraque")


def source_fingerprint():
    """
    Short hash of the C source, the ABI and the interpreter.

    Part of the library's filename, so editing ``_spectral.c`` or upgrading
    Python produces a *different* cache entry instead of silently reusing a
    stale one. A rebuilt kernel that is never loaded is the kind of bug that
    only shows up as a benchmark that stopped improving.
    """
    digest = hashlib.sha256()
    try:
        with open(SOURCE, "rb") as handle:
            digest.update(handle.read())
    except OSError:
        digest.update(b"missing")
    digest.update(str(ABI).encode())
    digest.update(sysconfig.get_platform().encode())
    return digest.hexdigest()[:16]


def library_path():
    """Full path of the cached shared library for this source."""
    suffix = ".dll" if sys.platform == "win32" else (
        ".dylib" if sys.platform == "darwin" else ".so")
    return os.path.join(cache_dir(),
                        f"poraque_spectral_{source_fingerprint()}{suffix}")


def enabled():
    """Whether the environment permits the C backend at all."""
    return os.environ.get("PORAQUE_C_BACKEND", "1").strip().lower() not in (
        "0", "off", "false", "no")


# ===================================================================== #
# Building
# ===================================================================== #
def compiler():
    """The C compiler to use, or ``None`` when there is none."""
    for name in (os.environ.get("CC"), "cc", "clang", "gcc"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _flag_sets():
    """
    Compiler flag sets to try, best first.

    Each is attempted in turn and the first that links wins, so a toolchain
    without ``libomp`` or without ``-mcpu=native`` degrades to a slower kernel
    rather than to no kernel. ``-ffast-math`` is deliberately absent: the
    vectorisation this kernel needs is across independent mode indices and
    requires no reassociation, so there is nothing to buy with it and it would
    put the C path's arithmetic out of step with PyTorch's.
    """
    base = ["-O3", "-funroll-loops", "-fPIC", "-shared", "-pthread"]
    tuning = ["-mcpu=native"] if _is_arm() else ["-march=native"]
    # Last resort drops threading rather than the whole kernel: a
    # serial C contraction still beats einsum by ~7x at batch 1.
    return [base + tuning, base,
            ["-O3", "-fPIC", "-shared", "-DPORAQUE_NO_PTHREADS"]]


def _is_arm():
    machine = (sysconfig.get_platform() or "").lower()
    return "arm" in machine or "aarch64" in machine


def build(force=False, log=None):
    """
    Compile the kernel into :func:`library_path`.

    Parameters
    ----------
    force : bool, optional
        Rebuild even when a cached library exists.
    log : callable, optional
        Sink for progress; silent when omitted.

    Returns
    -------
    str or None
        Path to the library, or ``None`` if it could not be built.
    """
    target = library_path()
    if os.path.exists(target) and not force:
        return target
    if not os.path.exists(SOURCE):
        if log:
            log(f"  C backend: no source at {SOURCE}")
        return None

    driver = compiler()
    if driver is None:
        if log:
            log("  C backend: no C compiler found (tried $CC, cc, clang, gcc)")
        return None

    try:
        os.makedirs(cache_dir(), exist_ok=True)
    except OSError as error:
        if log:
            log(f"  C backend: cache directory unusable ({error})")
        return None

    errors = []
    for flags in _flag_sets():
        # Compiled to a temporary name in the same directory and moved into
        # place, so a second process never loads a half-written library.
        handle, staging = tempfile.mkstemp(dir=cache_dir(), suffix=".partial")
        os.close(handle)
        command = [driver, *flags, SOURCE, "-o", staging]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=180)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{' '.join(flags)}: {error}")
            _unlink(staging)
            continue

        if completed.returncode == 0 and os.path.getsize(staging) > 0:
            os.replace(staging, target)
            if log:
                log(f"  C backend: built with {' '.join(flags)}")
            return target
        errors.append(f"{' '.join(flags)}: "
                      f"{(completed.stderr or '').strip().splitlines()[-1:]}")
        _unlink(staging)

    if log:
        log("  C backend: every flag set failed to compile:")
        for line in errors:
            log(f"      {line}")
    return None


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


# ===================================================================== #
# Loading
# ===================================================================== #
def load(log=None):
    """
    Load the kernel, building it on first use.

    Returns
    -------
    ctypes.CDLL or None
        ``None`` when the backend is unavailable for any reason; the caller
        then uses the PyTorch path.
    """
    global _HANDLE, _STATUS

    if _HANDLE is not None:
        return _HANDLE or None

    with _LOCK:
        if _HANDLE is not None:
            return _HANDLE or None

        if not enabled():
            _HANDLE, _STATUS = False, "disabled by PORAQUE_C_BACKEND"
            return None

        path = build(log=log)
        if path is None:
            _HANDLE, _STATUS = False, "unavailable (could not build)"
            return None

        try:
            library = ctypes.CDLL(path)
            _declare(library)
            if library.poraque_spectral_abi() != ABI:
                raise OSError(f"ABI {library.poraque_spectral_abi()} != {ABI}")
        except (OSError, AttributeError) as error:
            _HANDLE, _STATUS = False, f"unavailable ({error})"
            return None

        parallel = ("pthreads" if library.poraque_spectral_threaded()
                    else "serial")
        _HANDLE = library
        _STATUS = f"loaded from {path} ({parallel})"
        return library


def _declare(library):
    """Pin every signature, so a mismatch fails loudly instead of corrupting."""
    for name, real in (("poraque_spectral_contract_c64", ctypes.c_float),
                       ("poraque_spectral_contract_c128", ctypes.c_double)):
        function = getattr(library, name)
        function.restype = None
        function.argtypes = [ctypes.POINTER(real)] * 3 + [ctypes.c_long] * 4

        parallel = getattr(library, name + "_mt")
        parallel.restype = None
        parallel.argtypes = ([ctypes.POINTER(real)] * 3
                             + [ctypes.c_long] * 4 + [ctypes.c_int])

    for name in ("poraque_spectral_abi", "poraque_spectral_threaded"):
        getattr(library, name).restype = ctypes.c_int
        getattr(library, name).argtypes = []


def available():
    """Whether the C contraction can be used in this process."""
    return load() is not None


def describe():
    """One line saying what was loaded, or why nothing was."""
    load()
    return f"C spectral backend: {_STATUS}"


def reset():
    """Forget the loaded handle, so the next call reloads. For tests."""
    global _HANDLE, _STATUS
    with _LOCK:
        _HANDLE, _STATUS = None, "not yet loaded"


# ===================================================================== #
# The contraction
# ===================================================================== #
#: dtype -> (C element type, entry-point stem)
_KERNELS = {
    "torch.complex64": (ctypes.c_float, "poraque_spectral_contract_c64"),
    "torch.complex128": (ctypes.c_double, "poraque_spectral_contract_c128"),
}


def contract(x, weight, threads=None):
    r"""
    ``out[b, o, m] = sum_i x[b, i, m] * w[i, o, m]`` in C.

    The einsum of :class:`~poraque.ml.fno.SpectralConv3d`, with its three mode
    axes flattened. Flattening is exact: they are the fastest-varying axes of
    both operands and the contraction is elementwise across all three.

    Parameters
    ----------
    x : torch.Tensor
        ``(B, I, m1, m2, m3)`` complex, CPU.
    weight : torch.Tensor
        ``(I, O, m1, m2, m3)`` complex, CPU, same dtype as ``x``.
    threads : int, optional
        Threads for the pthread kernel. Defaults to
        :func:`torch.get_num_threads`, so the C path and the PyTorch path are
        compared on equal terms rather than one of them silently getting the
        whole machine.

    Returns
    -------
    torch.Tensor or None
        ``(B, O, m1, m2, m3)``, or ``None`` when this call cannot be served in
        C — a non-CPU tensor, an unsupported dtype, a shape mismatch — in which
        case the caller falls back to :func:`torch.einsum`.
    """
    import torch

    library = load()
    if library is None:
        return None

    kernel = _KERNELS.get(str(x.dtype))
    if kernel is None or x.dtype != weight.dtype:
        return None
    if x.device.type != "cpu" or weight.device.type != "cpu":
        return None
    if x.ndim != 5 or weight.ndim != 5:
        return None

    batch, in_x = x.shape[0], x.shape[1]
    in_w, out_channels = weight.shape[0], weight.shape[1]
    if in_x != in_w or x.shape[2:] != weight.shape[2:]:
        return None

    modes = 1
    for size in x.shape[2:]:
        modes *= int(size)
    if modes == 0:
        return None

    element, stem = kernel
    # `.contiguous()` is required, not defensive: the caller hands us strided
    # corner views, and the kernel walks raw memory. A strided operand read as
    # if contiguous is silently wrong, which is the one failure mode this
    # module must not have.
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((batch, out_channels) + tuple(x.shape[2:]),
                      dtype=x.dtype)

    if threads is None:
        threads = torch.get_num_threads()
    threads = max(1, int(threads))

    def pointer(tensor):
        return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(element))

    if threads > 1 and library.poraque_spectral_threaded():
        getattr(library, stem + "_mt")(
            pointer(x), pointer(weight), pointer(out),
            ctypes.c_long(batch), ctypes.c_long(in_x),
            ctypes.c_long(out_channels), ctypes.c_long(modes),
            ctypes.c_int(threads))
    else:
        getattr(library, stem)(
            pointer(x), pointer(weight), pointer(out),
            ctypes.c_long(batch), ctypes.c_long(in_x),
            ctypes.c_long(out_channels), ctypes.c_long(modes))
    return out


def selftest(tolerance=2e-5, seed=0):
    """
    Check the C kernel against :func:`torch.einsum` on random data.

    Returns
    -------
    dict
        ``available``, and when it is, ``max_absolute`` and ``max_relative``
        deviation from the PyTorch result.

    Raises
    ------
    AssertionError
        If the two disagree by more than ``tolerance`` in relative terms. A
        backend that is fast and wrong is worse than no backend, so this is an
        assertion rather than a returned flag.
    """
    import torch

    if not available():
        return {"available": False, "status": _STATUS}

    generator = torch.Generator().manual_seed(seed)
    report = {"available": True, "status": _STATUS,
              "max_absolute": 0.0, "max_relative": 0.0}
    for batch, in_channels, out_channels, modes in (
            (1, 8, 8, (4, 4, 3)), (2, 16, 32, (6, 5, 4)),
            (1, 32, 32, (12, 12, 12))):
        x = torch.randn(batch, in_channels, *modes, dtype=torch.cfloat,
                        generator=generator)
        weight = torch.randn(in_channels, out_channels, *modes,
                             dtype=torch.cfloat, generator=generator)
        expected = torch.einsum("bixyz,ioxyz->boxyz", x, weight)
        got = contract(x, weight)
        assert got is not None, "backend reported available but refused a call"
        absolute = (got - expected).abs().max().item()
        scale = expected.abs().max().item() or 1.0
        report["max_absolute"] = max(report["max_absolute"], absolute)
        report["max_relative"] = max(report["max_relative"], absolute / scale)

    assert report["max_relative"] <= tolerance, (
        f"C backend disagrees with torch.einsum by "
        f"{report['max_relative']:.2e} (> {tolerance:.0e})")
    return report

#: Aliases used by :mod:`poraque.ml`'s lazy import map, where the bare names
#: ``available``/``describe`` would be too generic to sit in a package
#: namespace shared with the operators, losses and datasets.
backend_available = available
backend_describe = describe


# ===================================================================== #
# Command line: python -m poraque.ml.backend
# ===================================================================== #
def _benchmark(repeats=20):
    """Time the kernel against ``torch.einsum`` at a production shape."""
    import time

    import torch

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 32, 12, 12, 12, dtype=torch.cfloat, generator=generator)
    weight = torch.randn(32, 32, 12, 12, 12, dtype=torch.cfloat,
                         generator=generator)

    def timed(function):
        for _ in range(5):
            function()
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - start) / repeats

    threads = max(1, __import__("torch").get_num_threads())
    return {
        "einsum": timed(lambda: torch.einsum("bixyz,ioxyz->boxyz", x, weight)),
        "serial": timed(lambda: contract(x, weight, threads=1)),
        "threaded": timed(lambda: contract(x, weight, threads=threads)),
        "threads": threads,
    }


def main(argv=None):
    """
    Build, verify and benchmark the C backend.

    ``python -m poraque.ml.backend`` is the answer to "how do I compile it":
    it does the compile that would otherwise happen on the first inference,
    checks the result against PyTorch, and prints where the library went.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poraque.ml.backend",
        description="Build and check the optional C backend for CPU inference.")
    parser.add_argument("--rebuild", action="store_true",
                        help="recompile even if a cached library exists")
    parser.add_argument("--benchmark", action="store_true",
                        help="also time the kernel against torch.einsum")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = (lambda *_: None) if args.quiet else print

    log("=" * 70)
    log("Poraque C backend")
    log("=" * 70)
    log(f"  source    : {SOURCE}")
    log(f"  compiler  : {compiler() or 'NONE FOUND'}")
    log(f"  cache dir : {cache_dir()}")
    log(f"  enabled   : {enabled()}  (PORAQUE_C_BACKEND)")
    log("")

    if args.rebuild:
        reset()
        build(force=True, log=log)
    reset()

    if not available():
        log(f"  {describe()}")
        log("")
        log("  Inference still works — it falls back to torch.einsum, about")
        log("  2-3x slower at cache resolutions. To enable the backend,")
        log("  install a C compiler (xcode-select --install on macOS,")
        log("  build-essential on Debian/Ubuntu) and run this again.")
        return 1

    log(f"  {describe()}")
    report = selftest()
    log(f"  agreement with torch.einsum: {report['max_relative']:.2e} "
        f"relative (float32 rounding is ~1e-7)")

    if args.benchmark:
        timing = _benchmark()
        baseline = timing["einsum"]
        rows = [
            (f"torch.einsum ({timing['threads']} threads)", baseline, None),
            ("C, serial", timing["serial"], baseline / timing["serial"]),
            (f"C, {timing['threads']} pthreads", timing["threaded"],
             baseline / timing["threaded"]),
        ]
        # Padded from the widest label rather than by hand, so the column
        # cannot drift when the thread count changes the label's length.
        width = max(len(label) for label, _, _ in rows)
        log("")
        log("  one contraction (batch 1, width 32, modes 12^3):")
        for label, seconds, speedup in rows:
            gain = "" if speedup is None else f"   {speedup:5.1f}x"
            log(f"    {label:<{width}s}  {seconds * 1e3:8.3f} ms{gain}")

    log("")
    log("  Ready. Nothing further is needed: inference picks this up")
    log("  automatically, and poraque-inference prints which path it used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
