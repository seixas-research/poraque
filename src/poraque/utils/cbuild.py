# -*- coding: utf-8 -*-
# file: cbuild.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Build and cache Poraquê's optional C kernels, on first use.

Every C source in the package is compiled the same way: on the machine that
runs it, by whatever compiler is there, into a per-user cache directory, and
loaded through :mod:`ctypes`. The package itself stays pure Python -- a wheel
needs no compiler -- and a machine without one, or with
``PORAQUE_C_BACKEND=0``, runs the Python path instead.

One owner for the mechanics, two clients: the spectral contraction of the
Fourier layer (:mod:`poraque.ml.backend`) and the text codec of the
volumetric files (:mod:`poraque.fields.vasp.codec`). Torch-free, so
``fields/`` can use it.

Environment
-----------
``PORAQUE_C_BACKEND``
    ``0``, ``off``, ``false`` or ``no`` disables every C kernel.
``PORAQUE_CACHE_DIR``
    Where the compiled libraries are kept. Defaults to
    ``$XDG_CACHE_HOME/poraque`` or ``~/.cache/poraque``.
``CC``
    The compiler to try first.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def cache_dir():
    """Directory holding the compiled libraries."""
    override = os.environ.get("PORAQUE_CACHE_DIR")
    if override:
        return os.path.abspath(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "poraque")


def enabled():
    """Whether the environment permits the C kernels at all."""
    return os.environ.get("PORAQUE_C_BACKEND", "1").strip().lower() not in (
        "0", "off", "false", "no")


def compiler():
    """The C compiler to use, or ``None`` when there is none."""
    for name in (os.environ.get("CC"), "cc", "clang", "gcc"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def is_arm():
    machine = (sysconfig.get_platform() or "").lower()
    return "arm" in machine or "aarch64" in machine


def native_tuning():
    """
    The flag that tunes code to the compiling CPU: ``-mcpu=native`` on ARM,
    ``-march=native`` elsewhere.

    One owner for the choice. Only a kernel whose hot loop the compiler
    vectorizes should ask for it, and only knowing the cost: the library's
    cache name records the platform, not the CPU model, so a tuned library
    built on one node can die with *illegal instruction* on another that
    shares the cache.
    """
    return ["-mcpu=native"] if is_arm() else ["-march=native"]


def source_fingerprint(source, abi, flag_sets=()):
    """
    Short hash of a C source, its ABI number, the flags and the platform.

    Part of the library's filename, so editing the source, changing the
    compiler flags or upgrading the interpreter produces a *different* cache
    entry instead of silently reusing a stale one -- a library built with
    ``-march=native`` must not survive the flag being dropped.
    """
    digest = hashlib.sha256()
    try:
        with open(source, "rb") as handle:
            digest.update(handle.read())
    except OSError:
        digest.update(b"missing")
    digest.update(str(abi).encode())
    digest.update(repr([list(flags) for flags in flag_sets]).encode())
    digest.update(sysconfig.get_platform().encode())
    return digest.hexdigest()[:16]


def library_path(source, stem, abi, flag_sets=()):
    """Full path of the cached shared library built from ``source``."""
    suffix = ".dll" if sys.platform == "win32" else (
        ".dylib" if sys.platform == "darwin" else ".so")
    fingerprint = source_fingerprint(source, abi, flag_sets)
    return os.path.join(cache_dir(), f"{stem}_{fingerprint}{suffix}")


def build(source, stem, abi, flag_sets, force=False, log=None):
    """
    Compile ``source`` into :func:`library_path`, trying ``flag_sets`` in turn.

    The first flag set that links wins, so a toolchain without ``-pthread`` or
    ``-march=native`` degrades to a slower library rather than to none. The
    library is compiled to a temporary name in the cache directory and moved
    into place, so a second process never loads a half-written file.

    Returns
    -------
    str or None
        Path to the library, or ``None`` if it could not be built.
    """
    target = library_path(source, stem, abi, flag_sets)
    if os.path.exists(target) and not force:
        return target
    if not os.path.exists(source):
        if log:
            log(f"  C backend: no source at {source}")
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
    for flags in flag_sets:
        handle, staging = tempfile.mkstemp(dir=cache_dir(), suffix=".partial")
        os.close(handle)
        command = [driver, *flags, source, "-o", staging]
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
                log(f"  C backend: built {stem} with {' '.join(flags)}")
            return target
        errors.append(f"{' '.join(flags)}: "
                      f"{(completed.stderr or '').strip().splitlines()[-1:]}")
        _unlink(staging)

    if log:
        log(f"  C backend: every flag set failed to compile {stem}:")
        for line in errors:
            log(f"      {line}")
    return None


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def stale_libraries(current):
    """
    Libraries in :func:`cache_dir` that no current kernel would load.

    A library's name hashes its source, ABI, flags and platform, so every
    edit, flag change or interpreter upgrade leaves the previous file behind;
    nothing reads it again. ``current`` lists the paths the kernels load now.

    Returns
    -------
    list of str
    """
    keep = {os.path.abspath(path) for path in current if path}
    try:
        names = sorted(os.listdir(cache_dir()))
    except OSError:
        return []
    stale = []
    for name in names:
        path = os.path.abspath(os.path.join(cache_dir(), name))
        if (name.startswith("poraque_")
                and name.endswith((".so", ".dylib", ".dll", ".partial"))
                and path not in keep):
            stale.append(path)
    return stale
