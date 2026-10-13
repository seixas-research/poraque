#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_compile.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Compile, check and report Poraquê's optional C kernels.

Two kernels ship as C sources and are compiled on the machine that runs them
(:mod:`poraque.utils.cbuild`), into ``~/.cache/poraque`` (or
``$PORAQUE_CACHE_DIR``):

``spectral``
    The spectral contraction of the Fourier layer, for CPU inference
    (:mod:`poraque.ml.backend`).
``codec``
    The text codec of volumetric grid blocks, for the dataset cache
    (:mod:`poraque.fields.vasp.codec`).

Each is otherwise compiled on its first use. This command does it up front --
in a cluster's setup step, say, so no job pays for or races over the compile --
then checks each kernel against its Python reference and says what loaded.
Without a compiler, or with ``PORAQUE_C_BACKEND=0``, everything still runs on
the Python path, more slowly; the command says so and exits 1.

Usage
-----
::

    poraque-compile                 # build what is missing, check, report
    poraque-compile --rebuild       # recompile even if cached
    poraque-compile --clean         # also delete libraries nothing loads
    poraque-compile --only codec    # one kernel

Running this file directly -- ``python scripts/poraque_compile.py`` -- is
equivalent, and needs nothing installed.
"""

import argparse
import os
import sys

# Run straight from a checkout, without installing, by preferring the in-tree
# package; installed as the ``poraque-compile`` console script this resolves
# to the installed package instead.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if os.path.isdir(os.path.join(_SRC, "poraque")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from poraque.utils import cbuild  # noqa: E402
from poraque.utils.logging import Logger  # noqa: E402

#: The kernels, in the order they are reported.
KERNELS = ("spectral", "codec")


def kernel_module(name):
    """The module that owns kernel ``name``."""
    if name == "spectral":
        from poraque.ml import backend

        return backend
    from poraque.fields.vasp import codec

    return codec


def check(name, module):
    """
    Run a kernel's self-test, returning one line for the report.

    The spectral test compares against ``torch.einsum``; the codec test
    against ``float()`` and ``fortran_exponential``.
    """
    try:
        report = module.selftest()
    except AssertionError as error:
        return False, f"FAILED: {error}"
    if not report.get("available"):
        return False, "not available"
    if name == "spectral":
        return True, (f"agrees with torch.einsum to "
                      f"{report['max_relative']:.2e} relative")
    return True, (f"reads bit-identical and writes byte-identical to the "
                  f"Python path ({report['values']} values)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="poraque-compile",
        description="Compile, check and report Poraque's optional C kernels.")
    parser.add_argument("--rebuild", action="store_true",
                        help="recompile even when a cached library exists")
    parser.add_argument("--clean", action="store_true",
                        help="delete cached libraries that no current kernel "
                             "loads (left behind by source, flag or "
                             "interpreter changes)")
    parser.add_argument("--only", choices=KERNELS, default=None,
                        help="handle one kernel only")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing; the exit status still says "
                             "whether every kernel loaded")
    return parser


def main(argv=None):
    """Build, check and report; exit 0 when every requested kernel loaded."""
    args = build_parser().parse_args(argv)
    log = Logger(silent=args.quiet)
    log.banner()
    log.title("Poraque C kernels")

    log.section("configuration")
    log.fields({"compiler": cbuild.compiler() or "NONE FOUND",
                "cache dir": cbuild.cache_dir(),
                "enabled": f"{cbuild.enabled()}  (PORAQUE_C_BACKEND)"})

    names = (args.only,) if args.only else KERNELS
    loaded = {}
    for name in names:
        module = kernel_module(name)
        log.section(f"{name} kernel")
        if cbuild.enabled():
            # A no-op for a cached library; otherwise the compile happens here,
            # where it is logged, rather than silently inside the check.
            module.reset()
            module.build(force=args.rebuild, log=log)
        module.reset()
        if module.available():
            ok, detail = check(name, module)
        else:
            ok, detail = False, module.describe().split(": ", 1)[-1]
        loaded[name] = ok
        log.fields({"status": "loaded" if ok else "unavailable",
                    "library": module.library_path() if ok else "-",
                    "check": detail})

    if args.clean:
        # Only libraries the *current* kernels would not load. Both kernels'
        # paths are kept even under --only, so cleaning one never deletes the
        # other's library.
        current = [kernel_module(name).library_path() for name in KERNELS]
        stale = cbuild.stale_libraries(current)
        log.section("clean")
        if not stale:
            log.text("nothing to delete")
        for path in stale:
            try:
                os.unlink(path)
                log.text(f"deleted {path}")
            except OSError as error:
                log.text(f"could not delete {path}: {error}")

    log.section("next steps")
    if all(loaded.values()):
        log.text("Ready. The kernels are picked up automatically: inference")
        log.text("and the cache builder use them without any setting.")
        return 0
    log.text("Everything still runs on the Python path, more slowly.")
    if not cbuild.enabled():
        log.text("PORAQUE_C_BACKEND disables the kernels; unset it to use them.")
        return 1
    log.text("To enable the kernels, install a C compiler (xcode-select --install")
    log.text("on macOS, build-essential on Debian/Ubuntu, `module load gcc` on")
    log.text("a cluster) and run this again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
