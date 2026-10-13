#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_atoms.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Build the isolated-atom database from single-atom calculations.

.. code-block:: text

    data/vasp/ref/Pt/CHGCAR  ---->  f_Pt(|G|)  +  its augmentation record
    data/vasp/ref/Ag/CHGCAR  ---->  f_Ag(|G|)  +  its augmentation record
                                        |
                                        v
                             atomic_reference.json

What the database is for
------------------------
**The superposition baseline.** Most of a crystal's valence density is its free
atoms placed side by side. Measured on this project's own platinum cells, the
superposition accounts for about 95 % of the field in :math:`L^2`. Training on
the residual :math:`\delta\rho = \rho - \rho_{\rm sup}` therefore removes the
part that was never in doubt --- including nearly all of the dynamic range the
``asinh`` transform exists to absorb --- and leaves the bonding charge.
``data.delta_density: true`` switches that on and reads the database from
``data.atomic_reference``.

**A last-resort source of PAW one-center terms.** Each atom's own augmentation
record travels with it, which is the only thing available for an element the
training set has never seen. It is *not* the better source when the element
**is** in the training set: measured here, a free Pt atom's record is 86.6 % RMS
away from a bulk Pt site while the training-set average is 9.9 % away. Nothing
selects it silently; ``poraque-inference --paw-source atomic`` is the only way
to ask for it.

What a reference calculation has to be
--------------------------------------
**One atom in a box**, converged, with ``LCHARG = .TRUE.`` so it writes a
``CHGCAR``. The box wants to be large enough that the atom does not see its own
images --- 10 Å is what this project's Pt reference uses. A cell with more than
one atom is refused rather than averaged: the form factor of a pair is not the
form factor of an atom.

The density is reduced to a **radial** table :math:`f(|G|)`, so how spherical
the reference atom actually was is a real question. It is measured and reported
per entry (``anisotropy`` in the output below, ``radial_scatter`` in the file)
rather than assumed away; the shipped platinum atom comes out at 0.48 %.

Usage
-----
.. code-block:: bash

    # One parent directory holding one subdirectory per element
    poraque-atoms data/vasp/ref --output atomic_reference.json

    # Or name the atoms explicitly, and extend an existing database
    poraque-atoms refs/Ag refs/Pt --output atomic_reference.json --append

    # Inspect what a database holds
    poraque-atoms --show atomic_reference.json
"""

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from poraque.fields.atomic import (                          # noqa: E402
    LIBRARY_FILENAME,
    AtomicReferenceLibrary,
    build_library,
)
from poraque.utils.logging import Logger, header, table_lines  # noqa: E402


def _section(log, name):
    """
    Open the block ``name`` on ``log``, whatever ``log`` is.

    The functions below take any callable, as they always have, so a test can
    hand them ``list.append`` and read back what they said; only :func:`main`
    hands them a :class:`~poraque.utils.logging.Logger`. A plain callable gets
    exactly what :meth:`Logger.section` would have written --- a blank line and
    the header --- so the layout does not depend on who is listening.
    """
    if isinstance(log, Logger):
        log.section(name)
    else:
        log("")
        log(header(name))


def show(path, log=print):
    """
    Print what a database holds, without building anything.

    Parameters
    ----------
    path : str
        The JSON file, or a directory holding one.
    log : callable, optional

    Returns
    -------
    AtomicReferenceLibrary
    """
    library = AtomicReferenceLibrary.load(path)
    _section(log, "database")
    if not len(library):
        log(f"  {path}: no isolated-atom references here.")
        return library

    log(f"  path:        {path}")
    log(f"  fingerprint: {library.fingerprint}")

    _section(log, "entries")
    rows = []
    for key in sorted(library.entries):
        entry = library.entries[key]
        rows.append((entry.element, f"{entry.valence_charge:.3f}",
                     f"{entry.g_max:.2f}", len(entry.g_grid),
                     f"{100 * entry.radial_scatter:.2f}%",
                     "yes" if entry.augmentation else "no",
                     entry.potcar_title or "(none)"))
    for line in table_lines(("element", "Zval", "g_max", "points",
                             "anisotropy", "aug", "potcar"), rows,
                            align="<>>>>><"):
        log(line)

    _section(log, "notes")
    log("  g_max is in 1/Ang; beyond it a species contributes zero rather than")
    log("  an extrapolation. anisotropy is the worst within-bin scatter of the")
    log("  form factor, i.e. how non-spherical the reference atom was.")
    return library


def build(args, log=print):
    """
    Ingest every named calculation and write the database.

    Parameters
    ----------
    args : argparse.Namespace
    log : callable, optional

    Returns
    -------
    AtomicReferenceLibrary
    """
    _section(log, "input")
    existing = None
    if args.append:
        existing = AtomicReferenceLibrary.load(args.output)
        if len(existing):
            log(f"  extending {args.output} ({len(existing)} entry/entries)")
    log(f"  reading isolated atoms from {len(args.directories)} path(s)")

    # One line per atom, written by the library as it ingests them: the block
    # is the record of what went in, and what was skipped and why.
    _section(log, "atoms")
    library = build_library(args.directories, filename=args.filename,
                            bins=args.bins, library=existing, log=log)

    if not len(library):
        log("  Nothing ingested. A reference needs a single-atom CHGCAR; "
            "check that\n  the paths hold one, and that the cells really do "
            "hold one atom each.")
        return library

    path = library.save(args.output)
    _section(log, "output files")
    log(f"  {len(library)} entry/entries -> {path}")
    log(f"  fingerprint {library.fingerprint}")

    _section(log, "usage")
    log("  data:")
    log("    delta_density: true")
    log(f"    atomic_reference: {path}")
    return library


def build_parser():
    parser = argparse.ArgumentParser(
        prog="poraque-atoms",
        description="Build the isolated-atom database used as the "
                    "delta-density baseline and as a fallback source of PAW "
                    "augmentation records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each reference must be ONE atom in a box, with a CHGCAR. A "
               "parent directory holding one subdirectory per element is "
               "expanded automatically.")
    parser.add_argument("directories", nargs="*",
                        help="isolated-atom calculation directories, or one "
                             "parent holding them")
    parser.add_argument("--output", default=LIBRARY_FILENAME,
                        help=f"database to write (default: {LIBRARY_FILENAME})")
    parser.add_argument("--filename", default="CHGCAR",
                        help="density file inside each directory "
                             "(default: CHGCAR)")
    parser.add_argument("--bins", type=int, default=None,
                        help="radial bins for the form-factor table "
                             "(default: 512)")
    parser.add_argument("--append", action="store_true",
                        help="extend the database at --output instead of "
                             "replacing it")
    parser.add_argument("--show", metavar="PATH", default=None,
                        help="print the contents of an existing database and "
                             "exit")
    return parser


def main(argv=None):
    """
    Console entry point for ``poraque-atoms``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())``.
    """
    log = Logger()
    log.banner()
    args = build_parser().parse_args(argv)

    if args.show:
        log.title("Isolated-atom database: contents")
        show(args.show, log)
        return 0

    if not args.directories:
        build_parser().error(
            "give at least one isolated-atom directory, or --show a database")

    log.title("Isolated-atom database: build")
    library = build(args, log)
    return 0 if len(library) else 1


if __name__ == "__main__":
    raise SystemExit(main())
