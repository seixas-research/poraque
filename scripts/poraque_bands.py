#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_bands.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Write the ``ICHARG = 11`` inputs that turn a predicted density into a band structure.

.. code-block:: text

    predicted CHGCAR ---> INCAR (ICHARG = 11) + KPOINTS (line mode) + POSCAR
                          |
                          |  you supply the POTCAR and run VASP
                          v
                          EIGENVAL / vasprun.xml -> bands

This is the point of the whole exercise. ``ICHARG = 11`` reads a charge density
and **holds it fixed** through the entire electronic minimization, so the
eigenvalues come from a single non-self-consistent diagonalization of the
Hamiltonian built from *that* density. If a Poraquê prediction is good enough,
that is a band structure obtained without an SCF cycle ever running.

What this command does **not** do
---------------------------------
It does not run VASP, and it does not write a ``POTCAR`` --- those cannot be
redistributed. What it produces is the deck that is otherwise retyped by hand
every time, with the two tags that are easiest to get wrong carried as comments:

``ENCUT`` **must match the run the density's FFT grid came from.** A different
cutoff means a different grid, and VASP will not read a ``CHGCAR`` whose grid
does not match the one it wants. The value is read off the source calculation
when ``--like`` points at one.

``LMAXMIX`` must match whatever wrote the augmentation occupancies, or the
one-centre terms are read into the wrong angular-momentum channels.

Checking the density first
--------------------------
Before writing anything the density is inspected and reported: its grid, its
electron count, and **whether it carries PAW augmentation records**. A
``CHGCAR`` without them is still readable, but the wiki is explicit that
"restarting calculations without one-center PAW occupancy matrices up to the
appropriate l-quantum number leads to loss of information" --- so a missing
block is called out rather than left to be discovered from the eigenvalues.
Use ``poraque-inference --add-paw`` to produce one that has them.

Usage
-----
.. code-block:: bash

    # A deck beside a prediction, with the cutoff taken from the source run
    poraque-bands predicted/CHGCAR --like data/vasp/struct_015 --output bands/

    # An explicit cutoff and a denser path
    poraque-bands predicted/CHGCAR --encut 450 --points 60 --output bands/

    # A path of your own, as fractional k-points
    poraque-bands predicted/CHGCAR --kpath "0,0,0  0.5,0,0.5  0.5,0.25,0.75" \
        --labels G X W --output bands/
"""

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from poraque.environment import banner                        # noqa: E402
from poraque.fields.vasp.templates import (                   # noqa: E402
    fcc_band_path,
    write_band_structure_deck,
)


def describe_density(path, log=print):
    """
    Report what the density is, and whether it can seed ``ICHARG = 11``.

    Parameters
    ----------
    path : str
        A ``CHGCAR``-format file.
    log : callable, optional

    Returns
    -------
    dict
        ``structure``, ``grid``, ``electrons``, ``augmentation_records``.
    """
    from poraque.fields import ChargeDensity, FieldGrid
    from poraque.fields.vasp.volumetric import (
        count_augmentation_records,
        read_augmentation,
    )

    grid = FieldGrid.from_file(path)
    density = ChargeDensity.read(path, grid=grid)

    try:
        _, block = read_augmentation(path)
        records = count_augmentation_records(block)
    except (OSError, ValueError):
        records = 0

    natoms = len(density.structure.scaled_positions)
    log(f"\n  density  {path}")
    log(f"    grid       {'x'.join(str(n) for n in grid.shape)}"
        f"   volume {grid.volume:.3f} Ang^3")
    log(f"    structure  {natoms} atoms, {density.structure.elements}")
    log(f"    electrons  {density.integrate():.6f}")

    if records == natoms:
        log(f"    PAW        {records} augmentation records (one per atom)")
    elif records:
        log(f"    PAW        !! {records} augmentation records for {natoms} "
            f"atoms -- VASP expects one each, and will not read this file")
    else:
        log("    PAW        none. VASP will read the grid, but the one-centre")
        log("               terms are lost -- rerun poraque-inference with")
        log("               --add-paw for a density that can seed ICHARG=11")
        log("               without discarding them.")

    return {"structure": density.structure, "grid": grid,
            "electrons": float(density.integrate()),
            "augmentation_records": records}


def resolve_encut(args, log=print):
    """
    The cutoff for the deck, and where it came from.

    Precedence: ``--encut``, then the ``INCAR`` of ``--like``, then a default
    that is announced loudly. Getting this wrong is the single commonest way an
    ``ICHARG = 11`` run fails to read its own density, so silence is not an
    option here.
    """
    if args.encut:
        log(f"    ENCUT      {args.encut:g} eV (from --encut)")
        return float(args.encut)

    if args.like:
        from poraque.fields.vasp.incar import Incar

        candidate = (args.like if os.path.isfile(args.like)
                     else os.path.join(args.like, "INCAR"))
        if os.path.exists(candidate):
            encut = Incar.from_file(candidate).encut
            if encut:
                log(f"    ENCUT      {encut:g} eV (from {candidate})")
                return float(encut)
        log(f"    ENCUT      !! no ENCUT found in {args.like}")

    log(f"    ENCUT      {args.default_encut:g} eV -- A GUESS. This MUST match")
    log("               the run the density's FFT grid came from, or VASP")
    log("               will refuse the CHGCAR. Pass --encut or --like.")
    return float(args.default_encut)


def parse_kpath(text):
    """
    ``"0,0,0  0.5,0,0.5  ..."`` to a continuous path of fractional k-points.

    Returns
    -------
    list of tuple
    """
    points = []
    for chunk in text.replace(";", " ").split():
        values = [float(v) for v in chunk.split(",")]
        if len(values) != 3:
            raise ValueError(
                f"{chunk!r} is not a k-point: expected three comma-separated "
                f"fractional coordinates, e.g. 0.5,0.25,0.75.")
        points.append(tuple(values))
    if len(points) < 2:
        raise ValueError("A band path needs at least two k-points.")
    return points


def run(args, log=print):
    """
    Write the deck.

    Returns
    -------
    dict
        ``{name: path}`` of the files written.
    """
    info = describe_density(args.chgcar, log)
    encut = resolve_encut(args, log)

    if args.kpath:
        path, labels = parse_kpath(args.kpath), (args.labels or None)
        origin = "--kpath"
    else:
        path, labels = fcc_band_path()
        origin = "the conventional FCC path G-X-W-K-G-L-U-W-L-K"

    log(f"\n  k-path     {origin}, {args.points} points per segment")
    if not args.kpath:
        log("             !! this is the FCC path. It is right for this "
            "project's gold")
        log("                cells and wrong for any other lattice -- pass "
            "--kpath for those.")

    written = write_band_structure_deck(
        args.output, chgcar=args.chgcar if args.copy_density else None,
        structure=info["structure"], kpath=path, labels=labels, encut=encut,
        points_per_segment=args.points, system=args.system,
        nbands=args.nbands)

    log(f"\n  wrote {len(written)} file(s) to {args.output}/")
    for name in sorted(written):
        log(f"    {name}")

    log("\n  Still needed before this runs:")
    if not args.copy_density:
        log("    CHGCAR   copy the predicted density in (or use --copy-density)")
    log("    POTCAR   the same pseudopotentials the density was built with")
    log("\n  Then: run VASP, and read the bands from EIGENVAL or vasprun.xml.")
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        prog="poraque-bands",
        description="Write the ICHARG = 11 inputs for a non-self-consistent "
                    "band structure from a predicted charge density.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="No VASP run happens here, and no POTCAR is written.")
    parser.add_argument("chgcar",
                        help="the predicted (or reference) CHGCAR")
    parser.add_argument("--output", default="bands",
                        help="directory to write the deck into "
                             "(default: bands)")
    parser.add_argument("--like", default=None,
                        help="a calculation directory (or INCAR) to take "
                             "ENCUT from -- normally the run the density's "
                             "grid came from")
    parser.add_argument("--encut", type=float, default=None,
                        help="plane-wave cutoff in eV; MUST match the "
                             "density's own grid")
    parser.add_argument("--default-encut", type=float, default=450.0,
                        help="fallback cutoff when neither --encut nor --like "
                             "supplies one (default: 450)")
    parser.add_argument("--points", type=int, default=40,
                        help="k-points per path segment (default: 40)")
    parser.add_argument("--kpath", default=None,
                        help='fractional k-points, e.g. '
                             '"0,0,0 0.5,0,0.5 0.5,0.25,0.75"; defaults to '
                             'the conventional FCC path')
    parser.add_argument("--labels", nargs="*", default=None,
                        help="high-symmetry labels, one per --kpath point")
    parser.add_argument("--nbands", type=int, default=None,
                        help="NBANDS; worth setting generously for a band "
                             "structure")
    parser.add_argument("--system", default="poraque",
                        help="SYSTEM label for the INCAR")
    parser.add_argument("--copy-density", action="store_true",
                        help="copy the CHGCAR into the output directory too")
    return parser


def main(argv=None):
    """
    Console entry point for ``poraque-bands``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())``.
    """
    banner()
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.chgcar):
        print(f"\n  {args.chgcar}: no such file.")
        return 1
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
