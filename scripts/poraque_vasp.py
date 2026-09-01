#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_vasp.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Write the VASP inputs that turn a predicted density into a DFT observable.

.. code-block:: text

    predicted CHGCAR ---> INCAR (ICHARG = 11) + KPOINTS + POSCAR
                          |
                          |  you supply the POTCAR and run VASP
                          v
                          bands / DOS / total energy

This is the point of the whole exercise. ``ICHARG = 11`` reads a charge density
and **holds it fixed** through the entire electronic minimization, so whatever
comes out comes from a single non-self-consistent diagonalization of the
Hamiltonian built from *that* density. If a Poraquê prediction is good enough,
that is a band structure — or a density of states, or an energy — obtained
without an SCF cycle ever running.

The three deck modes
--------------------
``bands``
    Eigenvalues along a **path** through the Brillouin zone: a line-mode
    ``KPOINTS``, ``ISMEAR = 0``, ``LORBIT = 11`` for fat bands.

``dos``
    The same eigenvalues integrated over a **mesh**: an automatic Γ-centred
    ``KPOINTS``, ``ISMEAR = -5`` (tetrahedron with Blöchl corrections) and a
    generous ``NEDOS``. A DOS wants a mesh and a band structure wants a line;
    they are not interchangeable, which is the whole difference between these
    two modes.

``energy``
    The energy from that same mesh. Read the caveat below before quoting it.

And one mode that writes no deck
--------------------------------
``chgcar``
    The density itself, written out as a ``CHGCAR`` VASP can open --- with its
    magnetisation block and both sets of PAW augmentation records. A Poraquê
    field store (``fields.h5``) holds everything a ``CHGCAR`` does and no VASP
    build reads one, so this is the step between a ``poraque-mp --hdf5``
    download and any ``ICHARG = 1`` or ``ICHARG = 11`` run at all.

    Every mode accepts a store where it accepts a ``CHGCAR`` --- address one
    field as ``fields.h5::CHGCAR`` when the store holds several --- and
    ``--copy-density`` converts rather than copies, so a deck never ends up
    holding a density VASP cannot read.

What ``energy`` actually computes
---------------------------------
``ICHARG = 11`` prints a ``TOTEN``, but it is the Harris–Foulkes functional
evaluated at the input density — a first-order estimate, correct to second
order in the density error, **not** a variational SCF energy. That is the right
number when the claim is "no SCF cycle ran", and the deck says so in a comment
rather than leaving it to be discovered.

``--scf`` asks the other question instead: ``ICHARG = 1`` reads the prediction
as the *starting* density and converges from there. The energy is then an
ordinary variational one, and what the prediction bought is the iteration
count.

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

    # A band deck beside a prediction, cutoff taken from the source run
    poraque-vasp bands predicted/CHGCAR --like ~/Simulations/vasp/metals/Pt/2.unitcell \
        --output bands/

    # An explicit cutoff and a denser path
    poraque-vasp bands predicted/CHGCAR --encut 450 --points 60 --output bands/

    # A path of your own, as fractional k-points
    poraque-vasp bands predicted/CHGCAR --kpath "0,0,0  0.5,0,0.5  0.5,0.25,0.75" \
        --labels G X W --output bands/

    # A density of states on a mesh chosen from the cell
    poraque-vasp dos predicted/CHGCAR --like <run dir> --kspacing 0.20 --output dos/

    # ... or on a mesh of your own
    poraque-vasp dos predicted/CHGCAR --encut 450 --mesh 16 16 16 --output dos/

    # A total energy at the predicted density (no SCF cycle)
    poraque-vasp energy predicted/CHGCAR --like <run dir> --output energy/

    # ... or converged *from* it, which measures what the prediction saved
    poraque-vasp energy predicted/CHGCAR --like <run dir> --scf --output energy_scf/
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
    kpoint_mesh_from_spacing,
    write_band_structure_deck,
    write_chgcar,
    write_dos_deck,
    write_total_energy_deck,
)

#: The subcommands, in the order they appear in ``--help``.
MODES = ("bands", "dos", "energy", "chgcar")


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


def resolve_mesh(args, structure, log=print):
    """
    The k-point mesh for a ``dos`` or ``energy`` deck, and where it came from.

    ``--mesh`` wins; otherwise ``--kspacing`` is applied to the density's own
    cell by VASP's own ``KSPACING`` rule. A vacuum direction needs no special
    case: its reciprocal vector is short, so the rule returns 1 there.
    """
    if args.mesh:
        mesh = tuple(int(n) for n in args.mesh)
        log(f"    k-mesh     {mesh[0]}x{mesh[1]}x{mesh[2]} (from --mesh)")
        return mesh

    mesh = kpoint_mesh_from_spacing(structure, args.kspacing)
    log(f"    k-mesh     {mesh[0]}x{mesh[1]}x{mesh[2]} "
        f"(from --kspacing {args.kspacing:g} 1/Ang)")
    if min(mesh) == 1 and max(mesh) > 1:
        log("               a direction sampled at one point -- expected for "
            "a slab")
        log("               or a cluster, wrong for a bulk cell.")
    return mesh


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


def run_bands(args, info, encut, log=print):
    """Write the ``ICHARG = 11`` band-structure deck."""
    if args.kpath:
        path, labels = parse_kpath(args.kpath), (args.labels or None)
        origin = "--kpath"
    else:
        path, labels = fcc_band_path()
        origin = "the conventional FCC path G-X-W-K-G-L-U-W-L-K"

    log(f"\n  k-path     {origin}, {args.points} points per segment")
    if not args.kpath:
        log("             !! this is the FCC path. It is right for this "
            "project's cubic")
        log("                cells and wrong for any other lattice -- pass "
            "--kpath for those.")

    return write_band_structure_deck(
        args.output, chgcar=args.chgcar if args.copy_density else None,
        structure=info["structure"], kpath=path, labels=labels, encut=encut,
        points_per_segment=args.points, system=args.system, ispin=args.ispin,
        nbands=args.nbands)


def run_dos(args, info, encut, log=print):
    """Write the ``ICHARG = 11`` density-of-states deck."""
    log("")
    mesh = resolve_mesh(args, info["structure"], log)
    if args.ismear == -5 and mesh[0] * mesh[1] * mesh[2] < 4:
        log("               !! ISMEAR = -5 (tetrahedron) needs at least four")
        log("                  k-points. Densify the mesh, or pass --ismear 0.")
    log(f"    NEDOS      {args.nedos} energy points")

    return write_dos_deck(
        args.output, chgcar=args.chgcar if args.copy_density else None,
        structure=info["structure"], mesh=mesh, encut=encut,
        ispin=args.ispin, nedos=args.nedos, lorbit=args.lorbit,
        ismear=args.ismear, sigma=args.sigma, emin=args.emin, emax=args.emax,
        gamma=not args.monkhorst_pack, system=args.system)


def run_energy(args, info, encut, log=print):
    """Write the total-energy deck, non-self-consistent unless ``--scf``."""
    log("")
    mesh = resolve_mesh(args, info["structure"], log)
    if args.ismear == -5 and mesh[0] * mesh[1] * mesh[2] < 4:
        log("               !! ISMEAR = -5 (tetrahedron) needs at least four")
        log("                  k-points. Densify the mesh, or pass --ismear 0.")

    if args.scf:
        log("    ICHARG     1 -- the prediction seeds an SCF cycle and is")
        log("               converged from there. The energy is variational;")
        log("               what the prediction buys is the iteration count.")
    else:
        log("    ICHARG     11 -- one diagonalization at the predicted")
        log("               density. TOTEN is then the Harris-Foulkes")
        log("               functional at that density, NOT a variational")
        log("               SCF energy. Pass --scf for one that is.")

    return write_total_energy_deck(
        args.output, chgcar=args.chgcar if args.copy_density else None,
        structure=info["structure"], mesh=mesh, encut=encut,
        ispin=args.ispin, selfconsistent=args.scf, ismear=args.ismear,
        sigma=args.sigma, ediff=args.ediff,
        gamma=not args.monkhorst_pack, system=args.system)


def run_chgcar(args, info, encut, log=print):
    """Write a VASP-readable ``CHGCAR`` and no deck at all."""
    log("")
    written = write_chgcar(args.chgcar, args.output)
    log(f"    wrote      {written}")
    if info["augmentation_records"]:
        log(f"               with {info['augmentation_records']} augmentation "
            f"records, so ICHARG = 1 can start from it")
    else:
        log("               with no augmentation records: fine for "
            "ICHARG = 11,")
        log("               not enough for ICHARG = 1.")
    return {"CHGCAR": written}


#: Which writer each subcommand dispatches to.
_RUNNERS = {"bands": run_bands, "dos": run_dos, "energy": run_energy,
            "chgcar": run_chgcar}


def run(args, log=print):
    """
    Write the deck for whichever mode was asked for.

    Returns
    -------
    dict
        ``{name: path}`` of the files written.
    """
    info = describe_density(args.chgcar, log)

    # `chgcar` writes one file and no deck, so it asks none of the questions a
    # deck asks -- there is no ENCUT to reconcile and no POTCAR to warn about.
    if args.mode == "chgcar":
        return _RUNNERS[args.mode](args, info, None, log)

    encut = resolve_encut(args, log)

    written = _RUNNERS[args.mode](args, info, encut, log)

    log(f"\n  wrote {len(written)} file(s) to {args.output}/")
    for name in sorted(written):
        log(f"    {name}")

    log("\n  Still needed before this runs:")
    if not args.copy_density:
        log("    CHGCAR   copy the predicted density in (or use --copy-density)")
    log("    POTCAR   the same pseudopotentials the density was built with")
    log(f"\n  Then: run VASP, and read the {_OUTPUTS[args.mode]}.")
    return written


#: What to look at once the run finishes, per mode.
_OUTPUTS = {
    "bands": "bands from EIGENVAL or vasprun.xml",
    "dos": "density of states from DOSCAR or vasprun.xml",
    "energy": "energy from OSZICAR or OUTCAR",
}


def _add_common(parser, default_output):
    """The options every mode shares: the density, the cutoff, the output."""
    parser.add_argument("chgcar",
                        help="the predicted (or reference) CHGCAR")
    parser.add_argument("--output", default=default_output,
                        help=f"directory to write the deck into "
                             f"(default: {default_output})")
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
    parser.add_argument("--ispin", type=int, default=1, choices=(1, 2),
                        help="ISPIN (default: 1)")
    parser.add_argument("--system", default="poraque",
                        help="SYSTEM label for the INCAR")
    parser.add_argument("--copy-density", action="store_true",
                        help="copy the CHGCAR into the output directory too")
    return parser


def _add_mesh_options(parser, kspacing):
    """The k-mesh options the ``dos`` and ``energy`` modes share."""
    parser.add_argument("--mesh", nargs=3, type=int, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="explicit k-point subdivisions; overrides "
                             "--kspacing")
    parser.add_argument("--kspacing", type=float, default=kspacing,
                        help=f"target k-point spacing in 1/Ang, applied to "
                             f"the density's own cell by VASP's own KSPACING "
                             f"rule (default: {kspacing:g})")
    parser.add_argument("--monkhorst-pack", action="store_true",
                        help="a Monkhorst-Pack mesh instead of a Gamma-centred "
                             "one")
    parser.add_argument("--ismear", type=int, default=-5,
                        help="ISMEAR; -5 is the tetrahedron method and needs "
                             "at least four k-points (default: -5)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="SIGMA, used when ISMEAR >= 0 (default: 0.05)")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="poraque-vasp",
        description="Write the VASP inputs that read a predicted charge "
                    "density back: a band structure, a density of states or "
                    "a total energy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="No VASP run happens here, and no POTCAR is written.")
    modes = parser.add_subparsers(dest="mode",
                                  metavar="{bands,dos,energy,chgcar}")

    bands = modes.add_parser(
        "bands", help="ICHARG = 11 band structure along a k-path",
        description="Write the ICHARG = 11 inputs for a non-self-consistent "
                    "band structure from a predicted charge density.")
    _add_common(bands, "bands")
    bands.add_argument("--points", type=int, default=40,
                       help="k-points per path segment (default: 40)")
    bands.add_argument("--kpath", default=None,
                       help='fractional k-points, e.g. '
                            '"0,0,0 0.5,0,0.5 0.5,0.25,0.75"; defaults to '
                            'the conventional FCC path')
    bands.add_argument("--labels", nargs="*", default=None,
                       help="high-symmetry labels, one per --kpath point")
    bands.add_argument("--nbands", type=int, default=None,
                       help="NBANDS; worth setting generously for a band "
                            "structure")

    dos = modes.add_parser(
        "dos", help="ICHARG = 11 density of states on a k-mesh",
        description="Write the ICHARG = 11 inputs for a non-self-consistent "
                    "density of states from a predicted charge density.")
    _add_common(dos, "dos")
    _add_mesh_options(dos, 0.25)
    dos.add_argument("--nedos", type=int, default=3001,
                     help="energy points written to DOSCAR (default: 3001; "
                          "VASP's own default of 301 hides structure)")
    dos.add_argument("--lorbit", type=int, default=11,
                     help="LORBIT; 11 gives the site- and l-projected DOS "
                          "(default: 11)")
    dos.add_argument("--emin", type=float, default=None,
                     help="EMIN, the bottom of the DOS window")
    dos.add_argument("--emax", type=float, default=None,
                     help="EMAX, the top of the DOS window")

    energy = modes.add_parser(
        "energy", help="total energy at (or converged from) the density",
        description="Write the inputs for a total energy from a predicted "
                    "charge density. Non-self-consistent by default, in which "
                    "case TOTEN is the Harris-Foulkes functional at that "
                    "density rather than a variational SCF energy.")
    _add_common(energy, "energy")
    _add_mesh_options(energy, 0.30)
    energy.add_argument("--scf", action="store_true",
                        help="ICHARG = 1: read the prediction as the starting "
                             "density and converge from it, for a variational "
                             "energy and an iteration count")
    energy.add_argument("--ediff", type=float, default=1e-6,
                        help="EDIFF, electronic convergence (default: 1e-6)")

    chgcar = modes.add_parser(
        "chgcar", help="write a VASP-readable CHGCAR from a field store",
        description="Write a density out as a CHGCAR that VASP can open, "
                    "carrying its magnetisation block and its PAW "
                    "augmentation records. No deck is written; this is the "
                    "step between a `poraque-mp --hdf5` store and any "
                    "ICHARG = 1 or ICHARG = 11 run, since no VASP build "
                    "reads HDF5.")
    chgcar.add_argument("chgcar",
                        help="the density: a field store (fields.h5, or "
                             "fields.h5::CHGCAR) or a CHGCAR")
    chgcar.add_argument("--output", default="CHGCAR",
                        help="the file to write (default: CHGCAR)")

    return parser


def _normalise(argv):
    """
    Let the pre-rename invocation keep working.

    ``poraque-bands predicted/CHGCAR ...`` became ``poraque-vasp bands
    predicted/CHGCAR ...``. A first argument that is not a mode but is a path
    is read as the old form and gets ``bands`` inserted, because the
    alternative is a stack trace for a command that worked last week.
    """
    argv = list(argv)
    first = next((token for token in argv if not token.startswith("-")), None)
    if first is not None and first not in MODES and os.path.exists(first):
        return ["bands"] + argv, True
    return argv, False


def main(argv=None):
    """
    Console entry point for ``poraque-vasp``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())``.
    """
    banner()
    argv, implied = _normalise(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode is None:
        parser.print_help()
        return 2
    if implied:
        print("\n  note: no mode given, assuming `bands` (this is the "
              "poraque-bands\n        spelling; poraque-vasp bands ... is the "
              "current one).")
    # A density may be addressed inside a store as `fields.h5::CHGCAR`, which
    # is a path plus a selector and not a filename anything can stat.
    from poraque.fields.hdf5 import split_target

    if not os.path.exists(split_target(args.chgcar)[0]):
        print(f"\n  {args.chgcar}: no such file.")
        return 1
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
