# -*- coding: utf-8 -*-
# file: espresso.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Quantum ESPRESSO reader — **skeleton**.

The class below fixes the file names, unit conversions and structural mapping
so that implementing it is a matter of filling in four method bodies, with no
changes anywhere else in Poraquê. It raises :class:`NotImplementedError` with
an actionable message until then.

Mapping from the neutral contract to Quantum ESPRESSO
-----------------------------------------------------

**Geometry** — from ``pw.x`` input (``&SYSTEM`` + ``ATOMIC_POSITIONS`` +
``CELL_PARAMETERS``) or, more robustly, from ``<prefix>.xml`` in the output
directory, which always carries the final cell in a single form. Watch for:

* ``CELL_PARAMETERS`` units: ``angstrom``, ``bohr`` or ``alat`` (multiply by
  ``celldm(1)``, itself in Bohr). :class:`Structure` wants **Ångström**.
* ``ATOMIC_POSITIONS`` units: ``crystal`` (already fractional), ``angstrom``,
  ``bohr``, or ``alat``.
* QE does **not** require atoms to be grouped by species, but
  :class:`Structure` does. Sort by species and keep the permutation if atom
  identity must be traced back.
* ``ibrav != 0`` builds the cell from ``celldm``/``A,B,C,cosAB,...`` rather
  than from ``CELL_PARAMETERS``; the XML output sidesteps this entirely and is
  the recommended source.

**Parameters** — ``ecutwfc`` is in **Rydberg**; multiply by
:data:`RY_TO_EV` for :attr:`CalculationParameters.cutoff`. ``ecutrho``
(default ``4 * ecutwfc``) is the density cutoff and is the direct analogue of
VASP's ``PREC``: the FFT grid is sized from it, so prefer
``grid_shape`` derived from ``ecutrho`` — or better, read ``nr1, nr2, nr3``
straight from the XML, which removes all guesswork.

**Pseudopotentials** — UPF files named in ``ATOMIC_SPECIES``, found under
``pseudo_dir``. Parse the header for ``z_valence`` (the required field) and
``rcloc``/``rcut`` for the core radius, converting **Bohr → Ångström**.
The UPF header is plain XML-ish text, so ``z_valence`` can be pulled out
without a full parser.

**Fields** — QE does not write CHGCAR-like files directly. Two viable routes:

1. ``pp.x`` post-processing with ``plot_num=0`` (charge density) and
   ``output_format=6`` (Gaussian cube) or ``=3`` (XCrySDen XSF). Cube files are
   the easiest neutral target and are also what GPAW can emit, so a shared
   :class:`~poraque.fields.io.base.CalculationReader` helper for cube I/O would
   serve both codes.
2. Read ``charge-density.dat``/``.hdf5`` from the output directory directly.
   This is the native G-space representation on the ``nr1 x nr2 x nr3`` grid;
   an inverse FFT gives the real-space field with no post-processing step.
   Faster and lossless, but the binary layout is version-dependent.

**Units of the fields themselves.** QE's charge density is in
:math:`e/\mathrm{Bohr}^3` and potentials from ``pp.x`` are in **Rydberg**, not
eV. Convert in :meth:`read_field` so that everything downstream keeps the
Å/eV convention of :mod:`poraque.fields`. There is no volume pre-factor, unlike
VASP's ``CHGCAR``, so :attr:`ScalarField.volume_scaled` must be ``False`` for
QE-sourced fields — subclass the field classes or pass the flag explicitly
rather than silently reusing the VASP convention.

**Kinetic energy density** — available via ``pp.x`` ``plot_num=6`` only for
meta-GGA runs; otherwise it must come from the wavefunctions.
"""

from .base import CalculationReader

#: Rydberg in electronvolt — QE's ``ecutwfc``/``ecutrho`` and ``pp.x``
#: potentials are in Ry, while Poraquê works in eV.
RY_TO_EV = 13.605693122994


class EspressoReader(CalculationReader):
    """
    Ingest a Quantum ESPRESSO calculation directory.

    .. warning::
       Not implemented yet. See the module docstring for the complete mapping
       from QE's files and units onto the neutral contract; each method below
       names exactly what it must produce.
    """

    code = "espresso"
    structure_files = ("pw.in", "scf.in", "espresso.pwi")
    field_files = {
        "external": "extpot.cube",
        "density": "charge-density.cube",
        "kinetic": "kinetic-density.cube",
    }

    def read_structure(self, directory):
        """Build a :class:`Structure` in Å with species-grouped atoms."""
        raise NotImplementedError(
            "EspressoReader.read_structure: parse CELL_PARAMETERS / "
            "ATOMIC_POSITIONS (or <prefix>.xml), convert to Angstrom and "
            "fractional coordinates, then sort atoms by species."
        )

    def read_parameters(self, directory):
        """Return :class:`CalculationParameters` with the cutoff in **eV**."""
        raise NotImplementedError(
            "EspressoReader.read_parameters: read ecutwfc (Rydberg) and "
            f"multiply by RY_TO_EV = {RY_TO_EV}; take grid_shape from nr1/nr2/nr3 "
            "in the XML when available, else derive it from ecutrho."
        )

    def read_pseudopotentials(self, directory):
        """Return ``{element: PseudopotentialInfo}`` from the UPF files."""
        raise NotImplementedError(
            "EspressoReader.read_pseudopotentials: resolve ATOMIC_SPECIES "
            "against pseudo_dir, parse z_valence from each UPF header, and "
            "convert radii Bohr -> Angstrom."
        )

    def read_field(self, path, field_class, grid=None):
        """Read a cube/XSF field, converting to Å and eV."""
        raise NotImplementedError(
            "EspressoReader.read_field: parse the cube written by pp.x, "
            "convert Bohr -> Angstrom and Rydberg -> eV, and note that QE "
            "stores no cell-volume pre-factor (volume_scaled=False)."
        )

    def write_field(self, field, path, comment=None):
        """Write a field in Gaussian cube format."""
        raise NotImplementedError(
            "EspressoReader.write_field: emit Gaussian cube format."
        )
