# -*- coding: utf-8 -*-
# file: aims.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
FHI-aims reader.

Unlike the VASP path this one is **implemented**, because ASE already parses
both file formats FHI-aims needs here — ``geometry.in`` and Gaussian cube — so
the reader is a units-and-conventions layer rather than a parser.

.. warning::
   :class:`AimsReader` reads FHI-aims correctly. Whether the result belongs in
   the *same dataset* as VASP data is a separate question, and the answer is
   usually **no**. See :ref:`the mismatch <aims-mismatch>` below before
   training on it.

Producing the fields
--------------------
FHI-aims writes no volumetric output unless asked. In ``control.in``::

    output cube total_density
    output cube elf                 # only if tau is wanted; see below
      cube origin   0.0 0.0 0.0
      cube edge 96  0.0568 0.0 0.0  # n_points and the step along each axis
      cube edge 96  0.0 0.0568 0.0
      cube edge 96  0.0 0.0 0.0568

The ``cube edge`` triple has to tile the cell exactly, or the field is not on
the lattice-periodic mesh everything downstream assumes.

Units, and a trap
-----------------
The cube format specifies atomic units, and FHI-aims does **not** use them by
default. From ``runtime_choices.f90``::

    cube_content_unit = 'legacy'   ! enabled by default. Causes output of
                                   ! cube files in the wrong units
                                   ! (1/A^3 instead of 1/bohr^3)

So the *same file* means :math:`e/\text{Å}^3` or :math:`e/a_0^3` depending on
a flag, the default is the non-standard one, and nothing in the file records
which. :attr:`AimsReader.content_unit` selects it; the reader also reads
``control.in`` when one sits beside the cube, which is the only way to get it
right without being told.

.. _aims-mismatch:

Why this is not simply another VASP
-----------------------------------
Two mismatches, and neither is about file formats.

**FHI-aims is all-electron.** Its ``total_density`` includes the core, while a
VASP ``CHGCAR`` is the *pseudo valence* density. Near a nucleus these differ by
orders of magnitude. Pooling them trains one operator on two different physical
quantities — the failure is silent, because both are smooth positive fields
with the right units. Poraquê's operators are trained on valence densities, so
an all-electron density is out of distribution for them.

FHI-aims does support pseudopotentials, and a run configured that way produces
a comparable valence density. That is the supported route into an existing
dataset; anything else should be its own dataset with its own model.

**There is no kinetic-energy-density cube.** ``read_control.f90`` accepts
``total_density``, ``delta_density``, ``spin_density``, ``eigenstate*``,
``hartree_potential``, ``xc_potential``, ``delta_v``, ``ion_dens``, ``elf``,
``stm``, ``dielec_func`` — and no :math:`\tau`. It can still be recovered,
because ELF is defined through it: with ``n_spin = 1`` FHI-aims computes

.. math::

    \mathrm{ELF} = \frac{t_1^2}{t_1^2 + t_2^2 + 10^{-6}}, \qquad
    t_1 = \rho\,\tau_{\rm TF}, \quad t_2 = \rho\,(\tau - \tau_{\rm vW}) ,

so :math:`\mathrm{ELF} \simeq 1/(1 + F^2)` with :math:`F` exactly the **Pauli
enhancement factor** Poraquê fits. Hence :func:`pauli_factor_from_elf`:

.. math::

    F = \sqrt{\frac{1 - \mathrm{ELF}}{\mathrm{ELF}}}, \qquad
    \tau = \tau_{\rm vW} + \tau_{\rm TF}\,F .

The sign is not lost — :math:`F \ge 0` by Hoffmann-Ostenhof. The *conditioning*
is: where ELF approaches 1 the numerator is a difference of nearly equal
numbers, so a fourth-digit error in ELF becomes a large relative error in
:math:`F`. That is precisely the von Weizsäcker limit, one of the two the
symbolic search is anchored on. Recovering :math:`\tau` this way is a real
option, not a good one; a meta-GGA run that reports :math:`\tau` directly is
better where it is available.
"""

import os

import numpy as np

from ..constants import BOHR_TO_ANGSTROM
from .base import CalculationParameters, CalculationReader

#: How ``cube_content_unit`` in ``control.in`` scales a density to e/Å³.
#:
#: ``legacy`` is FHI-aims' own default and already writes Å⁻³, despite the cube
#: format specifying atomic units; ``bohr`` writes the standard a₀⁻³.
CUBE_CONTENT_SCALE = {
    "legacy": 1.0,
    "bohr": 1.0 / BOHR_TO_ANGSTROM ** 3,
}


def read_content_unit(directory, default="legacy"):
    """
    ``cube_content_unit`` from a ``control.in``, when one is present.

    Parameters
    ----------
    directory : str
        Calculation directory.
    default : str, optional
        Returned when there is no ``control.in`` or it does not set the tag —
        ``"legacy"``, which is what FHI-aims itself defaults to.

    Returns
    -------
    str
        ``"legacy"`` or ``"bohr"``.
    """
    path = os.path.join(directory, "control.in")
    if not os.path.exists(path):
        return default
    with open(path, errors="replace") as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if line.lower().startswith("cube_content_unit"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].lower() in CUBE_CONTENT_SCALE:
                    return parts[1].lower()
    return default


def pauli_factor_from_elf(elf, floor=1e-8):
    r"""
    Pauli enhancement factor :math:`F` from an FHI-aims ELF field.

    Inverts :math:`\mathrm{ELF} = 1/(1 + F^2)`, the ``n_spin = 1`` Savin form
    FHI-aims writes.

    Parameters
    ----------
    elf : array_like
        ELF values in :math:`(0, 1]`.
    floor : float, optional
        ELF below this is treated as vacuum and yields ``0``. FHI-aims adds
        :math:`10^{-6}` to the denominator, so ELF tends to 0 where the density
        does and the inversion would otherwise diverge on empty space.

    Returns
    -------
    numpy.ndarray
        :math:`F \ge 0`.

    Notes
    -----
    Ill-conditioned as :math:`\mathrm{ELF} \to 1`: there :math:`1-\mathrm{ELF}`
    is a difference of nearly equal numbers, and that is the von Weizsäcker
    limit. Treat a recovered :math:`F` as indicative near 1.
    """
    values = np.clip(np.asarray(elf, dtype=float), 0.0, 1.0)
    safe = np.where(values > floor, values, 1.0)
    factor = np.sqrt(np.clip((1.0 - safe) / safe, 0.0, None))
    return np.where(values > floor, factor, 0.0)


class AimsReader(CalculationReader):
    """
    Ingest an FHI-aims calculation directory.

    Attributes
    ----------
    content_unit : str or None
        ``"legacy"``, ``"bohr"``, or ``None`` to read ``control.in`` and fall
        back to FHI-aims' own default. See :data:`CUBE_CONTENT_SCALE`.
    """

    code = "aims"
    structure_files = ("geometry.in", "geometry.in.next_step")
    field_files = {
        "external": "hartree_potential.cube",
        "density": "total_density.cube",
        "kinetic": "elf.cube",
    }

    content_unit = None

    # ------------------------------------------------------------------ #
    def read_structure(self, directory):
        """Geometry from ``geometry.in``, species-grouped and in Å."""
        import ase.io

        from ..structure import Structure

        return Structure.from_ase(
            ase.io.read(self.structure_path(directory), format="aims"))

    def read_parameters(self, directory):
        r"""
        Run settings, such as they are.

        FHI-aims uses numeric atom-centred orbitals, so there is no plane-wave
        cutoff to report and ``cutoff`` stays ``None``: the basis is chosen by
        the ``species_defaults`` level (``light``/``tight``/``really_tight``),
        which is recorded in ``extra`` rather than turned into an energy it is
        not.

        The grid comes from the cube itself when one exists — the ``cube edge``
        lines in ``control.in`` are the authority, and reading the header of
        the file they produced needs no reimplementation of that parsing.
        """
        shape, extra = None, {}
        density = self.field_path(directory, "density")
        if os.path.exists(density):
            shape = tuple(_cube_shape(density))

        control = os.path.join(directory, "control.in")
        if os.path.exists(control):
            with open(control, errors="replace") as handle:
                for line in handle:
                    line = line.split("#")[0].strip()
                    if line.lower().startswith("xc "):
                        extra["xc"] = line.split()[1]
                    elif line.lower().startswith("species_defaults"):
                        extra["species_defaults"] = line.split()[-1]
        extra["content_unit"] = self.content_unit or read_content_unit(directory)
        return CalculationParameters(cutoff=None, grid_shape=shape,
                                     xc=extra.get("xc"), extra=extra)

    def read_pseudopotentials(self, directory):
        """
        ``{}`` — FHI-aims is all-electron in its standard mode.

        The contract allows an empty mapping, and the caller must then be given
        valence charges explicitly. That is the honest answer here: there is no
        ``ZVAL`` to read, because there is no pseudopotential. A run that *does*
        use ``pseudopot`` species would need those parsed from ``control.in``;
        that is not implemented, and returning ``{}`` sends the caller to the
        explicit route rather than inventing a number.
        """
        return {}

    def read_field(self, path, field_class, grid=None):
        """
        Read one Gaussian cube, converted to Å and e/Å³.

        Parameters
        ----------
        path : str
            The ``.cube`` to read.
        field_class : type
            :class:`~poraque.fields.base.ScalarField` subclass to build.
        grid : FieldGrid, optional
            Shared grid to impose; a shape mismatch raises.

        Returns
        -------
        ScalarField
        """
        import ase.io

        from ..grid import FieldGrid
        from ..structure import Structure

        cube = ase.io.read(path, format="cube", read_data=True,
                           full_output=True)
        values = np.asarray(cube["data"], dtype=float)
        structure = Structure.from_ase(cube["atoms"])

        if grid is None:
            grid = FieldGrid(values.shape, cube["atoms"].cell.array)
        elif tuple(grid.shape) != values.shape:
            raise ValueError(
                f"{path}: cube shape {values.shape} does not match the "
                f"supplied shared grid {tuple(grid.shape)}. All fields of one "
                f"material must be defined on the same mesh."
            )

        unit = self.content_unit or read_content_unit(os.path.dirname(path))
        values = values * CUBE_CONTENT_SCALE[unit]

        # Built directly, not through `ScalarField.read`: that path applies
        # VASP's cell-volume convention, which a cube does not use.
        return field_class(values, grid, structure,
                           metadata={"source": str(path), "code": self.code,
                                     "cube_content_unit": unit})

    def write_field(self, field, path, comment=None):
        """Write a field as a Gaussian cube, in the units it is held in."""
        import ase.io

        ase.io.write(path, field.structure.to_ase(), data=np.asarray(field.data),
                     format="cube")
        return path


def _cube_shape(path):
    """``(N1, N2, N3)`` from a cube header, without reading the payload."""
    with open(path, errors="replace") as handle:
        handle.readline(), handle.readline()          # two comment lines
        natoms = int(handle.readline().split()[0])
        return [int(handle.readline().split()[0]) for _ in range(3)]
