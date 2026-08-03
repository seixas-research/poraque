# -*- coding: utf-8 -*-
# file: gpaw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
GPAW reader — **skeleton**.

GPAW is the easiest of the three to support, because it already speaks
Poraquê's units and its state lives in a single restart file rather than
scattered across text inputs.

Mapping from the neutral contract to GPAW
-----------------------------------------

**Everything comes from one ``.gpw`` file.** ``gpaw.GPAW(path)`` restores a
calculator carrying geometry, parameters, setups and densities, so all four
reader methods are thin accessors on one object. Cache the calculator per
directory rather than reopening it in each method.

**Geometry** — ``calc.get_atoms()`` returns an :class:`ase.Atoms` in
**Ångström**, and :meth:`Structure.from_ase` already handles the conversion
*and* the species grouping. This method is effectively two lines.

**Parameters** — GPAW has no single "cutoff" in real-space or LCAO mode:

* ``mode='pw'`` — ``calc.parameters.mode.ecut`` is already in **eV**, so it
  maps straight onto :attr:`CalculationParameters.cutoff`.
* ``mode='fd'`` — there is no cutoff, only a grid spacing ``h`` (Å). Leave
  ``cutoff=None`` and set ``grid_shape`` from ``calc.density.gd.N_c``; a
  spacing can be converted to an equivalent cutoff via
  :math:`E_{\rm cut} = \tfrac{1}{2}(\pi/h)^2` in atomic units if one is needed.
* ``mode='lcao'`` — same as ``fd`` for grid purposes.

In all three cases ``calc.density.gd.N_c`` (or ``finegd.N_c``) gives the exact
grid, so :attr:`CalculationParameters.grid_shape` should always be populated
and the cutoff-derived estimate never used.

**Pseudopotentials** — ``calc.setups`` holds one setup per atom;
``setup.Nv`` is the valence electron count (the ``ZVAL`` analogue) and
``setup.rcut_j`` the projector radii in **Bohr**. Deduplicate by element,
convert radii to Ångström.

**Fields**

* Density — ``calc.get_pseudo_density()`` (valence pseudo-density, the direct
  analogue of VASP's ``CHGCAR``) or ``calc.get_all_electron_density()`` for
  the PAW-reconstructed one. Both are in :math:`e/\text{Å}^3` **already**, so
  no conversion and no volume pre-factor: ``volume_scaled=False``.
  Be explicit about which of the two is used — mixing pseudo and all-electron
  densities across a dataset would be a silent physics error, since they differ
  by orders of magnitude near the nuclei.
* External potential — ``calc.get_electrostatic_potential()`` returns
  :math:`v_H + v_{\rm ext}` in eV, i.e. the ``LOCPOT`` analogue, not the bare
  ionic term. To obtain the ionic part alone, subtract the Hartree potential of
  the density (:func:`poraque.ml.physics.hartree_potential` computes it
  exactly), or generate it with
  :meth:`~poraque.fields.ExternalPotential.from_calculation` from the geometry
  and ``setup.Nv``.
* Kinetic energy density — not exposed directly; build it from
  ``calc.get_pseudo_wave_function()`` as
  :math:`\tau = \tfrac12\sum_i f_i |\nabla\psi_i|^2` using the spectral
  gradient, or use a meta-GGA run.

**Grid caveat.** GPAW distinguishes the coarse grid (``gd``) from the fine grid
(``finegd``, normally 2× denser) and the density lives on the fine one. Read
the shape from whichever grid the field being loaded actually uses, or fields
of one material will silently disagree.
"""

from .base import CalculationReader


class GpawReader(CalculationReader):
    """
    Ingest a GPAW calculation.

    .. warning::
       Not implemented yet. See the module docstring: every method is a thin
       accessor on a restored ``GPAW`` calculator, and GPAW's native units
       (Å, eV) already match Poraquê's.
    """

    code = "gpaw"
    structure_files = ("gpaw.gpw", "restart.gpw")
    field_files = {
        "external": "external.cube",
        "density": "density.cube",
        "kinetic": "kinetic.cube",
    }

    def read_structure(self, directory):
        """``Structure.from_ase(calc.get_atoms())``."""
        raise NotImplementedError(
            "GpawReader.read_structure: restore the .gpw with gpaw.GPAW(path) "
            "and pass calc.get_atoms() to Structure.from_ase (units already Angstrom)."
        )

    def read_parameters(self, directory):
        """Cutoff in eV for PW mode; ``grid_shape`` always from ``gd.N_c``."""
        raise NotImplementedError(
            "GpawReader.read_parameters: use calc.parameters.mode.ecut (already "
            "eV) for mode='pw', and always populate grid_shape from "
            "calc.density.finegd.N_c so no cutoff heuristic is needed."
        )

    def read_pseudopotentials(self, directory):
        """Valence charges from ``setup.Nv``, radii from ``setup.rcut_j``."""
        raise NotImplementedError(
            "GpawReader.read_pseudopotentials: iterate calc.setups, take Nv as "
            "the valence charge and rcut_j (Bohr -> Angstrom) as the core radius, "
            "deduplicating by element."
        )

    def read_field(self, path, field_class, grid=None):
        """Read a field from the calculator or a cube file."""
        raise NotImplementedError(
            "GpawReader.read_field: use calc.get_pseudo_density() (e/Ang^3, "
            "volume_scaled=False) and be explicit about pseudo vs all-electron."
        )

    def write_field(self, field, path, comment=None):
        """Write a field in Gaussian cube format."""
        raise NotImplementedError(
            "GpawReader.write_field: emit Gaussian cube format (ase.io.write "
            "supports it)."
        )
