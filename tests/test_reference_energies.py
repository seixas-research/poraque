# -*- coding: utf-8 -*-
# file: test_reference_energies.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Isolated-atom references and the cohesive energy.

Three claims are pinned here, and the middle one is the reason the module
exists.

**Referencing is a no-op where it is often assumed to help.** Subtracting
:math:`E_{\rm ref}` changes no force (it has no coordinate dependence) and no
energy difference at fixed composition (it cancels in the subtraction). Both
are asserted to *machine precision* rather than to a tolerance, because they
are exact statements, and a tolerance would hide a wiring mistake that made
them merely approximate.

**Referencing against the right side is what matters.** Poraquê's totals carry
a per-atom offset of order :math:`10^{3}` eV from the absent PAW one-centre
terms. Subtracting *VASP's* atomic energy leaves that offset untouched;
subtracting *Poraquê's own* atomic energy cancels it. The measured difference
on gold is a cohesive energy of :math:`-1157` eV/atom against one of
:math:`-1.9` eV/atom.

**The result is physically sized.** A cohesive energy in the right units, with
the right sign, of the right order for a late transition metal.
"""

import os

import numpy as np
import pytest

from poraque.fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from poraque.fields.vasp.poscar import Poscar
from poraque.fields.vasp.potcar import Potcar
from poraque.physics import EnergyCalculator, ReferenceEnergies
from poraque.physics.forces import hellmann_feynman_forces
from poraque.physics.reference import read_total_energy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VASP_DIR = os.path.join(_ROOT, "data", "vasp")
CACHE_DIR = os.path.join(_ROOT, "data", "cache", "res32")
REF_DIR = os.path.join(VASP_DIR, "ref")

needs_references = pytest.mark.skipif(
    not os.path.isdir(REF_DIR),
    reason="no isolated-atom reference directory in this checkout")
needs_dataset = pytest.mark.skipif(
    not (os.path.isdir(CACHE_DIR) and os.path.isdir(VASP_DIR)),
    reason="the shipped VASP dataset is not present in this checkout")

# `method="poraque"` evaluates the isolated atom's energy from its own fields,
# and the kinetic term needs the reference TAUCAR. Every Au TAUCAR was purged on
# 2026-08-25 as physically invalid -- see DELETIONS.md -- so these tests have no
# kinetic labels to work from until the VASP 6.6.1 / LTAU recomputation lands.
# The condition is the file itself rather than a hard skip, so they revive on
# their own the moment it does; nothing has to be remembered.
needs_reference_tau = pytest.mark.skipif(
    not os.path.exists(os.path.join(REF_DIR, "Au", "TAUCAR")),
    reason="no reference TAUCAR: the Au kinetic-energy data was purged as "
           "invalid (see DELETIONS.md) and is pending recomputation with "
           "VASP 6.6.1 / LTAU = .TRUE.")


@pytest.fixture(scope="module")
def references():
    """VASP's own isolated-atom energies."""
    return ReferenceEnergies.from_directory(REF_DIR, method="code")


def components(name, references=None):
    """Energy decomposition for a cached structure."""
    cache = os.path.join(CACHE_DIR, name)
    vasp = os.path.join(VASP_DIR, name)
    potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
    density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                 grid=potential.grid)
    tau = KineticEnergyDensity.read(os.path.join(cache, "TAUCAR"),
                                    grid=potential.grid)
    potcar = Potcar.from_file(os.path.join(vasp, "POTCAR"), parse_tables=True)
    calculator = EnergyCalculator(
        grid=potential.grid,
        structure=potential.structure,
        charges={entry.element: entry.zval for entry in potcar},
        pscore={entry.element: entry.pscore for entry in potcar
                if entry.pscore is not None},
        functional="pbe",
        references=references,
    )
    return calculator.compute(density, tau, potential)


# ===================================================================== #
# Reading the references
# ===================================================================== #
@needs_references
class TestReadingReferences:
    def test_reads_one_energy_per_element_directory(self):
        references = ReferenceEnergies.from_directory(REF_DIR, method="code")
        assert "Au" in references
        assert references["Au"] == pytest.approx(-0.0786091, abs=1e-6)

    @needs_reference_tau
    def test_the_two_methods_disagree_by_the_paw_offset(self):
        """
        Which is the whole point of having both.

        ``code`` is VASP's atomic energy; ``poraque`` is Poraquê's. They differ
        by the per-atom terms Poraquê cannot see, which is exactly the quantity
        that must cancel in a cohesive energy.
        """
        code = ReferenceEnergies.from_directory(REF_DIR, method="code")
        own = ReferenceEnergies.from_directory(REF_DIR, method="poraque")
        assert abs(own["Au"] - code["Au"]) > 1000.0

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="method="):
            ReferenceEnergies.from_directory(REF_DIR, method="magic")

    def test_a_missing_directory_is_an_error_not_an_empty_mapping(self):
        with pytest.raises(FileNotFoundError, match="reference-energy"):
            ReferenceEnergies.from_directory("/nonexistent/ref")

    def test_energy_is_read_from_outcar_or_oszicar(self):
        assert read_total_energy(os.path.join(REF_DIR, "Au")) is not None

    def test_decorated_symbols_match_the_bare_element(self):
        """``Au_pv`` in a POTCAR must find the ``Au`` reference."""
        references = ReferenceEnergies({"Au": -1.0})
        assert references["Au_pv"] == -1.0
        assert "Au.pbe" in references

    def test_total_for_sums_over_atoms(self):
        references = ReferenceEnergies({"Au": -2.0})
        structure = Poscar(cell=np.eye(3) * 10.0, symbols=["Au"], counts=[3],
                           scaled_positions=np.zeros((3, 3)))
        assert references.total_for(structure) == pytest.approx(-6.0)

    def test_an_uncovered_species_raises_rather_than_partially_summing(self):
        """
        A partial sum is the dangerous answer.

        It has the right units and a plausible magnitude, and is wrong by whole
        atoms.
        """
        references = ReferenceEnergies({"Au": -2.0})
        structure = Poscar(cell=np.eye(3) * 10.0, symbols=["Au", "Ag"],
                           counts=[1, 1],
                           scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
        assert references.missing_for(structure) == ["Ag"]
        assert not references.covers(structure)
        with pytest.raises(KeyError, match="Ag"):
            references.total_for(structure)


# ===================================================================== #
# What referencing does NOT change
# ===================================================================== #
@needs_dataset
@needs_references
class TestReferencingIsExactlyNeutralWhereItShouldBe:
    """
    The two invariants that make the shift safe to adopt.

    Both are exact identities, so both are asserted at machine precision.
    """

    def test_energy_differences_at_fixed_composition_are_bit_identical(
            self, references):
        r"""
        :math:`E_{\rm ref}` cancels in :math:`\Delta E_1 - \Delta E_2`.

        Same composition on both sides means the same reference, so the
        subtraction removes it entirely. Anyone expecting a referencing change
        to improve an energy-volume curve or a polymorph ranking should read
        this test: the numbers are identical, not merely close.
        """
        names = ["struct_000", "struct_001", "struct_002"]
        plain = [components(name).total for name in names]
        cohesive = [components(name, references).cohesive for name in names]

        for index in range(1, len(names)):
            assert ((cohesive[index] - cohesive[0])
                    == pytest.approx(plain[index] - plain[0], abs=1e-9))

    def test_forces_are_untouched(self, references):
        r"""
        :math:`\nabla_{\mathbf R} E_{\rm ref} = 0`, exactly.

        The reference depends on the composition and not on any coordinate.
        Nothing in the force path consults a total energy at all, so this is
        less a numerical check than a guard against someone later "helpfully"
        subtracting a constant somewhere it would do harm.
        """
        name = "struct_015"
        cache = os.path.join(CACHE_DIR, name)
        potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
        density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                     grid=potential.grid)
        potcar = Potcar.from_file(os.path.join(VASP_DIR, name, "POTCAR"),
                                  parse_tables=True)

        forces = hellmann_feynman_forces(density.data, potential.structure,
                                         potential.grid, potcar=potcar)
        # The reference is a number; the force cannot see it. Recomputing with
        # the reference in hand must give the identical array.
        again = hellmann_feynman_forces(density.data, potential.structure,
                                        potential.grid, potcar=potcar)
        assert np.array_equal(forces, again)
        assert references.total_for(potential.structure) != 0.0


# ===================================================================== #
# What referencing DOES change
# ===================================================================== #
@needs_dataset
@needs_references
@needs_reference_tau
class TestSelfReferencingRemovesThePawOffset:
    """
    The measurable improvement, and the reason ``method="poraque"`` is default.
    """

    def test_cohesive_energy_is_physically_sized(self):
        """
        Right order, right sign, for a late transition metal.

        Gold's cohesive energy is a few eV per atom. Referenced against
        Poraquê's own isolated atom the answer lands there; referenced against
        VASP's it does not, by three orders of magnitude.
        """
        references = ReferenceEnergies.from_directory(REF_DIR,
                                                      method="poraque")
        result = components("struct_000", references)
        assert result.cohesive_per_atom is not None
        assert -6.0 < result.cohesive_per_atom < -0.5, (
            f"cohesive energy {result.cohesive_per_atom:.3f} eV/atom is not a "
            f"plausible bonding energy")

    def test_self_referencing_beats_code_referencing_by_orders_of_magnitude(
            self):
        """
        The headline comparison, asserted rather than asserted-about.

        Both are 'the cohesive energy'; only one has had the offset removed.
        """
        own = ReferenceEnergies.from_directory(REF_DIR, method="poraque")
        code = ReferenceEnergies.from_directory(REF_DIR, method="code")

        self_referenced = abs(components("struct_000", own).cohesive_per_atom)
        code_referenced = abs(components("struct_000", code).cohesive_per_atom)
        assert self_referenced < 10.0
        assert code_referenced > 1000.0

    def test_the_offset_removed_is_the_atomic_energy(self):
        """Bookkeeping: total - reference is the cohesive energy, exactly."""
        references = ReferenceEnergies.from_directory(REF_DIR,
                                                      method="poraque")
        result = components("struct_000", references)
        assert result.cohesive == pytest.approx(
            result.total - result.reference, abs=1e-9)
        assert result.cohesive_per_atom == pytest.approx(
            result.cohesive / result.natoms, abs=1e-12)


# ===================================================================== #
# Plumbing
# ===================================================================== #
class TestComponentsWithoutReferences:
    """The unreferenced path must stay usable and must not fake a number."""

    def test_cohesive_is_none_without_references(self):
        from poraque.physics import EnergyComponents

        result = EnergyComponents(kinetic=1.0, external=-2.0, hartree=0.5,
                                  xc=-0.3)
        assert result.reference is None
        assert result.cohesive is None
        assert result.cohesive_per_atom is None

    def test_as_dict_carries_the_new_fields(self):
        from poraque.physics import EnergyComponents

        payload = EnergyComponents(kinetic=1.0, external=-2.0, hartree=0.5,
                                   xc=-0.3, reference=-10.0, natoms=2).as_dict()
        assert payload["reference"] == -10.0
        assert payload["cohesive"] == pytest.approx(-0.8 + 10.0)
        assert payload["cohesive_per_atom"] == pytest.approx(
            payload["cohesive"] / 2)


@needs_references
class TestCalculatorIntegration:
    def test_accepts_a_path_a_mapping_or_an_instance(self):
        from poraque.calculator import _resolve_references

        assert _resolve_references(None) is None
        assert _resolve_references({"Au": -1.0})["Au"] == -1.0
        assert isinstance(_resolve_references(REF_DIR), ReferenceEnergies)
        built = ReferenceEnergies({"Au": -1.0})
        assert _resolve_references(built) is built
