# -*- coding: utf-8 -*-
# file: test_atomic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The isolated-atom database, the superposition it builds, and delta-density mode.

The superposition is the baseline a δ-density model's whole target is defined
against, so the properties asserted here are the ones that would corrupt every
weight in such a model if they broke:

**The electron count is exact, not approximate.** :math:`f_s(0) = Z^{\rm val}_s`
makes :math:`\int\rho_{\rm sup} = \sum_a Z^{\rm val}_a` by construction. If it
ever became approximate, the residual would silently absorb the difference and
the electron-count constraint would be fighting the baseline.

**Translation invariance and supercell consistency.** A reciprocal-space
placement is exactly periodic and exactly translation-covariant. Both are
asserted to near machine precision rather than to a tolerance, because both are
identities — a tolerance would hide a phase convention error in
:func:`~poraque.fields.external.structure_factor`, which is shared with the
external potential and would therefore be wrong in two places at once.

**Round-trip.** A single atom superposed back onto its own reference grid must
reproduce the density it came from, to within the radial approximation.

The fixtures are synthetic Gaussian atoms, so the "reference density" is known
in closed form and the tests do not depend on the repository's data being
present. The real gold reference is used where it exists, and skipped where it
does not.
"""

import json
import os

import numpy as np
import pytest

from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.atomic import (
    AtomicReference,
    AtomicReferenceLibrary,
    atomic_superposition,
    augmentation_from_atoms,
    base_element,
    build_library,
    form_factor_from_density,
    reference_from_calculation,
)
from poraque.fields.vasp.poscar import Poscar

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AU = os.path.join(_ROOT, "data", "vasp", "ref", "Au")
VASP_DIR = os.path.join(_ROOT, "data", "vasp")

needs_reference_atom = pytest.mark.skipif(
    not os.path.exists(os.path.join(REF_AU, "CHGCAR")),
    reason="the shipped isolated-atom reference is not in this checkout")


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def gaussian_atom(grid, position, n_electrons=4.0, sigma=0.7):
    """A normalised Gaussian at a fractional position, in e/Å³."""
    coords = grid.cartesian_coordinates()
    cell = np.asarray(grid.cell, dtype=float)
    centre = np.asarray(position, dtype=float) @ cell

    # The nearest periodic image, so an atom near a face is not smeared by the
    # tail of the one across the boundary being measured from the wrong side.
    delta = coords - centre
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    r2 = ((fractional @ cell) ** 2).sum(axis=-1)

    amplitude = n_electrons / (2 * np.pi * sigma ** 2) ** 1.5
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def synthetic_reference(element="Si", n_electrons=4.0, sigma=0.7, side=8.0,
                        n=48):
    """An :class:`AtomicReference` built from a Gaussian atom in a box."""
    grid = FieldGrid((n, n, n), np.eye(3) * side)
    structure = Poscar(np.eye(3) * side, [element], [1], [[0.5, 0.5, 0.5]])
    density = ChargeDensity(gaussian_atom(grid, (0.5, 0.5, 0.5), n_electrons,
                                          sigma), grid, structure)
    table = form_factor_from_density(density)
    return AtomicReference(
        element=element, valence_charge=table["valence_charge"],
        g_grid=table["g_grid"], form_factor=table["form_factor"],
        g_max=table["g_max"], radial_scatter=table["radial_scatter"],
        potcar_title=f"SYNTHETIC {element}", potcar_sha256="a" * 64)


@pytest.fixture
def library():
    return AtomicReferenceLibrary({r.key: r for r in [synthetic_reference()]})


def cell_of(structure_positions, side=10.0, element="Si"):
    return Poscar(np.eye(3) * side, [element], [len(structure_positions)],
                  np.asarray(structure_positions, dtype=float))


# ===================================================================== #
# The form factor
# ===================================================================== #
class TestTheFormFactor:
    def test_f_of_zero_is_the_valence_charge(self):
        table = form_factor_from_density(
            ChargeDensity(
                gaussian_atom(FieldGrid((48, 48, 48), np.eye(3) * 8.0),
                              (0.5, 0.5, 0.5), 6.0),
                FieldGrid((48, 48, 48), np.eye(3) * 8.0),
                cell_of([[0.5, 0.5, 0.5]], side=8.0)))
        assert table["valence_charge"] == pytest.approx(6.0, rel=1e-6)
        assert table["form_factor"][0] == pytest.approx(6.0, rel=1e-6)

    def test_a_gaussian_atom_is_radial_to_machine_precision(self):
        """
        A Gaussian *is* spherical, so any scatter here is the binning or a bug
        in the recentring phase — not physics. This is the control for the real
        atom's 0.48 %, which is genuine anisotropy.
        """
        reference = synthetic_reference()
        assert reference.radial_scatter < 1e-6

    def test_the_table_matches_the_analytic_transform(self):
        r"""
        For a Gaussian, :math:`f(G) = N e^{-G^2\sigma^2/2}` exactly.

        Checked against mathematics rather than against another array, so a
        drift in the FFT normalisation or the volume factor cannot hide.
        """
        reference = synthetic_reference(n_electrons=4.0, sigma=0.7)
        g = np.linspace(0.0, 5.0, 400)
        analytic = 4.0 * np.exp(-0.5 * g ** 2 * 0.7 ** 2)
        # 0.5 % of f(0). The table is built on an 8 Angstrom box, whose first
        # reciprocal shell sits at 0.79 1/Ang, so everything below that is
        # interpolation -- which is exactly the regime `evaluate` interpolates
        # in G^2 to handle, and the reason this bound is 0.02 and not 0.2.
        assert np.allclose(reference.evaluate(g), analytic, atol=0.02)

    def test_more_than_one_atom_is_refused(self):
        grid = FieldGrid((16, 16, 16), np.eye(3) * 8.0)
        structure = cell_of([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], side=8.0)
        with pytest.raises(ValueError, match="exactly one atom"):
            form_factor_from_density(np.zeros(grid.shape), grid, structure)

    def test_the_table_is_zero_beyond_its_range(self):
        reference = synthetic_reference()
        assert reference.evaluate([reference.g_max * 2]) == pytest.approx(0.0)


# ===================================================================== #
# The superposition
# ===================================================================== #
class TestSuperposition:
    def test_a_single_atom_round_trips_onto_its_own_grid(self):
        """The reference density, reconstructed from its stored table."""
        side, n = 8.0, 48
        grid = FieldGrid((n, n, n), np.eye(3) * side)
        structure = cell_of([[0.5, 0.5, 0.5]], side=side)
        original = gaussian_atom(grid, (0.5, 0.5, 0.5), 4.0, 0.7)

        reference = synthetic_reference()
        library = AtomicReferenceLibrary({reference.key: reference})
        rebuilt = atomic_superposition(structure, grid, library)

        error = (np.linalg.norm(rebuilt.data - original)
                 / np.linalg.norm(original))
        # A Gaussian is exactly spherical, so the only error left is the
        # binning and the interpolation. Measured at 3.6e-8 when this was
        # written; the bound is loose enough not to be a precision test and
        # tight enough that a regression in either would trip it.
        assert error < 1e-5

    def test_the_electron_count_is_exact(self, library):
        """
        Not approximate: f(0) = Z_val makes the integral a sum of integers.

        This is what lets the electron-count constraint act on the residual
        without fighting the baseline.
        """
        grid = FieldGrid((32, 32, 32), np.eye(3) * 10.0)
        structure = cell_of([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4],
                             [0.9, 0.1, 0.7]])
        rho = atomic_superposition(structure, grid, library)
        assert rho.integrate() == pytest.approx(3 * 4.0, rel=1e-6)

    def test_translating_every_atom_translates_the_field(self, library):
        r"""
        :math:`\rho_{\rm sup}` is covariant under a rigid shift, exactly.

        Asserted against a *rolled* array rather than a tolerance, so the shift
        is by whole grid points and the comparison is an identity. A phase
        convention error in the shared structure factor would break this — and
        would break the external potential in the same way.
        """
        n, side = 32, 10.0
        grid = FieldGrid((n, n, n), np.eye(3) * side)
        positions = np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]])
        shift = np.array([4, 0, 0]) / n

        first = atomic_superposition(cell_of(positions), grid, library).data
        second = atomic_superposition(cell_of(positions + shift), grid,
                                      library).data
        assert np.allclose(np.roll(first, 4, axis=0), second, atol=1e-10)

    def test_it_is_periodic_across_the_cell_boundary(self, library):
        """An atom at the face and the same atom at the opposite face agree."""
        grid = FieldGrid((32, 32, 32), np.eye(3) * 10.0)
        at_zero = atomic_superposition(cell_of([[0.0, 0.5, 0.5]]), grid,
                                       library).data
        at_one = atomic_superposition(cell_of([[1.0, 0.5, 0.5]]), grid,
                                      library).data
        assert np.allclose(at_zero, at_one, atol=1e-10)

    def test_a_supercell_repeats_the_unit_cell(self, library):
        r"""
        Doubling the cell and the atoms reproduces the field, tile for tile.

        The property that makes one table serve every cell: the form factor is
        a function of physical :math:`|G|`, so nothing about it knows which
        cell it is being evaluated in.
        """
        small = FieldGrid((24, 24, 24), np.eye(3) * 5.0)
        large = FieldGrid((48, 24, 24),
                          np.diag([10.0, 5.0, 5.0]))

        unit = atomic_superposition(
            cell_of([[0.3, 0.4, 0.5]], side=5.0), small, library).data

        doubled = Poscar(np.diag([10.0, 5.0, 5.0]), ["Si"], [2],
                         [[0.15, 0.4, 0.5], [0.65, 0.4, 0.5]])
        supercell = atomic_superposition(doubled, large, library).data

        assert np.allclose(supercell[:24], unit, atol=1e-8)
        assert np.allclose(supercell[24:], unit, atol=1e-8)

    def test_it_does_not_depend_on_the_grid_it_is_evaluated_on(self, library):
        """
        The same physical field on two meshes, compared where they coincide.

        A finer grid resolves more of the atom, so the two are not identical —
        but on the coarse grid's own points they must agree to the truncation,
        which is what "grid-independent table" has to mean.
        """
        structure = cell_of([[0.25, 0.25, 0.25]])
        coarse = atomic_superposition(
            structure, FieldGrid((24, 24, 24), np.eye(3) * 10.0), library).data
        fine = atomic_superposition(
            structure, FieldGrid((48, 48, 48), np.eye(3) * 10.0), library).data
        assert np.allclose(coarse, fine[::2, ::2, ::2], atol=5e-3)

    def test_a_missing_element_raises_rather_than_partially_summing(self,
                                                                   library):
        """
        A partial superposition has the right units and a plausible shape and
        is wrong by whole atoms — the worst kind of answer to return.
        """
        grid = FieldGrid((16, 16, 16), np.eye(3) * 10.0)
        structure = Poscar(np.eye(3) * 10.0, ["Si", "Ge"], [1, 1],
                           [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
        with pytest.raises(KeyError, match="Ge"):
            atomic_superposition(structure, grid, library)

    def test_the_field_records_the_library_it_came_from(self, library):
        grid = FieldGrid((16, 16, 16), np.eye(3) * 10.0)
        rho = atomic_superposition(cell_of([[0.5, 0.5, 0.5]]), grid, library)
        assert rho.metadata["library_fingerprint"] == library.fingerprint


# ===================================================================== #
# The database
# ===================================================================== #
class TestTheLibrary:
    def test_it_round_trips_through_json(self, library, tmp_path):
        path = library.save(tmp_path / "atoms.json")
        again = AtomicReferenceLibrary.load(path)
        assert again.fingerprint == library.fingerprint
        assert again.elements() == library.elements()

    def test_loading_a_missing_file_gives_an_empty_library(self, tmp_path):
        assert len(AtomicReferenceLibrary.load(tmp_path / "nothing.json")) == 0

    def test_a_directory_resolves_to_the_conventional_filename(self, library,
                                                               tmp_path):
        library.save(tmp_path / "atomic_reference.json")
        assert len(AtomicReferenceLibrary.load(tmp_path)) == 1

    def test_a_newer_schema_raises_rather_than_being_half_read(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({"version": 99, "entries": {}}))
        with pytest.raises(ValueError, match="schema version"):
            AtomicReferenceLibrary.load(path)

    def test_the_fingerprint_changes_with_the_table(self, library):
        before = library.fingerprint
        entry = list(library.entries.values())[0]
        entry.form_factor = [v * 1.01 for v in entry.form_factor]
        assert library.fingerprint != before

    def test_two_potcar_variants_of_one_element_coexist(self):
        """
        ``Au`` and ``Au_pv`` are different atoms with different valence counts.
        Merging them would give a baseline wrong for both.
        """
        plain = synthetic_reference(element="Au", n_electrons=11.0)
        plain.potcar_title, plain.potcar_sha256 = "PAW_PBE Au", "1" * 64
        semicore = synthetic_reference(element="Au", n_electrons=17.0)
        semicore.potcar_title, semicore.potcar_sha256 = "PAW_PBE Au_pv", "2" * 64

        library = AtomicReferenceLibrary()
        library.add(plain)
        library.add(semicore)
        assert len(library) == 2
        assert library.lookup("Au", "PAW_PBE Au_pv").valence_charge == \
            pytest.approx(17.0, rel=1e-6)

    def test_an_ambiguous_lookup_returns_nothing_rather_than_guessing(self):
        plain = synthetic_reference(element="Au", n_electrons=11.0)
        plain.potcar_title, plain.potcar_sha256 = "PAW_PBE Au", "1" * 64
        semicore = synthetic_reference(element="Au", n_electrons=17.0)
        semicore.potcar_title, semicore.potcar_sha256 = "PAW_PBE Au_pv", "2" * 64
        library = AtomicReferenceLibrary({plain.key: plain,
                                          semicore.key: semicore})
        assert library.lookup("Au") is None

    def test_a_decorated_symbol_finds_the_bare_element(self, library):
        assert library.lookup("Si_pv") is not None
        assert library.lookup("Si.pbe") is not None

    def test_base_element_strips_every_decoration_this_tree_uses(self):
        assert base_element("Au_pv") == "Au"
        assert base_element("O.pbe-n-kjpaw") == "O"
        assert base_element("Fe1") == "Fe"
        assert base_element("Au") == "Au"


# ===================================================================== #
# Against the real reference atom
# ===================================================================== #
@needs_reference_atom
class TestTheShippedGoldAtom:
    """
    The real thing, where the synthetic fixtures cannot reach.

    A Gaussian is exactly spherical and exactly band-limited; a PAW gold atom
    in a box is neither. These are the numbers ``DESIGN_PAW.md`` quotes.
    """

    def test_it_ingests_with_its_valence_charge_and_its_augmentation(self):
        reference = reference_from_calculation(REF_AU)
        assert reference.element == "Au"
        assert reference.valence_charge == pytest.approx(11.0, rel=1e-5)
        assert reference.augmentation is not None
        assert len(reference.augmentation) == 138

    def test_the_measured_anisotropy_is_the_documented_one(self):
        """
        ~0.5 %, and small enough that a radial table is a fair representation.

        Asserted as a band rather than a value: it is a property of the
        reference calculation, and pinning it exactly would break on a
        recomputed atom for no reason.
        """
        reference = reference_from_calculation(REF_AU)
        assert 0.0 < reference.radial_scatter < 0.02

    def test_superposing_it_back_reproduces_its_own_density(self):
        """
        3.0e-4 relative L2 — the total cost of the radial reduction plus the
        binning, and the error the baseline contributes to every delta target.
        """
        library = build_library([REF_AU])
        grid = FieldGrid.from_file(os.path.join(REF_AU, "CHGCAR"))
        original = ChargeDensity.read(os.path.join(REF_AU, "CHGCAR"), grid=grid)
        rebuilt = atomic_superposition(original.structure, grid, library)

        error = (np.linalg.norm(rebuilt.data - original.data)
                 / np.linalg.norm(original.data))
        assert error < 2e-3

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(VASP_DIR, "struct_000", "CHGCAR")),
        reason="the shipped gold supercells are not in this checkout")
    def test_the_residual_is_far_smaller_than_the_density(self):
        """
        The claim delta-density mode rests on: ~96 % of the field is just the
        free atoms. Measured at 0.036 on struct_000 when this was written.
        """
        library = build_library([REF_AU])
        path = os.path.join(VASP_DIR, "struct_000", "CHGCAR")
        grid = FieldGrid.from_file(path)
        rho = ChargeDensity.read(path, grid=grid)
        baseline = atomic_superposition(rho.structure, grid, library)

        residual = (np.linalg.norm(rho.data - baseline.data)
                    / np.linalg.norm(rho.data))
        assert residual < 0.15
        assert baseline.integrate() == pytest.approx(rho.integrate(), rel=1e-5)

    def test_the_free_atom_augmentation_is_offered_but_not_pretended_about(self):
        """
        It builds a valid block — and ``DESIGN_PAW.md`` §3.2 records that it is
        86.6 % RMS from a bulk site, which is why nothing selects it silently.
        """
        library = build_library([REF_AU])
        structure = Poscar(np.eye(3) * 6.0, ["Au"], [2],
                           [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        lines, missing = augmentation_from_atoms(structure, library)
        assert not missing
        assert sum(1 for line in lines
                   if "augmentation occupancies" in line) == 2

    def test_an_uncovered_element_yields_no_partial_block(self):
        library = build_library([REF_AU])
        structure = Poscar(np.eye(3) * 6.0, ["Au", "Ag"], [1, 1],
                           [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        lines, missing = augmentation_from_atoms(structure, library)
        assert lines == []
        assert missing == ["Ag"]
