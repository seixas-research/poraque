# -*- coding: utf-8 -*-
# file: test_charges.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Charge conservation and population analysis.

The partitionings are tested against densities whose answer is known by
construction rather than against each other. Two Gaussians of charge 8 and 4
sitting far apart have an unambiguous split, and any scheme that fails to
recover it is broken regardless of what the others say.

The property every scheme must satisfy, whatever the weights, is that the
partition is **exhaustive**: :math:`\sum_A w_A = 1` everywhere, so the
populations sum to :math:`\int\rho` exactly. That is asserted for all three,
because it is the one thing a partitioning cannot be allowed to get wrong --- a
scheme that loses charge in the interstitial produces plausible per-atom
numbers that do not add up, and nothing else in the output would show it.

Symmetry is the other lever: in an elemental solid every atom is the same
species, so a Hirshfeld promolecule built from identical free atoms must give
charges of essentially zero. That catches reference-handling mistakes which the
Gaussian tests cannot.
"""

import os
import shutil

import numpy as np
import pytest

from poraque.analysis import (
    PARTITION_METHODS,
    atomic_radial_profile,
    bader_charges,
    hirshfeld_charges,
    partial_charges,
    verify_total_charge,
    voronoi_charges,
)
from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.vasp.poscar import Poscar

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "data", "cache", "res32")
VASP_DIR = os.path.join(_ROOT, "data", "vasp")
REF_DIR = os.path.join(VASP_DIR, "ref")

needs_dataset = pytest.mark.skipif(
    not os.path.isdir(CACHE_DIR),
    reason="the shipped dataset is not present in this checkout")
needs_references = pytest.mark.skipif(
    not os.path.isdir(REF_DIR),
    reason="no isolated-atom reference directory in this checkout")


# ===================================================================== #
# Fixtures: densities with a known answer
# ===================================================================== #
LENGTH = 10.0
POINTS = 40


@pytest.fixture
def two_gaussians():
    r"""
    Charges 8 and 4 on two well-separated sites.

    Well separated relative to the width, so the true partition is unambiguous
    and every scheme should agree with it and with each other.
    """
    cell = np.eye(3) * LENGTH
    grid = FieldGrid((POINTS,) * 3, cell)
    structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                       scaled_positions=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])

    coordinates = grid.cartesian_coordinates()

    def blob(centre, charge, width=0.8):
        squared = ((coordinates - np.asarray(centre)) ** 2).sum(-1)
        return (charge * np.exp(-squared / (2 * width ** 2))
                / (2 * np.pi * width ** 2) ** 1.5)

    values = blob([2.5, 5.0, 5.0], 8.0) + blob([7.5, 5.0, 5.0], 4.0)
    return ChargeDensity(values, grid, structure)


@pytest.fixture
def skewed_gaussians():
    """The same idea in a non-orthogonal cell, where wrapping alone is wrong."""
    cell = np.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 10.0]])
    grid = FieldGrid((24, 24, 24), cell)
    structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                       scaled_positions=[[0.2, 0.2, 0.5], [0.7, 0.7, 0.5]])

    coordinates = grid.cartesian_coordinates()
    centres = np.asarray(structure.scaled_positions) @ cell

    values = np.zeros(grid.shape)
    for centre, charge in zip(centres, (6.0, 3.0)):
        squared = ((coordinates - centre) ** 2).sum(-1)
        values += (charge * np.exp(-squared / (2 * 0.7 ** 2))
                   / (2 * np.pi * 0.7 ** 2) ** 1.5)
    return ChargeDensity(values, grid, structure)


# ===================================================================== #
# Task 1: charge conservation
# ===================================================================== #
class TestVerifyTotalCharge:
    def test_integrates_correctly(self, two_gaussians):
        check = verify_total_charge(two_gaussians, two_gaussians.grid.cell,
                                    12.0)
        assert check.integrated == pytest.approx(12.0, abs=0.02)
        assert check.ok

    def test_voxel_volume_is_cell_volume_over_the_point_count(self):
        r""":math:`dV = \Omega / (N_x N_y N_z)`, stated as a test."""
        cell = np.eye(3) * 2.0
        grid = FieldGrid((4, 4, 4), cell)
        structure = Poscar(cell=cell, symbols=["H"], counts=[1],
                           scaled_positions=[[0, 0, 0]])
        density = ChargeDensity(np.full(grid.shape, 3.0), grid, structure)
        # 3 e/A^3 over 8 A^3 is 24 electrons, whatever the mesh.
        assert verify_total_charge(density, cell, 24.0).integrated == \
            pytest.approx(24.0)

    def test_reports_the_drift_and_fails_outside_tolerance(self, two_gaussians):
        check = verify_total_charge(two_gaussians, two_gaussians.grid.cell,
                                    24.0, warn=False)
        assert not check.ok
        assert check.relative_error == pytest.approx(-0.5, abs=0.01)
        assert check.absolute_error == pytest.approx(-12.0, abs=0.1)

    def test_warns_when_charge_is_not_conserved(self, two_gaussians):
        with pytest.warns(RuntimeWarning, match="not conserved"):
            verify_total_charge(two_gaussians, two_gaussians.grid.cell, 24.0)

    def test_silent_when_asked_to_be(self, two_gaussians):
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            verify_total_charge(two_gaussians, two_gaussians.grid.cell, 24.0,
                                warn=False)

    def test_accepts_a_grid_or_bare_lattice_vectors(self, two_gaussians):
        from_cell = verify_total_charge(two_gaussians,
                                        two_gaussians.grid.cell, 12.0)
        from_grid = verify_total_charge(two_gaussians.data,
                                        two_gaussians.grid, 12.0)
        assert from_cell.integrated == pytest.approx(from_grid.integrated)

    def test_normalization_makes_the_check_pass(self, two_gaussians):
        """The repair and the check must agree on what 'correct' means."""
        fixed = two_gaussians.normalized(12.0)
        check = verify_total_charge(fixed, fixed.grid.cell, 12.0)
        assert check.relative_error == pytest.approx(0.0, abs=1e-12)


# ===================================================================== #
# The partitionings
# ===================================================================== #
@pytest.mark.parametrize("method", PARTITION_METHODS)
class TestEveryPartitioning:
    """Properties that hold for all three, asserted for all three."""

    def test_recovers_a_known_split(self, two_gaussians, method):
        """Charges 8 and 4, far apart: no scheme has room to disagree."""
        result = partial_charges(two_gaussians, method=method,
                                 valence={"Au": 6.0})
        assert result.populations[0] == pytest.approx(8.0, abs=0.1)
        assert result.populations[1] == pytest.approx(4.0, abs=0.1)

    def test_conserves_charge(self, two_gaussians, method):
        r"""
        :math:`\sum_A N_A = \int\rho`, exactly.

        The partition is exhaustive, so this holds whatever the weights are. A
        scheme that lost charge would still produce per-atom numbers, and only
        this assertion would notice.
        """
        result = partial_charges(two_gaussians, method=method,
                                 valence={"Au": 6.0})
        assert result.total_population == pytest.approx(
            two_gaussians.integrate(), rel=1e-9)

    def test_net_charge_is_valence_minus_population(self, two_gaussians,
                                                    method):
        result = partial_charges(two_gaussians, method=method,
                                 valence={"Au": 6.0})
        assert np.allclose(result.charges,
                           result.valence - result.populations)

    def test_handles_a_non_orthogonal_cell(self, skewed_gaussians, method):
        """
        Where a naive fractional wrap gives the wrong neighbour.

        In a skewed lattice the nearest image can be a diagonal one, so the
        minimum image needs a search over the surrounding images rather than a
        single rounding.
        """
        result = partial_charges(skewed_gaussians, method=method,
                                 valence={"Au": 4.5})
        assert result.total_population == pytest.approx(
            skewed_gaussians.integrate(), rel=1e-9)
        assert result.populations[0] > result.populations[1]

    def test_reports_one_entry_per_atom(self, two_gaussians, method):
        result = partial_charges(two_gaussians, method=method)
        assert len(result.symbols) == 2
        assert result.populations.shape == (2,)

    def test_as_dict_round_trips(self, two_gaussians, method):
        import json

        payload = partial_charges(two_gaussians, method=method,
                                  valence={"Au": 6.0}).as_dict()
        json.dumps(payload)
        assert payload["method"] == method
        assert len(payload["charges"]) == 2


class TestUnknownMethod:
    def test_names_the_alternatives(self, two_gaussians):
        with pytest.raises(ValueError, match="voronoi"):
            partial_charges(two_gaussians, method="mulliken")


# ===================================================================== #
# Task 2: Voronoi
# ===================================================================== #
class TestVoronoi:
    def test_boundary_sits_at_the_midpoint(self):
        """
        Geometric by definition: a uniform density splits exactly in half.

        This is also the scheme's weakness, and the reason it is a baseline
        rather than a default -- the boundary ignores the density entirely.
        """
        cell = np.eye(3) * 8.0
        grid = FieldGrid((16, 16, 16), cell)
        structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                           scaled_positions=[[0.25, 0.5, 0.5],
                                             [0.75, 0.5, 0.5]])
        density = ChargeDensity(np.ones(grid.shape), grid, structure)
        result = voronoi_charges(density)
        assert result.populations[0] == pytest.approx(result.populations[1])

    def test_equidistant_voxels_are_shared_not_awarded_by_index(self):
        """
        The symmetric case, where a naive tie-break would be visibly wrong.

        Two atoms at x = 1/4 and x = 3/4 of a cubic cell put whole *planes* of
        voxels exactly on the two boundaries. Giving them all to atom 0 --
        which a strict ``<`` comparison does -- hands it a 64-electron excess
        out of 512 in this cell, on a structure whose answer is symmetric by
        construction.
        """
        cell = np.eye(3) * 8.0
        grid = FieldGrid((16, 16, 16), cell)
        structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                           scaled_positions=[[0.25, 0.5, 0.5],
                                             [0.75, 0.5, 0.5]])
        density = ChargeDensity(np.ones(grid.shape), grid, structure)
        result = voronoi_charges(density)

        assert result.details["shared_voxels"] > 0, (
            "this geometry is chosen to produce exact ties")
        assert result.populations[0] == pytest.approx(result.populations[1],
                                                      rel=1e-12)
        assert result.total_population == pytest.approx(density.integrate(),
                                                        rel=1e-12)

    def test_assigns_every_voxel(self, two_gaussians):
        result = voronoi_charges(two_gaussians)
        assert result.total_population == pytest.approx(
            two_gaussians.integrate(), rel=1e-12)


# ===================================================================== #
# Task 3: Hirshfeld
# ===================================================================== #
class TestHirshfeld:
    def test_falls_back_to_the_exponential_promolecule(self, two_gaussians):
        """Without references it still works, and says that it did."""
        result = hirshfeld_charges(two_gaussians, valence={"Au": 6.0})
        assert result.details["promolecule"]["Au"] == "exponential model"
        assert result.total_population == pytest.approx(
            two_gaussians.integrate(), rel=1e-9)

    @needs_references
    def test_uses_a_real_isolated_atom_density_when_available(self,
                                                              two_gaussians):
        result = hirshfeld_charges(two_gaussians, valence={"Au": 6.0},
                                   references=REF_DIR)
        assert result.details["promolecule"]["Au"] == "isolated-atom CHGCAR"

    @needs_dataset
    @needs_references
    def test_an_elemental_solid_has_near_zero_charges(self):
        r"""
        Symmetry, and the sharpest available check on the reference handling.

        Every atom is the same species, so a promolecule of identical free
        atoms weights them almost equally and the charges must come out near
        zero. A mis-scaled or misplaced reference density would break this
        while leaving charge conservation intact.
        """
        density = ChargeDensity.read(os.path.join(CACHE_DIR, "struct_000",
                                                  "CHGCAR"))
        result = hirshfeld_charges(density, valence={"Au": 11.0},
                                   references=REF_DIR)
        assert np.abs(result.charges).max() < 0.05

    def test_weights_sum_to_one_even_in_empty_space(self):
        """
        Far from every atom the promolecule underflows.

        Sharing the weight equally there keeps the partition exhaustive; simply
        dividing would give NaN and silently lose the charge in that region.
        """
        cell = np.eye(3) * 30.0                 # mostly vacuum
        grid = FieldGrid((20, 20, 20), cell)
        structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                           scaled_positions=[[0.1, 0.1, 0.1],
                                             [0.2, 0.2, 0.2]])
        density = ChargeDensity(np.full(grid.shape, 1e-6), grid, structure)
        result = hirshfeld_charges(density, valence={"Au": 1.0})
        assert np.isfinite(result.populations).all()
        assert result.total_population == pytest.approx(density.integrate(),
                                                        rel=1e-9)

    @needs_references
    def test_radial_profile_integrates_to_the_electron_count(self):
        r"""
        :math:`\int\rho^{\rm at}(r)\,4\pi r^2 dr \approx N`.

        The spherical average must preserve the norm, or the promolecule is
        mis-scaled and every Hirshfeld charge shifts with it.
        """
        reference = ChargeDensity.read(os.path.join(REF_DIR, "Au", "CHGCAR"))
        radii, profile = atomic_radial_profile(reference)
        integral = np.trapezoid(profile * 4 * np.pi * radii ** 2, radii)
        assert integral == pytest.approx(11.0, rel=0.05)


# ===================================================================== #
# Task 4: Bader
# ===================================================================== #
class TestBaderNative:
    def test_finds_one_maximum_per_well_separated_atom(self, two_gaussians):
        result = bader_charges(two_gaussians, valence={"Au": 6.0},
                               backend="native")
        assert result.details["backend"] == "native"
        assert result.details["maxima"] == 2

    def test_basins_follow_the_density_not_the_geometry(self):
        r"""
        What separates Bader from Voronoi.

        Two atoms of very unequal size: the zero-flux surface sits away from
        the geometric midpoint, toward the smaller atom, so Bader gives the
        large atom more than half while Voronoi gives exactly half.
        """
        cell = np.eye(3) * 12.0
        grid = FieldGrid((36, 36, 36), cell)
        structure = Poscar(cell=cell, symbols=["Au"], counts=[2],
                           scaled_positions=[[0.35, 0.5, 0.5],
                                             [0.65, 0.5, 0.5]])
        coordinates = grid.cartesian_coordinates()

        def blob(centre, charge, width):
            squared = ((coordinates - np.asarray(centre)) ** 2).sum(-1)
            return (charge * np.exp(-squared / (2 * width ** 2))
                    / (2 * np.pi * width ** 2) ** 1.5)

        values = blob([4.2, 6, 6], 10.0, 1.1) + blob([7.8, 6, 6], 2.0, 0.6)
        density = ChargeDensity(values, grid, structure)

        bader = bader_charges(density, backend="native")
        voronoi = voronoi_charges(density)
        assert bader.populations[0] > voronoi.populations[0]

    def test_tolerates_more_maxima_than_atoms(self):
        """
        A metal can have non-nuclear attractors, and grid noise makes more.

        The assignment is many-to-one by design: every maximum goes to its
        nearest atom, so extra maxima redistribute charge instead of raising an
        index error.
        """
        cell = np.eye(3) * 8.0
        grid = FieldGrid((20, 20, 20), cell)
        structure = Poscar(cell=cell, symbols=["Au"], counts=[1],
                           scaled_positions=[[0.5, 0.5, 0.5]])
        rng = np.random.default_rng(0)
        values = 1.0 + 0.5 * rng.random(grid.shape)     # many local maxima
        density = ChargeDensity(values, grid, structure)

        result = bader_charges(density, backend="native")
        assert result.details["maxima"] > 1
        assert result.total_population == pytest.approx(density.integrate(),
                                                        rel=1e-9)

    def test_conserves_charge_on_a_real_density(self):
        if not os.path.isdir(CACHE_DIR):
            pytest.skip("dataset not present")
        density = ChargeDensity.read(os.path.join(CACHE_DIR, "struct_000",
                                                  "CHGCAR"))
        result = bader_charges(density, valence={"Au": 11.0}, backend="native")
        assert result.total_population == pytest.approx(density.integrate(),
                                                        rel=1e-9)
        assert result.total_charge == pytest.approx(0.0, abs=1e-5)


class TestBaderExternal:
    def test_rejects_an_unknown_backend(self, two_gaussians):
        with pytest.raises(ValueError, match="backend="):
            bader_charges(two_gaussians, backend="quantum")

    def test_explicit_external_raises_when_absent(self, two_gaussians):
        """
        Asking for the external program when it is missing must not silently
        fall back --- the two backends are not identical, and a paper that says
        "Bader (Henkelman)" should not quietly have used something else.
        """
        if shutil.which("bader") is not None:
            pytest.skip("the bader executable is installed here")
        with pytest.raises(FileNotFoundError, match="not on PATH"):
            bader_charges(two_gaussians, backend="external")

    def test_auto_falls_back_to_native(self, two_gaussians):
        result = bader_charges(two_gaussians, backend="auto")
        expected = ("external" if shutil.which("bader") else "native")
        assert result.details["backend"] == expected

    def test_acf_parsing(self, tmp_path):
        """The external path's only fragile step, tested without the program."""
        from poraque.analysis.charges import _parse_acf

        content = (
            "    #         X           Y           Z       CHARGE      "
            "MIN DIST   ATOMIC VOL\n"
            " --------------------------------------------------------\n"
            "    1      0.0000      0.0000      0.0000     10.1234      "
            "1.0000      20.0000\n"
            "    2      2.0000      0.0000      0.0000     11.8766      "
            "1.0000      20.0000\n"
            " --------------------------------------------------------\n"
            "    VACUUM CHARGE:               0.0000\n"
        )
        path = tmp_path / "ACF.dat"
        path.write_text(content)
        assert np.allclose(_parse_acf(str(path), 2), [10.1234, 11.8766])

    def test_acf_parsing_rejects_a_short_table(self, tmp_path):
        from poraque.analysis.charges import _parse_acf

        path = tmp_path / "ACF.dat"
        path.write_text("    1  0.0 0.0 0.0  10.0  1.0  20.0\n")
        with pytest.raises(RuntimeError, match="lists 1 atoms"):
            _parse_acf(str(path), 4)


# ===================================================================== #
# Task 5: the ASE calculator
# ===================================================================== #
class TestCalculatorCharges:
    @pytest.fixture
    def calculator(self, tmp_path):
        from poraque.calculator import Poraque
        from poraque.ml import BUNDLE_FILENAME, FieldOperator, save_bundle

        bundle = save_bundle(
            str(tmp_path / BUNDLE_FILENAME),
            {task: FieldOperator(task, width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu",
                                 training_resolution=16)
             for task in ("ext2chg", "chg2tau")})
        return Poraque(bundle, charges={"Au": 11.0}, device="cpu")

    @pytest.fixture
    def atoms(self):
        ase = pytest.importorskip("ase")

        return ase.Atoms("Au2", cell=np.eye(3) * 4.08, pbc=True,
                         scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_matches_the_ase_interface(self, calculator, atoms):
        """
        ``atoms.get_charges()`` calls ``calc.get_charges(atoms)`` positionally,
        so the default partitioning has to work with no other argument.
        """
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            charges = atoms.get_charges()
        assert charges.shape == (len(atoms),)
        assert np.isfinite(charges).all()

    @pytest.mark.parametrize("method", PARTITION_METHODS)
    def test_every_method_is_reachable(self, calculator, atoms, method):
        with pytest.warns(RuntimeWarning):
            charges = calculator.get_charges(atoms, method=method)
        assert charges.shape == (len(atoms),)

    def test_keeps_the_full_analysis(self, calculator, atoms):
        with pytest.warns(RuntimeWarning):
            calculator.get_charges(atoms, method="voronoi")
        analysis = calculator.charge_analysis
        assert analysis.method == "voronoi"
        assert "voronoi" in str(analysis)
        assert np.allclose(analysis.charges,
                           analysis.valence - analysis.populations)

    def test_subtracts_the_valence_from_the_calculator(self, calculator,
                                                       atoms):
        with pytest.warns(RuntimeWarning):
            calculator.get_charges(atoms, method="voronoi")
        assert np.allclose(calculator.charge_analysis.valence, 11.0)

    def test_verify_charge_reports_the_density_actually_used(self, calculator,
                                                             atoms):
        """
        The check must describe the field the energy will be integrated from.

        Asserting ``ok`` here would be flaky, and for an honest reason: an
        *untrained* operator can emit an everywhere-negative density, which no
        rescaling can repair, so the calculator warns and falls back to the raw
        prediction. The invariant that always holds is that the check reports
        that state truthfully rather than the state that was hoped for.
        """
        with pytest.warns(RuntimeWarning):
            check = calculator.verify_charge(atoms)
            density = calculator.get_charge_density(atoms)

        assert check.expected == pytest.approx(22.0)
        assert check.integrated == pytest.approx(density.integrate(), rel=1e-9)

    @needs_dataset
    def test_a_normalized_density_passes(self):
        """On a real density the normalization holds and the check passes."""
        density = ChargeDensity.read(os.path.join(CACHE_DIR, "struct_000",
                                                  "CHGCAR"))
        check = verify_total_charge(density, density.grid.cell, 297.0)
        assert check.ok
