# -*- coding: utf-8 -*-
# file: test_forces.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Hellmann-Feynman forces: implementation, then physics.

The two are worth separating, because they give different answers.

**The implementation is exact.** Each analytic force is checked against a
central finite difference of *the energy term it differentiates*, which is a
complete test of the derivative with no physics in it: if
:math:`\mathbf F = -\partial E/\partial\mathbf R` fails here, a sign or a
convention is wrong. Translation invariance
(:math:`\sum_I\mathbf F_I = 0`) is checked separately on the Ewald term, where
it holds identically.

**The physics is incomplete.** The Hellmann-Feynman force of a local
pseudopotential is the whole force only when the pseudopotential *is* local.
Every PAW dataset for a transition metal is strongly non-local, and the
projector and one-centre force terms are not recoverable from :math:`\rho`,
:math:`\tau` and :math:`V_{\rm loc}` sampled on a grid. The comparison against
VASP's ``TOTAL-FORCE`` is therefore written as a characterization: it pins what
the method currently achieves so a regression is visible, and it does not
pretend the number is small.

The stretched-molecule case the task asks for needs reference calculations this
repository does not ship; see :mod:`tests.test_energy_differences` for the
layout and set ``PORAQUE_REFERENCE_DATA``.
"""

import os

import numpy as np
import pytest

from poraque.fields import ChargeDensity, ExternalPotential
from poraque.fields.structure import Structure
from poraque.fields.vasp.potcar import Potcar
from poraque.physics import (
    ewald_energy,
    ewald_forces,
    force_consistency_error,
    hellmann_feynman_forces,
    local_potential_forces,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VASP_DIR = os.path.join(_ROOT, "data", "vasp")
CACHE_DIR = os.path.join(_ROOT, "data", "cache", "res32")
REFERENCE_DIR = os.environ.get("PORAQUE_REFERENCE_DATA",
                               os.path.join(_ROOT, "data", "reference"))

#: A rattled structure — a perfect lattice has zero forces by symmetry and
#: would pass any implementation, correct or not.
RATTLED = "struct_015"

needs_dataset = pytest.mark.skipif(
    not (os.path.isdir(CACHE_DIR) and os.path.isdir(VASP_DIR)),
    reason="the shipped VASP dataset is not present in this checkout")


# ===================================================================== #
# Helpers
# ===================================================================== #
def displaced(structure, index, axis, delta):
    """Copy of ``structure`` with atom ``index`` moved ``delta`` Å along ``axis``."""
    cell = np.asarray(structure.cell, dtype=float)
    cartesian = np.asarray(structure.scaled_positions, dtype=float) @ cell
    cartesian[index, axis] += delta
    return Structure(cell=structure.cell, symbols=structure.symbols,
                     counts=structure.counts,
                     scaled_positions=cartesian @ np.linalg.inv(cell))


def outcar_forces(path):
    """The last ``TOTAL-FORCE`` block of an ``OUTCAR``, as ``(natoms, 3)``."""
    block = open(path).read().split("TOTAL-FORCE (eV/Angst)")[-1]
    rows = []
    for line in block.splitlines()[2:]:
        parts = line.split()
        if len(parts) != 6:
            break
        rows.append([float(x) for x in parts[3:6]])
    return np.array(rows)


@pytest.fixture(scope="module")
def rattled_case():
    """VASP's own density, potential and geometry for a rattled platinum cell."""
    cache = os.path.join(CACHE_DIR, RATTLED)
    vasp = os.path.join(VASP_DIR, RATTLED)
    if not os.path.isdir(cache):
        pytest.skip(f"{RATTLED} is not cached")

    potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
    density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                 grid=potential.grid)
    potcar = Potcar.from_file(os.path.join(vasp, "POTCAR"), parse_tables=True)
    return {
        "density": density,
        "grid": potential.grid,
        "structure": potential.structure,
        "potcar": potcar,
        "charges": {entry.element: entry.zval for entry in potcar},
        "outcar": os.path.join(vasp, "OUTCAR"),
    }


# ===================================================================== #
# Ewald: the term with an exact answer
# ===================================================================== #
class TestEwaldForces:
    @pytest.fixture
    def cell(self):
        rng = np.random.default_rng(0)
        return Structure(cell=np.eye(3) * 5.0, symbols=["Na"], counts=[4],
                         scaled_positions=rng.random((4, 3)))

    def test_matches_finite_differences(self, cell):
        """
        Every component, against a central difference of :func:`ewald_energy`.

        This is the whole derivative: the Ewald energy is analytic and needs no
        grid, so there is no discretisation to blame and the agreement should
        be at the finite-difference floor.
        """
        analytic = ewald_forces(cell, {"Na": 1.0})
        step = 1e-5
        for index in range(cell.natoms):
            for axis in range(3):
                plus = ewald_energy(displaced(cell, index, axis, step),
                                    {"Na": 1.0})
                minus = ewald_energy(displaced(cell, index, axis, -step),
                                     {"Na": 1.0})
                assert analytic[index, axis] == pytest.approx(
                    -(plus - minus) / (2 * step), abs=1e-6)

    def test_net_force_vanishes(self, cell):
        """Newton's third law, exactly — not to within a tolerance on physics."""
        assert force_consistency_error(ewald_forces(cell, {"Na": 1.0})) < 1e-9

    def test_a_symmetric_lattice_feels_nothing(self):
        """Every site of a simple cubic lattice is an inversion centre."""
        lattice = Structure(cell=np.eye(3) * 3.0, symbols=["Na"], counts=[1],
                            scaled_positions=[[0.0, 0.0, 0.0]])
        assert np.abs(ewald_forces(lattice, {"Na": 1.0})).max() < 1e-9

    def test_like_charges_repel(self):
        """A displaced pair must be pushed apart, not together."""
        pair = Structure(cell=np.eye(3) * 12.0, symbols=["Na"], counts=[2],
                         scaled_positions=[[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
        forces = ewald_forces(pair, {"Na": 1.0})
        assert forces[0, 0] < 0.0 and forces[1, 0] > 0.0


# ===================================================================== #
# Electron-ion: the term that needs a density
# ===================================================================== #
@needs_dataset
class TestLocalPotentialForces:
    def test_matches_finite_differences_at_fixed_density(self, rattled_case):
        r"""
        Against a central difference of :math:`\int\rho V_{\rm ext}`.

        The density is held fixed while the ions move, which is precisely the
        derivative the Hellmann-Feynman term claims to be — so this tests the
        implementation exactly, and says nothing about whether discarding
        :math:`\partial\rho/\partial\mathbf R` was justified.
        """
        case = rattled_case
        grid, potcar = case["grid"], case["potcar"]
        rho = case["density"].data

        analytic = local_potential_forces(rho, case["structure"], grid, potcar)

        def energy(structure):
            potential = ExternalPotential.from_potcar_tables(structure, grid,
                                                             potcar)
            return grid.integrate(rho * potential.data)

        step = 1e-4
        for index in range(3):                       # three atoms is plenty
            for axis in range(3):
                reference = -(energy(displaced(case["structure"], index, axis, step))
                              - energy(displaced(case["structure"], index, axis, -step))
                              ) / (2 * step)
                assert analytic[index, axis] == pytest.approx(reference,
                                                              rel=1e-5, abs=1e-5)

    def test_a_perfect_lattice_feels_nothing(self):
        """
        Symmetry, on the structure that has it.

        ``struct_010`` is an undistorted cubic platinum cell, so every force must
        vanish. It is a weak test on its own — an implementation returning zero
        always would pass — which is why it sits beside the finite-difference
        check rather than instead of it.
        """
        name = "struct_010"
        cache = os.path.join(CACHE_DIR, name)
        if not os.path.isdir(cache):
            pytest.skip(f"{name} is not cached")

        potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
        density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                     grid=potential.grid)
        potcar = Potcar.from_file(os.path.join(VASP_DIR, name, "POTCAR"),
                                  parse_tables=True)
        forces = local_potential_forces(density.data, potential.structure,
                                        potential.grid, potcar)
        assert np.abs(forces).max() < 1e-3

    def test_rejects_a_density_on_the_wrong_grid(self, rattled_case):
        case = rattled_case
        with pytest.raises(ValueError, match="shape"):
            local_potential_forces(np.zeros((4, 4, 4)), case["structure"],
                                   case["grid"], case["potcar"])


# ===================================================================== #
# The two together, against VASP
# ===================================================================== #
@needs_dataset
class TestAgainstVasp:
    """
    Characterization against ``TOTAL-FORCE``, using VASP's own density.

    No model error enters here, so what is measured is the method's ceiling.
    """

    def test_the_two_terms_very_nearly_cancel(self, rattled_case):
        """
        Why forces are delicate, stated as a number.

        Electron-ion and ion-ion are each ~100 eV/Å and the total is ~0.5, so
        the answer is a half-percent residual of two large opposing terms. A
        relative error in :math:`\\rho` is amplified by that ratio on its way
        into the force, which is the quantitative reason a 1 %-accurate density
        cannot give a usable force.
        """
        case = rattled_case
        electronic = local_potential_forces(case["density"].data,
                                            case["structure"], case["grid"],
                                            case["potcar"])
        ionic = ewald_forces(case["structure"], case["charges"])
        total = electronic + ionic

        magnitude = np.linalg.norm(electronic, axis=1).mean()
        residual = np.linalg.norm(total, axis=1).mean()
        assert magnitude > 10.0
        assert residual < 0.05 * magnitude

    def test_net_force_is_small(self, rattled_case):
        r"""
        :math:`\sum_I\mathbf F_I` should vanish, and nearly does.

        It is not exactly zero because the density is VASP's PAW density while
        the force is evaluated from the local potential alone; the residue is
        the missing non-local term, not a bug in the sum.
        """
        case = rattled_case
        total = hellmann_feynman_forces(case["density"].data, case["structure"],
                                        case["grid"], potcar=case["potcar"])
        assert force_consistency_error(total) < 1.0

    def test_error_against_vasp_is_pinned(self, rattled_case):
        """
        The honest number.

        The mean absolute error is comparable to the forces themselves, because
        the projector force is missing. The bound is a regression guard: it is
        set where the method currently sits, not where a usable force would be.
        Tightening it is the deliverable of implementing the non-local term.
        """
        case = rattled_case
        if not os.path.isfile(case["outcar"]):
            pytest.skip("no OUTCAR for the rattled structure")

        reference = outcar_forces(case["outcar"])
        total = hellmann_feynman_forces(case["density"].data, case["structure"],
                                        case["grid"], potcar=case["potcar"])
        assert total.shape == reference.shape

        error = np.abs(total - reference).mean()
        assert error < 2.0, f"mean |dF| = {error:.4f} eV/Ang"

    def test_magnitude_is_the_right_order(self, rattled_case):
        """Right order of magnitude, which is as much as can be claimed."""
        case = rattled_case
        if not os.path.isfile(case["outcar"]):
            pytest.skip("no OUTCAR for the rattled structure")

        reference = np.linalg.norm(outcar_forces(case["outcar"]), axis=1).mean()
        total = hellmann_feynman_forces(case["density"].data, case["structure"],
                                        case["grid"], potcar=case["potcar"])
        predicted = np.linalg.norm(total, axis=1).mean()
        assert 0.1 * reference < predicted < 10.0 * reference


# ===================================================================== #
# Calculator wiring
# ===================================================================== #
class TestCalculatorForces:
    def test_forces_are_advertised(self):
        from poraque.calculator import Poraque

        assert "forces" in Poraque.implemented_properties

    def test_refuses_without_a_potcar(self):
        """
        The Gaussian fallback has no form factor to differentiate.

        Returning the Ewald force alone would be ~100 eV/Å of pure nonsense
        that looks like a force, so it raises instead.
        """
        pytest.importorskip("ase")
        from ase import Atoms

        from poraque.calculator import Poraque
        from poraque.ml import BUNDLE_FILENAME, FieldOperator, save_bundle

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            bundle = save_bundle(
                os.path.join(directory, BUNDLE_FILENAME),
                {task: FieldOperator(task, width=4, modes=2, n_layers=1,
                                     projection_channels=8, device="cpu",
                                     training_resolution=16)
                 for task in ("ext2chg", "chg2tau")})
            calculator = Poraque(bundle, charges={"Pt": 11.0}, device="cpu")
            atoms = Atoms("Pt2", cell=np.eye(3) * 4.08, pbc=True,
                          scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
            with pytest.warns(RuntimeWarning):
                with pytest.raises(ValueError, match="POTCAR"):
                    calculator.compute_forces(atoms)


# ===================================================================== #
# Stretched molecules — skip until the reference data exists
# ===================================================================== #
class TestStretchedMolecules:
    r"""
    Forces on a perturbed :math:`N_2` / :math:`CO`, against VASP.

    A stretched diatomic is the cleanest possible force test: one bond, one
    non-zero component per atom, equal and opposite. It needs a reference
    calculation this repository does not ship.
    """

    @pytest.mark.parametrize("case", ["n2_stretched", "co_stretched"])
    def test_forces_against_vasp(self, case):
        directory = os.path.join(REFERENCE_DIR, case)
        outcar = os.path.join(directory, "OUTCAR")
        if not os.path.isfile(outcar):
            pytest.skip(
                f"no reference calculation under {directory}; see "
                f"tests/test_energy_differences.py for the expected layout")

        pytest.importorskip("ase")
        from ase.io import read as ase_read

        from poraque.calculator import Poraque
        from poraque.ml import BUNDLE_FILENAME, resolve_bundle_path

        bundle = resolve_bundle_path(os.path.join(_ROOT, "models",
                                                  BUNDLE_FILENAME))
        if not os.path.isfile(bundle):
            pytest.skip("no trained model bundle to run inference with")

        atoms = ase_read(os.path.join(directory, "POSCAR"))
        atoms.pbc = True
        atoms.calc = Poraque(models=bundle,
                             potcar=os.path.join(directory, "POTCAR"),
                             device="cpu")
        predicted = atoms.get_forces()
        reference = outcar_forces(outcar)

        assert predicted.shape == reference.shape
        assert np.abs(predicted - reference).max() < 0.5, (
            f"{case}: max |dF| = {np.abs(predicted - reference).max():.4f} eV/Ang")
