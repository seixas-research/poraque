# -*- coding: utf-8 -*-
# file: test_hartree.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The Hartree potential, solved from the density by FFT.

Unlike the learned fields this one has an *exact* answer, so it is tested
against closed-form solutions rather than against tolerances chosen to
accommodate a model:

* a single Fourier mode :math:`\rho = \rho_0\cos(\mathbf G\cdot\mathbf r)` has
  :math:`v_{\rm H} = 4\pi e^2\rho_0\cos(\mathbf G\cdot\mathbf r)/G^2` exactly,
  which pins the prefactor, the units and the sign in one assertion;
* a uniform density gives exactly zero in the neutralizing-background
  convention;
* the potential must satisfy :math:`\nabla^2 v_{\rm H} = -4\pi e^2\rho`, which
  is checked directly in reciprocal space and is independent of how the
  solution was obtained.

Also covered: that ``LOCPOT`` is written **unscaled**. ``CHGCAR`` stores
:math:`\rho\Omega` and ``LOCPOT`` stores the potential itself; writing one with
the other's convention produces a file that opens cleanly and is wrong by a
factor of the cell volume.
"""

import os

import numpy as np
import pytest

from poraque.fields import (
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    HartreePotential,
)
from poraque.fields.constants import COULOMB_CONSTANT_EV_ANGSTROM as KE
from poraque.fields.vasp.poscar import Poscar
from poraque.physics import hartree_energy, hartree_potential

LENGTH = 8.0


@pytest.fixture
def grid():
    return FieldGrid((24, 24, 24), np.eye(3) * LENGTH)


@pytest.fixture
def poscar():
    return Poscar(cell=np.eye(3) * LENGTH, symbols=["Pt"], counts=[1],
                  scaled_positions=[[0.5, 0.5, 0.5]])


@pytest.fixture
def cosine(grid, poscar):
    """A single-mode density, whose Hartree potential is known in closed form."""
    fractional = grid.scaled_coordinates()[..., 0]
    amplitude = 0.3
    density = ChargeDensity(amplitude * np.cos(2 * np.pi * fractional),
                            grid, poscar)
    wavevector = 2 * np.pi / LENGTH
    exact = (4 * np.pi * KE * amplitude
             * np.cos(2 * np.pi * fractional) / wavevector ** 2)
    return density, exact


# ===================================================================== #
# The solver
# ===================================================================== #
class TestPoissonSolver:
    def test_matches_the_analytic_single_mode(self, cosine):
        r"""
        The one assertion that pins prefactor, units and sign together.

        :math:`4\pi e^2/G^2` with :math:`e^2` in eV·Å is the whole content of
        the solver; a missing factor of :math:`4\pi`, a Hartree/eV slip or a
        sign error all fail here and nowhere else in this file.
        """
        density, exact = cosine
        computed = HartreePotential.from_density(density)
        assert np.allclose(computed.data, exact, atol=1e-10)

    def test_a_uniform_density_gives_exactly_zero(self, grid, poscar):
        """
        The neutralizing background, stated as a number.

        With G = 0 removed a uniform density is exactly cancelled by its
        compensating background, so the potential is zero — not small.
        """
        density = ChargeDensity(np.full(grid.shape, 0.7), grid, poscar)
        assert np.abs(HartreePotential.from_density(density).data).max() == 0.0

    def test_satisfies_poissons_equation(self, grid, poscar):
        r"""
        :math:`\nabla^2 v_{\rm H} = -4\pi e^2\rho`, checked independently.

        The Laplacian is taken spectrally on the result, so this does not
        re-run the solver's own arithmetic — it tests the output against the
        equation it claims to solve. Only the G != 0 components are compared,
        since G = 0 was deliberately dropped from both sides.
        """
        rng = np.random.default_rng(0)
        values = rng.random(grid.shape)
        values -= values.mean()
        density = ChargeDensity(values, grid, poscar)

        potential = HartreePotential.from_density(density).data
        g2 = grid.get_g2()
        laplacian = np.fft.fftn(potential) * (-g2)
        expected = -4.0 * np.pi * KE * np.fft.fftn(density.data)

        nonzero = g2 > 1e-12
        assert np.allclose(laplacian[nonzero], expected[nonzero], rtol=1e-8)

    def test_is_linear_in_the_density(self, cosine, grid, poscar):
        """Poisson is linear; a solver that were not would be wrong."""
        density, _ = cosine
        single = HartreePotential.from_density(density).data
        doubled = HartreePotential.from_density(
            ChargeDensity(density.data * 2.0, grid, poscar)).data
        assert np.allclose(doubled, 2.0 * single)

    def test_accepts_lattice_vectors_instead_of_a_grid(self, cosine, grid):
        """The (rho, lattice_vectors) call form, without building a grid."""
        density, exact = cosine
        computed = hartree_potential(density.data, np.eye(3) * LENGTH)
        assert np.allclose(computed, exact, atol=1e-10)
        assert np.allclose(computed, hartree_potential(density.data, grid))

    def test_rejects_a_malformed_cell(self, cosine):
        density, _ = cosine
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            hartree_potential(density.data, np.eye(2))

    def test_is_consistent_with_the_hartree_energy(self, cosine, grid):
        r""":math:`E_{\rm H} = \tfrac12\int\rho v_{\rm H}` on the same field."""
        density, _ = cosine
        potential = HartreePotential.from_density(density)
        assert hartree_energy(density.data, grid) == pytest.approx(
            0.5 * grid.integrate(density.data * potential.data), rel=1e-10)


# ===================================================================== #
# The field object
# ===================================================================== #
class TestHartreeField:
    def test_declares_the_locpot_conventions(self, cosine):
        density, _ = cosine
        field = HartreePotential.from_density(density)
        assert field.default_filename == "LOCPOT"
        assert field.unit == "eV"
        assert field.volume_scaled is False, (
            "a LOCPOT stores the potential itself, not potential * volume")

    def test_writes_and_reads_back_unscaled(self, tmp_path, cosine):
        """
        The convention that is easy to get backwards and hard to notice.

        If ``volume_scaled`` were true the round trip would still succeed --
        it would divide by the same volume it multiplied -- so the check is
        against the *values on disk*, not against a round trip.
        """
        density, _ = cosine
        field = HartreePotential.from_density(density)
        path = str(tmp_path / "LOCPOT")
        field.write(path)

        back = HartreePotential.read(path)
        assert np.allclose(back.data, field.data, rtol=1e-9)

        # Read the first data value straight out of the file.
        with open(path) as handle:
            lines = handle.read().splitlines()
        blank = next(i for i, line in enumerate(lines) if not line.strip())
        first = float(lines[blank + 2].split()[0])
        assert first == pytest.approx(field.data.ravel(order="F")[0], rel=1e-6)

    def test_inherits_the_grid_and_structure(self, cosine, grid, poscar):
        density, _ = cosine
        field = HartreePotential.from_density(density)
        assert field.grid is grid
        assert field.structure is poscar

    def test_records_its_provenance(self, cosine):
        density, _ = cosine
        assert HartreePotential.from_density(density).metadata["source"] == "poisson"

    def test_a_bare_array_needs_a_grid_and_structure(self, cosine):
        density, _ = cosine
        with pytest.raises(ValueError, match="bare density array"):
            HartreePotential.from_density(density.data)

    def test_total_with_external_adds_on_the_shared_grid(self, cosine, grid,
                                                         poscar):
        r"""Both fields drop G = 0, so they are directly addable."""
        density, _ = cosine
        hartree = HartreePotential.from_density(density)
        external = ExternalPotential(np.full(grid.shape, -3.0), grid, poscar)
        total = hartree.total_with(external)
        assert np.allclose(total.data, hartree.data - 3.0)
        assert total.metadata["source"] == "hartree + external"

    def test_total_with_rejects_a_grid_mismatch(self, cosine, poscar):
        density, _ = cosine
        other = FieldGrid((8, 8, 8), np.eye(3) * LENGTH)
        external = ExternalPotential(np.zeros(other.shape), other, poscar)
        with pytest.raises(ValueError, match="share one mesh"):
            HartreePotential.from_density(density).total_with(external)

    def test_a_spin_density_contributes_through_its_total(self, grid, poscar):
        r"""
        The Hartree term is blind to polarisation.

        It is the classical repulsion of the whole charge, so
        :math:`v_{\rm H}[\rho_\uparrow, \rho_\downarrow]` depends only on
        :math:`\rho_\uparrow + \rho_\downarrow`.
        """
        from poraque.fields import SpinDensity

        rng = np.random.default_rng(1)
        up, down = rng.random(grid.shape), rng.random(grid.shape)
        spin = SpinDensity.from_up_down(up, down, grid, poscar)
        plain = ChargeDensity(up + down, grid, poscar)
        assert np.allclose(HartreePotential.from_density(spin).data,
                           HartreePotential.from_density(plain).data)


# ===================================================================== #
# Calculator wiring
# ===================================================================== #
class TestCalculatorAccessors:
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
        return Poraque(bundle, charges={"Pt": 11.0}, device="cpu")

    @pytest.fixture
    def atoms(self):
        ase = pytest.importorskip("ase")

        return ase.Atoms("Pt2", cell=np.eye(3) * 4.08, pbc=True,
                         scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_hartree_potential_is_a_field_like_the_others(self, calculator,
                                                          atoms):
        with pytest.warns(RuntimeWarning):
            potential = calculator.get_hartree_potential(atoms)
        assert isinstance(potential, HartreePotential)
        assert potential.data.shape == tuple(potential.grid.shape)
        assert potential.unit == "eV"

    def test_the_sibling_accessors_exist_and_share_one_grid(self, calculator,
                                                            atoms):
        """`get_hartree_potential` must behave like its neighbours."""
        with pytest.warns(RuntimeWarning):
            fields = [
                calculator.get_external_potential(atoms),
                calculator.get_charge_density(atoms),
                calculator.get_kinetic_energy_density(atoms),
                calculator.get_hartree_potential(atoms),
            ]
        shapes = {tuple(field.grid.shape) for field in fields}
        assert len(shapes) == 1

    def test_solves_from_the_normalized_density(self, calculator, atoms):
        """
        The field it is derived from is the one the energy uses.

        Solving from the raw prediction instead would give a potential
        inconsistent with the reported Hartree energy.
        """
        with pytest.warns(RuntimeWarning):
            density = calculator.get_charge_density(atoms)
            potential = calculator.get_hartree_potential(atoms)
        assert np.allclose(potential.data,
                           hartree_potential(density.data, density.grid))

    def test_with_external_returns_the_total_local_potential(self, calculator,
                                                             atoms):
        with pytest.warns(RuntimeWarning):
            hartree = calculator.get_hartree_potential(atoms)
            external = calculator.get_external_potential(atoms)
            total = calculator.get_hartree_potential(atoms, with_external=True)
        assert np.allclose(total.data, hartree.data + external.data)


# ===================================================================== #
# CLI
# ===================================================================== #
class TestInferenceCli:
    def test_locpot_flags_are_exposed(self):
        """
        Asserted on the parser rather than by running inference.

        A full run needs a trained bundle and a POTCAR; the flag's existence
        and its default are what can regress silently.
        """
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "poraque_inference",
            os.path.join(root, "scripts", "poraque_inference.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        parser = module.build_parser() if hasattr(module, "build_parser") else None
        if parser is None:
            pytest.skip("the inference script has no separable parser factory")
        options = {action.dest for action in parser._actions}
        assert "write_locpot" in options
        assert "locpot_total" in options
