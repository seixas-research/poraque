# -*- coding: utf-8 -*-
# file: test_paw_energy.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The PAW total energy (:mod:`poraque.physics.paw`) and its differentiable twin
with the forces (:mod:`poraque.ml.energy`).

The expression reconstructs VASP's ``TOTEN`` from the ``CHGCAR`` grid, its
occupancy records, the ``TAUCAR`` and the ``POTCAR``. It was measured against
static VASP 6.6.1 runs: diamond and COS (``LASPH = .FALSE.``) to 0.2 and 0.6 meV,
a displaced CN cell (``LASPH = .TRUE.``) to 5 µeV, and the 32-atom Pt bulk cell
of this repository to 1.2 meV. Getting there turned on one point the VASP
source makes easy to miss: the one-center strength :math:`\bar D` must be built
with the reference-atom potentials VASP **recomputes** at start-up, not the
``POTCAR`` blocks; with the blocks, the Pt cell was 1.17 eV off.

Tests that need the POTCAR library run when ``PORAQUE_TEST_POTCAR_DIR`` points
at one, and skip otherwise.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.physics.paw import (  # noqa: E402
    AUTOA,
    compensation_shapes,
    radial_hartree,
    simpson_weights,
)
from poraque.physics.xc import pbe_xc  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIBRARY = os.environ.get("PORAQUE_TEST_POTCAR_DIR") or None
needs_potcar = pytest.mark.skipif(
    LIBRARY is None or not os.path.exists(os.path.join(LIBRARY or "", "Pt",
                                                       "POTCAR")),
    reason="PORAQUE_TEST_POTCAR_DIR with a Pt POTCAR is not set")
PT_RUN = os.path.join(_ROOT, "data", "vasp", "structures", "structure_0000")


def _log_grid(n=401, r0=1e-5, r1=8.0):
    return r0 * (r1 / r0) ** (np.arange(n) / (n - 1))


class TestTheFunctionalIsOneExpressionForBothBackends:
    """NumPy and torch evaluate the same PBE, and it is the grid's PBE."""

    def test_numpy_and_torch_agree_and_differentiate(self):
        rng = np.random.default_rng(0)
        n, m, g = rng.random(40) * 0.5, rng.random(40) * 0.1, rng.random(40)
        args = (n / 2 + m / 2, n / 2 - m / 2, g / 2, g / 3, g)
        reference = sum(pbe_xc(*args))
        tensors = [torch.tensor(a, dtype=torch.float64, requires_grad=True)
                   for a in args]
        value = sum(pbe_xc(*tensors))
        value.sum().backward()
        np.testing.assert_allclose(value.detach().numpy(), reference,
                                   atol=1e-14)
        assert all(torch.isfinite(t.grad).all() for t in tensors)

    def test_the_unpolarized_limit_is_the_grid_functional(self):
        from poraque.fields import FieldGrid
        from poraque.physics import energy

        grid = FieldGrid((20, 20, 20), np.eye(3) * 5.0)
        x = np.stack(np.meshgrid(*[np.arange(20) / 20] * 3,
                                 indexing="ij"), -1) * 5.0
        density = 0.05 + 0.4 * np.exp(-((x - 2.5) ** 2).sum(-1))
        expected = (energy.pbe_exchange_energy(density, grid)
                    + energy.pbe_correlation_energy(density, grid))
        bohr, squared = energy._density_and_gradient(density, grid)
        gradient = np.sqrt(squared)
        lda, correction = pbe_xc(bohr / 2, bohr / 2, gradient / 2,
                                 gradient / 2, gradient)
        mine = grid.integrate((lda + correction)
                              * energy._HA_PER_BOHR3_TO_EV_PER_ANG3)
        assert mine == pytest.approx(expected, rel=1e-12)


class TestTheRadialNumericsAreVasps:
    """The radial rules reproduce closed forms on VASP's logarithmic grid."""

    def test_the_hartree_energy_of_a_gaussian(self):
        r = _log_grid()
        weights = simpson_weights(r)
        sigma = 0.5
        density = np.exp(-r ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** 1.5
        rho00 = np.sqrt(4 * np.pi) * r ** 2 * density     # r^2 n_00, one electron
        from poraque.physics.paw import FELECT

        exact = FELECT / (2 * np.sqrt(np.pi) * sigma)
        assert radial_hartree(0, r, weights, rho00) == pytest.approx(
            exact, rel=1e-6)

    def test_the_compensation_shapes_are_normalized_and_vanish_at_the_radius(self):
        r = _log_grid()
        rmax = float(r[np.argmax(r > 1.0)])
        shapes = compensation_shapes(r, rmax, 4)
        weights = simpson_weights(r)
        for L in range(5):
            moment = np.sum(shapes[L] * r ** L * weights)
            assert moment == pytest.approx(1.0, rel=2e-4)
            # q_i r_max are zeros of j_L and g'(r_max) = 0: a double zero.
            inside = r <= rmax
            g = shapes[L, inside] / r[inside] ** 2
            assert abs(g[-1]) < 1e-3 * np.abs(g).max()
            assert not shapes[L, r > rmax].any()

    def test_units_are_vasps(self):
        assert AUTOA == 0.529177249


@needs_potcar
class TestTheEnergyIsVaspsTotalEnergy:
    """Pt bulk, 32 atoms, spin-polarized, records to L = 4."""

    def test_the_pt_cell_reproduces_toten(self):
        from poraque.fields import KineticEnergyDensity, SpinDensity
        from poraque.fields.vasp.augmentation import occupancy_arrays
        from poraque.fields.vasp.paw_setup import PawSetup
        from poraque.fields.vasp.volumetric import read_augmentation_blocks
        from poraque.physics.paw import paw_total_energy

        chgcar = os.path.join(PT_RUN, "CHGCAR")
        if not os.path.exists(os.path.join(PT_RUN, "OUTCAR")):
            pytest.skip("the Pt reference run is not in this checkout")
        density = SpinDensity.read(chgcar)
        tau = KineticEnergyDensity.read(os.path.join(PT_RUN, "TAUCAR"),
                                        grid=density.grid).data
        setups = {"Pt": PawSetup.from_file(os.path.join(LIBRARY, "Pt",
                                                        "POTCAR"))[0]}
        _, blocks = read_augmentation_blocks(chgcar)
        occupancies, lengths = occupancy_arrays(blocks, channels=2)
        records = [[occupancies[a, s, :lengths[a]] for s in range(2)]
                   for a in range(len(lengths))]
        terms = paw_total_energy(density.structure, density.grid,
                                 np.asarray(density.data[0]),
                                 np.asarray(density.data[1]), np.asarray(tau),
                                 records, setups, encut=400.0, lasph=False,
                                 entropy=-0.58236)
        assert terms["total"] == pytest.approx(-195.34322336, abs=5e-3)


@needs_potcar
class TestTheTorchEnergyIsTheNumpyEnergy:
    """The differentiable one-center energy is the validated one."""

    @pytest.mark.parametrize("lasph", [False, True])
    def test_random_occupancies_give_the_same_one_center_energy(self, lasph):
        from poraque.fields.vasp.paw_setup import PawSetup
        from poraque.ml.energy import OneCenterTorch
        from poraque.physics.paw import OneCenterEnergy

        setup = PawSetup.from_file(os.path.join(LIBRARY, "Pt", "POTCAR"))[0]
        numpy_energy = OneCenterEnergy(setup, lasph=lasph)
        rng = np.random.default_rng(1)
        records = numpy_energy.atomic_record()[None, None, :] * (
            1.0 + 0.05 * rng.standard_normal((2, 2, len(numpy_energy.layout))))
        records[:, 1] *= 0.02                               # a small moment
        expected = [numpy_energy.energy(list(atom))["total"] for atom in records]
        module = OneCenterTorch(numpy_energy)
        got = module(torch.as_tensor(records)).detach().numpy()
        np.testing.assert_allclose(got, expected, atol=1e-8)

    def test_the_recomputed_reference_potentials_match_the_file_away_from_the_nucleus(self):
        from poraque.fields.vasp.paw_setup import PawSetup
        from poraque.physics.paw import OneCenterEnergy

        setup = PawSetup.from_file(os.path.join(LIBRARY, "Pt", "POTCAR"))[0]
        energy = OneCenterEnergy(setup, lasph=False)
        ae, _ = energy.reference
        away = energy.r > 0.05
        difference = np.abs(ae[away] - setup.tables["aepotential"][away])
        assert difference.max() < 0.5


def _untrained_pt_operators(tmp_path):
    """Small untrained ext2chg and chg2tau operators for Pt, on the CPU."""
    from poraque.fields.atomic import resolve_library
    from poraque.fields.vasp.augmentation import record_layout
    from poraque.ml import FieldOperator
    from poraque.ml.paw import OccupancyTransform
    from poraque.ml.transforms import Asinh, Channelwise, Standardize

    atoms = os.path.join(_ROOT, "data", "vasp", "isolated_atoms")
    if not os.path.exists(os.path.join(atoms, "Pt", "CHGCAR")):
        pytest.skip("the Pt reference atom is not in this checkout")
    library = resolve_library(atoms, cache=str(tmp_path))
    from poraque.fields.vasp.paw_setup import PawSetup
    from poraque.physics.paw import OneCenterEnergy

    layouts = {78: (2, 2, 0, 0, 1, 1)}
    length = len(record_layout(layouts[78]))
    # Occupancies near the free atom's, so the one-center densities stay
    # physical (positive) and the energy is smooth where it is probed.
    atom = OneCenterEnergy(PawSetup.from_file(
        os.path.join(LIBRARY, "Pt", "POTCAR"))[0], lasph=False)
    mean = np.stack([atom.atomic_record(), np.zeros(length)])
    transform = OccupancyTransform(layouts, 2, {78: mean},
                                   {78: np.full((2, length), 1e-3)})
    common = dict(width=4, modes=3, n_layers=1, projection_channels=8,
                  device="cpu", use_coordinates=False, site_layouts=layouts,
                  site_transform=transform, init_seed=0)
    torch.manual_seed(0)
    density = FieldOperator(
        "ext2chg", in_channels=2, out_channels=2, readout_g_max=3.0,
        input_transform=Channelwise([Standardize(0.0, 50.0), Asinh(0.1)]),
        target_transform=Channelwise([Asinh(0.05), Asinh(0.05)]),
        baseline=library, structure_input={"superposition": True,
                                           "elements": []}, **common)
    kinetic = FieldOperator(
        "chg2tau", in_channels=2, out_channels=1, occupancy_channels=2,
        input_transform=Channelwise([Asinh(0.1), Asinh(0.1)]),
        target_transform=Asinh(5.0), **common)
    return density, kinetic


@needs_potcar
class TestTheForcesAreTheGradientOfTheEnergy:
    """
    Autograd through V_ext, ext2chg, chg2tau and the energy equals a central
    difference of the same energy: the forces are conservative by construction.
    Untrained operators are enough; the property is of the chain.
    """

    def test_autograd_matches_a_central_difference(self, tmp_path):
        from poraque.fields import FieldGrid
        from poraque.fields.structure import Structure
        from poraque.ml.energy import PawEnergyChain

        density, kinetic = _untrained_pt_operators(tmp_path)
        cell = np.eye(3) * 3.92 * np.array([1.0, 1.0, 1.0])
        structure = Structure(cell, ["Pt"], [2],
                              np.array([[0.02, 0.0, 0.01], [0.5, 0.48, 0.5]]))
        chain = PawEnergyChain(density, kinetic, structure,
                               FieldGrid((16, 16, 16), cell), LIBRARY,
                               lasph=False, device="cpu")
        # The chain's fields are the operator's own predictions.
        from poraque.fields import ExternalPotential
        from poraque.fields.vasp.potcar import Potcar

        external = ExternalPotential.from_potcar_tables(
            structure, chain.grid,
            Potcar.from_library(LIBRARY, ["Pt"], parse_tables=True))
        frac = torch.as_tensor(np.asarray(structure.scaled_positions))
        chain.normalize_density = False         # predict() does not rescale
        with torch.no_grad():
            n, m, records, _ = chain.fields(frac)
        chain.normalize_density = True
        predicted = density.predict(external)
        np.testing.assert_allclose(n.numpy(), predicted.data[0], atol=1e-8)
        np.testing.assert_allclose(m.numpy(), predicted.data[1], atol=1e-8)
        occupancies = density.predict_occupancies(external)
        np.testing.assert_allclose(records[0, 0].numpy(), occupancies[0][0],
                                   atol=1e-8)

        _, forces = chain.forces()
        frac = np.asarray(structure.scaled_positions)
        h = 1e-4
        step = np.array([h, 0.0, 0.0]) @ np.linalg.inv(cell)
        plus, minus = frac.copy(), frac.copy()
        plus[0] += step
        minus[0] -= step
        with torch.no_grad():
            e_plus = float(chain.energy(torch.as_tensor(plus)))
            e_minus = float(chain.energy(torch.as_tensor(minus)))
        assert forces[0, 0] == pytest.approx(-(e_plus - e_minus) / (2 * h),
                                             rel=1e-4, abs=1e-5)

    def test_the_ase_calculator_is_the_chain(self, tmp_path):
        ase = pytest.importorskip("ase")
        from poraque.calculator import PoraquePAW
        from poraque.fields.structure import Structure
        from poraque.ml.energy import PawEnergyChain

        density, kinetic = _untrained_pt_operators(tmp_path)
        atoms = ase.Atoms("Pt2", cell=np.eye(3) * 3.92, pbc=True,
                          scaled_positions=[[0.02, 0.0, 0.01],
                                            [0.5, 0.48, 0.5]])
        atoms.calc = PoraquePAW(potcar_dir=LIBRARY, lasph=False,
                                resolution=16, device="cpu",
                                density_operator=density,
                                kinetic_operator=kinetic)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        chain = next(iter(atoms.calc._chains.values()))
        structure = Structure(np.asarray(atoms.cell), ["Pt"], [2],
                              atoms.get_scaled_positions())
        direct = PawEnergyChain(density, kinetic, structure, chain.grid,
                                LIBRARY, lasph=False, device="cpu")
        expected_energy, expected_forces = direct.forces()
        assert energy == pytest.approx(expected_energy, abs=1e-8)
        np.testing.assert_allclose(forces, expected_forces, atol=1e-8)
