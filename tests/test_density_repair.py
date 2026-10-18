# -*- coding: utf-8 -*-
# file: test_density_repair.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Sanity and repair of a predicted density before a Harris–Foulkes evaluation.

The repair is the one place three callers (the Gamma evaluator bridge, the
PAW Bloch solver and the benchmark's SCF diagnostic) agree on what a
"nonnegative and normalized" density is and what to do about one that is
not. These tests pin the contract rather than the arithmetic: the raw array
is never touched, the floor acts on the physically non-negative quantities
(the eigen-densities, not :math:`m`), one factor scales density,
magnetization and PAW records alike, and the diagnostics describe exactly
the defect the repair removed.
"""

import numpy as np
import pytest

from poraque.fields import ChargeDensity, FieldGrid, SpinDensity
from poraque.physics.density_repair import (
    DensityRepair,
    inspect_density,
    repair_density,
    require_valid_density,
)
from poraque.physics.harris import integrate


@pytest.fixture
def grid():
    return FieldGrid((6, 5, 4), np.diag([5.0, 4.0, 3.0]))


def ringing_density(grid, electrons, depth=0.02, seed=0):
    """A positive density with a few voxels pushed below zero, like FNO ringing."""
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.3 * rng.normal(size=grid.shape)
    rho = np.abs(rho)
    rho *= electrons / integrate(rho, grid)
    rho.flat[[0, 7, 19]] = -depth
    return rho


class TestTheInspectionDescribesTheDefectWithoutTouchingTheArray:
    def test_a_clean_density_has_no_problems(self, grid):
        rho = np.full(grid.shape, 8.0 / grid.volume)
        d = inspect_density(rho, grid, 8.0)
        assert d.nonnegative and d.normalized() and d.problems() == []
        assert d.charge_defect == pytest.approx(0.0, abs=1e-12)
        assert require_valid_density(rho, grid, 8.0) == d

    def test_negative_voxels_and_a_charge_leak_are_both_named(self, grid):
        rho = ringing_density(grid, 8.0) * 0.98  # three voxels at -0.0196, 2 % leaked
        before = rho.copy()
        d = inspect_density(rho, grid, 8.0)
        assert np.array_equal(rho, before)
        assert not d.nonnegative and not d.normalized()
        assert d.negative_voxel_fraction == pytest.approx(3 / rho.size)
        assert d.minimum == pytest.approx(-0.02 * 0.98)
        assert d.negative_electrons == pytest.approx(3 * 0.02 * 0.98 * grid.volume / rho.size)
        assert d.charge_defect < 0 and d.relative_charge_defect < 0
        problems = d.problems()
        assert len(problems) == 2
        assert "below 0" in problems[0] and "valence count" in problems[1]
        with pytest.raises(ValueError, match="nonnegative and normalized"):
            require_valid_density(rho, grid, 8.0)

    def test_the_spin_check_is_on_the_eigen_densities_not_on_m(self, grid):
        """m may take either sign; what must not go negative is (rho - |m|)/2."""
        rho = np.full(grid.shape, 8.0 / grid.volume)
        m = -0.5 * rho  # negative everywhere, but |m| < rho: fine
        assert inspect_density(rho, grid, 8.0, m).nonnegative
        m.flat[3] = 1.5 * rho.flat[3]  # |m| > rho at one voxel: not a density matrix
        d = inspect_density(rho, grid, 8.0, m)
        assert not d.nonnegative and d.spin
        assert "eigen-densities" in d.problems()[0]

    @pytest.mark.parametrize("bad", ["shape", "nan", "electrons", "floor", "mshape"])
    def test_malformed_input_is_refused(self, grid, bad):
        rho = np.full(grid.shape, 1.0)
        kwargs = dict(electrons=8.0)
        if bad == "shape":
            rho = rho[0]
        elif bad == "nan":
            rho.flat[0] = np.nan
        elif bad == "electrons":
            kwargs["electrons"] = 0.0
        elif bad == "floor":
            kwargs["floor"] = -1e-3
        else:
            kwargs["magnetization"] = np.zeros((2, *grid.shape))
        with pytest.raises(ValueError):
            inspect_density(rho, grid, magnetization=kwargs.pop("magnetization", None), **kwargs)


class TestTheRepairIsAFloorAndOneGlobalFactor:
    def test_scalar_repair_matches_the_field_normalizer(self, grid):
        """The fields' `normalized` and this module must not disagree on the answer."""
        rho = ringing_density(grid, 8.0) * 1.03
        raw = rho.copy()
        repair = repair_density(rho, grid, 8.0)
        assert np.array_equal(rho, raw), "the raw prediction is never modified"
        assert repair.repaired and repair.magnetization is None and repair.records == ()
        assert integrate(repair.charge, grid) == pytest.approx(8.0, rel=1e-13)
        assert repair.input_electrons == pytest.approx(8.0, rel=1e-13)
        assert repair.charge.min() >= 0.0
        reference = ChargeDensity(raw, grid, None).normalized(8.0)
        assert np.allclose(repair.charge, reference.data, rtol=1e-13, atol=0)
        assert repair.clipped_electrons == pytest.approx(repair.diagnostics.negative_electrons)
        assert repair.normalization_factor == pytest.approx(
            8.0 / integrate(np.maximum(raw, 0), grid))
        record = repair.as_dict()
        assert record["raw_electrons"] == pytest.approx(integrate(raw, grid))
        assert record["repaired"] is True
        assert not any(isinstance(value, np.ndarray) for value in record.values())

    def test_the_floor_is_a_parameter(self, grid):
        rho = ringing_density(grid, 8.0)
        repair = repair_density(rho, grid, 8.0, floor=1e-6)
        assert repair.charge.min() >= 1e-6 * repair.normalization_factor
        assert repair.diagnostics.floor == 1e-6

    def test_spin_repair_keeps_the_density_matrix_positive_and_the_polarization(self, grid):
        rho = ringing_density(grid, 8.0, depth=0.05)
        m = 0.4 * np.abs(rho) * np.sign(np.cos(np.arange(rho.size)).reshape(rho.shape))
        m.flat[11] = 1.3 * rho.flat[11]  # |m| > rho: the minus eigen-density goes negative
        untouched = np.ones(rho.shape, bool)
        untouched.flat[[0, 7, 19, 11]] = False
        rec = [np.arange(5.0) + i for i in range(3)]
        repair = repair_density(rho, grid, 8.0, m, rec)
        n_plus = (repair.charge + np.abs(repair.magnetization)) / 2
        n_minus = (repair.charge - np.abs(repair.magnetization)) / 2
        assert n_plus.min() >= 0 and n_minus.min() >= -1e-15
        assert integrate(repair.charge, grid) == pytest.approx(8.0, rel=1e-13)
        f = repair.normalization_factor
        # Where nothing was clipped, m/rho is the prediction's own polarization.
        assert np.allclose(repair.magnetization[untouched] / repair.charge[untouched],
                           m[untouched] / rho[untouched])
        assert np.allclose(repair.charge[untouched], f * rho[untouched])
        # The sign of m survives wherever m survives; where both eigen-densities
        # were clipped (rho itself negative) m is zero, as rho_up = rho_down = 0.
        kept = repair.magnetization != 0
        assert np.all(np.sign(repair.magnetization[kept]) == np.sign(m[kept]))
        assert np.all(repair.charge[~kept] == 0)
        # One factor for the records too.
        assert len(repair.records) == 3
        for before, after in zip(rec, repair.records):
            assert np.allclose(after, f * before)
        # The same arithmetic as SpinDensity.normalized, which clips rho_up and rho_down.
        reference = SpinDensity(rho, m, grid, None).normalized(8.0)
        assert np.allclose(repair.charge, reference.total, rtol=1e-12, atol=1e-15)
        assert np.allclose(repair.magnetization, reference.magnetization, rtol=1e-12, atol=1e-15)

    def test_vector_magnetization_keeps_its_direction(self, grid):
        rho = np.full(grid.shape, 8.0 / grid.volume)
        direction = np.array([0.6, 0.0, 0.8])
        m = 0.5 * direction[:, None, None, None] * rho
        m[:, 0, 0, 0] *= 3.0  # |m| = 1.5 rho at one voxel
        repair = repair_density(rho * 1.01, grid, 8.0, m)
        unit = repair.magnetization / np.linalg.norm(repair.magnetization, axis=0)
        assert np.allclose(unit, direction[:, None, None, None])
        assert np.linalg.norm(repair.magnetization, axis=0)[0, 0, 0] <= repair.charge[0, 0, 0] + 1e-15

    def test_a_density_with_no_positive_charge_cannot_be_normalized(self, grid):
        with pytest.raises(ValueError, match="no positive charge"):
            repair_density(np.full(grid.shape, -1.0), grid, 8.0)

    def test_the_unrepaired_record_reads_as_identity(self, grid):
        rho = np.full(grid.shape, 8.0 / grid.volume)
        d = inspect_density(rho, grid, 8.0)
        plain = DensityRepair.unrepaired(rho, None, (np.ones(4),), d)
        assert not plain.repaired
        assert plain.normalization_factor == 1.0 and plain.clipped_electrons == 0.0
        assert plain.input_electrons == pytest.approx(8.0)
        assert plain.as_dict()["repaired"] is False
