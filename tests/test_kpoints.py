# -*- coding: utf-8 -*-
# file: test_kpoints.py
"""Tests for Brillouin-zone sampling and the plane-wave grid helpers."""

import numpy as np
import pytest

from poraque.core import Grid, monkhorst_pack_kpoints
from poraque.core.kpoints import gamma_only


class TestMonkhorstPack:
    def test_gamma_only(self):
        kpts, w = gamma_only()
        assert kpts.shape == (1, 3)
        assert np.allclose(kpts, 0.0)
        assert w.sum() == pytest.approx(1.0)

    def test_weights_normalized(self):
        kpts, w = monkhorst_pack_kpoints((4, 4, 4))
        assert w.sum() == pytest.approx(1.0)

    def test_time_reversal_reduces_count(self):
        full, wf = monkhorst_pack_kpoints((4, 4, 4), reduce_time_reversal=False)
        red, wr = monkhorst_pack_kpoints((4, 4, 4), reduce_time_reversal=True)
        assert len(red) < len(full)
        # Folding preserves the total weight.
        assert wr.sum() == pytest.approx(wf.sum())

    def test_scalar_size_broadcasts(self):
        a, _ = monkhorst_pack_kpoints(2, reduce_time_reversal=False)
        b, _ = monkhorst_pack_kpoints((2, 2, 2), reduce_time_reversal=False)
        assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))


class TestPlaneWaveGrid:
    def test_from_ecut_denser_for_higher_cutoff(self):
        cell = np.eye(3) * 10.0
        low = Grid.from_ecut(cell, ecut=10.0)
        high = Grid.from_ecut(cell, ecut=40.0)
        assert all(h >= l for h, l in zip(high.shape, low.shape))
        assert any(h > l for h, l in zip(high.shape, low.shape))
        assert high.ecut == 40.0

    def test_from_ecut_even_shape(self):
        grid = Grid.from_ecut(np.eye(3) * 8.0, ecut=20.0)
        assert all(n % 2 == 0 for n in grid.shape)

    def test_kinetic_g2_gamma_matches_g2(self):
        grid = Grid((12, 12, 12), np.eye(3) * 6.0)
        assert np.allclose(grid.kinetic_g2(), grid.get_g2())

    def test_kinetic_g2_shifted_by_kpoint(self):
        grid = Grid((10, 10, 10), np.eye(3) * 6.0)
        kcart = grid.kpoint_to_cartesian([0.5, 0.0, 0.0])
        shifted = grid.kinetic_g2(kcart)
        # The minimum of |G + k|^2 is no longer zero for a non-Gamma k-point.
        assert shifted.min() > 1e-8
