# -*- coding: utf-8 -*-
# file: test_pseudopotentials.py
"""Tests for the modular pseudopotential layer."""

import numpy as np
import pytest

from poraque.core import Grid, System
from poraque.pseudopotentials import (
    GaussianCorePP,
    SoftCoulombPP,
    build_pseudopotential_potential,
    default_pseudopotential,
    read_pseudopotential,
    resolve_pseudopotentials,
    valence_electrons,
)


@pytest.fixture
def si_system():
    cell = np.eye(3) * 6.0
    return System([[3.0, 3.0, 3.0]], [14], cell, pbc=True)


class TestValence:
    def test_known_element(self):
        assert valence_electrons("Si") == 4
        assert valence_electrons("O") == 6

    def test_all_electron_fallback(self):
        # Unlisted element falls back to the full nuclear charge.
        assert valence_electrons("U", atomic_number=92) == 92


class TestLocalForms:
    def test_soft_coulomb_tail(self):
        pp = SoftCoulombPP("Si", 4.0, rc=0.8)
        r = np.array([10.0])
        assert pp.radial_potential(r)[0] == pytest.approx(-4.0 / np.sqrt(100 + 0.64))

    def test_gaussian_core_is_finite_at_origin(self):
        pp = GaussianCorePP("Si", 4.0, rc=0.5)
        v0 = pp.radial_potential(np.array([0.0]))[0]
        assert np.isfinite(v0)
        assert v0 < 0


class TestResolution:
    def test_auto_uses_valence_charge(self, si_system):
        table = resolve_pseudopotentials(si_system, "auto")
        assert table["Si"].z_valence == 4.0

    def test_build_potential_counts_valence(self, si_system):
        grid = Grid((20, 20, 20), si_system.cell, pbc=True)
        v_ext, n_valence = build_pseudopotential_potential(grid, si_system, "auto")
        assert v_ext.shape == grid.shape
        assert n_valence == pytest.approx(4.0)
        assert v_ext.min() < 0  # attractive

    def test_explicit_object_applied_to_all(self, si_system):
        pp = SoftCoulombPP("Si", 4.0, rc=1.0)
        table = resolve_pseudopotentials(si_system, pp)
        assert table["Si"] is pp


class TestIO:
    def test_read_text_format(self, tmp_path):
        path = tmp_path / "Si.psp"
        path.write_text(
            "# silicon local pseudopotential\n"
            "element: Si\n"
            "z_valence: 4\n"
            "form: soft_coulomb\n"
            "rc: 0.8\n"
        )
        pp = read_pseudopotential(str(path))
        assert isinstance(pp, SoftCoulombPP)
        assert pp.symbol == "Si"
        assert pp.z_valence == 4.0
        assert pp.rc == 0.8

    def test_read_json_format(self, tmp_path):
        path = tmp_path / "Si.json"
        path.write_text('{"element": "Si", "z_valence": 4, "form": "gaussian", "rc": 0.5}')
        pp = read_pseudopotential(str(path))
        assert isinstance(pp, GaussianCorePP)
        assert pp.z_valence == 4.0
