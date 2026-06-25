# -*- coding: utf-8 -*-
# file: test_upf.py
"""Tests for the UPF (.upf) pseudopotential reader and the bundled registry.

These exercise compatibility with PseudoDojo / Quantum ESPRESSO UPF v2 files:
parsing the header (element, valence charge, functional), reading the local
potential onto its radial mesh (Rydberg -> Hartree), and selecting the
functional-specific file (LDA vs PBE) from the local ``pseudos/`` registry.
"""

import numpy as np
import pytest

from poraque.core import Grid, System
from poraque.pseudopotentials import (
    UPFLocalPseudopotential,
    build_pseudopotential_potential,
    find_pseudo_dir,
    read_pseudopotential,
    read_upf,
    registry_pseudopotential,
    resolve_pseudopotentials,
)
from poraque.pseudopotentials.registry import normalize_functional, pseudo_path


@pytest.fixture
def si_lda_path():
    return pseudo_path("Si", "LDA")


@pytest.fixture
def si_system():
    cell = np.eye(3) * 6.0
    return System([[3.0, 3.0, 3.0]], [14], cell, pbc=True)


class TestRegistryLayout:
    def test_pseudo_dir_has_functional_subdirs(self):
        root = find_pseudo_dir()
        assert (root / "LDA").is_dir()
        assert (root / "PBE").is_dir()

    def test_functional_normalization(self):
        assert normalize_functional("lda") == "LDA"
        assert normalize_functional("PBE") == "PBE"
        assert normalize_functional("gga") == "PBE"

    def test_unknown_functional_raises(self):
        with pytest.raises(KeyError):
            normalize_functional("scan")


class TestUPFParsing:
    def test_header_fields(self, si_lda_path):
        pp = read_upf(str(si_lda_path))
        assert isinstance(pp, UPFLocalPseudopotential)
        assert pp.symbol == "Si"
        assert pp.z_valence == pytest.approx(4.0)
        assert pp.functional == "LDA"
        assert pp.r.size == pp.v_loc.size > 100

    def test_local_potential_is_attractive(self, si_lda_path):
        pp = read_upf(str(si_lda_path))
        # The local channel is negative (attractive) near the core.
        assert pp.radial_potential(np.array([0.0]))[0] < 0.0

    def test_coulomb_tail_in_hartree(self, si_lda_path):
        # Far from the core, v_loc(r) -> -Z_v / r in Hartree atomic units.
        pp = read_upf(str(si_lda_path))
        for r in (4.0, 6.0, 8.0):
            assert pp.radial_potential(np.array([r]))[0] == pytest.approx(
                -pp.z_valence / r, rel=2e-2
            )

    def test_beyond_mesh_uses_coulomb_tail(self, si_lda_path):
        pp = read_upf(str(si_lda_path))
        r_far = pp._r_max + 5.0
        assert pp.radial_potential(np.array([r_far]))[0] == pytest.approx(
            -pp.z_valence / r_far
        )

    def test_read_pseudopotential_dispatches_to_upf(self, si_lda_path):
        pp = read_pseudopotential(str(si_lda_path))
        assert isinstance(pp, UPFLocalPseudopotential)

    def test_lda_and_pbe_differ(self):
        lda = registry_pseudopotential("Si", "LDA")
        pbe = registry_pseudopotential("Si", "PBE")
        assert lda.functional == "LDA"
        assert pbe.functional == "PBE"
        # Same valence, but distinct functionals / local channels.
        assert lda.z_valence == pbe.z_valence == pytest.approx(4.0)
        assert not np.allclose(lda.v_loc[: min(lda.v_loc.size, pbe.v_loc.size)],
                               pbe.v_loc[: min(lda.v_loc.size, pbe.v_loc.size)])


class TestUPFResolution:
    def test_resolve_upf_uses_functional(self, si_system):
        table = resolve_pseudopotentials(si_system, "upf", functional="PBE")
        assert isinstance(table["Si"], UPFLocalPseudopotential)
        assert table["Si"].functional == "PBE"

    def test_build_potential_counts_valence(self, si_system):
        grid = Grid((24, 24, 24), si_system.cell, pbc=True)
        v_ext, n_valence = build_pseudopotential_potential(
            grid, si_system, "upf", functional="LDA"
        )
        assert v_ext.shape == grid.shape
        assert n_valence == pytest.approx(4.0)
        assert v_ext.min() < 0.0  # attractive

    def test_per_element_upf_mapping(self, si_system):
        table = resolve_pseudopotentials(si_system, {"Si": "upf"}, functional="LDA")
        assert isinstance(table["Si"], UPFLocalPseudopotential)
        assert table["Si"].functional == "LDA"
