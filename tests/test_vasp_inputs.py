# -*- coding: utf-8 -*-
# file: test_vasp_inputs.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
``poraque-vasp``: the three decks that read a predicted density back.

``ICHARG = 11`` is one trick asked three questions. The eigenvalues along a
**path** are a band structure; the same eigenvalues on a **mesh** are a density
of states; the energy from that mesh is a total energy. The differences are
small and every one of them is load-bearing, which is what this module pins:

**A DOS gets a mesh and a band structure gets a line.** Handing a line-mode
``KPOINTS`` to a tetrahedron integration is not a worse DOS, it is not a DOS;
handing a Monkhorst-Pack mesh to a band plot is not a worse plot, it is not a
plot. The two files come from two functions for exactly this reason, and the
tests check the right one reaches the right deck.

**The energy deck says which energy it means.** ``ICHARG = 11`` prints a
``TOTEN`` that is the Harris–Foulkes functional evaluated at the input density,
not a variational SCF energy. That caveat is written into the deck, and a test
holds it there — it is the sentence most likely to be lost to a tidy-up, and
the number is quoted in papers.

**Comments describe the tags that were actually written.** A deck asked for
``--ismear 0`` annotated with an explanation of the tetrahedron method argues
with itself, and a file that argues with itself is worse than an uncommented
one.

The command itself is exercised as a dry run: every mode is driven through
``build_parser`` and ``run`` exactly as the console script drives it, since
until now no test touched any script's ``run()`` body at all.
"""

import os
import sys

import numpy as np
import pytest

from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.vasp.poscar import Poscar
from poraque.fields.vasp.templates import (
    automatic_kpoints,
    dos_incar,
    kpoint_mesh_from_spacing,
    total_energy_incar,
    write_band_structure_deck,
    write_dos_deck,
    write_total_energy_deck,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import poraque_vasp                                            # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def tag(deck, name):
    """The value written for one ``INCAR`` tag, comment stripped."""
    for line in deck.splitlines():
        if line.split("=")[0].strip() == name:
            return line.split("=", 1)[1].split("#")[0].strip()
    return None


def comment(deck, name):
    """The comment written beside one ``INCAR`` tag, or ``""``."""
    for line in deck.splitlines():
        if line.split("=")[0].strip() == name:
            return line.split("#", 1)[1].strip() if "#" in line else ""
    return None


@pytest.fixture
def density(tmp_path):
    """A two-atom CHGCAR on a small cubic grid, written to disk."""
    cell = np.eye(3) * 6.0
    grid = FieldGrid((12, 12, 12), cell)
    structure = Poscar(cell, ["Si"], [2], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    values = 1.0 + 0.1 * np.cos(
        2 * np.pi * np.arange(12)[:, None, None] / 12)
    path = tmp_path / "CHGCAR"
    ChargeDensity(np.broadcast_to(values, grid.shape).copy(),
                  grid, structure).write(path)
    return str(path)


def invoke(argv, log=None):
    """Drive the console script the way its entry point does, without VASP."""
    args = poraque_vasp.build_parser().parse_args(argv)
    return poraque_vasp.run(args, log=(log if log is not None else lambda *_: None))


# ===================================================================== #
# The density-of-states deck
# ===================================================================== #
class TestTheDosDeck:
    """A DOS is the band trick asked for an integral instead of a curve."""

    def test_it_holds_the_density_fixed(self):
        assert tag(dos_incar(), "ICHARG") == "11"
        assert tag(dos_incar(), "ISTART") == "0"

    def test_it_uses_the_tetrahedron_method(self):
        """ISMEAR = -5 is what makes a DOS smooth without smearing it flat."""
        assert tag(dos_incar(), "ISMEAR") == "-5"

    def test_the_energy_grid_is_finer_than_vasps_default(self):
        """301 points hide the structure a DOS is computed to look at."""
        assert int(tag(dos_incar(), "NEDOS")) >= 3001

    def test_it_asks_for_the_projected_weights(self):
        assert tag(dos_incar(), "LORBIT") == "11"

    def test_it_does_not_write_a_density_back(self):
        assert tag(dos_incar(), "LCHARG") == ".FALSE."

    def test_the_cutoff_is_carried_with_its_warning(self):
        deck = dos_incar(encut=520)
        assert tag(deck, "ENCUT") == "520"
        assert "MUST match" in comment(deck, "ENCUT")

    def test_the_dos_window_appears_only_when_asked_for(self):
        assert tag(dos_incar(), "EMIN") is None
        deck = dos_incar(emin=-12.0, emax=8.0)
        assert tag(deck, "EMIN") == "-12.0" and tag(deck, "EMAX") == "8.0"

    def test_the_smearing_comment_describes_the_smearing_written(self):
        """A deck that explains a tag it is not using argues with itself."""
        assert "tetrahedron" in comment(dos_incar(ismear=-5), "ISMEAR")
        assert "Gaussian" in comment(dos_incar(ismear=0), "ISMEAR")
        assert "tetrahedron" not in comment(dos_incar(ismear=0), "SIGMA")

    def test_extra_tags_override_defaults(self):
        deck = dos_incar(extra={"NEDOS": 601, "NCORE": 4})
        assert tag(deck, "NEDOS") == "601"
        assert tag(deck, "NCORE") == "4"


# ===================================================================== #
# The total-energy deck
# ===================================================================== #
class TestTheTotalEnergyDeck:
    """Two different energies, and the deck has to say which one it means."""

    def test_it_is_non_self_consistent_by_default(self):
        assert tag(total_energy_incar(), "ICHARG") == "11"

    def test_the_non_variational_caveat_is_in_the_file(self):
        """
        ``ICHARG = 11`` prints the Harris-Foulkes functional at the input
        density. That is the number people quote, so the deck says so where it
        will be read rather than leaving it to a docstring.
        """
        note = comment(total_energy_incar(), "ICHARG")
        assert "Harris-Foulkes" in note
        assert "NOT a variational" in note

    def test_scf_mode_reads_the_prediction_as_a_starting_guess(self):
        deck = total_energy_incar(selfconsistent=True)
        assert tag(deck, "ICHARG") == "1"
        assert "starting" in comment(deck, "ICHARG")
        assert "Harris-Foulkes" not in comment(deck, "ICHARG")

    def test_only_the_scf_run_writes_a_density_back(self):
        """The converged density is what the prediction gets compared with."""
        assert tag(total_energy_incar(), "LCHARG") == ".FALSE."
        assert tag(total_energy_incar(selfconsistent=True), "LCHARG") == ".TRUE."

    def test_it_asks_for_aspherical_corrections(self):
        """The project's elements are d metals; LASPH is not optional there."""
        assert tag(total_energy_incar(), "LASPH") == ".TRUE."

    def test_it_does_not_ask_for_projected_weights(self):
        """Nothing in a total energy needs LORBIT, and it costs a PROCAR."""
        assert tag(total_energy_incar(), "LORBIT") is None

    def test_the_convergence_criterion_is_carried(self):
        assert tag(total_energy_incar(ediff=1e-8), "EDIFF") == "1e-08"

    def test_the_banner_names_the_mode(self):
        assert "ICHARG = 11 total energy" in total_energy_incar()
        assert "ICHARG = 1 total energy" in total_energy_incar(
            selfconsistent=True)


# ===================================================================== #
# The k-point mesh
# ===================================================================== #
class TestTheAutomaticMesh:
    def test_it_is_gamma_centred_by_default(self):
        lines = automatic_kpoints((8, 8, 8)).splitlines()
        assert lines[1].strip() == "0"
        assert lines[2].strip() == "Gamma"
        assert lines[3].split() == ["8", "8", "8"]

    def test_monkhorst_pack_when_asked_for(self):
        assert "Monkhorst-Pack" in automatic_kpoints((4, 4, 4), gamma=False)

    def test_a_mesh_needs_three_subdivisions(self):
        with pytest.raises(ValueError, match="three subdivisions"):
            automatic_kpoints((8, 8))

    def test_a_subdivision_below_one_is_refused(self):
        """A zero would be read by VASP as an automatic length, not a mesh."""
        with pytest.raises(ValueError, match=">= 1"):
            automatic_kpoints((8, 0, 8))


class TestTheSpacingRule:
    """VASP's own KSPACING rule, applied here so the mesh appears in the deck."""

    def test_a_cube_is_sampled_isotropically(self):
        assert kpoint_mesh_from_spacing(np.eye(3) * 10.0, 0.25) == (3, 3, 3)

    def test_a_denser_spacing_gives_a_denser_mesh(self):
        coarse = kpoint_mesh_from_spacing(np.eye(3) * 4.0, 0.50)
        fine = kpoint_mesh_from_spacing(np.eye(3) * 4.0, 0.10)
        assert all(f > c for f, c in zip(fine, coarse))

    def test_a_vacuum_direction_needs_no_special_case(self):
        """
        A slab axis has a short reciprocal vector, so the rule returns one
        point there without ever being told the cell is a slab. This is the
        property that makes the same rule right for bulk, slab and cluster.
        """
        assert kpoint_mesh_from_spacing(np.diag([4.0, 4.0, 30.0]), 0.25) \
            == (7, 7, 1)

    def test_a_cluster_collapses_to_the_gamma_point(self):
        assert kpoint_mesh_from_spacing(np.eye(3) * 30.0, 0.25) == (1, 1, 1)

    def test_it_reads_a_structures_own_cell(self):
        structure = Poscar(np.eye(3) * 10.0, ["Si"], [1], [[0.0, 0.0, 0.0]])
        assert kpoint_mesh_from_spacing(structure, 0.25) == (3, 3, 3)

    def test_a_nonsense_spacing_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            kpoint_mesh_from_spacing(np.eye(3) * 4.0, 0.0)


# ===================================================================== #
# Writing the decks
# ===================================================================== #
class TestWritingTheNewDecks:
    def test_the_dos_deck_is_incar_kpoints_and_poscar(self, tmp_path, density):
        written = write_dos_deck(str(tmp_path / "dos"), chgcar=density,
                                 encut=400)
        assert set(written) == {"INCAR", "KPOINTS", "POSCAR", "CHGCAR"}
        for path in written.values():
            assert os.path.exists(path)

    def test_the_energy_deck_is_incar_kpoints_and_poscar(self, tmp_path,
                                                         density):
        written = write_total_energy_deck(str(tmp_path / "energy"),
                                          chgcar=density, encut=400)
        assert set(written) == {"INCAR", "KPOINTS", "POSCAR", "CHGCAR"}

    def test_no_potcar_is_written_by_either(self, tmp_path, density):
        """It cannot be redistributed, and a stub would be worse than nothing."""
        for target, writer in (("dos", write_dos_deck),
                               ("energy", write_total_energy_deck)):
            writer(str(tmp_path / target), chgcar=density, encut=400)
            assert not os.path.exists(tmp_path / target / "POTCAR")

    def test_the_structure_comes_from_the_density_when_not_given(self, tmp_path,
                                                                 density):
        written = write_dos_deck(str(tmp_path / "dos"), chgcar=density,
                                 encut=400)
        poscar = Poscar.from_file(written["POSCAR"])
        assert poscar.symbols == ["Si"] and list(poscar.counts) == [2]

    def test_the_mesh_is_derived_from_that_structure(self, tmp_path, density):
        """6 A cube at the default 0.25 1/Ang: ceil(2pi/6/0.25) = 5."""
        written = write_dos_deck(str(tmp_path / "dos"), chgcar=density,
                                 encut=400)
        with open(written["KPOINTS"]) as handle:
            assert handle.read().splitlines()[3].split() == ["5", "5", "5"]

    def test_an_explicit_mesh_wins(self, tmp_path, density):
        written = write_dos_deck(str(tmp_path / "dos"), chgcar=density,
                                 encut=400, mesh=(2, 3, 4))
        with open(written["KPOINTS"]) as handle:
            assert handle.read().splitlines()[3].split() == ["2", "3", "4"]

    def test_a_mesh_with_nothing_to_derive_it_from_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="explicit `mesh`"):
            write_dos_deck(str(tmp_path / "dos"), encut=400)

    def test_the_density_is_left_alone_unless_asked_for(self, tmp_path):
        structure = Poscar(np.eye(3) * 6.0, ["Si"], [1], [[0.0, 0.0, 0.0]])
        written = write_total_energy_deck(str(tmp_path / "energy"),
                                          structure=structure, encut=400)
        assert "CHGCAR" not in written

    def test_a_mesh_never_reaches_the_band_deck(self, tmp_path, density):
        """
        A band structure needs a path. If line mode ever stopped being written
        the deck would still run and would silently produce the wrong thing.
        """
        written = write_band_structure_deck(str(tmp_path / "bands"),
                                            chgcar=density, encut=400)
        with open(written["KPOINTS"]) as handle:
            text = handle.read()
        assert "Line-mode" in text and "Gamma" not in text

    def test_a_line_never_reaches_the_dos_deck(self, tmp_path, density):
        written = write_dos_deck(str(tmp_path / "dos"), chgcar=density,
                                 encut=400)
        with open(written["KPOINTS"]) as handle:
            text = handle.read()
        assert "Gamma" in text and "Line-mode" not in text


# ===================================================================== #
# The command line
# ===================================================================== #
class TestTheCommandLineWritesEachMode:
    """A dry run of every mode: no VASP, no POTCAR, just the deck."""

    def test_bands_writes_a_line_mode_deck(self, tmp_path, density):
        out = str(tmp_path / "bands")
        written = invoke(["bands", density, "--encut", "400", "--output", out])
        assert set(written) == {"INCAR", "KPOINTS", "POSCAR"}
        incar = open(written["INCAR"]).read()
        assert tag(incar, "ICHARG") == "11"
        assert tag(incar, "LORBIT") == "11"
        assert "Line-mode" in open(written["KPOINTS"]).read()

    def test_dos_writes_a_mesh_deck(self, tmp_path, density):
        out = str(tmp_path / "dos")
        written = invoke(["dos", density, "--encut", "400", "--output", out])
        assert set(written) == {"INCAR", "KPOINTS", "POSCAR"}
        incar = open(written["INCAR"]).read()
        assert tag(incar, "ICHARG") == "11"
        assert tag(incar, "ISMEAR") == "-5"
        assert int(tag(incar, "NEDOS")) >= 3001
        assert "Gamma" in open(written["KPOINTS"]).read()

    def test_energy_writes_a_mesh_deck_without_lorbit(self, tmp_path, density):
        out = str(tmp_path / "energy")
        written = invoke(["energy", density, "--encut", "400", "--output", out])
        incar = open(written["INCAR"]).read()
        assert tag(incar, "ICHARG") == "11"
        assert tag(incar, "LORBIT") is None
        assert tag(incar, "LASPH") == ".TRUE."
        assert "Gamma" in open(written["KPOINTS"]).read()

    def test_energy_scf_switches_to_icharg_one(self, tmp_path, density):
        out = str(tmp_path / "energy")
        written = invoke(["energy", density, "--encut", "400", "--scf",
                          "--output", out])
        assert tag(open(written["INCAR"]).read(), "ICHARG") == "1"

    def test_each_mode_has_its_own_default_output(self, density):
        parse = poraque_vasp.build_parser().parse_args
        assert parse(["bands", density]).output == "bands"
        assert parse(["dos", density]).output == "dos"
        assert parse(["energy", density]).output == "energy"

    def test_the_density_is_copied_only_when_asked_for(self, tmp_path, density):
        out = str(tmp_path / "dos")
        written = invoke(["dos", density, "--encut", "400", "--output", out,
                          "--copy-density"])
        assert "CHGCAR" in written and os.path.exists(written["CHGCAR"])

    def test_the_mesh_can_be_stated_outright(self, tmp_path, density):
        out = str(tmp_path / "dos")
        written = invoke(["dos", density, "--encut", "400", "--output", out,
                          "--mesh", "9", "9", "9"])
        assert open(written["KPOINTS"]).read().splitlines()[3].split() \
            == ["9", "9", "9"]

    def test_spin_reaches_every_deck(self, tmp_path, density):
        """The platinum runs are ISPIN = 2; a deck that forgets it is wrong."""
        for mode in ("bands", "dos", "energy"):
            written = invoke([mode, density, "--encut", "400", "--ispin", "2",
                              "--output", str(tmp_path / mode)])
            assert tag(open(written["INCAR"]).read(), "ISPIN") == "2"

    def test_a_sparse_mesh_is_called_out_for_tetrahedra(self, tmp_path,
                                                        density):
        """ISMEAR = -5 with fewer than four k-points is a run that dies."""
        lines = []
        invoke(["dos", density, "--encut", "400", "--mesh", "1", "1", "1",
                "--output", str(tmp_path / "dos")], log=lines.append)
        assert any("at least four" in line for line in lines)

    def test_the_energy_mode_says_which_energy_it_means(self, tmp_path,
                                                        density):
        lines = []
        invoke(["energy", density, "--encut", "400",
                "--output", str(tmp_path / "energy")], log=lines.append)
        assert any("Harris-Foulkes" in line for line in lines)


class TestTheCutoffIsNeverSilent:
    """Getting ENCUT wrong is the commonest way ICHARG = 11 fails to start."""

    def test_a_guessed_cutoff_is_announced(self, tmp_path, density):
        lines = []
        invoke(["dos", density, "--output", str(tmp_path / "dos")],
               log=lines.append)
        assert any("A GUESS" in line for line in lines)

    def test_a_cutoff_taken_from_a_run_is_attributed(self, tmp_path, density):
        run_dir = tmp_path / "source"
        run_dir.mkdir()
        (run_dir / "INCAR").write_text("ENCUT = 520\nPREC = Accurate\n")
        lines = []
        written = invoke(["dos", density, "--like", str(run_dir),
                          "--output", str(tmp_path / "dos")], log=lines.append)
        assert tag(open(written["INCAR"]).read(), "ENCUT") == "520.0"
        assert any("520" in line and "INCAR" in line for line in lines)


class TestTheOldSpellingStillWorks:
    """
    ``poraque-bands <CHGCAR>`` became ``poraque-vasp bands <CHGCAR>``.

    The rename would otherwise turn every existing script and note into a
    stack trace, so a first argument that is a path rather than a mode is read
    as the old form.
    """

    def test_a_bare_path_is_read_as_the_band_mode(self, density):
        argv, implied = poraque_vasp._normalise([density, "--encut", "400"])
        assert implied and argv[0] == "bands"

    def test_an_explicit_mode_is_left_alone(self, density):
        argv, implied = poraque_vasp._normalise(["dos", density])
        assert not implied and argv[0] == "dos"

    def test_a_missing_file_is_not_silently_turned_into_a_mode(self):
        """Otherwise a typo'd mode becomes a confusing `bands` failure."""
        argv, implied = poraque_vasp._normalise(["dosx", "CHGCAR"])
        assert not implied and argv[0] == "dosx"
