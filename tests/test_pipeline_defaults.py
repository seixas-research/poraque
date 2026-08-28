# -*- coding: utf-8 -*-
# file: test_pipeline_defaults.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The three defaults that decide what ``ext2chg`` is actually trained on.

Changed together on 2026-08-26, and grouped here because they are one decision
seen from three sides: the isolated atom becomes the reference for everything
that needs a per-element quantity.

**The target is the charge-density variation** (``data.delta_density``, now on
by default). The operator learns
:math:`\delta\rho = \rho - \rho_{\rm sup}` and *returns*
:math:`\delta\rho + \rho_{\rm sup}`, so no caller sees a residual.

**The PAW augmentation comes from the isolated atom** (``data.paw_source``, now
``"atomic"``), not from averaging the training set's material records.

**The external potential is computed, never imported.** This one was already
true; the tests here pin it, because the Pt ``INCAR``s carry an
``EXTCAR = .TRUE.`` line and will start depositing an ``EXTCAR`` file next to
the density. The guarantee that Poraquê ignores it is exactly what stops the
input channel silently changing meaning when that happens.

The fixtures are synthetic so nothing here depends on the training data, which
has not landed yet. Where a real isolated atom exists it is used, and skipped
where it does not.
"""

import json
import os
import sys
from unittest import mock

import numpy as np
import pytest

from poraque.data.cache import build_paw_reference
from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid
from poraque.fields.atomic import (
    AtomicReference,
    AtomicReferenceLibrary,
    atomic_superposition,
    augmentation_reference,
    form_factor_from_density,
    resolve_library,
)
from poraque.fields.vasp.poscar import Poscar
from poraque.ml.config import DataConfig, TrainingConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

PT_ATOM = os.path.expanduser("~/Simulations/vasp/metals/Pt/1.atom")
needs_pt_atom = pytest.mark.skipif(
    not os.path.exists(os.path.join(PT_ATOM, "CHGCAR")),
    reason="the isolated Pt reference calculation is not on this machine")


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def gaussian_atom(grid, position, n_electrons=10.0, sigma=0.7):
    """A normalised Gaussian at a fractional position, in e/Å³."""
    coords = grid.cartesian_coordinates()
    cell = np.asarray(grid.cell, dtype=float)
    delta = coords - np.asarray(position, dtype=float) @ cell
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    r2 = ((fractional @ cell) ** 2).sum(axis=-1)
    amplitude = n_electrons / (2 * np.pi * sigma ** 2) ** 1.5
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def synthetic_atom_run(directory, element="Pt", n_electrons=10.0, sigma=0.7,
                       side=8.0, n=40):
    """An isolated-atom calculation directory, as ``poraque-atoms`` wants one."""
    os.makedirs(directory, exist_ok=True)
    cell = np.eye(3) * side
    grid = FieldGrid((n, n, n), cell)
    structure = Poscar(cell, [element], [1], [[0.5, 0.5, 0.5]])
    ChargeDensity(gaussian_atom(grid, (0.5, 0.5, 0.5), n_electrons, sigma),
                  grid, structure).write(os.path.join(directory, "CHGCAR"))
    structure.write(os.path.join(directory, "POSCAR"))
    with open(os.path.join(directory, "INCAR"), "w") as handle:
        handle.write("ENCUT = 400\nLTAU = .TRUE.\nLCHARG = .TRUE.\n")
    with open(os.path.join(directory, "OUTCAR"), "w") as handle:
        handle.write(" vasp.6.6.1 17Jul2026 (build ...) complex\n")
    return str(directory)


def vacuum_of(grid, structure, distance=4.0):
    r"""
    Boolean mask of the grid points farther than ``distance`` from every atom.

    "Vacuum" defined by the geometry rather than by a hand-picked corner of the
    array. A corner is a few dozen points and can easily come out single-signed
    by chance, which would make the sign test below report leakage where there
    is only ringing — a false alarm on the one check that is meant to tell the
    two apart.
    """
    coords = grid.cartesian_coordinates()
    cell = np.asarray(grid.cell, dtype=float)
    inverse = np.linalg.inv(cell)

    mask = np.ones(grid.shape, dtype=bool)
    for position in np.atleast_2d(structure.scaled_positions):
        delta = coords - position @ cell
        fractional = delta @ inverse
        fractional -= np.round(fractional)          # nearest image, not this one
        mask &= ((fractional @ cell) ** 2).sum(axis=-1) > distance ** 2
    return mask


def _is_ringing_not_leakage(vacuum, peak, tolerance=1e-3):
    r"""
    Whether a vacuum region holds band-limiting ringing rather than real charge.

    The distinction is the whole point of testing vacuum at all, and a bare
    magnitude bound does not make it. Two things can put density where there
    should be none:

    **Ringing** — the form factor is tabulated to a finite ``g_max``, so the
    superposition is a band-limited reconstruction and oscillates about zero
    wherever the true field has a sharp edge. Harmless, an artefact of the
    representation, and it shrinks as the reference atom's own grid gets finer:
    measured at 3.3e-4 of the peak with this file's 40³/8 Å synthetic atom and
    8.9e-5 with the real 140³/10 Å Pt one.

    **Image leakage** — a real-space placement with a cutoff, or a tail wrapped
    across the cell boundary. That is a *bug*, it is strictly positive (density
    is), and it would not shrink with a finer table.

    So the test is: small **and** both-signed. The true Gaussian tail in these
    fixtures' vacuum is of order 1e-18, i.e. nothing, which is what makes the
    sign the discriminating observation rather than the magnitude.
    """
    vacuum = np.asarray(vacuum)
    small = np.abs(vacuum).max() < tolerance * peak
    oscillates = vacuum.min() < 0.0 < vacuum.max()
    return small and oscillates


def synthetic_library(element="Pt", n_electrons=10.0, augmentation=True):
    """A one-element database, optionally carrying an augmentation record."""
    grid = FieldGrid((40, 40, 40), np.eye(3) * 8.0)
    structure = Poscar(np.eye(3) * 8.0, [element], [1], [[0.5, 0.5, 0.5]])
    density = ChargeDensity(
        gaussian_atom(grid, (0.5, 0.5, 0.5), n_electrons, 0.7), grid, structure)
    table = form_factor_from_density(density)
    entry = AtomicReference(
        element=element, valence_charge=table["valence_charge"],
        g_grid=table["g_grid"], form_factor=table["form_factor"],
        g_max=table["g_max"], radial_scatter=table["radial_scatter"],
        augmentation=([0.5, -0.25, 0.125] if augmentation else None),
        potcar_title=f"PAW_PBE {element}", potcar_sha256="a" * 64)
    return AtomicReferenceLibrary({entry.key: entry})


# ===================================================================== #
# Task 1 — the delta-density default
# ===================================================================== #
class TestDeltaDensityIsTheDefaultTarget:
    def test_the_config_default_is_on(self):
        """
        The behaviour change itself. Asserted on the dataclass rather than on a
        parsed YAML, because a config that states nothing is the case that
        matters — every committed config in this repo states nothing.
        """
        assert DataConfig().delta_density is True

    def test_a_config_that_says_nothing_still_gets_it(self, tmp_path):
        path = tmp_path / "train.yaml"
        path.write_text("task: ext2chg\n")
        assert TrainingConfig.from_yaml(path).data.delta_density is True

    def test_it_can_still_be_turned_off_for_the_ablation(self, tmp_path):
        path = tmp_path / "train.yaml"
        path.write_text("data:\n  delta_density: false\n")
        assert TrainingConfig.from_yaml(path).data.delta_density is False

    @pytest.mark.parametrize("reference", [None, "no/such/directory"])
    def test_without_a_resolvable_atomic_reference_the_run_fails_loudly(
            self, tmp_path, reference):
        """
        Not silently back to the absolute density.

        The whole hazard of flipping a training *target* by default is that a
        run which quietly does the old thing looks identical to one that did
        the new thing. The error names the key, the CLI that builds a database,
        and the way to opt out.

        Both spellings of the same failure are checked. ``atomic_reference``
        carries a default path, so on any machine but the one holding the data
        the realistic case is not an unset key but a key pointing somewhere
        that does not exist -- and a bare ``FileNotFoundError`` there would
        name the path without naming the opt-out.
        """
        from poraque_train import resolve_baseline
        from poraque.ml.tasks import resolve_task

        settings = {"task": "ext2chg", "data": {"atomic_reference": reference}}
        config = TrainingConfig.from_dict(settings)
        with pytest.raises(ValueError) as excinfo:
            resolve_baseline(resolve_task("ext2chg"), config, str(tmp_path),
                             lambda *_: None)

        message = str(excinfo.value)
        assert "atomic_reference" in message
        assert "poraque-atoms" in message
        assert "delta_density: false" in message

    def test_chg2tau_is_unaffected_whatever_the_flag_says(self, tmp_path):
        """There is no atomic superposition of a kinetic energy density."""
        from poraque_train import resolve_baseline
        from poraque.ml.tasks import resolve_task

        config = TrainingConfig.from_dict({"data": {"delta_density": True}})
        assert resolve_baseline(resolve_task("chg2tau"), config,
                                str(tmp_path), lambda *_: None) is None

    def test_an_uncovered_element_is_caught_before_training_starts(self,
                                                                   tmp_path):
        """
        The pair (dataset, library) is checkable before a single epoch runs.
        Discovering it two hours in, as a KeyError inside a DataLoader worker,
        is the same information delivered uselessly.
        """
        from poraque_train import resolve_baseline
        from poraque.ml.tasks import resolve_task

        reference = synthetic_atom_run(tmp_path / "atoms" / "Pt")
        cache = tmp_path / "cache" / "PtSi"
        cache.mkdir(parents=True)
        grid = FieldGrid((12, 12, 12), np.eye(3) * 6.0)
        # Two DIFFERENT elements, only one of which the library covers -- the
        # whole point is the uncovered one.
        structure = Poscar(np.eye(3) * 6.0, ["Pt", "Si"], [1, 1],
                           [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
        ChargeDensity(np.ones(grid.shape) * 0.1, grid, structure).write(
            cache / "CHGCAR")

        config = TrainingConfig.from_dict(
            {"data": {"atomic_reference": os.path.dirname(reference)}})
        with pytest.raises(ValueError, match="Si"):
            resolve_baseline(resolve_task("ext2chg"), config,
                             str(tmp_path / "cache"), lambda *_: None)


class TestTheSuperpositionWorksForEveryGeometry:
    """
    Bulk, slab and cluster, which is what the dataset is about to contain.

    Nothing in the reciprocal-space construction assumes a filled cell, but
    "nothing assumes it" is a claim about code and these are the measurements.
    The invariant that matters in all three is the electron count: it is exact
    by construction (:math:`f(0) = Z_{\\rm val}`), and if it ever became
    approximate the residual would silently absorb the difference.
    """

    @staticmethod
    def _count(structure, grid, library):
        return atomic_superposition(structure, grid, library).integrate()

    def test_bulk(self):
        library = synthetic_library()
        grid = FieldGrid((24, 24, 24), np.eye(3) * 8.0)
        structure = Poscar(np.eye(3) * 8.0, ["Pt"], [4],
                           [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5],
                            [0, 0.5, 0.5]])
        assert self._count(structure, grid, library) == pytest.approx(40.0,
                                                                      rel=1e-5)

    def test_slab_with_vacuum(self):
        """
        Atoms occupying a third of the cell, vacuum above and below. The
        superposition must not leak across the vacuum gap into the images.
        """
        library = synthetic_library()
        cell = np.diag([6.0, 6.0, 24.0])
        grid = FieldGrid((20, 20, 80), cell)
        z = [0.30, 0.38, 0.46]
        positions = [[0.25, 0.25, zi] for zi in z] + [[0.75, 0.75, zi]
                                                      for zi in z]
        structure = Poscar(cell, ["Pt"], [6], positions)
        density = atomic_superposition(structure, grid, library)

        assert density.integrate() == pytest.approx(60.0, rel=1e-5)
        vacuum = density.data[vacuum_of(grid, structure, 4.0)]
        assert vacuum.size > 1000
        assert _is_ringing_not_leakage(vacuum, density.data.max())

    def test_cluster_in_a_box(self):
        library = synthetic_library()
        cell = np.eye(3) * 18.0
        grid = FieldGrid((48, 48, 48), cell)
        offsets = np.array([[0, 0, 0], [0.13, 0, 0], [0, 0.13, 0],
                            [0, 0, 0.13]])
        structure = Poscar(cell, ["Pt"], [4], 0.4 + offsets)
        density = atomic_superposition(structure, grid, library)

        assert density.integrate() == pytest.approx(40.0, rel=1e-5)
        vacuum = density.data[vacuum_of(grid, structure, 5.0)]
        assert vacuum.size > 1000
        assert _is_ringing_not_leakage(vacuum, density.data.max())

    def test_a_slab_is_translation_covariant_along_the_surface_normal(self):
        """
        Moving the slab through the vacuum moves the field with it, exactly.
        A real-space placement with a cutoff would fail this at the boundary.
        """
        library = synthetic_library()
        cell = np.diag([6.0, 6.0, 24.0])
        grid = FieldGrid((16, 16, 64), cell)
        z = np.array([0.30, 0.38, 0.46])
        first = atomic_superposition(
            Poscar(cell, ["Pt"], [3], [[0.5, 0.5, zi] for zi in z]),
            grid, library).data
        shift = 8 / 64
        second = atomic_superposition(
            Poscar(cell, ["Pt"], [3], [[0.5, 0.5, zi + shift] for zi in z]),
            grid, library).data
        assert np.allclose(np.roll(first, 8, axis=2), second, atol=1e-10)


# ===================================================================== #
# Task 2 — the PAW augmentation source
# ===================================================================== #
class TestThePawRecordsComeFromTheIsolatedAtom:
    def test_the_config_default_is_atomic(self):
        assert DataConfig().paw_source == "atomic"

    def test_the_table_has_the_shape_the_bundle_already_stores(self):
        """
        Same shape as the averaged table, so `records_for_structure`, the
        bundle metadata and the inference writer are all unchanged and only the
        provenance of the numbers differs.
        """
        reference = augmentation_reference(synthetic_library())
        assert set(reference) == {"Pt"}
        assert reference["Pt"]["values"] == [0.5, -0.25, 0.125]
        assert reference["Pt"]["source"] == "isolated_atom"

    def test_it_says_one_atom_in_one_structure_and_means_it(self):
        """
        A free atom is one atom in one calculation. Anyone comparing this
        against a table averaged over 32 bulk sites should see that at a
        glance rather than having to know which code path wrote it.
        """
        reference = augmentation_reference(synthetic_library())
        assert reference["Pt"]["atoms"] == 1
        assert reference["Pt"]["structures"] == 1

    def test_an_atom_without_a_record_contributes_nothing(self):
        assert augmentation_reference(
            synthetic_library(augmentation=False)) == {}

    def test_build_paw_reference_prefers_the_atoms(self, tmp_path):
        reference = build_paw_reference([], str(tmp_path),
                                        library=synthetic_library(),
                                        source="atomic")
        assert reference["Pt"]["source"] == "isolated_atom"
        with open(tmp_path / "paw_reference.json") as handle:
            assert json.load(handle)["Pt"]["source"] == "isolated_atom"

    def test_without_a_library_it_falls_back_and_says_so(self, tmp_path):
        lines = []
        reference = build_paw_reference([], str(tmp_path), log=lines.append,
                                        library=None, source="atomic")
        assert reference == {}
        assert any("falling back" in line for line in lines)

    def test_switching_source_rebuilds_rather_than_reusing(self, tmp_path):
        """
        The cached table from the *other* source is not this run's table.

        Same failure `cache_tag` exists to prevent one level up: a silently
        reused artefact answers a question nobody asked.
        """
        (tmp_path / "paw_reference.json").write_text(json.dumps(
            {"Pt": {"values": [9.0], "atoms": 32, "structures": 1}}))

        lines = []
        reference = build_paw_reference([], str(tmp_path), log=lines.append,
                                        library=synthetic_library(),
                                        source="atomic")
        assert reference["Pt"]["source"] == "isolated_atom"
        assert any("rebuilding" in line for line in lines)

    def test_a_matching_cached_table_is_reused(self, tmp_path):
        build_paw_reference([], str(tmp_path), library=synthetic_library(),
                            source="atomic")
        lines = []
        again = build_paw_reference([], str(tmp_path), log=lines.append,
                                    library=synthetic_library(),
                                    source="atomic")
        assert again["Pt"]["source"] == "isolated_atom"
        assert any("cached" in line for line in lines)


# ===================================================================== #
# Task 3 — EXTCAR is computed, never imported
# ===================================================================== #
class TestTheExternalPotentialIsComputedNotRead:
    """
    The Pt ``INCAR``s carry ``EXTCAR = .TRUE.``, so runs will start depositing
    an ``EXTCAR`` beside the density. Poraquê must keep computing its own.

    Not a stylistic preference: at inference time there is no DFT run to read
    an ``EXTCAR`` from, so the training input has to be exactly what
    :class:`~poraque.fields.ExternalPotential` produces. A pipeline that reads
    the file when it happens to exist trains on one field and predicts from
    another.
    """

    @staticmethod
    def _run(directory, with_extcar_file):
        from poraque.data.sources import CalculationSource

        os.makedirs(directory, exist_ok=True)
        cell = np.eye(3) * 6.0
        grid = FieldGrid((16, 16, 16), cell)
        structure = Poscar(cell, ["Si"], [2],
                           [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
        structure.write(os.path.join(directory, "POSCAR"))
        with open(os.path.join(directory, "INCAR"), "w") as handle:
            handle.write("ENCUT = 300\nPREC = Accurate\nEXTCAR = .TRUE.\n")
        ChargeDensity(np.ones(grid.shape) * 0.1, grid, structure).write(
            os.path.join(directory, "CHGCAR"))

        if with_extcar_file:
            # A decoy: a constant field nothing could mistake for the real one.
            ExternalPotential(np.full(grid.shape, -999.0), grid,
                              structure).write(
                os.path.join(directory, "EXTCAR"))

        source = CalculationSource(os.path.dirname(directory),
                                   charges={"Si": 4.0})
        record = source.discover()[0]
        return source, record, grid

    def test_a_calculation_never_lists_an_extcar_among_its_files(self,
                                                                tmp_path):
        _, record, _ = self._run(str(tmp_path / "runs" / "s0"), True)
        assert "EXTCAR" not in record.files
        assert "CHGCAR" in record.files

    def test_extcar_is_offered_even_when_no_file_exists(self, tmp_path):
        source, record, _ = self._run(str(tmp_path / "runs" / "s0"), False)
        assert "EXTCAR" in source.provides(record)

    def test_a_decoy_extcar_on_disk_is_ignored(self, tmp_path):
        """The guarantee, stated as a measurement rather than as a comment."""
        source, record, grid = self._run(str(tmp_path / "runs" / "s0"), True)
        field = source.read(record, "EXTCAR", grid)

        assert not np.allclose(field.data, -999.0)
        # The computed potential is a neutralised lattice sum: zero mean.
        assert abs(field.data.mean()) < 1e-8 * max(np.abs(field.data).max(), 1.0)

    def test_the_computed_field_is_the_same_with_or_without_the_decoy(self,
                                                                      tmp_path):
        with_file = self._run(str(tmp_path / "a" / "s0"), True)
        without = self._run(str(tmp_path / "b" / "s0"), False)
        first = with_file[0].read(with_file[1], "EXTCAR", with_file[2]).data
        second = without[0].read(without[1], "EXTCAR", without[2]).data
        assert np.allclose(first, second, atol=1e-12)


# ===================================================================== #
# Against the real isolated Pt atom
# ===================================================================== #
@needs_pt_atom
class TestTheShippedPlatinumAtom:
    """
    The real reference for the current dataset: VASP 6.6.1, ``LTAU = .TRUE.``,
    ``ISPIN = 2``, one Pt in a 10 Å cube.
    """

    def test_a_bare_run_directory_ingests(self, tmp_path):
        library = resolve_library(PT_ATOM, cache=str(tmp_path))
        entry = library.lookup("Pt")
        assert entry is not None
        assert entry.valence_charge == pytest.approx(10.0, rel=1e-5)
        assert entry.vasp_version == "6.6.1"

    def test_the_parent_directory_works_too(self, tmp_path):
        """``metals/Pt`` holding ``1.atom/`` is what a person actually types."""
        library = resolve_library(os.path.dirname(PT_ATOM),
                                  cache=str(tmp_path))
        assert library.elements() == ["Pt"]

    def test_the_vacuum_of_a_slab_holds_only_ringing(self):
        """
        The number that matters for the incoming slab and cluster data.

        Most of such a cell is vacuum, so whatever the baseline puts there is
        what delta-density mode will ask the operator to cancel. Measured at
        ~9e-5 of the peak with the real Pt table — an order of magnitude below
        the synthetic fixture's, because this atom was solved on a 140³ grid
        and its form factor runs to a much higher ``g_max``.
        """
        library = resolve_library(PT_ATOM)
        cell = np.diag([6.0, 6.0, 24.0])
        grid = FieldGrid((24, 24, 96), cell)
        z = (0.30, 0.38, 0.46)
        positions = ([[0.25, 0.25, zi] for zi in z]
                     + [[0.75, 0.75, zi] for zi in z])
        density = atomic_superposition(Poscar(cell, ["Pt"], [6], positions),
                                       grid, library)

        structure = Poscar(cell, ["Pt"], [6], positions)
        assert density.integrate() == pytest.approx(60.0, rel=1e-5)
        vacuum = density.data[vacuum_of(grid, structure, 4.0)]
        assert np.abs(vacuum).max() < 5e-4 * density.data.max()
        assert vacuum.min() < 0.0 < vacuum.max()

    def test_the_ingest_is_memoised_into_the_cache(self, tmp_path):
        first = resolve_library(PT_ATOM, cache=str(tmp_path))
        assert os.path.exists(tmp_path / "atomic_reference.json")

        lines = []
        second = resolve_library(PT_ATOM, cache=str(tmp_path),
                                 log=lines.append)
        assert second.fingerprint == first.fingerprint
        assert not any("ingesting" in line for line in lines)

    def test_the_spin_polarised_chgcar_yields_one_augmentation_record(self):
        """
        ``ISPIN = 2`` writes the records twice — total, then magnetisation.
        Only the first belongs to a density.
        """
        library = resolve_library(PT_ATOM)
        entry = library.lookup("Pt")
        assert entry.augmentation is not None
        assert len(entry.augmentation) == 139

    def test_superposing_it_back_reproduces_its_own_density(self):
        library = resolve_library(PT_ATOM)
        path = os.path.join(PT_ATOM, "CHGCAR")
        grid = FieldGrid.from_file(path)
        original = ChargeDensity.read(path, grid=grid)
        rebuilt = atomic_superposition(original.structure, grid, library)

        error = (np.linalg.norm(rebuilt.data - original.data)
                 / np.linalg.norm(original.data))
        assert error < 2e-3
        assert rebuilt.integrate() == pytest.approx(original.integrate(),
                                                    rel=1e-5)

    def test_a_bulk_run_is_refused_as_a_reference(self, tmp_path):
        """
        The commonest way to get this wrong is to point at a bulk directory.
        The form factor of four atoms is not the form factor of an atom.
        """
        cell = np.eye(3) * 8.0
        grid = FieldGrid((16, 16, 16), cell)
        structure = Poscar(cell, ["Pt"], [4],
                           [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5],
                            [0, 0.5, 0.5]])
        bulk = tmp_path / "atoms" / "bulk"
        bulk.mkdir(parents=True)
        ChargeDensity(np.ones(grid.shape) * 0.5, grid, structure).write(
            bulk / "CHGCAR")

        with pytest.raises(ValueError, match="single atom"):
            resolve_library(str(tmp_path / "atoms"))


class TestTheGeometryComesFromTheDensityNotTheStructureFile:
    """
    Regression: `V_ext` was built at the geometry a run *started* from.

    ``VaspReader.structure_files`` is ``("POSCAR", "CONTCAR")`` and the first
    wins, which is right for a static run and wrong for a relaxation: there the
    ``POSCAR`` is the input geometry while the ``CHGCAR`` holds the density at
    the output geometry. Pairing them trains the operator on a potential and a
    density from two different systems.

    Found on the platinum set. ``structure_0042`` relaxed by 0.12 Ang rms, which
    put a 2.5 % relative L2 error into its external potential and made it the
    worst structure in the training set by a factor of 23 -- while its
    ``chg2tau`` error, which never reads a ``POSCAR``, stayed unremarkable.
    """

    @staticmethod
    def _run(directory, drift=0.0):
        """A run whose POSCAR is `drift` (fractional) away from its density."""
        from poraque.data.sources import CalculationSource

        os.makedirs(directory, exist_ok=True)
        cell = np.eye(3) * 6.0
        grid = FieldGrid((16, 16, 16), cell)
        relaxed = np.array([[0.10, 0.20, 0.30], [0.60, 0.70, 0.80]])

        # The density carries the geometry it was computed at.
        ChargeDensity(np.ones(grid.shape) * 0.1, grid,
                      Poscar(cell, ["Si"], [2], relaxed)).write(
            os.path.join(directory, "CHGCAR"))
        # The structure file is left wherever the run started.
        Poscar(cell, ["Si"], [2], relaxed + drift).write(
            os.path.join(directory, "POSCAR"))
        with open(os.path.join(directory, "INCAR"), "w") as handle:
            handle.write("ENCUT = 300\nPREC = Accurate\nIBRION = 2\nNSW = 100\n")

        lines = []
        source = CalculationSource(os.path.dirname(directory),
                                   charges={"Si": 4.0}, log=lines.append)
        return source, source.discover()[0], grid, lines

    def test_the_resolved_geometry_is_the_densitys_own(self, tmp_path):
        source, record, _, _ = self._run(str(tmp_path / "runs" / "s0"),
                                         drift=0.02)
        density = ChargeDensity.read(record.files["CHGCAR"])

        assert source.geometry(record).scaled_positions == pytest.approx(
            density.structure.scaled_positions)

    def test_a_stale_structure_file_no_longer_changes_the_potential(self,
                                                                    tmp_path):
        """The whole point: the same density must give the same input field
        however far behind its POSCAR has been left."""
        clean, record_a, grid, _ = self._run(str(tmp_path / "a" / "s0"))
        stale, record_b, _, _ = self._run(str(tmp_path / "b" / "s0"), drift=0.02)

        a = clean.read(record_a, "EXTCAR", grid).data
        b = stale.read(record_b, "EXTCAR", grid).data
        assert np.allclose(a, b)

    def test_the_disagreement_is_reported(self, tmp_path):
        source, record, grid, lines = self._run(str(tmp_path / "runs" / "s0"),
                                                drift=0.02)
        source.read(record, "EXTCAR", grid)
        message = " ".join(lines)

        assert "geometry differs" in message
        assert record.identifier in message
        # 0.02 fractional on a 6 Ang cell is 0.12 Ang, which the line quotes so
        # a reader can judge whether it matters.
        assert "0.120" in message

    def test_a_static_run_is_silent(self, tmp_path):
        """Text precision alone must never trip the warning."""
        source, record, grid, lines = self._run(str(tmp_path / "runs" / "s0"))
        source.read(record, "EXTCAR", grid)

        assert not [line for line in lines if "geometry differs" in line]

    def test_it_falls_back_when_the_format_carries_no_geometry(self, tmp_path):
        """
        A reader whose volumetric format embeds no structure answers ``None``
        from ``read_field_structure``, and the directory's structure file is
        then the only geometry there is.
        """
        from poraque.fields.io.vasp import VaspReader

        source, record, _, _ = self._run(str(tmp_path / "runs" / "s0"),
                                         drift=0.02)
        poscar = Poscar.from_file(os.path.join(record.directory, "POSCAR"))

        with mock.patch.object(VaspReader, "read_field_structure",
                               return_value=None):
            assert source.geometry(record).scaled_positions == pytest.approx(
                poscar.scaled_positions)
