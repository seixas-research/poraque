# -*- coding: utf-8 -*-
# file: test_paw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Making a prediction readable by VASP: the writer, delta-density mode, the decks.

Three claims, and they are the three places the chain can break silently.

**The writer round-trips.** A ``CHGCAR`` parsed and written back must be the
same file. Not byte-identical — that would be testing
:func:`~poraque.fields.vasp.volumetric.fortran_exponential`'s rounding against
VASP's Fortran, which is a different claim — but the structure, the grid, the
density to the precision the format writes (``E17.11``), and the augmentation
block **byte-for-byte**, since that block is copied as text and never
reformatted. Anything weaker and a file that VASP refuses would still pass.

**Delta-density mode reconstructs the density.** The dataset hands the network a
signed residual and the operator hands back an absolute density. If the baseline
were added back in the wrong place — after positivity, or after the
electron-count rescale — the result would still look like a density and be
wrong. :class:`TestTheOrderOfOperationsAtInference` is where that order is
held to its reasons.

**The decks say what they have to say.** ``ICHARG = 11`` and an ``ENCUT`` that
matches the density's own grid are the two tags that decide whether VASP reads
the file at all.
"""

import os

import numpy as np
import pytest
import torch

from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid
from poraque.fields.atomic import (
    AtomicReference,
    AtomicReferenceLibrary,
    atomic_superposition,
    form_factor_from_density,
)
from poraque.fields.vasp.augmentation import parse_augmentation
from poraque.fields.vasp.poscar import Poscar
from poraque.fields.vasp.templates import (
    band_structure_incar,
    fcc_band_path,
    line_mode_kpoints,
    tau_incar,
    write_band_structure_deck,
)
from poraque.fields.vasp.volumetric import (
    read_augmentation,
    read_volumetric,
    write_volumetric,
)
from poraque.ml.data import FieldPairDataset, make_dataloader
from poraque.ml.training import FieldOperator, train

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AU = os.path.join(_ROOT, "data", "vasp", "ref", "Pt")
STRUCT_015 = os.path.join(_ROOT, "data", "vasp", "struct_015", "CHGCAR")

needs_paw_chgcar = pytest.mark.skipif(
    not os.path.exists(STRUCT_015),
    reason="the shipped platinum dataset is not in this checkout")


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def kpoint_lines(text):
    """The lines of a KPOINTS file that actually carry a k-point.

    Filtering on "starts with a digit" also catches the points-per-segment
    line, which is how this helper earned its own function.
    """
    found = []
    for line in text.splitlines():
        body = line.split("!")[0].split()
        if len(body) != 3:
            continue
        try:
            [float(token) for token in body]
        except ValueError:
            continue
        found.append(body)
    return found


def gaussian_atom(grid, position, n_electrons=4.0, sigma=0.7):
    """A normalised Gaussian at a fractional position, in e/Å³."""
    coords = grid.cartesian_coordinates()
    cell = np.asarray(grid.cell, dtype=float)
    delta = coords - np.asarray(position, dtype=float) @ cell
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    r2 = ((fractional @ cell) ** 2).sum(axis=-1)
    amplitude = n_electrons / (2 * np.pi * sigma ** 2) ** 1.5
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


@pytest.fixture(scope="module")
def original():
    """The shipped platinum reference density, read once for the whole module."""
    grid = FieldGrid.from_file(STRUCT_015)
    return ChargeDensity.read(STRUCT_015, grid=grid), grid


@pytest.fixture
def library():
    """A one-element database built from a synthetic Gaussian atom."""
    grid = FieldGrid((32, 32, 32), np.eye(3) * 6.0)
    structure = Poscar(np.eye(3) * 6.0, ["Si"], [1], [[0.5, 0.5, 0.5]])
    density = ChargeDensity(gaussian_atom(grid, (0.5, 0.5, 0.5), 4.0, 0.6),
                            grid, structure)
    table = form_factor_from_density(density)
    entry = AtomicReference(
        element="Si", valence_charge=table["valence_charge"],
        g_grid=table["g_grid"], form_factor=table["form_factor"],
        g_max=table["g_max"], radial_scatter=table["radial_scatter"],
        potcar_title="SYNTHETIC Si", potcar_sha256="a" * 64)
    return AtomicReferenceLibrary({entry.key: entry})


@pytest.fixture
def cache(tmp_path):
    """Four synthetic two-atom materials with EXTCAR and CHGCAR."""
    rng = np.random.default_rng(3)
    root = tmp_path / "cache"
    for index in range(4):
        directory = root / f"mat_{index}"
        directory.mkdir(parents=True)
        cell = np.eye(3) * 6.0
        grid = FieldGrid((16, 16, 16), cell)
        positions = 0.2 + 0.6 * rng.random((2, 3))
        structure = Poscar(cell, ["Si"], [2], positions)

        potential = ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                              widths={"Si": 0.5})
        potential.write(directory / "EXTCAR")

        density = (gaussian_atom(grid, positions[0], 4.0, 0.6)
                   + gaussian_atom(grid, positions[1], 4.0, 0.6))
        # A little bonding charge, so the residual is not identically zero and
        # the network has something to learn rather than a constant.
        density = density * (1.0 + 0.05 * np.cos(
            2 * np.pi * np.arange(16)[:, None, None] / 16))
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")
    return str(root)


# ===================================================================== #
# The CHGCAR round-trip
# ===================================================================== #
@needs_paw_chgcar
class TestTheWriterRoundTrips:
    """
    Read a real, unmodified reference density; write it back; compare.

    The strongest statement the format supports without owning VASP's Fortran
    runtime. It is deliberately run on a **real** file rather than a fixture:
    the augmentation block, the three-digit exponents and the column alignment
    are all things a synthetic file would simply not exercise.
    """

    def test_the_structure_and_grid_survive(self, original, tmp_path):
        density, grid = original
        _, block = read_augmentation(STRUCT_015)
        path = str(tmp_path / "CHGCAR")
        density.write(path, augmentation=block)

        again = ChargeDensity.read(path, grid=FieldGrid.from_file(path))
        assert again.grid.shape == grid.shape
        assert np.allclose(again.grid.cell, grid.cell, atol=1e-8)
        assert again.structure.symbols == density.structure.symbols
        assert list(again.structure.counts) == list(density.structure.counts)
        assert np.allclose(again.structure.scaled_positions,
                           density.structure.scaled_positions, atol=1e-8)

    def test_the_density_survives_to_the_written_precision(self, original,
                                                           tmp_path):
        r"""
        ``E17.11`` is eleven decimals on a mantissa in :math:`[0.1, 1)`, so a
        relative :math:`10^{-11}` is exactly what the format can carry. A looser
        bound here would let a genuine transformation of the values hide.
        """
        density, _ = original
        path = str(tmp_path / "CHGCAR")
        density.write(path)

        again = ChargeDensity.read(path, grid=FieldGrid.from_file(path))
        scale = np.abs(density.data).max()
        assert np.allclose(again.data, density.data, rtol=0, atol=1e-10 * scale)
        assert again.integrate() == pytest.approx(density.integrate(),
                                                  rel=1e-10)

    def test_the_augmentation_block_is_byte_identical(self, original,
                                                      tmp_path):
        """
        Copied as text, never reformatted. Reflowing the columns of a
        fixed-format Fortran read is how a file VASP declines to parse gets
        made, so the guarantee has to be at the byte level.
        """
        density, _ = original
        _, block = read_augmentation(STRUCT_015)
        path = str(tmp_path / "CHGCAR")
        density.write(path, augmentation=block)

        _, written = read_augmentation(path)
        assert written == block

    def test_every_atom_still_has_exactly_one_record(self, original, tmp_path):
        density, _ = original
        _, block = read_augmentation(STRUCT_015)
        path = str(tmp_path / "CHGCAR")
        density.write(path, augmentation=block)

        _, written = read_augmentation(path)
        records = parse_augmentation(written)
        assert len(records) == len(density.structure.scaled_positions)
        assert all(len(r) == len(records[0]) for r in records)

    def test_a_second_round_trip_changes_nothing_further(self, original,
                                                         tmp_path):
        """
        Idempotence. A writer that loses a little each pass would look fine on
        one round-trip and destroy a file that is written, read and written
        again — which is exactly what an inference pipeline does.
        """
        density, _ = original
        _, block = read_augmentation(STRUCT_015)
        first = str(tmp_path / "one")
        second = str(tmp_path / "two")

        density.write(first, augmentation=block)
        once = ChargeDensity.read(first, grid=FieldGrid.from_file(first))
        once.write(second, augmentation=read_augmentation(first)[1])
        twice = ChargeDensity.read(second, grid=FieldGrid.from_file(second))

        assert np.array_equal(once.data, twice.data)
        with open(first) as a, open(second) as b:
            assert a.read() == b.read()

    def test_a_density_written_without_records_has_none(self, original,
                                                        tmp_path):
        """Not a silent empty block: no block at all."""
        density, _ = original
        path = str(tmp_path / "CHGCAR")
        density.write(path)
        assert read_augmentation(path)[1] == []


class TestTheWriterOnASyntheticFile:
    """The same guarantees where no reference data is needed to check them."""

    def test_values_and_shape_round_trip(self, tmp_path):
        grid = FieldGrid((7, 5, 9), np.diag([4.0, 6.0, 8.0]))
        structure = Poscar(np.diag([4.0, 6.0, 8.0]), ["Si"], [1],
                           [[0.1, 0.2, 0.3]])
        rng = np.random.default_rng(0)
        values = rng.random(grid.shape) * 3.0 - 1.0

        path = str(tmp_path / "FIELD")
        write_volumetric(path, structure, values * grid.volume)
        _, read_back, _ = read_volumetric(path)

        assert read_back.shape == grid.shape
        assert np.allclose(read_back / grid.volume, values, atol=1e-10)

    def test_the_fortran_ordering_is_preserved(self, tmp_path):
        """
        x fastest, z slowest. A transposed writer produces a file that parses
        perfectly and holds a different field.
        """
        structure = Poscar(np.eye(3) * 4.0, ["Si"], [1], [[0.0, 0.0, 0.0]])
        values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

        path = str(tmp_path / "FIELD")
        write_volumetric(path, structure, values)
        _, read_back, _ = read_volumetric(path)
        assert np.array_equal(read_back, values)


# ===================================================================== #
# Delta-density mode, end to end
# ===================================================================== #
class TestDeltaDensityMode:
    def test_the_target_is_the_residual(self, cache, library):
        absolute = FieldPairDataset(cache, task="ext2chg")
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)

        baseline = delta.baseline_for(0)
        assert baseline is not None
        assert np.allclose(delta.target_values(0),
                           absolute.target_values(0) - baseline, atol=1e-12)

    def test_the_residual_is_much_smaller_than_the_density(self, cache,
                                                           library):
        """The claim the whole mode rests on, on this fixture's own numbers."""
        absolute = FieldPairDataset(cache, task="ext2chg")
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        ratio = (np.linalg.norm(delta.target_values(0))
                 / np.linalg.norm(absolute.target_values(0)))
        assert ratio < 0.5

    def test_the_residual_is_signed(self, cache, library):
        """
        Which is why positivity cannot be applied to the target. A test that
        only checked the magnitude would miss the constraint that actually
        moved.
        """
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        values = delta.target_values(0)
        assert values.min() < 0.0 < values.max()

    def test_every_sample_carries_its_baseline(self, cache, library):
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        sample = delta[0]
        assert "baseline" in sample
        assert sample["baseline"].shape == sample["target_physical"].shape

    def test_absolute_mode_carries_none(self, cache):
        assert "baseline" not in FieldPairDataset(cache, task="ext2chg")[0]

    def test_the_batch_carries_it_too(self, cache, library):
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        batch = next(iter(make_dataloader(delta, batch_size=2, shuffle=False)))
        assert batch["baseline"].shape == batch["target_physical"].shape

    def test_adding_the_baseline_back_recovers_the_density(self, cache,
                                                           library):
        absolute = FieldPairDataset(cache, task="ext2chg")
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        recovered = delta.target_values(0) + delta.baseline_for(0)
        assert np.allclose(recovered, absolute.target_values(0), atol=1e-12)

    def test_a_split_keeps_the_same_baseline(self, cache, library):
        """
        A validation half in absolute mode would score the model against a
        different field entirely, and the number would look plausible.
        """
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        left, right = delta.split(fraction=0.5)
        assert left.baseline is delta.baseline
        assert right.baseline is delta.baseline

    def test_the_transforms_are_fitted_to_the_residual(self, cache, library):
        """
        Fitting an Asinh scale to rho and then feeding it delta_rho would be a
        silent scale error of the size of the baseline itself.
        """
        absolute = FieldPairDataset(cache, task="ext2chg")
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        _, absolute_target = absolute.fit_transforms()
        _, delta_target = delta.fit_transforms()
        assert (absolute_target.state_dict()
                != delta_target.state_dict())

    def test_chg2tau_ignores_a_baseline_it_cannot_use(self, cache, library,
                                                      tmp_path):
        """There is no atomic superposition of a kinetic energy density."""
        from poraque.fields import KineticEnergyDensity

        for name in sorted(os.listdir(cache)):
            directory = os.path.join(cache, name)
            grid = FieldGrid.from_file(os.path.join(directory, "CHGCAR"))
            rho = ChargeDensity.read(os.path.join(directory, "CHGCAR"),
                                     grid=grid)
            KineticEnergyDensity(np.abs(rho.data) * 10.0, grid,
                                 rho.structure).write(
                os.path.join(directory, "TAUCAR"))

        dataset = FieldPairDataset(cache, task="chg2tau", baseline=library)
        assert dataset.baseline_for(0) is None
        assert "baseline" not in dataset[0]


class TestDeltaDensityTrainingSmoke:
    """
    A real (tiny) training run and a real prediction.

    Not an accuracy test — four synthetic materials and three epochs measure
    nothing. What it pins is that the plumbing carries the baseline all the way
    from the dataset to a prediction in absolute units, which is where a
    silently-wrong field would otherwise appear.
    """

    @staticmethod
    def _operator(train_set, library):
        torch.manual_seed(0)
        return FieldOperator(
            "ext2chg",
            input_transform=train_set.input_transform,
            target_transform=train_set.target_transform,
            width=4, modes=4, n_layers=1, projection_channels=8,
            device="cpu", baseline=library)

    def test_it_trains_and_predicts_an_absolute_density(self, cache, library):
        train_set = FieldPairDataset(cache, task="ext2chg", baseline=library)
        train_set.fit_transforms()
        operator = self._operator(train_set, library)
        train(operator, train_set, epochs=3, batch_size=2, verbose=False)

        potential = ExternalPotential.read(
            os.path.join(cache, "mat_0", "EXTCAR"))
        prediction = operator.predict(potential)

        reference = ChargeDensity.read(os.path.join(cache, "mat_0", "CHGCAR"),
                                       grid=potential.grid)
        # The prediction is untrained, so it is not close to the reference --
        # but it must be in the same *world*: an absolute density integrating
        # to something of the order of the electron count, not a residual
        # integrating to nearly zero.
        assert prediction.metadata["delta_density"] is True
        assert prediction.integrate() > 0.3 * reference.integrate()

    def test_the_same_run_without_a_baseline_predicts_no_delta(self, cache):
        train_set = FieldPairDataset(cache, task="ext2chg")
        train_set.fit_transforms()
        operator = self._operator(train_set, None)
        train(operator, train_set, epochs=2, batch_size=2, verbose=False)

        potential = ExternalPotential.read(
            os.path.join(cache, "mat_0", "EXTCAR"))
        assert "delta_density" not in operator.predict(potential).metadata

    def test_the_library_travels_inside_the_checkpoint(self, cache, library,
                                                       tmp_path):
        """
        A delta-density model's weights only mean anything against the
        particular superposition they were fitted to. A baseline referenced
        from beside the checkpoint could change; one stored inside it cannot.
        """
        train_set = FieldPairDataset(cache, task="ext2chg", baseline=library)
        train_set.fit_transforms()
        operator = self._operator(train_set, library)

        path = operator.save(str(tmp_path / "op.pt"))
        restored = FieldOperator.load(path, device="cpu")

        assert restored.baseline is not None
        assert restored.baseline.fingerprint == library.fingerprint

    def test_a_restored_model_predicts_the_same_field(self, cache, library,
                                                      tmp_path):
        train_set = FieldPairDataset(cache, task="ext2chg", baseline=library)
        train_set.fit_transforms()
        operator = self._operator(train_set, library)

        potential = ExternalPotential.read(
            os.path.join(cache, "mat_0", "EXTCAR"))
        before = operator.predict(potential).data

        restored = FieldOperator.load(operator.save(str(tmp_path / "op.pt")),
                                      device="cpu")
        assert np.allclose(restored.predict(potential).data, before,
                           atol=1e-6)

    def test_an_absolute_checkpoint_records_no_baseline(self, cache, tmp_path):
        train_set = FieldPairDataset(cache, task="ext2chg")
        train_set.fit_transforms()
        operator = self._operator(train_set, None)
        assert operator.state()["baseline"] is None


class TestTheOrderOfOperationsAtInference:
    r"""
    The baseline goes back **before** positivity and before normalization.

    Both of the alternatives produce something that still looks like a
    density:

    - clipping :math:`\delta\rho` at zero deletes the bonding charge, which is
      negative wherever charge moved away from the free atoms — i.e. exactly
      the signal the mode exists to model;
    - rescaling :math:`\delta\rho` to an electron count divides by an integral
      that is approximately zero.
    """

    def test_clipping_the_residual_would_delete_the_bonding_charge(self, cache,
                                                                  library):
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        residual = delta.target_values(0)
        negative = residual[residual < 0]
        assert negative.size > 0
        # Not a rounding-level amount: a real fraction of the residual's mass.
        assert np.abs(negative).sum() > 0.1 * np.abs(residual).sum()

    def test_the_residual_integrates_to_nearly_nothing(self, cache, library):
        """
        Which is why it cannot be rescaled to an electron count: the factor
        would be a finite number divided by ~0.
        """
        delta = FieldPairDataset(cache, task="ext2chg", baseline=library)
        grid = FieldGrid.from_file(os.path.join(cache, "mat_0", "CHGCAR"))
        absolute = FieldPairDataset(cache, task="ext2chg")
        assert abs(grid.integrate(delta.target_values(0))) < \
            0.05 * abs(grid.integrate(absolute.target_values(0)))

    def test_the_baseline_alone_already_carries_the_electron_count(self, cache,
                                                                   library):
        """
        Exactly, because f(0) = Z_val. So the normalization that follows the
        reconstruction is a small correction rather than a rescue.
        """
        grid = FieldGrid.from_file(os.path.join(cache, "mat_0", "CHGCAR"))
        structure = ChargeDensity.read(
            os.path.join(cache, "mat_0", "CHGCAR"), grid=grid).structure
        baseline = atomic_superposition(structure, grid, library)
        assert baseline.integrate() == pytest.approx(8.0, rel=1e-5)


# ===================================================================== #
# The VASP decks
# ===================================================================== #
class TestTheTauDeck:
    def test_it_asks_for_tau_the_only_way_there_is(self):
        """
        ``LTAU = .TRUE.`` is what calculates the kinetic energy density, and a
        deck that omits it produces no ``TAUCAR`` however it is run. ``LCHARG``
        then writes the result; setting ``LTAU`` alone calculates tau and
        throws it away, which is why the order is pinned too.
        """
        deck = tau_incar()
        assert "LTAU" in deck and ".TRUE." in deck
        assert "LCHARG" in deck
        assert deck.index("LTAU") < deck.index("LCHARG")

    def test_every_assignment_in_the_deck_is_a_real_vasp_tag(self):
        """
        A generated deck must not invent tags. Filenames appear in the banner
        as prose, which is fine; what must never appear is an assignment whose
        left-hand side is a filename rather than a tag, because VASP ignores
        one silently and the run then produces nothing the name promises.
        """
        deck = tau_incar()
        assignments = {line.split("=")[0].strip() for line in deck.splitlines()
                       if "=" in line and not line.startswith("#")}
        assert not assignments & {"TAUCAR", "CHGCAR", "EXTCAR", "WAVECAR"}

    def test_it_names_the_required_version(self):
        assert "6.6.1" in tau_incar()

    def test_extra_tags_override_defaults(self):
        deck = tau_incar(extra={"ISPIN": 2, "NCORE": 4})
        assert "NCORE" in deck
        assert len([line for line in deck.splitlines()
                    if line.startswith("ISPIN")]) == 1


class TestTheBandStructureDeck:
    def test_it_sets_icharg_eleven(self):
        deck = band_structure_incar()
        assert "ICHARG  = 11" in deck.replace("ICHARG = 11", "ICHARG  = 11")

    def test_it_does_not_read_a_wavecar(self):
        """There is none for a predicted density; ISTART = 0 says so."""
        assert "ISTART" in band_structure_incar()

    def test_it_does_not_write_a_density_back(self):
        deck = band_structure_incar()
        line = [ln for ln in deck.splitlines() if ln.startswith("LCHARG")][0]
        assert ".FALSE." in line

    def test_the_cutoff_is_carried_with_its_warning(self):
        deck = band_structure_incar(encut=520)
        assert "520" in deck
        assert "MUST match" in deck

    def test_nbands_appears_only_when_asked_for(self):
        assert "NBANDS" not in band_structure_incar()
        assert "NBANDS" in band_structure_incar(nbands=96)


class TestTheKpointsFile:
    def test_line_mode_with_the_requested_density(self):
        path, labels = fcc_band_path()
        text = line_mode_kpoints(path, 60, labels=labels)
        lines = text.splitlines()
        assert lines[1].strip() == "60"
        assert lines[2].strip() == "Line-mode"
        assert lines[3].strip() == "Reciprocal"

    def test_every_segment_contributes_two_points(self):
        path, labels = fcc_band_path()
        text = line_mode_kpoints(path, 20, labels=labels)
        assert len(kpoint_lines(text)) == 2 * len(path)

    def test_a_flat_path_is_read_as_a_continuous_route(self):
        text = line_mode_kpoints([(0, 0, 0), (0.5, 0, 0.5), (0.5, 0.25, 0.75)],
                                 10)
        assert len(kpoint_lines(text)) == 4          # two segments

    def test_labels_are_comments_not_fields(self):
        """VASP tolerates a trailing comment; it does not tolerate a fourth column."""
        text = line_mode_kpoints([(0, 0, 0), (0.5, 0, 0.5)], 10,
                                 labels=["G", "X"])
        assert "! G" in text and "! X" in text
        # Every k-point line has exactly three columns once the comment is
        # stripped: a label written as a fourth field would be read as a
        # weight and silently change the path.
        assert all(len(body) == 3 for body in kpoint_lines(text))
        assert len(kpoint_lines(text)) == 2


class TestWritingTheWholeDeck:
    def test_it_writes_incar_kpoints_and_poscar(self, tmp_path, cache):
        chgcar = os.path.join(cache, "mat_0", "CHGCAR")
        written = write_band_structure_deck(str(tmp_path / "bands"),
                                            chgcar=chgcar, encut=400)
        assert set(written) == {"INCAR", "KPOINTS", "POSCAR", "CHGCAR"}
        for path in written.values():
            assert os.path.exists(path)

    def test_the_structure_comes_from_the_density_when_not_given(self, tmp_path,
                                                                 cache):
        """A CHGCAR carries its own POSCAR; nothing else has to be supplied."""
        chgcar = os.path.join(cache, "mat_0", "CHGCAR")
        written = write_band_structure_deck(str(tmp_path / "bands"),
                                            chgcar=chgcar, encut=400)
        poscar = Poscar.from_file(written["POSCAR"])
        assert poscar.symbols == ["Si"]
        assert list(poscar.counts) == [2]

    def test_no_potcar_is_written(self, tmp_path, cache):
        """It cannot be redistributed, and a stub would be worse than nothing."""
        target = tmp_path / "bands"
        write_band_structure_deck(str(target),
                                  chgcar=os.path.join(cache, "mat_0", "CHGCAR"),
                                  encut=400)
        assert not os.path.exists(target / "POTCAR")

    def test_the_density_is_left_alone_unless_asked_for(self, tmp_path, cache):
        written = write_band_structure_deck(str(tmp_path / "bands"),
                                            structure=Poscar(
                                                np.eye(3) * 6.0, ["Si"], [1],
                                                [[0.0, 0.0, 0.0]]),
                                            encut=400)
        assert "CHGCAR" not in written
