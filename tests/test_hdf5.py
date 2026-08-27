# -*- coding: utf-8 -*-
# file: test_hdf5.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Fields stored as HDF5: the same numbers, read by the same loader.

The whole design rests on one claim — that an HDF5 store is a *storage* change
and not a *format* change. A ``CHGCAR`` and a ``fields.h5`` hold the same
values in the same convention, so
:func:`~poraque.fields.vasp.volumetric.read_volumetric` answers for both and
everything above it (``ScalarField.read``, ``FieldGrid.from_file``,
``SpinDensity.read``, ``FieldPairDataset``, the bucket sampler) is unchanged.
There is one loader in this codebase, not two that have to agree.

That claim is what these tests hold in place, at every level it could break:

**The values.** A round trip through HDF5 is exact to a float64 ulp, against
the text format's eleven significant digits — the arithmetic is one multiply by
the cell volume and one divide back, and nothing is ever rendered as a decimal
string. A test asserting mere closeness would pass just as happily on a
silently lossy store.

**The convention.** The stored numbers are :math:`\rho\Omega`, not
:math:`\rho`. That is not an aesthetic choice: it is what makes the shared
reader work, and a store that helpfully wrote physical units would read back
scaled by the cell volume with no error anywhere.

**Compression is a filter, not a format.** gzip and lzf files are read by the
same code, and a compressed field equals an uncompressed one bit for bit.

**Ragged grids.** Materials Project cells differ in size, so the dataset built
over an HDF5 cache must bucket by shape exactly as it does over a text one, and
no batch may mix two shapes.
"""

import os

import numpy as np
import pytest

from poraque.data.cache import build_field_cache, cached_paths
from poraque.fields import (
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
    SpinDensity,
    is_spin_polarized,
)
from poraque.fields.hdf5 import (
    chunk_shape,
    compression_options,
    describe,
    field_names,
    is_hdf5_path,
    join_target,
    peek_shape,
    split_target,
    write_field,
    write_fields,
)
from poraque.fields.vasp.poscar import Poscar
from poraque.ml.data import (
    FieldPairDataset,
    ShapeBucketSampler,
    discover_materials,
    make_dataloader,
    prepared_fields,
)

h5py = pytest.importorskip("h5py")

CODECS = [None, "gzip", "lzf"]


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def structure(lattice=6.0, sites=2):
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]][:sites]
    return Poscar(np.eye(3) * lattice, ["Si"], [sites], positions)


def density(shape=(12, 12, 12), lattice=6.0, seed=0):
    """A smooth, strictly positive density — what a real one looks like."""
    grid = FieldGrid(shape, np.eye(3) * lattice)
    axes = np.meshgrid(*[np.linspace(0, 2 * np.pi, n, endpoint=False)
                         for n in shape], indexing="ij")
    values = 1.0 + 0.3 * np.cos(axes[0]) * np.sin(axes[1]) + 0.1 * np.cos(axes[2])
    values += 0.01 * np.random.default_rng(seed).random(shape)
    return ChargeDensity(values, grid, structure(lattice), dtype="float64")


@pytest.fixture
def store(tmp_path):
    """One material's density in an HDF5 store."""
    field = density()
    path = str(tmp_path / "fields.h5")
    write_fields(path, {"CHGCAR": field}, compression="gzip")
    return path, field


@pytest.fixture
def raw_runs(tmp_path):
    """Three prepared materials on three different grids."""
    root = tmp_path / "runs"
    for index, (n, a) in enumerate(((12, 6.0), (16, 7.0), (20, 8.0))):
        directory = root / f"mat_{index}"
        directory.mkdir(parents=True)
        field = density((n, n, n), a, seed=index)
        field.write(directory / "CHGCAR")
        ExternalPotential.compute(field.structure, field.grid, {"Si": 4.0},
                                  widths={"Si": 0.5}).write(directory / "EXTCAR")
    return str(root)


# ===================================================================== #
# Paths and options
# ===================================================================== #
class TestAddressingAFieldInsideAStore:
    def test_a_field_is_named_after_its_file(self):
        assert split_target("c/fields.h5::CHGCAR") == ("c/fields.h5", "CHGCAR")

    def test_a_bare_file_names_no_field(self):
        assert split_target("c/fields.h5") == ("c/fields.h5", None)

    def test_the_two_halves_round_trip(self):
        assert split_target(join_target("a/b.h5", "TAUCAR")) == ("a/b.h5",
                                                                 "TAUCAR")

    @pytest.mark.parametrize("path", ["a.h5", "a.hdf5", "A.H5",
                                      "a/fields.h5::CHGCAR"])
    def test_hdf5_paths_are_recognised(self, path):
        assert is_hdf5_path(path)

    @pytest.mark.parametrize("path", ["CHGCAR", "CHGCAR.gz", "a/EXTCAR",
                                      "notes.h5.txt"])
    def test_other_paths_are_not(self, path):
        assert not is_hdf5_path(path)

    def test_it_answers_for_a_file_that_does_not_exist_yet(self, tmp_path):
        """A writer asks this *before* creating the file, so a signature check
        could not answer it at all."""
        assert is_hdf5_path(str(tmp_path / "nothing-here.h5"))


class TestCompressionOptions:
    def test_none_means_no_filter(self):
        assert compression_options(None) == {}
        assert compression_options("none") == {}
        assert compression_options(False) == {}

    def test_gzip_carries_its_level(self):
        assert compression_options("gzip", 7)["compression_opts"] == 7

    def test_lzf_has_no_level_and_does_not_complain_about_one(self):
        """A run that changes codec should not also have to drop the level."""
        assert compression_options("lzf", 9) == {"compression": "lzf",
                                                 "shuffle": True}

    def test_shuffle_is_on_by_default(self):
        assert compression_options("gzip")["shuffle"] is True
        assert compression_options("gzip", shuffle=False)["shuffle"] is False

    def test_an_unknown_codec_names_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="gzip, lzf"):
            compression_options("zstd")

    def test_an_impossible_gzip_level_is_refused(self):
        with pytest.raises(ValueError, match="between 0 and 9"):
            compression_options("gzip", 12)

    def test_a_mapping_is_accepted_as_written(self):
        assert compression_options({"codec": "gzip", "level": 2}) == {
            "compression": "gzip", "compression_opts": 2, "shuffle": True}


class TestChunking:
    def test_a_small_grid_is_one_chunk(self):
        assert chunk_shape((16, 16, 16)) == (16, 16, 16)

    def test_a_large_grid_is_split_towards_cubes(self):
        chunks = chunk_shape((140, 140, 140))
        assert max(chunks) / min(chunks) <= 2
        assert np.prod(chunks) * 8 <= 1 << 20

    def test_no_chunk_axis_exceeds_the_grid(self):
        for shape in ((140, 140, 140), (32, 32, 200), (7, 512, 3)):
            assert all(c <= n for c, n in zip(chunk_shape(shape), shape))

    def test_an_anisotropic_grid_is_not_sliced_into_slabs(self):
        """
        A compressed chunk decompresses whole, so a slab chunk would make a
        strided read along the wrong axis touch the entire field.
        """
        chunks = chunk_shape((32, 32, 512))
        assert chunks[2] < 512


# ===================================================================== #
# The round trip
# ===================================================================== #
class TestTheValuesSurviveExactly:
    @pytest.mark.parametrize("codec", CODECS)
    def test_a_density_round_trips_to_a_float64_ulp(self, tmp_path, codec):
        field = density()
        path = str(tmp_path / f"{codec}.h5")
        write_fields(path, {"CHGCAR": field}, compression=codec)
        back = ChargeDensity.read(path + "::CHGCAR", dtype="float64")

        error = np.abs(field.data - back.data).max() / np.abs(field.data).max()
        assert error < 1e-15, "HDF5 stores doubles; only rho*V/V costs anything"

    def test_it_beats_the_text_format_it_replaces(self, tmp_path):
        """
        E17.11 keeps eleven digits. This is not a stylistic difference: it is
        five orders of magnitude, and it is the reason a store is not merely a
        smaller CHGCAR.
        """
        field = density()
        write_fields(str(tmp_path / "f.h5"), {"CHGCAR": field}, compression="gzip")
        field.write(tmp_path / "CHGCAR")

        def error(other):
            return np.abs(field.data - other.data).max() / np.abs(field.data).max()

        binary = error(ChargeDensity.read(str(tmp_path / "f.h5") + "::CHGCAR",
                                          dtype="float64"))
        text = error(ChargeDensity.read(tmp_path / "CHGCAR", dtype="float64"))
        assert binary < text / 1e3

    @pytest.mark.parametrize("codec", CODECS)
    def test_every_codec_gives_identical_values(self, tmp_path, codec):
        field = density()
        plain = str(tmp_path / "plain.h5")
        write_fields(plain, {"CHGCAR": field}, compression=None)
        other = str(tmp_path / "other.h5")
        write_fields(other, {"CHGCAR": field}, compression=codec)

        assert np.array_equal(
            ChargeDensity.read(plain + "::CHGCAR", dtype="float64").data,
            ChargeDensity.read(other + "::CHGCAR", dtype="float64").data)

    def test_the_stored_numbers_are_the_file_convention(self, tmp_path):
        r"""
        A store holds :math:`\rho\Omega`, exactly as a ``CHGCAR`` does. Writing
        physical units instead would read back scaled by the cell volume, with
        no error raised anywhere — the electron count would simply be wrong by
        a factor of 216.
        """
        field = density()
        path = str(tmp_path / "f.h5")
        write_fields(path, {"CHGCAR": field})
        with h5py.File(path, "r") as handle:
            stored = handle["CHGCAR"][...]
        assert np.allclose(stored, field.data * field.grid.volume)

    def test_the_structure_survives(self, store):
        path, field = store
        back = ChargeDensity.read(path + "::CHGCAR")
        assert back.structure.symbols == field.structure.symbols
        assert list(back.structure.counts) == list(field.structure.counts)
        assert np.allclose(back.structure.scaled_positions,
                           field.structure.scaled_positions)
        assert np.allclose(back.grid.cell, field.grid.cell)

    def test_a_potential_is_not_volume_scaled_and_still_round_trips(self,
                                                                    tmp_path):
        """EXTCAR and CHGCAR use opposite conventions; one code path serves both."""
        field = density()
        potential = ExternalPotential.compute(field.structure, field.grid,
                                              {"Si": 4.0}, widths={"Si": 0.5})
        path = str(tmp_path / "f.h5")
        write_fields(path, {"EXTCAR": potential}, compression="gzip")
        back = ExternalPotential.read(path + "::EXTCAR", dtype="float64")
        assert np.allclose(potential.data, back.data, rtol=0, atol=1e-12)

    def test_a_kinetic_energy_density_round_trips(self, tmp_path):
        field = density()
        tau = KineticEnergyDensity(field.data * 2.0, field.grid,
                                   field.structure, dtype="float64")
        path = str(tmp_path / "f.h5")
        write_fields(path, {"TAUCAR": tau}, compression="lzf")
        assert np.allclose(
            KineticEnergyDensity.read(path + "::TAUCAR", dtype="float64").data,
            tau.data, rtol=0, atol=1e-12)


class TestShapesAndDtypes:
    def test_the_grid_comes_back_through_the_shared_reader(self, store):
        path, field = store
        assert FieldGrid.from_file(path + "::CHGCAR").shape == field.grid.shape

    def test_the_shape_is_read_without_touching_a_value(self, store):
        path, field = store
        assert peek_shape(path + "::CHGCAR") == field.grid.shape

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_the_reader_honours_the_requested_precision(self, store, dtype):
        path, _ = store
        assert ChargeDensity.read(path + "::CHGCAR",
                                  dtype=dtype).data.dtype == np.dtype(dtype)

    def test_a_shared_grid_is_imposed_not_re_derived(self, store):
        path, field = store
        wrong = FieldGrid((8, 8, 8), field.grid.cell)
        with pytest.raises(ValueError, match="does not match"):
            ChargeDensity.read(path + "::CHGCAR", grid=wrong)

    def test_a_mismatched_cell_is_refused(self, store):
        """Shape alone is not identity: a volume-scaled field would rescale."""
        path, field = store
        wrong = FieldGrid(field.grid.shape, np.eye(3) * 9.0)
        with pytest.raises(ValueError, match="cell"):
            ChargeDensity.read(path + "::CHGCAR", grid=wrong)


class TestSeveralFieldsInOneStore:
    def test_three_fields_share_one_file(self, tmp_path):
        field = density()
        path = str(tmp_path / "fields.h5")
        potential = ExternalPotential.compute(field.structure, field.grid,
                                              {"Si": 4.0}, widths={"Si": 0.5})
        tau = KineticEnergyDensity(field.data * 2, field.grid, field.structure)
        write_fields(path, {"EXTCAR": potential, "CHGCAR": field,
                            "TAUCAR": tau}, compression="gzip")
        assert field_names(path) == ["CHGCAR", "EXTCAR", "TAUCAR"]

    def test_an_unnamed_field_in_a_crowded_store_is_refused(self, tmp_path):
        """Guessing between CHGCAR and TAUCAR would train on the wrong target."""
        field = density()
        path = str(tmp_path / "fields.h5")
        write_fields(path, {"CHGCAR": field,
                            "TAUCAR": KineticEnergyDensity(
                                field.data, field.grid, field.structure)})
        with pytest.raises(ValueError, match="names none of them"):
            ChargeDensity.read(path)

    def test_a_lone_field_needs_no_selector(self, tmp_path):
        field = density()
        path = str(tmp_path / "fields.h5")
        write_fields(path, {"CHGCAR": field})
        assert np.allclose(ChargeDensity.read(path, dtype="float64").data,
                           field.data)

    def test_a_missing_field_lists_what_is_there(self, store):
        path, _ = store
        with pytest.raises(KeyError, match="CHGCAR"):
            ChargeDensity.read(path + "::TAUCAR")

    def test_a_second_cell_in_one_store_is_refused(self, tmp_path):
        """
        Every field of one material must share one mesh in one cell. Two cells
        in one file would be invisible on disk and wrong in every integral.
        """
        path = str(tmp_path / "fields.h5")
        write_fields(path, {"CHGCAR": density(lattice=6.0)})
        with pytest.raises(ValueError, match="different cell"):
            write_fields(path, {"TAUCAR": density(lattice=7.0)})

    def test_rewriting_one_field_leaves_the_others(self, tmp_path):
        field = density()
        path = str(tmp_path / "fields.h5")
        write_fields(path, {"CHGCAR": field, "TAUCAR": KineticEnergyDensity(
            field.data, field.grid, field.structure)})
        write_field(path, "CHGCAR", density(seed=5), compression="lzf")
        assert field_names(path) == ["CHGCAR", "TAUCAR"]


class TestSpinDensities:
    def test_both_channels_survive(self, tmp_path):
        field = density()
        spin = SpinDensity(field.data, field.data * 0.1, field.grid,
                           field.structure, dtype="float64")
        path = str(tmp_path / "fields.h5")
        write_fields(path, {"CHGCAR": spin}, compression="gzip")
        back = SpinDensity.read(path + "::CHGCAR")
        assert np.allclose(spin.total, back.total, rtol=0, atol=1e-12)
        assert np.allclose(spin.magnetization, back.magnetization, rtol=0,
                           atol=1e-12)

    def test_it_is_detected_without_reading_a_value(self, tmp_path):
        field = density()
        spin = SpinDensity(field.data, field.data * 0.1, field.grid,
                           field.structure)
        path = str(tmp_path / "spin.h5")
        write_fields(path, {"CHGCAR": spin})
        assert is_spin_polarized(path + "::CHGCAR")

    def test_a_single_channel_store_is_not_mistaken_for_one(self, store):
        path, _ = store
        assert not is_spin_polarized(path + "::CHGCAR")


class TestWhatAStoreCannotDo:
    def test_augmentation_records_are_refused_rather_than_dropped(self,
                                                                  tmp_path):
        """
        PAW occupancies are what make a density readable by VASP. Silently
        dropping them would produce a file that looks complete and cannot seed
        an ICHARG=1 restart.
        """
        with pytest.raises(ValueError, match="augmentation"):
            density().write(tmp_path / "f.h5", augmentation=["1 2 3"])


class TestCompressionActuallyCompresses:
    def test_a_smooth_field_gets_smaller(self, tmp_path):
        field = density((32, 32, 32))
        sizes = {}
        for codec in CODECS:
            path = str(tmp_path / f"{codec}.h5")
            write_fields(path, {"CHGCAR": field}, compression=codec)
            sizes[codec] = describe(path)["fields"]["CHGCAR"]["stored_bytes"]
        assert sizes["gzip"] < sizes[None]
        assert sizes["lzf"] < sizes[None]

    def test_the_report_says_what_was_used(self, tmp_path):
        path = str(tmp_path / "f.h5")
        write_fields(path, {"CHGCAR": density()}, compression="gzip", level=7)
        info = describe(path)["fields"]["CHGCAR"]
        assert info["compression"] == "gzip"
        assert info["compression_opts"] == 7
        assert info["shuffle"] is True
        assert info["ratio"] > 0

    def test_a_filter_is_never_silently_dropped(self, tmp_path):
        """
        HDF5 cannot filter a contiguous dataset, so asking for compression with
        chunks off has to chunk anyway rather than report a saving that did not
        happen.
        """
        path = str(tmp_path / "f.h5")
        write_field(path, "CHGCAR", density(), compression="gzip", chunks=False)
        assert describe(path)["fields"]["CHGCAR"]["chunks"] is not None


# ===================================================================== #
# Through the cache and the loaders
# ===================================================================== #
class TestTheCacheCanWriteEitherLayout:
    def test_an_hdf5_cache_holds_one_file_per_material(self, tmp_path, raw_runs):
        cache = str(tmp_path / "cache")
        build_field_cache(raw_runs, cache, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="hdf5",
                          compression="gzip")
        entries = sorted(os.listdir(os.path.join(cache, "mat_0")))
        assert "fields.h5" in entries
        assert "CHGCAR" not in entries

    def test_both_layouts_give_the_same_tensors(self, tmp_path, raw_runs):
        samples = {}
        for storage, codec in (("files", None), ("hdf5", None),
                               ("hdf5", "gzip"), ("hdf5", "lzf")):
            cache = str(tmp_path / f"{storage}-{codec}")
            build_field_cache(raw_runs, cache, resolution=0,
                              fields=("EXTCAR", "CHGCAR"), storage=storage,
                              compression=codec)
            dataset = FieldPairDataset(cache, "ext2chg")
            samples[(storage, codec)] = dataset[0]

        reference = samples[("files", None)]
        for key, sample in samples.items():
            assert np.array_equal(sample["input"].numpy(),
                                  reference["input"].numpy()), key
            assert np.array_equal(sample["target"].numpy(),
                                  reference["target"].numpy()), key

    def test_an_hdf5_cache_is_smaller(self, tmp_path, raw_runs):
        def total(cache):
            return sum(os.path.getsize(os.path.join(root, name))
                       for root, _, files in os.walk(cache) for name in files)

        text = str(tmp_path / "text")
        build_field_cache(raw_runs, text, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="files")
        binary = str(tmp_path / "binary")
        build_field_cache(raw_runs, binary, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="hdf5",
                          compression="gzip")
        assert total(binary) < total(text)

    def test_compression_without_hdf5_is_refused(self, tmp_path, raw_runs):
        """A flag that quietly does nothing reports a saving that never happened."""
        with pytest.raises(ValueError, match="storage"):
            build_field_cache(raw_runs, str(tmp_path / "c"), resolution=0,
                              storage="files", compression="gzip")

    def test_an_unknown_storage_is_refused(self, tmp_path, raw_runs):
        with pytest.raises(ValueError, match="storage"):
            build_field_cache(raw_runs, str(tmp_path / "c"), resolution=0,
                              storage="parquet")

    def test_the_layout_is_part_of_the_fingerprint(self, tmp_path, raw_runs):
        """
        Half a cache as text and half as HDF5 would load, and every reuse check
        would then be answering about the wrong files.
        """
        cache = str(tmp_path / "cache")
        build_field_cache(raw_runs, cache, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="files")
        with pytest.raises(Exception):
            build_field_cache(raw_runs, cache, resolution=0,
                              fields=("EXTCAR", "CHGCAR"), storage="hdf5")

    def test_a_rebuild_reuses_what_is_there(self, tmp_path, raw_runs):
        cache = str(tmp_path / "cache")
        for _ in range(2):
            build_field_cache(raw_runs, cache, resolution=0,
                              fields=("EXTCAR", "CHGCAR"), storage="hdf5",
                              compression="gzip")
        assert len(FieldPairDataset(cache, "ext2chg")) == 3

    def test_a_half_written_store_is_not_taken_for_complete(self, tmp_path,
                                                            raw_runs):
        """
        The fields share a file, so the file existing says nothing about which
        datasets are in it. An interrupted build must be resumed, not skipped.
        """
        cache = str(tmp_path / "cache")
        build_field_cache(raw_runs, cache, resolution=0, fields=("CHGCAR",),
                          storage="hdf5")
        assert field_names(os.path.join(cache, "mat_0", "fields.h5")) == ["CHGCAR"]

        from poraque.data.cache import _all_present

        targets = cached_paths(os.path.join(cache, "mat_0"),
                               ("EXTCAR", "CHGCAR"), "hdf5")
        assert not _all_present(targets, "hdf5")


class TestTheLoadersFindBothLayouts:
    def test_discovery_addresses_fields_inside_a_store(self, tmp_path, raw_runs):
        cache = str(tmp_path / "cache")
        build_field_cache(raw_runs, cache, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="hdf5")
        records = discover_materials(cache, required=("EXTCAR", "CHGCAR"))
        assert len(records) == 3
        assert records[0].files["CHGCAR"].endswith("fields.h5::CHGCAR")

    def test_a_text_file_wins_a_collision(self, tmp_path):
        """
        A directory holding both was converted by hand. The text file is the
        one every other tool can read, so it is the safer of two disagreeing
        copies to serve.
        """
        directory = tmp_path / "mat"
        directory.mkdir()
        field = density()
        field.write(directory / "CHGCAR")
        write_fields(str(directory / "fields.h5"), {"CHGCAR": density(seed=9)})
        files = prepared_fields(str(directory), ("CHGCAR",))
        assert files["CHGCAR"].endswith(os.path.join("mat", "CHGCAR"))

    def test_a_mixed_cache_works_without_anyone_intending_it(self, tmp_path,
                                                             raw_runs):
        text = str(tmp_path / "text")
        build_field_cache(raw_runs, text, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="files")
        binary = str(tmp_path / "binary")
        build_field_cache(raw_runs, binary, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="hdf5")

        mixed = tmp_path / "mixed"
        mixed.mkdir()
        import shutil

        shutil.copytree(os.path.join(text, "mat_0"), mixed / "mat_0")
        shutil.copytree(os.path.join(binary, "mat_1"), mixed / "mat_1")
        dataset = FieldPairDataset(str(mixed), "ext2chg")
        assert len(dataset) == 2
        assert {tuple(dataset[i]["shape"]) for i in range(2)} == {
            (12, 12, 12), (16, 16, 16)}

    def test_an_unreadable_store_does_not_hide_the_text_beside_it(self,
                                                                  tmp_path):
        directory = tmp_path / "mat"
        directory.mkdir()
        density().write(directory / "CHGCAR")
        (directory / "fields.h5").write_bytes(b"not an HDF5 file at all")
        assert "CHGCAR" in prepared_fields(str(directory), ("CHGCAR",))


class TestRaggedGridsAcrossMaterials:
    """
    Materials Project cells differ in size. Nothing about that is new here —
    the point is that HDF5 storage does not break it.
    """

    @pytest.fixture
    def cache(self, tmp_path, raw_runs):
        path = str(tmp_path / "cache")
        build_field_cache(raw_runs, path, resolution=0,
                          fields=("EXTCAR", "CHGCAR"), storage="hdf5",
                          compression="gzip")
        return path

    def test_three_shapes_are_carried_as_they_are(self, cache):
        dataset = FieldPairDataset(cache, "ext2chg")
        assert dataset.shapes() == [(12, 12, 12), (16, 16, 16), (20, 20, 20)]

    def test_no_batch_mixes_two_shapes(self, cache):
        dataset = FieldPairDataset(cache, "ext2chg")
        sampler = ShapeBucketSampler(dataset, batch_size=3, shuffle=False)
        shapes = dataset.shapes()
        for batch in sampler:
            assert len({shapes[index] for index in batch}) == 1

    def test_a_dataloader_yields_every_material(self, cache):
        dataset = FieldPairDataset(cache, "ext2chg")
        loader = make_dataloader(dataset, batch_size=2, shuffle=False)
        seen = [name for batch in loader for name in batch["material"]]
        assert sorted(seen) == ["mat_0", "mat_1", "mat_2"]

    def test_shapes_are_read_from_headers_alone(self, cache):
        """
        Bucketing must not require decoding the fields. HDF5 keeps the shape in
        the object header, so this stays as cheap as the text peek it replaces.
        """
        dataset = FieldPairDataset(cache, "ext2chg")
        for record in dataset.materials:
            record.shape = None
        assert dataset.shapes() == [(12, 12, 12), (16, 16, 16), (20, 20, 20)]
