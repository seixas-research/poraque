# -*- coding: utf-8 -*-
# file: test_mp_data.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Materials Project ingestion: compressed reads, and the fields MP does not ship.

Nothing here touches the network. The fixtures write small ``CHGCAR`` files of
known composition and known electron count, which is enough to exercise every
inference the pipeline makes from an MP download — the structure out of the
header, the valence charges out of the densities, the external potential out of
the structure, and the tasks that are and are not trainable.
"""

import bz2
import gzip
import lzma
import os
import zipfile

import numpy as np
import pytest

from poraque.data import (
    MPChargeDensityDataset,
    available_tasks,
    build_mp_cache,
    discover_mp_chgcars,
    infer_valence_charges,
)
from poraque.data.materials_project import Estimate, MPDataFetcher, load_api_key
from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.io.compressed import (
    is_compressed,
    open_text,
    strip_compression_suffix,
)
from poraque.fields.vasp.volumetric import (
    read_structure_header,
    read_volumetric,
    write_volumetric,
)
from poraque.fields.vasp.poscar import Poscar


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def make_chgcar(path, symbols, counts, electrons, shape=(8, 8, 8),
                lattice=4.0, spin=False, augmentation=None):
    """
    Write a ``CHGCAR`` whose density integrates to exactly ``electrons``.

    A flat density is enough: every quantity the ingestion infers — the
    composition, the electron count, the cell — comes from the header and the
    block's *sum*, none of it from the shape of the field.
    """
    positions = np.linspace(0.0, 0.9, sum(counts)).reshape(-1, 1) * np.ones((1, 3))
    structure = Poscar(np.eye(3) * lattice, list(symbols), list(counts),
                       positions, comment="test")
    # VASP stores rho * Omega, so a block whose mean is N integrates to N.
    data = np.full(shape, float(electrons))
    extra = ["  {:d}  {:d}  {:d}".format(*shape),
             " ".join("0.0" for _ in range(int(np.prod(shape))))] if spin else None
    write_volumetric(path, structure, data,
                     augmentation=(list(augmentation or []) + (extra or []))
                     or None)
    return path


def compress(path, suffix):
    """Rewrite ``path`` as ``path + suffix`` and remove the original."""
    target = str(path) + suffix
    with open(path, "rb") as source:
        payload = source.read()
    if suffix == ".gz":
        with gzip.open(target, "wb") as sink:
            sink.write(payload)
    elif suffix == ".bz2":
        with bz2.open(target, "wb") as sink:
            sink.write(payload)
    elif suffix == ".xz":
        with lzma.open(target, "wb") as sink:
            sink.write(payload)
    elif suffix == ".zip":
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(os.path.basename(path), payload)
    else:                                                   # pragma: no cover
        raise ValueError(suffix)
    os.remove(path)
    return target


def _material_dir(root, identifier):
    """``<root>/<id>/CHGCAR`` — the path one material's density belongs at."""
    directory = root / identifier
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "CHGCAR"


@pytest.fixture
def mp_download(tmp_path):
    """
    A miniature Materials Project download: one directory per material, each
    holding a gzipped density and nothing else.

    Compositions are chosen so the electron counts determine all three valence
    charges — Ag 11, Pd 10, Pt 10, the values the POTCARs state.
    """
    root = tmp_path / "MP"
    root.mkdir(parents=True)
    entries = [
        ("mp-124", ["Ag"], [1], 11),
        ("mp-81", ["Pd"], [1], 10),
        ("mp-126", ["Pt"], [1], 10),
        ("mp-1183137", ["Ag", "Pd"], [3, 1], 43),
        ("mp-30353", ["Ag", "Pt"], [1, 3], 41),
    ]
    for identifier, symbols, counts, electrons in entries:
        (root / identifier).mkdir()
        plain = root / identifier / "CHGCAR"
        make_chgcar(plain, symbols, counts, electrons)
        compress(plain, ".gz")
    return root


# ---------------------------------------------------------------------- #
# Compressed reading
# ---------------------------------------------------------------------- #
class TestCompressedIO:
    """Every codec reaches the parser as the same stream of lines."""

    @pytest.mark.parametrize("suffix", [".gz", ".bz2", ".xz", ".zip"])
    def test_volumetric_round_trip(self, tmp_path, suffix):
        plain = tmp_path / "CHGCAR"
        make_chgcar(plain, ["Ag"], [2], 22, shape=(6, 6, 6))
        reference = read_volumetric(plain)

        path = compress(plain, suffix)
        structure, data, _ = read_volumetric(path)

        assert structure.symbols == reference[0].symbols
        assert structure.counts == reference[0].counts
        np.testing.assert_allclose(data, reference[1])
        np.testing.assert_allclose(structure.cell, reference[0].cell)

    @pytest.mark.parametrize("suffix", [".gz", ".zip"])
    def test_field_and_grid_read_compressed(self, tmp_path, suffix):
        plain = tmp_path / "CHGCAR"
        make_chgcar(plain, ["Pt"], [1], 10, shape=(8, 8, 8), lattice=4.0)
        path = compress(plain, suffix)

        grid = FieldGrid.from_file(path)
        density = ChargeDensity.read(path, grid=grid)

        assert grid.shape == (8, 8, 8)
        # The file stores rho * Omega, so the field integrates to the count.
        assert density.integrate() == pytest.approx(10.0)

    def test_header_read_is_cheap_and_correct(self, tmp_path):
        plain = tmp_path / "CHGCAR"
        make_chgcar(plain, ["Ag", "Pt"], [3, 1], 44)
        path = compress(plain, ".gz")

        structure = read_structure_header(path)

        assert structure.symbols == ["Ag", "Pt"]
        assert structure.counts == [3, 1]

    def test_poscar_reads_compressed(self, tmp_path):
        plain = tmp_path / "POSCAR"
        Poscar(np.eye(3) * 3.0, ["Si"], [1], np.zeros((1, 3))).write(plain)
        path = compress(plain, ".gz")

        assert Poscar.from_file(path).symbols == ["Si"]

    def test_suffix_helpers(self):
        assert is_compressed("CHGCAR_mp-126.gz")
        assert is_compressed("CHGCAR.ZIP")
        assert not is_compressed("CHGCAR")
        assert strip_compression_suffix("CHGCAR_mp-126.gz") == "CHGCAR_mp-126"
        assert strip_compression_suffix("CHGCAR") == "CHGCAR"

    def test_ambiguous_zip_is_an_error(self, tmp_path):
        """Two members and no name match: guessing would be the wrong help."""
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("CHGCAR_a", "x")
            handle.writestr("CHGCAR_b", "y")

        with pytest.raises(ValueError, match="ambiguous"):
            with open_text(archive) as handle:
                handle.read()

    def test_zip_member_matching_the_archive_name_wins(self, tmp_path):
        archive = tmp_path / "CHGCAR_mp-1.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("README", "ignore me")
            handle.writestr("CHGCAR_mp-1", "the payload")

        with open_text(archive) as handle:
            assert handle.read() == "the payload"

    def test_truncated_file_names_itself(self, tmp_path):
        path = tmp_path / "CHGCAR"
        path.write_text("c\n1.0\n1 0 0\n0 1 0\n0 0 1\nAg\n1\nDirect\n0 0 0\n\n"
                        "4 4 4\n1.0 2.0\n")

        with pytest.raises(ValueError, match="truncated"):
            read_volumetric(path)


# ---------------------------------------------------------------------- #
# Discovery and valence charges
# ---------------------------------------------------------------------- #
class TestDiscovery:
    def test_finds_every_density_and_names_it_by_material_id(self, mp_download):
        records = discover_mp_chgcars(mp_download)

        assert [r.identifier for r in records] == [
            "mp-1183137", "mp-124", "mp-126", "mp-30353", "mp-81"]
        assert all(set(r.files) == {"CHGCAR"} for r in records), \
            "a record must not name files the download does not have"

    def test_a_prefix_filters_the_material_directories(self, mp_download):
        assert [r.identifier for r in
                discover_mp_chgcars(mp_download, pattern="mp-12")] == [
                    "mp-124", "mp-126"]

    def test_empty_directory_says_what_to_do(self, tmp_path):
        (tmp_path / "MP").mkdir()
        with pytest.raises(FileNotFoundError, match="poraque-mp"):
            discover_mp_chgcars(tmp_path / "MP")


class TestValenceCharges:
    """Z_val is recovered from the densities, because no POTCAR ships with them."""

    def test_recovers_the_potcar_values(self, mp_download):
        charges = infer_valence_charges(discover_mp_chgcars(mp_download))

        assert charges == {"Ag": 11.0, "Pd": 10.0, "Pt": 10.0}

    def test_reads_only_as_many_densities_as_it_needs(self, mp_download,
                                                      monkeypatch):
        """
        Rank, not exhaustiveness.

        Three unknown charges need three independent compositions, and a
        density is opened only for a composition that adds one. On a real
        download the difference is a handful of small reads against a full pass
        over hundreds of megabytes.
        """
        from poraque.data import mp_dataset

        opened = []
        original = mp_dataset._electron_count
        monkeypatch.setattr(mp_dataset, "_electron_count",
                            lambda path: opened.append(path) or original(path))

        charges = infer_valence_charges(discover_mp_chgcars(mp_download))

        assert len(opened) == 3, "five materials, three elements"
        assert charges == {"Ag": 11.0, "Pd": 10.0, "Pt": 10.0}

    def test_overrides_win_and_reduce_the_system(self, mp_download):
        charges = infer_valence_charges(discover_mp_chgcars(mp_download),
                                        overrides={"Pt": 18.0})

        assert charges["Pt"] == 18.0
        assert charges["Ag"] == 11.0

    def test_undetermined_system_explains_itself(self, tmp_path):
        """One binary cannot split its electrons between its two elements."""
        root = tmp_path / "chgcar"
        root.mkdir()
        make_chgcar(_material_dir(root, "mp-1"), ["Ag", "Pt"], [1, 1], 22)

        with pytest.raises(ValueError, match="independent directions"):
            infer_valence_charges(discover_mp_chgcars(root))

    def test_that_same_system_is_solvable_with_one_override(self, tmp_path):
        root = tmp_path / "chgcar"
        root.mkdir()
        make_chgcar(_material_dir(root, "mp-1"), ["Ag", "Pt"], [1, 1], 22)

        charges = infer_valence_charges(discover_mp_chgcars(root),
                                        overrides={"Ag": 11.0})

        assert charges == {"Ag": 11.0, "Pt": 11.0}


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class TestMPChargeDensityDataset:
    def test_serves_aligned_pairs_at_the_requested_resolution(self, mp_download):
        data = MPChargeDensityDataset(mp_download, resolution=4)

        sample = data[0]

        assert sample["input"].shape == (1, 4, 4, 4)
        assert sample["target"].shape == (1, 4, 4, 4)
        assert data.channels == (1, 1)

    def test_the_potential_comes_from_the_density_header(self, mp_download):
        """No POSCAR exists; the structure is the one the CHGCAR carries."""
        data = MPChargeDensityDataset(mp_download, resolution=4)
        potential, density = data.load_fields(0)

        assert potential.structure.symbols == density.structure.symbols
        assert potential.grid.shape == density.grid.shape
        # The structure is the density's, which is the property that matters;
        # `derived_from` was a label the removed bulk source wrote and there is
        # no second geometry left for it to distinguish this one from.
        assert np.allclose(potential.structure.cell, density.structure.cell)
        # A neutralising background fixes the cell average at zero.
        assert potential.data.mean() == pytest.approx(0.0, abs=1e-8)

    def test_native_resolution_keeps_the_mp_grid(self, mp_download):
        data = MPChargeDensityDataset(mp_download)

        assert data.shapes() == [(8, 8, 8)] * 5

    def test_missing_energies_are_none_not_zero(self, mp_download):
        """MP ships no OUTCAR, and a fabricated zero would enter a mean."""
        data = MPChargeDensityDataset(mp_download, resolution=4)

        assert data[0]["reference_energy"] is None
        assert data.reference_energy(0) is None

    def test_chg2tau_is_refused_with_a_reason(self, mp_download):
        with pytest.raises(ValueError, match="kinetic energy density"):
            MPChargeDensityDataset(mp_download, task="chg2tau")

    def test_available_tasks_names_only_what_is_trainable(self):
        assert available_tasks() == ["ext2chg"]

    def test_split_carries_the_settings_to_both_halves(self, mp_download):
        data = MPChargeDensityDataset(mp_download, resolution=4,
                                      charges={"Ag": 11, "Pd": 10, "Pt": 10})

        train, held_out = data.split(fraction=0.6, seed=0)

        assert len(train) + len(held_out) == len(data)
        assert train.resolution == held_out.resolution == 4
        assert train.charges == data.charges

    def test_transforms_fit_on_this_dataset(self, mp_download):
        """The normalizations are derived from fields that exist only in memory."""
        data = MPChargeDensityDataset(mp_download, resolution=4, cache=True)

        source, target = data.fit_transforms(max_materials=3, max_points=500)

        assert source is data.input_transform
        assert target is data.target_transform

    def test_spin_channel_is_available_when_the_file_has_one(self, tmp_path):
        root = tmp_path / "chgcar"
        root.mkdir()
        make_chgcar(_material_dir(root, "mp-1"), ["Pt"], [1], 10, spin=True)
        make_chgcar(_material_dir(root, "mp-2"), ["Pt"], [2], 20, spin=True)

        total_only = MPChargeDensityDataset(root)
        both = MPChargeDensityDataset(root, spin=True)

        assert total_only.channels == (1, 1)
        assert both.channels == (1, 2)

    def test_spin_true_on_collinear_data_is_an_error(self, mp_download):
        with pytest.raises(ValueError, match="no magnetisation block"):
            MPChargeDensityDataset(mp_download, spin=True)


# ---------------------------------------------------------------------- #
# Cache
# ---------------------------------------------------------------------- #
class TestBuildMPCache:
    def test_writes_the_standard_material_layout(self, mp_download, tmp_path):
        cache = build_mp_cache(mp_download, tmp_path / "cache", resolution=4)

        for identifier in ("mp-124", "mp-81", "mp-126"):
            assert os.path.exists(os.path.join(cache, identifier, "EXTCAR"))
            assert os.path.exists(os.path.join(cache, identifier, "CHGCAR"))
            assert not os.path.exists(os.path.join(cache, identifier, "TAUCAR"))

    def test_the_standard_dataset_reads_it_unchanged(self, mp_download, tmp_path):
        from poraque.ml.data import FieldPairDataset

        cache = build_mp_cache(mp_download, tmp_path / "cache", resolution=4)
        data = FieldPairDataset(cache, task="ext2chg")

        assert len(data) == 5
        assert data[0]["input"].shape == (1, 4, 4, 4)

    def test_electron_count_survives_the_downsample(self, mp_download, tmp_path):
        """Fourier truncation is exact for a band-limited field; check it is."""
        cache = build_mp_cache(mp_download, tmp_path / "cache", resolution=4)
        path = os.path.join(cache, "mp-124", "CHGCAR")

        density = ChargeDensity.read(path, grid=FieldGrid.from_file(path))

        assert density.integrate() == pytest.approx(11.0, rel=1e-9)

    def test_rebuilding_reuses_what_is_there(self, mp_download, tmp_path):
        cache = build_mp_cache(mp_download, tmp_path / "cache", resolution=4)
        stamp = os.path.getmtime(os.path.join(cache, "mp-124", "CHGCAR"))

        build_mp_cache(mp_download, cache, resolution=4)

        assert os.path.getmtime(os.path.join(cache, "mp-124", "CHGCAR")) == stamp

    def test_limit_takes_the_smallest_files(self, mp_download, tmp_path):
        cache = build_mp_cache(mp_download, tmp_path / "cache", resolution=4,
                               limit=2)

        # Directories only: a cache also holds the build summary beside them.
        assert len([entry for entry in os.listdir(cache)
                    if os.path.isdir(os.path.join(cache, entry))]) == 2


# ---------------------------------------------------------------------- #
# The operator on MP data
# ---------------------------------------------------------------------- #
class TestOperatorOnMPData:
    """The FNO ingests an MP download with no shape mismatch and no missing file."""

    def test_ragged_mp_grids_batch_without_padding(self, tmp_path):
        """MP cells differ in shape; the sampler buckets rather than pads."""
        from poraque.ml.data import ShapeBucketSampler

        root = tmp_path / "chgcar"
        root.mkdir()
        make_chgcar(_material_dir(root, "mp-1"), ["Ag"], [1], 11, shape=(8, 8, 8))
        make_chgcar(_material_dir(root, "mp-2"), ["Ag"], [2], 22, shape=(8, 8, 12))
        make_chgcar(_material_dir(root, "mp-3"), ["Pt"], [1], 11, shape=(8, 8, 8))

        data = MPChargeDensityDataset(root, charges={"Ag": 11, "Pt": 11})
        sampler = ShapeBucketSampler(data, batch_size=4, shuffle=False)

        assert sorted(len(batch) for batch in sampler) == [1, 2]

    def test_prediction_round_trips_through_a_chgcar(self, mp_download, tmp_path):
        """A prediction is written in the same format the input came in."""
        from poraque.ml import FieldOperator

        data = MPChargeDensityDataset(mp_download, resolution=4)
        operator = FieldOperator("ext2chg", width=8, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        potential, _ = data.load_fields(0)

        prediction = operator.predict(potential)
        path = prediction.write(tmp_path / "CHGCAR_predicted")

        assert isinstance(prediction, ChargeDensity)
        assert prediction.data.shape == (4, 4, 4)
        assert read_volumetric(path)[1].shape == (4, 4, 4)


# ---------------------------------------------------------------------- #
# Fetcher (no network)
# ---------------------------------------------------------------------- #
class TestMPDataFetcher:
    def test_chemical_space_spans_every_subsystem(self, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        fetcher = MPDataFetcher("Pt-Pd-Ni")

        assert fetcher.elements == sorted(["Pt", "Pd", "Ni"])
        assert sorted(fetcher.chemical_systems) == sorted([
            "Ni", "Pd", "Pt", "Ni-Pd", "Ni-Pt", "Pd-Pt", "Ni-Pd-Pt"])

    def test_filters_are_pushed_into_the_query_where_they_can_be(self, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        fetcher = MPDataFetcher(["Si", "O"], band_gap=(0.5, 6.0),
                                num_sites=(1, 8), is_stable=True,
                                crystal_system="cubic")

        assert fetcher._query_filters() == {
            "band_gap": (0.5, 6.0), "num_sites": (1, 8), "is_stable": True}
        # Crystal system lives under a sub-document, so it is applied locally.
        assert fetcher.crystal_system == {"Cubic"}

    def test_no_elements_means_the_whole_database(self, monkeypatch):
        """
        This used to raise. It now selects every material the index says has a
        charge density, which is what ``--all`` asks for — the element list
        stops being a requirement and becomes one more filter.

        The *command line* still refuses a bare invocation
        (:func:`~poraque.data.materials_project.main`): omitting a flag by
        accident should not silently become a multi-terabyte question.
        """
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        fetcher = MPDataFetcher([])

        assert fetcher.elements == []
        assert fetcher.label == "all elements"
        assert fetcher.chemical_systems == []

    def test_api_key_precedence(self, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "from-env")

        assert load_api_key() == "from-env"
        assert load_api_key("explicit") == "explicit"

    def test_missing_api_key_says_where_to_put_one(self, monkeypatch):
        """
        The dotenv lookup is stubbed out rather than pointed somewhere empty.

        ``load_dotenv`` searches upward from the calling module, so on a real
        checkout it can reach a developer's own ``~/.env`` and quietly succeed
        — which would make this test pass for the wrong reason on one machine
        and fail on another.
        """
        import dotenv

        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

        with pytest.raises(ValueError, match=r"\.env"):
            load_api_key()

    def test_estimate_reports_the_three_storage_figures(self):
        estimate = Estimate(elements=["Ag"], n_materials=3, n_advertised=3,
                            n_available=2,
                            sizes={"mp-1": 1_000_000, "mp-2": 3_000_000,
                                   "mp-3": -1},
                            rows=[])

        assert estimate.download_bytes == 4_000_000
        assert estimate.gz_disk_bytes == pytest.approx(3_516_000)
        assert estimate.unzipped_bytes == pytest.approx(3_516_000 * 2.99)
        assert estimate.n_missing == 1
        assert estimate.files_under(2.0) == (1, 1_000_000)
