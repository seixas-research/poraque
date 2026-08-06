# -*- coding: utf-8 -*-
# file: test_data_sources.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Format detection, and training across a mixture of them.

The fixtures below build the three layouts side by side — a DFT calculation
directory, a bulk archive of standalone densities, and a prepared cache — so
the tests exercise the thing that actually matters: that a directory is
recognised for what it is, that a field is *offered* only when it can really be
produced, and that a mixture reaches the model as one dataset.
"""

import gzip
import os
import warnings

import numpy as np
import pytest

from poraque.data import (
    BulkDensitySource,
    CalculationSource,
    MixedFieldDataset,
    PreparedFieldsSource,
    available_formats,
    build_field_cache,
    detect_source,
    discover_records,
    resolve_source,
)
from poraque.data.cache import build_paw_reference, load_paw_reference
from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid
from poraque.fields import KineticEnergyDensity
from poraque.fields.vasp.poscar import Poscar

CHARGES = {"Si": 4.0, "Au": 11.0}

#: The reference POTCAR shipped with the repository's gold dataset. Tests that
#: need a *complete* local-potential table use it and skip when it is absent —
#: no fixture can fabricate one, since a truncated table parses fine and then
#: cannot be splined onto the PSGMAX mesh.
REFERENCE_POTCAR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vasp", "struct_000", "POTCAR")

#: A POTCAR whose ``local part`` block stops after three values instead of
#: NPSPTS of them. It parses without error, which is exactly the problem.
TRUNCATED_POTCAR = """ PAW_PBE Si 05Jan2001
   4.00000000000000
 parameters from PSCTR are:
   VRHFIN =Si: s2p2
   LEXCH  = PE
   TITEL  = PAW_PBE Si 05Jan2001
   POMASS =   28.085; ZVAL   =    4.000    mass and valenz
   RCORE  =    1.900    outmost cutoff radius
   ENMAX  =  245.345; ENMIN  = 184.009 eV
 END of PSCTR-controll parameters
  local part
             4.00000000000000
  0.4899969775558059E+03  0.4897893015154668E+03  0.4891667097127825E+03
 gradient corrections used for XC
    1
 End of Dataset
"""


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def _material(cell, shape, element="Si", counts=(2,), seed=0):
    """A structure, its grid, a potential and a plausible density on it."""
    rng = np.random.default_rng(seed)
    grid = FieldGrid(shape, cell)
    structure = Poscar(cell, [element], list(counts),
                       rng.random((sum(counts), 3)))
    potential = ExternalPotential.compute(structure, grid, CHARGES,
                                          widths={element: 0.5})
    density = np.exp(-(potential.data - potential.data.min()) / 20.0) * 0.2 + 0.01
    return grid, structure, potential, ChargeDensity(density, grid, structure)


@pytest.fixture
def potcar_library(tmp_path):
    """A one-element POTCAR library laid out the way VASP ships one."""
    if not os.path.exists(REFERENCE_POTCAR):
        pytest.skip("reference POTCAR not available")
    root = tmp_path / "potcars"
    (root / "Au").mkdir(parents=True)
    with open(REFERENCE_POTCAR, "r", errors="replace") as source:
        (root / "Au" / "POTCAR").write_text(source.read())
    return str(root)


def write_calculation(directory, shape=(12, 12, 12), cell=None, seed=0,
                      tau=True, encut=300.0, element="Si", potcar=None):
    """
    A VASP run directory: inputs plus outputs.

    No ``POTCAR`` is written, so the potential falls back to the Gaussian model
    with explicit charges — enough to exercise the source, and it keeps the
    fixture free of a pseudopotential file the repository does not ship.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    cell = np.eye(3) * 5.0 if cell is None else cell
    grid, structure, potential, density = _material(cell, shape, seed=seed,
                                                    element=element)

    structure.write(os.path.join(directory, "POSCAR"))
    if potcar is not None:
        with open(potcar, "r", errors="replace") as source:
            with open(os.path.join(directory, "POTCAR"), "w") as sink:
                sink.write(source.read())
    with open(os.path.join(directory, "INCAR"), "w") as handle:
        handle.write(f"ENCUT = {encut}\nPREC = Accurate\n")
    density.write(os.path.join(directory, "CHGCAR"))
    if tau:
        values = 2.871234 * density.data ** (5.0 / 3.0) * 51.42 + 0.01
        KineticEnergyDensity(values, grid, structure).write(
            os.path.join(directory, "TAUCAR"))
    return directory


def write_bulk(directory, identifiers=("mp-1", "mp-2"), shape=(12, 12, 12),
               compress=True, element="Si"):
    """A flat archive of standalone densities, gzipped like a real download."""
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    for index, identifier in enumerate(identifiers):
        _, _, _, density = _material(np.eye(3) * (5.0 + index), shape,
                                     seed=index + 10, element=element)
        path = os.path.join(directory, f"CHGCAR_{identifier}")
        density.write(path)
        if compress:
            with open(path, "rb") as source, \
                    gzip.open(path + ".gz", "wb") as sink:
                sink.write(source.read())
            os.remove(path)
    return directory


def write_prepared(directory, names=("a", "b"), shape=(12, 12, 12), tau=True):
    """A cache: one directory per material, holding the fields themselves."""
    directory = str(directory)
    for index, name in enumerate(names):
        child = os.path.join(directory, name)
        os.makedirs(child, exist_ok=True)
        grid, structure, potential, density = _material(
            np.eye(3) * (5.0 + index), shape, seed=index + 20)
        potential.write(os.path.join(child, "EXTCAR"))
        density.write(os.path.join(child, "CHGCAR"))
        if tau:
            values = 2.871234 * density.data ** (5.0 / 3.0) * 51.42 + 0.01
            KineticEnergyDensity(values, grid, structure).write(
                os.path.join(child, "TAUCAR"))
    return directory


@pytest.fixture
def calculations(tmp_path):
    root = tmp_path / "runs"
    for index in range(3):
        write_calculation(root / f"struct_{index:03d}", seed=index)
    return str(root)


@pytest.fixture
def bulk(tmp_path):
    return write_bulk(tmp_path / "chgcar")


@pytest.fixture
def prepared(tmp_path):
    return write_prepared(tmp_path / "cache")


# ---------------------------------------------------------------------- #
# Detection
# ---------------------------------------------------------------------- #
class TestDetection:
    def test_each_layout_is_recognised(self, calculations, bulk, prepared):
        assert detect_source(calculations) is CalculationSource
        assert detect_source(bulk) is BulkDensitySource
        assert detect_source(prepared) is PreparedFieldsSource

    def test_a_single_calculation_directory_is_one_material(self, calculations):
        one = os.path.join(calculations, "struct_000")

        source = resolve_source(one)

        assert isinstance(source, CalculationSource)
        assert [record.identifier for record in source.discover()] == ["struct_000"]

    def test_a_calculation_is_not_mistaken_for_a_bulk_archive(self, calculations):
        """It contains a CHGCAR too; the POSCAR is what tells them apart."""
        one = os.path.join(calculations, "struct_000")

        assert not BulkDensitySource.detect(one)
        assert CalculationSource.detect(one)

    def test_metadata_files_are_not_mistaken_for_densities(self, tmp_path):
        """`chgcar_estimate.csv` sits in every real download."""
        root = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",))
        with open(os.path.join(root, "chgcar_estimate.csv"), "w") as handle:
            handle.write("material_id,size_mb\nmp-1,1.0\n")

        records = resolve_source(root).discover()

        assert [record.identifier for record in records] == ["mp-1"]

    def test_the_conventional_chgcar_subdirectory_is_followed(self, tmp_path):
        """Pointing a config at `data/MP` should find `data/MP/chgcar`."""
        download = tmp_path / "MP"
        write_bulk(download / "chgcar", identifiers=("mp-1", "mp-2"))
        (download / "summary.csv").write_text("material_id\nmp-1\n")

        source = resolve_source(download)

        assert isinstance(source, BulkDensitySource)
        assert len(source.discover()) == 2

    def test_an_unrecognisable_directory_says_what_was_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(ValueError, match="POSCAR or CONTCAR"):
            detect_source(tmp_path / "empty")

    def test_a_missing_directory_is_not_an_unknown_format(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_source(tmp_path / "nope")

    def test_an_explicit_format_is_checked_against_the_directory(self, bulk):
        with pytest.raises(ValueError, match="not laid out as a vasp dataset"):
            resolve_source(bulk, format="vasp")

    def test_an_unknown_format_lists_the_known_ones(self, bulk):
        with pytest.raises(ValueError, match="Unknown data format"):
            resolve_source(bulk, format="hdf5")

    def test_available_formats_are_in_detection_order(self):
        assert available_formats() == ["vasp", "prepared", "bulk"]


# ---------------------------------------------------------------------- #
# What each source offers
# ---------------------------------------------------------------------- #
class TestProvides:
    def test_a_calculation_always_offers_a_potential(self, calculations):
        """It is computed from the inputs, so no EXTCAR file is needed."""
        source = resolve_source(calculations)
        record = source.discover()[0]

        assert "EXTCAR" not in os.listdir(record.directory)
        assert "EXTCAR" in source.provides(record)

    def test_a_calculation_without_tau_offers_the_rest(self, tmp_path):
        """A density-only run is a first-class ext2chg dataset, not an error."""
        write_calculation(tmp_path / "runs" / "struct_000", tau=False)
        source = resolve_source(tmp_path / "runs")

        assert source.provides(source.discover()[0]) == ("EXTCAR", "CHGCAR")

    def test_tau_is_offered_per_material_not_per_directory(self, tmp_path):
        write_calculation(tmp_path / "runs" / "struct_000", tau=True)
        write_calculation(tmp_path / "runs" / "struct_001", tau=False)
        source = resolve_source(tmp_path / "runs")

        offered = {record.identifier: source.provides(record)
                   for record in source.discover()}

        assert "TAUCAR" in offered["struct_000"]
        assert "TAUCAR" not in offered["struct_001"]

    def test_a_bulk_archive_never_offers_tau(self, bulk):
        source = resolve_source(bulk)

        assert source.provides(source.discover()[0]) == ("EXTCAR", "CHGCAR")

    def test_asking_a_bulk_archive_for_tau_explains_why_not(self, bulk):
        source = resolve_source(bulk, charges=CHARGES)
        record = source.discover()[0]

        with pytest.raises(FileNotFoundError, match="publishes no TAUCAR"):
            source.read(record, "TAUCAR", source.grid(record))

    def test_a_prepared_cache_offers_what_is_on_disk(self, tmp_path):
        write_prepared(tmp_path / "cache", names=("a",), tau=False)
        source = resolve_source(tmp_path / "cache")

        assert source.provides(source.discover()[0]) == ("EXTCAR", "CHGCAR")

    def test_the_pattern_filter_excludes_sibling_directories(self, tmp_path):
        """A `ref/` of isolated atoms must not be trained on."""
        write_calculation(tmp_path / "runs" / "struct_000")
        write_calculation(tmp_path / "runs" / "ref_Si")

        every = resolve_source(tmp_path / "runs").discover()
        filtered = resolve_source(tmp_path / "runs", pattern="struct").discover()

        assert len(every) == 2
        assert [record.identifier for record in filtered] == ["struct_000"]


# ---------------------------------------------------------------------- #
# Reading
# ---------------------------------------------------------------------- #
class TestReading:
    def test_every_source_puts_its_fields_on_one_grid(self, calculations):
        source = resolve_source(calculations, charges=CHARGES)
        record = source.discover()[0]
        grid = source.grid(record)

        fields = [source.read(record, name, grid)
                  for name in source.provides(record)]

        assert {tuple(field.grid.shape) for field in fields} == {tuple(grid.shape)}

    def test_shape_is_read_from_a_header_alone(self, bulk):
        source = resolve_source(bulk, charges=CHARGES)
        record = source.discover()[0]

        assert source.shape(record) == (12, 12, 12)
        assert source.shape(record) == tuple(source.grid(record).shape)

    def test_the_bulk_potential_comes_from_the_density_header(self, bulk):
        source = resolve_source(bulk, charges=CHARGES)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["derived_from"] == "CHGCAR header"
        assert potential.data.mean() == pytest.approx(0.0, abs=1e-8)

    def test_bulk_charges_are_inferred_when_not_supplied(self, bulk):
        source = resolve_source(bulk)

        assert set(source.charges) == {"Si"}


# ---------------------------------------------------------------------- #
# The external potential a bulk archive gets
# ---------------------------------------------------------------------- #
class TestPotcarLibrary:
    """
    ``potcar_dir`` closes the one real gap in training on a public archive.

    The archive supplies the structure; the library supplies the
    pseudopotentials. Together they are everything the exact tabulated
    construction needs, so the potential stops being a model of VASP's and
    becomes VASP's.
    """

    def test_without_a_library_the_potential_is_the_gaussian_model(self, tmp_path):
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")
        source = resolve_source(archive, charges=CHARGES)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "gaussian"
        assert source.potential_model() == "gaussian"

    def test_with_a_library_the_potential_is_tabulated(self, tmp_path,
                                                       potcar_library):
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")
        source = resolve_source(archive, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "potcar"
        assert potential.metadata["derived_from"] == "CHGCAR header + POTCAR library"
        assert source.potential_model() == "tabulated"

    def test_the_two_constructions_genuinely_differ(self, tmp_path,
                                                    potcar_library):
        """
        If they agreed, the option would be pointless.

        The published residual is of order 0.1 relative L2; asserting only that
        it is well clear of zero keeps the test about the mechanism rather than
        about one number on one fixture.
        """
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")
        record = resolve_source(archive).discover()[0]
        grid = resolve_source(archive).grid(record)

        gaussian = resolve_source(archive, charges=CHARGES).read(
            record, "EXTCAR", grid)
        tabulated = resolve_source(archive, potcar_dir=potcar_library).read(
            record, "EXTCAR", grid)

        difference = np.linalg.norm(tabulated.data - gaussian.data)
        assert difference / np.linalg.norm(tabulated.data) > 0.01
        # Both are neutralised the same way, so both average to zero.
        assert tabulated.data.mean() == pytest.approx(0.0, abs=1e-8)

    def test_charges_come_from_the_library_rather_than_inference(
            self, tmp_path, potcar_library):
        """Z_val is stated by the POTCAR; inferring it would be a detour."""
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")

        source = resolve_source(archive, potcar_dir=potcar_library)

        assert source.charges["Au"] == 11.0

    def test_a_missing_element_warns_and_falls_back(self, tmp_path,
                                                    potcar_library):
        """A quality difference must not become an outage."""
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Si")
        source = resolve_source(archive, potcar_dir=potcar_library,
                                charges=CHARGES)
        record = source.discover()[0]

        with pytest.warns(RuntimeWarning, match="No usable 'Si' POTCAR"):
            potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "gaussian"

    def test_a_truncated_table_warns_and_falls_back(self, tmp_path):
        """A partial `local part` parses fine but cannot be splined."""
        library = tmp_path / "potcars"
        (library / "Si").mkdir(parents=True)
        (library / "Si" / "POTCAR").write_text(TRUNCATED_POTCAR)
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",))
        source = resolve_source(archive, potcar_dir=str(library),
                                charges=CHARGES)
        record = source.discover()[0]

        with pytest.warns(RuntimeWarning, match="no complete local-potential"):
            potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "gaussian"

    def test_a_stripped_calculation_is_rescued_by_the_library(
            self, tmp_path, potcar_library):
        """POTCARs are routinely removed from archived runs for licensing."""
        runs = os.path.dirname(write_calculation(
            tmp_path / "runs" / "struct_000", element="Au"))
        source = resolve_source(runs, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "potcar"
        assert potential.metadata["derived_from"] == "POSCAR + POTCAR library"

    def test_a_run_with_its_own_potcar_ignores_the_library(self, tmp_path,
                                                           potcar_library):
        runs = os.path.dirname(write_calculation(
            tmp_path / "runs" / "struct_000", element="Au",
            potcar=REFERENCE_POTCAR))
        source = resolve_source(runs, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert "potcar_dir" not in potential.metadata

    def test_the_cache_records_which_construction_was_used(self, tmp_path,
                                                           potcar_library):
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")
        lines = []

        build_field_cache(archive, tmp_path / "out", resolution=8,
                          potcar_dir=potcar_library, log=lines.append)

        assert any("V_ext tabulated from" in line for line in lines)


# ---------------------------------------------------------------------- #
# Enumeration across sources
# ---------------------------------------------------------------------- #
class TestDiscoverRecords:
    def test_pools_every_source(self, calculations, bulk):
        sources = [resolve_source(calculations), resolve_source(bulk,
                                                                charges=CHARGES)]

        records = discover_records(sources, required=("CHGCAR",))

        assert len(records) == 5

    def test_colliding_identifiers_are_disambiguated(self, tmp_path):
        """Two archives can easily both contain a `struct_000`."""
        first = write_calculation(tmp_path / "runs_a" / "struct_000")
        second = write_calculation(tmp_path / "runs_b" / "struct_000")
        sources = [resolve_source(os.path.dirname(first)),
                   resolve_source(os.path.dirname(second))]

        records = discover_records(sources, required=("CHGCAR",))
        identifiers = [record.identifier for record in records]

        assert len(identifiers) == 2
        assert len(set(identifiers)) == 2, "a collision would drop a material"
        assert "runs_b:struct_000" in identifiers

    def test_required_fields_filter_per_material(self, tmp_path):
        write_calculation(tmp_path / "runs" / "struct_000", tau=True)
        write_calculation(tmp_path / "runs" / "struct_001", tau=False)
        sources = [resolve_source(tmp_path / "runs")]

        assert len(discover_records(sources, required=("CHGCAR",))) == 2
        assert len(discover_records(sources, required=("CHGCAR", "TAUCAR"))) == 1


# ---------------------------------------------------------------------- #
# The unified dataset
# ---------------------------------------------------------------------- #
class TestMixedFieldDataset:
    def test_serves_materials_from_every_path(self, calculations, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     resolution=8, charges=CHARGES)

        assert len(data) == 5
        assert data.contributions() == {calculations: 3, bulk: 2}
        assert data[0]["input"].shape == (1, 8, 8, 8)

    def test_a_single_path_needs_no_list(self, calculations):
        data = MixedFieldDataset(calculations, task="ext2chg", resolution=8)

        assert len(data) == 3

    def test_available_tasks_is_the_union_over_sources(self, calculations, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     resolution=8, charges=CHARGES)

        assert sorted(data.available_tasks()) == ["chg2tau", "ext2chg"]

    def test_chg2tau_keeps_only_the_materials_that_have_tau(self, calculations,
                                                            bulk):
        """The bulk archive has no tau, so it contributes nothing here."""
        data = MixedFieldDataset([calculations, bulk], task="chg2tau",
                                 resolution=8, charges=CHARGES)

        assert len(data) == 3
        assert data.contributions()[bulk] == 0

    def test_a_task_nothing_supports_is_refused_up_front(self, bulk):
        with pytest.raises(ValueError, match="Available tasks here"):
            MixedFieldDataset(bulk, task="chg2tau", resolution=8,
                              charges=CHARGES)

    def test_mixing_potential_conventions_warns(self, tmp_path, bulk):
        """Two definitions of V_ext under one name is never a good accident."""
        runs = write_calculation(tmp_path / "runs" / "struct_000",
                                 potcar=REFERENCE_POTCAR
                                 if os.path.exists(REFERENCE_POTCAR) else None)
        if not os.path.exists(os.path.join(runs, "POTCAR")):
            pytest.skip("reference POTCAR not available")

        with pytest.warns(UserWarning, match="define the external potential"):
            MixedFieldDataset([os.path.dirname(runs), bulk], task="ext2chg",
                              resolution=8, charges=CHARGES)

    def test_a_potcar_library_makes_the_mixture_one_quantity(
            self, tmp_path, potcar_library):
        """
        With the library both sources build the tabulated potential.

        That is the whole point of `potcar_dir`: it does not merely improve the
        bulk archive's potential, it makes it the *same physical quantity* the
        calculations use, at which point the mixture is no longer a mixture.
        """
        runs = os.path.dirname(write_calculation(
            tmp_path / "runs" / "struct_000", element="Au",
            potcar=REFERENCE_POTCAR))
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Au")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data = MixedFieldDataset([runs, archive], task="ext2chg",
                                     resolution=8, potcar_dir=potcar_library)

        assert len(data) == 2
        assert {source.potential_model() for source in data.sources} == {
            "tabulated"}

    def test_one_convention_does_not_warn(self, calculations):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MixedFieldDataset(calculations, task="ext2chg", resolution=8)

    def test_chg2tau_does_not_warn_about_a_potential_it_never_reads(
            self, calculations, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MixedFieldDataset([calculations, bulk], task="chg2tau",
                              resolution=8, charges=CHARGES)

    def test_the_split_mixes_the_archives(self, calculations, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     resolution=8, charges=CHARGES)

        train, held_out = data.split(0.6, seed=0)

        assert len(train) + len(held_out) == len(data)
        assert train.resolution == 8
        assert train[0]["input"].shape == (1, 8, 8, 8)

    def test_shapes_come_from_headers_not_from_a_full_load(self, calculations,
                                                           bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     resolution=8, charges=CHARGES)

        assert data.shapes() == [(8, 8, 8)] * 5

    def test_native_resolution_keeps_each_grid(self, calculations):
        data = MixedFieldDataset(calculations, task="ext2chg")

        assert data.shapes() == [(12, 12, 12)] * 3

    def test_one_format_per_path_may_be_given(self, calculations, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     format=["vasp", "bulk"], resolution=8,
                                     charges=CHARGES)

        assert len(data) == 5

    def test_a_format_list_of_the_wrong_length_is_refused(self, calculations,
                                                          bulk):
        with pytest.raises(ValueError, match="one per path"):
            MixedFieldDataset([calculations, bulk], format=["vasp"],
                              resolution=8)


# ---------------------------------------------------------------------- #
# The cache
# ---------------------------------------------------------------------- #
def cached_materials(cache):
    """
    The material directories in a cache, ignoring what sits beside them.

    A cache also holds bookkeeping files -- the PAW reference table, the build
    summary -- so counting directory entries would count those too.
    """
    return [entry for entry in os.listdir(cache)
            if os.path.isdir(os.path.join(cache, entry))]


class TestBuildFieldCache:
    def test_writes_one_layout_from_a_mixture(self, calculations, bulk,
                                              tmp_path):
        cache = build_field_cache([calculations, bulk], tmp_path / "out",
                                  resolution=8, charges=CHARGES)

        assert sorted(cached_materials(cache)) == [
            "mp-1", "mp-2", "struct_000", "struct_001", "struct_002"]

    def test_writes_every_field_a_source_offers(self, calculations, bulk,
                                                tmp_path):
        """Not only the current task's: a rebuild for chg2tau is then free."""
        cache = build_field_cache([calculations, bulk], tmp_path / "out",
                                  resolution=8, charges=CHARGES)

        assert sorted(os.listdir(os.path.join(cache, "struct_000"))) == [
            "CHGCAR", "EXTCAR", "TAUCAR"]
        assert sorted(os.listdir(os.path.join(cache, "mp-1"))) == [
            "CHGCAR", "EXTCAR"]

    def test_a_run_without_tau_caches_without_complaint(self, tmp_path):
        write_calculation(tmp_path / "runs" / "struct_000", tau=False)

        cache = build_field_cache(tmp_path / "runs", tmp_path / "out",
                                  resolution=8, charges=CHARGES)

        assert sorted(os.listdir(os.path.join(cache, "struct_000"))) == [
            "CHGCAR", "EXTCAR"]

    def test_the_standard_dataset_reads_the_result(self, calculations, bulk,
                                                   tmp_path):
        from poraque.ml.data import FieldPairDataset

        cache = build_field_cache([calculations, bulk], tmp_path / "out",
                                  resolution=8, charges=CHARGES)
        data = FieldPairDataset(cache, task="ext2chg")

        assert len(data) == 5
        assert data[0]["input"].shape == (1, 8, 8, 8)

    def test_a_prepared_cache_is_itself_a_source(self, prepared, tmp_path):
        """So a cache can be re-cached at a lower resolution."""
        cache = build_field_cache(prepared, tmp_path / "out", resolution=6)

        assert FieldGrid.from_file(
            os.path.join(cache, "a", "CHGCAR")).shape == (6, 6, 6)

    def test_rebuilding_reuses_what_is_there(self, calculations, tmp_path):
        cache = build_field_cache(calculations, tmp_path / "out",
                                  resolution=8, charges=CHARGES)
        stamp = os.path.getmtime(os.path.join(cache, "struct_000", "CHGCAR"))

        build_field_cache(calculations, cache, resolution=8, charges=CHARGES)

        assert os.path.getmtime(
            os.path.join(cache, "struct_000", "CHGCAR")) == stamp

    def test_limit_takes_the_smallest_sources(self, calculations, tmp_path):
        cache = build_field_cache(calculations, tmp_path / "out", resolution=8,
                                  charges=CHARGES, limit=2)

        assert len(cached_materials(cache)) == 2

    def test_the_paw_reference_survives_an_empty_source(self, bulk, tmp_path):
        """These fixtures carry no augmentation records; that is not an error."""
        sources = [resolve_source(bulk, charges=CHARGES)]
        cache = build_field_cache(bulk, tmp_path / "out", resolution=8,
                                  charges=CHARGES)

        reference = build_paw_reference(
            discover_records(sources, required=("CHGCAR",)), cache)

        assert reference == {}
        assert load_paw_reference(cache) == {}
