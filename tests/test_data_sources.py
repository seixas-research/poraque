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

import glob
import gzip
import os
import warnings

import numpy as np
import pytest

from poraque.data import (
    DATA_FORMATS,
    CalculationSource,
    MixedFieldDataset,
    build_field_cache,
    discover_records,
    resolve_source,
)
from poraque.data.cache import build_paw_reference, load_paw_reference
from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid
from poraque.fields import KineticEnergyDensity
from poraque.fields.vasp.poscar import Poscar

CHARGES = {"Si": 4.0, "Pt": 11.0}

#: What a run that actually wrote a TAUCAR asked for. `LTAU` evaluates tau and
#: `LCHARG` writes it; a fixture that omits either is not a fixture of a tau
#: run, whatever file it leaves behind.
TAU_INCAR = "LTAU = .TRUE.\nLCHARG = .TRUE.\n"
OUTCAR_FIRST_LINE = (" vasp.6.6.1 18Jan21 (build Jul 30 2026 13:44:49) complex\n")


def plausible_tau(density, grid):
    r"""
    A kinetic energy density that is physically consistent with ``density``.

    :math:`\tau_{\rm TF} + \tau_{\rm vW}`, in Poraquê's own units. Two
    properties matter and neither is decoration: it is :math:`\ge \tau_{\rm
    vW}` everywhere, because both terms are non-negative, and its integral is a
    small multiple of the Thomas-Fermi one. So it is a tau a real calculation
    could have produced, rather than an array of the right shape.

    The literal these fixtures used before -- ``C_TF * rho**(5/3) * 51.42`` --
    was itself about 6.7x the Thomas-Fermi estimate.
    """
    from poraque.fields.density import thomas_fermi_tau, von_weizsacker_tau

    return thomas_fermi_tau(density) + von_weizsacker_tau(density, grid)

#: The reference POTCAR shipped with the repository's platinum dataset. Tests that
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


#: A real VASP library on this machine, if the runner points at one.
#:
#: Pseudopotentials are licensed and this repository ships none, so the
#: POTCAR-dependent tests skip in a bare checkout — which is right, and which
#: also means the *positive* half of every "did the library serve this
#: element?" question goes unexercised wherever they skip. `PORAQUE_TEST_POTCAR_DIR`
#: is the opt-in, in the spirit `PORAQUE_TEST_PYSR` used to set: a path stated by
#: the person running the suite rather than a home directory committed to git.
LIBRARY_POTCAR_DIR = os.environ.get("PORAQUE_TEST_POTCAR_DIR") or None


def _library_element(element):
    """That element's ``POTCAR`` text from ``PORAQUE_TEST_POTCAR_DIR``, or None."""
    if not LIBRARY_POTCAR_DIR:
        return None
    for candidate in (os.path.join(LIBRARY_POTCAR_DIR, element, "POTCAR"),
                      os.path.join(LIBRARY_POTCAR_DIR, f"POTCAR.{element}")):
        if os.path.exists(candidate):
            with open(candidate, "r", errors="replace") as handle:
                return handle.read()
    return None


def _pt_potcar_text():
    """A Pt ``POTCAR``: the shipped reference, or the opted-in library."""
    if os.path.exists(REFERENCE_POTCAR):
        with open(REFERENCE_POTCAR, "r", errors="replace") as source:
            return source.read()
    return _library_element("Pt")


@pytest.fixture
def potcar_library(tmp_path):
    """A one-element POTCAR library laid out the way VASP ships one."""
    text = _pt_potcar_text()
    if text is None:
        pytest.skip("no Pt POTCAR: ship one, or set PORAQUE_TEST_POTCAR_DIR")
    root = tmp_path / "potcars"
    (root / "Pt").mkdir(parents=True)
    (root / "Pt" / "POTCAR").write_text(text)
    return str(root)


@pytest.fixture
def pt_potcar(tmp_path):
    """The same POTCAR as a loose file, for ``write_calculation(potcar=...)``."""
    text = _pt_potcar_text()
    if text is None:
        pytest.skip("no Pt POTCAR: ship one, or set PORAQUE_TEST_POTCAR_DIR")
    path = tmp_path / "REFERENCE_POTCAR"
    path.write_text(text)
    return str(path)


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
        if tau:
            handle.write(TAU_INCAR)
    density.write(os.path.join(directory, "CHGCAR"))
    if tau:
        # An OUTCAR, for its first line only: that is where the code version
        # is, and `poraque.data.provenance` reads nothing else from it.
        with open(os.path.join(directory, "OUTCAR"), "w") as handle:
            handle.write(OUTCAR_FIRST_LINE)
        KineticEnergyDensity(plausible_tau(density.data, grid), grid,
                             structure).write(
            os.path.join(directory, "TAUCAR"))
    return directory


def write_bulk(directory, identifiers=("mp-1", "mp-2"), shape=(12, 12, 12),
               compress=True, element="Si"):
    """
    Densities with no inputs beside them: one directory each, gzipped.

    What a published archive ships, and what a ``poraque-mp --no-vasp-inputs``
    download looks like. The directory layout is the same as a run tree's --
    that is the point of there being one schema -- and the *content* is what
    makes it different: no ``POSCAR``, so the structure comes from the
    density's own header, and no ``POTCAR``, so the valence charges are
    inferred unless a library is configured.
    """
    directory = str(directory)
    for index, identifier in enumerate(identifiers):
        child = os.path.join(directory, identifier)
        os.makedirs(child, exist_ok=True)
        _, _, _, density = _material(np.eye(3) * (5.0 + index), shape,
                                     seed=index + 10, element=element)
        path = os.path.join(child, "CHGCAR")
        density.write(path)
        if compress:
            with open(path, "rb") as source, \
                    gzip.open(path + ".gz", "wb") as sink:
                sink.write(source.read())
            os.remove(path)
    return directory


def write_download(directory, identifiers=("mp-1", "mp-2"), shape=(12, 12, 12),
                   compress=True, element="Si"):
    """
    A download in the current layout: one directory per material, named by it.

    Deliberately not built from :func:`write_bulk` -- the two layouts are what
    is under test, and expressing one in terms of the other would make them
    agree by construction.
    """
    directory = str(directory)
    for index, identifier in enumerate(identifiers):
        child = os.path.join(directory, identifier)
        os.makedirs(child, exist_ok=True)
        _, _, _, density = _material(np.eye(3) * (5.0 + index), shape,
                                     seed=index + 10, element=element)
        path = os.path.join(child, "CHGCAR")
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
            KineticEnergyDensity(plausible_tau(density.data, grid), grid,
                                 structure).write(
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
def download(tmp_path):
    return write_download(tmp_path / "MP")


@pytest.fixture
def prepared(tmp_path):
    return write_prepared(tmp_path / "cache")


# ---------------------------------------------------------------------- #
# Detection
# ---------------------------------------------------------------------- #
class TestDetection:
    r"""
    There is one source class, and one question: does this directory hold
    materials?

    A material is **a directory with a density in it**. Until 2026-08-31 three
    classes competed to claim a directory — ``CalculationSource`` on a
    ``POSCAR``, ``BulkDensitySource`` on a bare density, ``PreparedFieldsSource``
    on cached fields — and detection order decided which won. All that is gone:
    the density identifies the material, and what else is in its directory
    changes how the *fields* are built, not who builds them.
    """

    def test_every_dataset_is_read_by_the_one_source(self, calculations, bulk,
                                                     prepared):
        for root in (calculations, bulk, prepared):
            assert isinstance(resolve_source(root), CalculationSource)

    def test_a_single_calculation_directory_is_one_material(self, calculations):
        one = os.path.join(calculations, "struct_000")

        source = resolve_source(one)

        assert isinstance(source, CalculationSource)
        assert [record.identifier for record in source.discover()] == ["struct_000"]

    def test_a_directory_with_inputs_and_no_density_holds_no_material(self,
                                                                      tmp_path):
        """A calculation that has not run. The density is the qualification."""
        run = tmp_path / "runs" / "structure_000"
        run.mkdir(parents=True)
        (run / "POSCAR").write_text("Si\n1.0\n")
        (run / "INCAR").write_text("ENCUT = 400\n")

        with pytest.raises(ValueError, match="No materials under"):
            resolve_source(str(tmp_path / "runs"))

    def test_metadata_files_are_not_mistaken_for_densities(self, tmp_path):
        """`chgcar_estimate.csv` sits in every real download."""
        root = write_bulk(tmp_path / "MP", identifiers=("mp-1",))
        with open(os.path.join(root, "chgcar_estimate.csv"), "w") as handle:
            handle.write("material_id,size_mb\nmp-1,1.0\n")

        records = resolve_source(root).discover()

        assert [record.identifier for record in records] == ["mp-1"]

    def test_an_empty_directory_says_what_was_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(ValueError, match="one subdirectory per material"):
            resolve_source(tmp_path / "empty")

    def test_a_missing_directory_is_not_an_empty_one(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_source(tmp_path / "nope")

    def test_an_unknown_format_lists_the_two_that_exist(self, bulk):
        with pytest.raises(ValueError, match="Unknown data format"):
            resolve_source(bulk, format="hdf5")

    @pytest.mark.parametrize("name", ("bulk", "prepared", "mp", "cache"))
    def test_the_retired_layout_names_are_not_accepted(self, bulk, name):
        """
        They named *layouts*, and there are none to choose between. Silently
        accepting one would suggest it still selected something.
        """
        with pytest.raises(ValueError, match="Unknown data format"):
            resolve_source(bulk, format=name)

    def test_there_are_exactly_two_formats(self):
        assert DATA_FORMATS == ("auto", "vasp")


class TestADownloadIsOneDirectoryPerMaterial:
    r"""
    ``poraque-mp`` writes ``data/MP/mp-124/CHGCAR.gz``: the id names the
    directory, not the file.

    The shape is the same as a VASP run and the same as a prepared cache, which
    is the point — and also the hazard, because those three are then
    indistinguishable as directory *trees* and only their contents say which is
    which. Getting it wrong is not cosmetic: read as a prepared cache, a
    download would offer ``CHGCAR`` and no ``EXTCAR``, and every ``ext2chg``
    run on it would fail for want of an input field that was never missing —
    it is *computed*, and only :class:`BulkDensitySource` knows to compute it.
    """

    def test_it_is_read_like_any_other_dataset(self, download):
        assert isinstance(resolve_source(download), CalculationSource)

    def test_the_directory_names_the_material(self, download):
        records = resolve_source(download).discover()
        assert [record.identifier for record in records] == ["mp-1", "mp-2"]

    def test_each_record_points_into_its_own_directory(self, download):
        for record in resolve_source(download).discover():
            assert os.path.basename(record.directory) == record.identifier
            assert record.files["CHGCAR"].startswith(record.directory)

    def test_the_potential_is_still_computed_not_expected_on_disk(self, download):
        """The failure a prepared-cache misdetection would produce."""
        source = resolve_source(download)
        assert "EXTCAR" in source.provides(source.discover()[0])

    def test_the_deck_changes_nothing_about_how_it_is_read(self, tmp_path):
        """
        With or without the reconstructed inputs, the same materials come back
        and the same fields are offered. The deck is for VASP.
        """
        bare = write_download(tmp_path / "bare", identifiers=("mp-1",))
        source = resolve_source(bare)
        record = source.discover()[0]

        assert record.identifier == "mp-1"
        assert set(record.files) == {"CHGCAR"}
        assert "EXTCAR" in source.provides(record)

    def test_loose_files_beside_directories_are_all_found(self, tmp_path):
        """
        An archive assembled by hand is not obliged to pick one arrangement,
        and a source that ignored the other would drop materials silently.
        """
        root = str(tmp_path / "mixed")
        write_download(root, identifiers=("mp-1",))
        write_bulk(root, identifiers=("mp-2",))

        records = resolve_source(root).discover()

        assert sorted(r.identifier for r in records) == ["mp-1", "mp-2"]

    def test_a_directory_of_two_densities_is_not_one_material(self, tmp_path):
        """
        Exactly one density is what makes a directory a material's. Two is
        something else, and picking one of them to train on is not a decision
        to make silently.
        """
        root = tmp_path / "odd"
        good = root / "mp-1"
        good.mkdir(parents=True)
        _, _, _, density = _material(np.eye(3) * 5.0, (8, 8, 8))
        density.write(str(good / "CHGCAR"))

        ambiguous = root / "two"
        ambiguous.mkdir()
        density.write(str(ambiguous / "CHGCAR"))
        density.write(str(ambiguous / "CHGCAR_other"))

        found = [r.identifier for r in resolve_source(root).discover()]
        assert found == ["mp-1"]

    def test_an_hdf5_download_is_read_the_same_way(self, tmp_path):
        pytest.importorskip("h5py")
        from poraque.fields.hdf5 import write_fields

        root = tmp_path / "MPh5"
        for index, identifier in enumerate(("mp-1", "mp-2")):
            child = root / identifier
            child.mkdir(parents=True)
            _, _, _, density = _material(np.eye(3) * (5.0 + index),
                                         (8, 8, 8), seed=index)
            write_fields(str(child / "fields.h5"), {"CHGCAR": density})

        records = resolve_source(root).discover()
        assert [r.identifier for r in records] == ["mp-1", "mp-2"]
        assert all(r.files["CHGCAR"].endswith("fields.h5::CHGCAR")
                   for r in records)


class TestOneSchemaForEveryOrigin:
    r"""
    ``data.data_paths`` names directories, and every entry has the same shape:
    subdirectories, one per material, each holding that material's volumetric
    files. A local run tree, a ``poraque-mp`` download and a cache are the same
    thing to a config.

    What differs is the *content* of a material's directory, and content is
    read rather than declared. These tests pin the three properties that makes
    load-bearing: the same directory yields the same materials whatever
    produced it, ``TAUCAR`` is optional per material, and an external potential
    can be built in every case.
    """

    def test_the_three_layouts_agree_on_what_a_material_is(
            self, calculations, download, prepared):
        """One subdirectory per material, in all three."""
        for root, expected in ((calculations, 3), (download, 2), (prepared, 2)):
            source = resolve_source(root)
            records = source.discover()
            assert len(records) == expected
            assert all(os.path.isdir(record.directory) for record in records)
            assert all(os.path.basename(record.directory) == record.identifier
                       for record in records)

    def test_every_origin_offers_an_external_potential(
            self, calculations, download, prepared):
        """
        Computed from the inputs, computed from the density's own header, or
        read from a cached EXTCAR -- three routes to one field, and a config
        chooses none of them.
        """
        for root in (calculations, download, prepared):
            source = resolve_source(root)
            assert "EXTCAR" in source.provides(source.discover()[0])

    def test_tau_is_optional_material_by_material(self, tmp_path):
        """
        Not directory by directory. One calculation in a tree may have written
        a TAUCAR while its neighbour did not, and the one that did should still
        reach `chg2tau`.
        """
        root = tmp_path / "runs"
        write_calculation(root / "structure_000", seed=0)
        write_calculation(root / "structure_001", seed=1, tau=False)

        source = resolve_source(str(root))
        records = {record.identifier: set(source.provides(record))
                   for record in source.discover()}

        assert records["structure_000"] >= {"EXTCAR", "CHGCAR", "TAUCAR"}
        assert "TAUCAR" not in records["structure_001"]

    def test_a_pooled_set_keeps_everything_for_ext2chg(self, tmp_path):
        """The point of the unified schema: a download with no tau still
        trains the task it can."""
        runs = tmp_path / "runs"
        write_calculation(runs / "structure_000", seed=0)
        write_calculation(runs / "structure_001", seed=1, tau=False)
        mp = write_download(tmp_path / "MP", identifiers=("mp-1", "mp-2"))

        sources = [resolve_source(str(runs)), resolve_source(mp)]

        assert len(discover_records(sources, required=("CHGCAR",))) == 4
        assert len(discover_records(sources, required=("EXTCAR", "CHGCAR"))) == 4
        for_tau = discover_records(sources, required=("CHGCAR", "TAUCAR"))
        assert [record.identifier for record in for_tau] == ["structure_000"]

    def test_a_single_run_directly_under_a_path_is_one_material(self,
                                                                calculations):
        """So a lone calculation needs no wrapper directory."""
        one = os.path.join(calculations, "struct_000")
        assert [r.identifier for r in resolve_source(one).discover()] \
            == ["struct_000"]


class TestTheValenceChargesDoNotDependOnHowAPathIsRead:
    r"""
    A directory of bare densities infers :math:`Z^{
m val}` from them; the
    same directory with structure files beside the densities used to raise
    ``No valence charge for ['Si']``.

    Nothing about the physics differs between those two — the densities are
    identical and the inference reads only them — so the answer should not
    have. The fallback now lives on :class:`MaterialSource` and every source
    reaches it, which is what makes "the external potential still works" true
    of the unified schema rather than of one layout in it.
    """

    def _stripped(self, tmp_path):
        """
        Calculations with no ``POTCAR`` -- what an archived run ships, since
        pseudopotentials are routinely stripped for licensing. ``TAUCAR`` is
        left out so the directories reduce to densities plus inputs, which is
        the pair the argument here is about.
        """
        root = tmp_path / "runs"
        for index in range(2):
            write_calculation(root / f"structure_{index:03d}", seed=index,
                              tau=False)
        return str(root)

    def test_a_stripped_calculation_still_builds_a_potential(self, tmp_path):
        source = resolve_source(self._stripped(tmp_path))
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert np.isfinite(np.asarray(potential.data)).all()
        assert np.ptp(np.asarray(potential.data)) > 0

    def test_it_agrees_with_reading_the_same_densities_as_an_archive(
            self, tmp_path):
        """
        The counterfactual, made concrete: delete the structure files and the
        same directories are a bulk archive. The potential must not move.
        """
        runs = self._stripped(tmp_path)
        source = resolve_source(runs)
        record = source.discover()[0]
        from_calculation = np.asarray(
            source.read(record, "EXTCAR", source.grid(record)).data)

        for entry in sorted(os.listdir(runs)):
            for name in ("POSCAR", "CONTCAR", "INCAR"):
                path = os.path.join(runs, entry, name)
                if os.path.exists(path):
                    os.remove(path)

        archive = resolve_source(runs)
        record = archive.discover()[0]
        from_archive = np.asarray(
            archive.read(record, "EXTCAR", archive.grid(record)).data)

        assert np.allclose(from_calculation, from_archive, rtol=1e-10)

    def test_an_explicit_charge_still_wins(self, tmp_path):
        source = resolve_source(self._stripped(tmp_path), charges={"Si": 4.0})
        assert source.charges == {"Si": 4.0}

    def test_a_run_with_its_own_potcar_infers_nothing(self, tmp_path,
                                                      potcar_library):
        """
        The tabulated route reads ZVAL from the POTCAR and needs no inference,
        and inference parses densities -- so it must not run where its answer
        would be thrown away.
        """
        run = tmp_path / "runs" / "structure_000"
        write_calculation(run, tau=False, element="Pt",
                          potcar=os.path.join(potcar_library, "Pt", "POTCAR"))
        source = resolve_source(str(tmp_path / "runs"))

        assert source._modelled_charges(source.discover()[0]) is None


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

    def test_asking_for_a_tau_that_is_not_there_explains_why_not(self, bulk):
        source = resolve_source(bulk, charges=CHARGES)
        record = source.discover()[0]

        with pytest.raises(FileNotFoundError, match="has no TAUCAR"):
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

    def test_a_density_only_potential_comes_from_the_density_header(self,
                                                                    bulk):
        """No POSCAR to read, so the structure is the one the CHGCAR carries."""
        source = resolve_source(bulk, charges=CHARGES)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.data.mean() == pytest.approx(0.0, abs=1e-8)
        assert np.ptp(np.asarray(potential.data)) > 0

    def test_charges_are_inferred_when_not_supplied(self, bulk):
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
                             element="Pt")
        source = resolve_source(archive, charges=CHARGES)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "gaussian"
        assert source.potential_model() == "gaussian"

    def test_with_a_library_the_potential_is_tabulated(self, tmp_path,
                                                       potcar_library):
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        source = resolve_source(archive, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "potcar"
        # "density geometry", not "CHGCAR header": since 2026-08-28 the
        # potential is built at the geometry `CalculationSource.geometry`
        # resolves, which is the density's own header when it has one and the
        # structure file otherwise -- one string for both routes.
        assert potential.metadata["derived_from"] == "density geometry + POTCAR library"
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
                             element="Pt")
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
        """
        Z_val is stated by the POTCAR; inferring it would be a detour.

        Read off the fixture's own POTCAR rather than written as a literal:
        Pt ships as `Pt` (ZVAL 10) and `Pt_pv` (ZVAL 11) among others, so a
        literal asserts which pseudopotential the runner happens to have, not
        that the library was consulted at all. The densities in `write_bulk`
        are random, so an inferred charge would not be either number.
        """
        from poraque.fields.vasp.potcar import Potcar

        expected = float(Potcar.from_library(potcar_library, ["Pt"],
                                             parse_tables=False)[0].zval)
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")

        source = resolve_source(archive, potcar_dir=potcar_library)

        assert source.charges["Pt"] == expected

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
            tmp_path / "runs" / "struct_000", element="Pt"))
        source = resolve_source(runs, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert potential.metadata["model"] == "potcar"
        # Was "POSCAR + POTCAR library". A run directory and a bare archive now
        # give the *same* string, because both build the potential at the
        # geometry the density carries -- which is the point of that change,
        # not an accident of wording.
        assert potential.metadata["derived_from"] == "density geometry + POTCAR library"

    def test_a_run_with_its_own_potcar_ignores_the_library(self, tmp_path,
                                                           potcar_library,
                                                           pt_potcar):
        runs = os.path.dirname(write_calculation(
            tmp_path / "runs" / "struct_000", element="Pt",
            potcar=pt_potcar))
        source = resolve_source(runs, potcar_dir=potcar_library)
        record = source.discover()[0]

        potential = source.read(record, "EXTCAR", source.grid(record))

        assert "potcar_dir" not in potential.metadata

    def test_the_cache_records_which_construction_was_used(self, tmp_path,
                                                           potcar_library):
        """
        The log line this asserted, ``V_ext tabulated from``, was removed in
        26.8.46 and the assertion went stale unnoticed because the fixture
        skips wherever no POTCAR is available. Its replacement says more: the
        construction **that was resolved**, per element, which is the same
        string the cache fingerprint now records.
        """
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        lines = []

        build_field_cache(archive, tmp_path / "out", resolution=8,
                          potcar_dir=potcar_library, log=lines.append)

        assert any("POTCAR library: Pt=library" in line for line in lines)


class TestTheCacheRecordsWhatTheLibraryServedNotWhatWasAsked:
    """
    ``potcar_dir`` in the fingerprint is a request; ``potcar_source`` is an
    answer.

    Found on Santos Dumont: a control run landed on a compute node where the
    library's filesystem was not mounted, warned six times on stderr, trained
    against analytic pseudo-ion potentials, and wrote a
    ``cache_fingerprint.json`` byte-identical to one built from real
    pseudopotentials — in a directory *named* ``res32_potcar``. Every later run
    reused it silently, including on nodes where the library was mounted, and
    the warning appeared only during the build that wrote it. The validation
    error that cache produced had already been quoted as a result.

    The two constructions are different physical quantities, not different
    approximations to one (see :class:`TestPotcarLibrary`), which is what makes
    a fingerprint that cannot tell them apart a correctness problem rather than
    a tidiness one.
    """

    @pytest.fixture
    def unreachable_library(self, tmp_path):
        """A configured library that cannot be read — the /prj case exactly."""
        return str(tmp_path / "not" / "mounted" / "POTCARs")

    def test_the_fingerprint_says_gaussian_when_the_library_was_unreadable(
            self, tmp_path, unreachable_library):
        import json

        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        with pytest.warns(RuntimeWarning, match="No usable 'Pt' POTCAR"):
            build_field_cache(archive, tmp_path / "out", resolution=8,
                              potcar_dir=unreachable_library,
                              charges=CHARGES)

        with open(tmp_path / "out" / "cache_fingerprint.json") as handle:
            recorded = json.load(handle)

        assert recorded["potcar_source"] == {"Pt": "gaussian"}
        # And the path that was asked for is still there beside it, because
        # both facts are wanted -- the gap between them is the whole point.
        assert recorded["potcar_dir"].endswith("POTCARs")

    def test_a_gaussian_cache_is_not_reused_by_a_run_with_the_library(
            self, tmp_path, unreachable_library):
        """
        The consequence that matters. Written by hand rather than by building
        twice, because a second build needs a real POTCAR this checkout does
        not ship — and the mechanism under test is the fingerprint comparison,
        not the pseudopotential parser.
        """
        import json

        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        cache = tmp_path / "out"
        cache.mkdir()
        with pytest.warns(RuntimeWarning, match="No usable 'Pt' POTCAR"):
            build_field_cache(archive, cache, resolution=8,
                              potcar_dir=unreachable_library, charges=CHARGES)

        # Now pretend the library came back: the same request, a different
        # answer. Before `potcar_source` existed the two were indistinguishable
        # and the second run silently reused the first's analytic potentials.
        path = cache / "cache_fingerprint.json"
        recorded = json.loads(path.read_text())
        recorded["potcar_source"] = {"Pt": "library"}
        path.write_text(json.dumps(recorded))

        with pytest.raises(ValueError, match="potcar_source"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                build_field_cache(archive, cache, resolution=8,
                                  potcar_dir=unreachable_library,
                                  charges=CHARGES)

    def test_a_dataset_with_no_library_is_fingerprinted_exactly_as_before(
            self, tmp_path):
        """A key that is `null` whenever it has nothing to say costs nothing
        and invalidates nothing."""
        import json

        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        build_field_cache(archive, tmp_path / "out", resolution=8,
                          charges=CHARGES)

        with open(tmp_path / "out" / "cache_fingerprint.json") as handle:
            assert json.load(handle)["potcar_source"] is None

    def test_a_cache_written_before_the_key_existed_is_adopted_not_refused(
            self, tmp_path):
        """
        An upgrade that invalidated every cache in existence would mean
        rebuilding hundreds of densities to learn nothing: the older build
        never asked the question, so it cannot have disagreed about the answer.
        It is adopted with a warning naming the key, because "unrecorded" and
        "recorded as the same" are different states.
        """
        import json

        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        cache = tmp_path / "out"
        build_field_cache(archive, cache, resolution=8, charges=CHARGES)

        path = cache / "cache_fingerprint.json"
        recorded = json.loads(path.read_text())
        del recorded["potcar_source"]
        path.write_text(json.dumps(recorded))

        with pytest.warns(RuntimeWarning, match="potcar_source"):
            build_field_cache(archive, cache, resolution=8, charges=CHARGES)

        assert "potcar_source" in json.loads(path.read_text())

    def test_a_real_mismatch_still_raises(self, tmp_path):
        """The narrow adoption above must not become "ignore any difference"."""
        import json

        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        cache = tmp_path / "out"
        build_field_cache(archive, cache, resolution=8, charges=CHARGES)

        path = cache / "cache_fingerprint.json"
        recorded = json.loads(path.read_text())
        del recorded["potcar_source"]
        recorded["resolution"] = 16
        path.write_text(json.dumps(recorded))

        with pytest.raises(ValueError, match="resolution"):
            build_field_cache(archive, cache, resolution=8, charges=CHARGES)

    def test_strict_potcar_refuses_the_fallback_and_names_the_element(
            self, tmp_path, unreachable_library):
        """
        The same failure shape as `training.strict_device`, pointing at the
        other resource: a job that holds its allocation and quietly computes
        something other than what was configured.
        """
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")

        with pytest.raises(RuntimeError, match=r"\['Pt'\]"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                build_field_cache(archive, tmp_path / "out", resolution=8,
                                  potcar_dir=unreachable_library,
                                  charges=CHARGES, strict_potcar=True)

    def test_the_potential_model_reports_what_was_resolved(
            self, tmp_path, unreachable_library):
        """
        `potential_model` answered "tabulated" from the *presence* of a
        `potcar_dir` option, which is the same mistake the fingerprint was
        making — and it is what `MixedFieldDataset` consults before deciding
        whether a mixture of constructions needs a warning.
        """
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")
        source = resolve_source(archive, potcar_dir=unreachable_library,
                                charges=CHARGES)

        with pytest.warns(RuntimeWarning, match="No usable 'Pt' POTCAR"):
            assert source.potential_model() == "gaussian"

    def test_coverage_is_none_when_no_library_is_configured(self, tmp_path):
        """Nothing was asked for, so nothing can have failed."""
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")

        assert resolve_source(archive, charges=CHARGES).potcar_coverage() is None


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
            self, tmp_path, potcar_library, pt_potcar):
        """
        With the library both sources build the tabulated potential.

        That is the whole point of `potcar_dir`: it does not merely improve the
        bulk archive's potential, it makes it the *same physical quantity* the
        calculations use, at which point the mixture is no longer a mixture.
        """
        runs = os.path.dirname(write_calculation(
            tmp_path / "runs" / "struct_000", element="Pt",
            potcar=pt_potcar))
        archive = write_bulk(tmp_path / "chgcar", identifiers=("mp-1",),
                             element="Pt")

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

    def test_one_format_covers_every_path(self, calculations, bulk):
        """
        It names the *code*, not a layout, so there is nothing to vary between
        paths: one dataset is one code's files.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MixedFieldDataset([calculations, bulk], task="ext2chg",
                                     format="vasp", resolution=8,
                                     charges=CHARGES)

        assert len(data) == 5

    def test_an_unknown_format_is_refused(self, calculations, bulk):
        with pytest.raises(ValueError, match="Unknown data format"):
            MixedFieldDataset([calculations, bulk], format="bulk",
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


class TestACacheEntryAppearsWholeOrNotAtAll:
    """
    Every field is written beside its final name and moved with ``os.replace``.

    Found on Santos Dumont: two jobs in one allocation started against the same
    cold cache directory and one of them left a **zero-length** ``CHGCAR``
    behind, which the reader then correctly refused. The text format carries no
    length field, so a torn file is not always so obvious — a truncated density
    parses, into a field of the wrong shape — and :func:`_all_present` is a
    plain ``os.path.exists``, so whatever is at the final name is the answer
    forever after. It is the between-jobs form of the race
    :mod:`poraque.ml.distributed`'s barrier already handles inside one job, and
    nothing guarded it.

    What a rename buys is that a reader sees the old state or the new one and
    never a half-written one. What it deliberately does *not* buy is
    exclusion: two builds still do the work twice, because a lock that behaves
    on a shared parallel filesystem is a far larger promise than this function
    should make, and duplicated work costs minutes where a torn field costs the
    dataset.
    """

    def test_no_temporary_survives_a_finished_build(self, bulk, tmp_path):
        cache = build_field_cache(bulk, tmp_path / "out", resolution=8,
                                  charges=CHARGES)

        litter = glob.glob(os.path.join(cache, "*", "*partial*"))
        assert litter == []
        assert os.path.getsize(os.path.join(cache, "mp-1", "CHGCAR")) > 0

    def test_every_field_arrives_by_rename(self, bulk, tmp_path, monkeypatch):
        """
        Asserted at the moment of the move, which is the only moment it can be.

        A finished cache looks identical either way; what distinguishes the two
        implementations is whether the destination was ever open for writing.
        """
        from poraque.data import cache as cache_module

        moves, replace = [], os.replace

        def record(source, destination):
            assert os.path.exists(source), source
            assert not os.path.exists(destination), (
                f"{destination} already existed, so it was written in place")
            moves.append((str(source), str(destination)))
            replace(source, destination)

        monkeypatch.setattr(cache_module.os, "replace", record)
        cache = build_field_cache(bulk, tmp_path / "out", resolution=8,
                                  charges=CHARGES)

        assert moves, "nothing was moved into place"
        for _, destination in moves:
            assert os.path.exists(destination)
        assert {os.path.basename(destination) for _, destination in moves} == \
            {"EXTCAR", "CHGCAR"}
        assert len(cached_materials(cache)) == 2

    def test_a_writer_that_raises_leaves_no_file_at_the_final_name(
            self, bulk, tmp_path, monkeypatch):
        """
        The counterfactual, and the regression stated exactly.

        The failing writer here still *creates* its file before dying, which is
        what the interrupted job did. In place, that file is the cache entry
        from then on; beside it, it is litter — and it has to be cleaned up
        too, or the next run inherits a directory of temporaries named after
        processes that no longer exist.
        """
        from poraque.fields.base import ScalarField

        def torn(self, path=None, *args, **kwargs):
            with open(path, "w") as handle:
                handle.write("half a header\n")
            raise RuntimeError("the job was cancelled mid-write")

        monkeypatch.setattr(ScalarField, "write", torn)
        with pytest.raises(RuntimeError, match="cancelled mid-write"):
            build_field_cache(bulk, tmp_path / "out", resolution=8,
                              charges=CHARGES)

        out = str(tmp_path / "out")
        assert glob.glob(os.path.join(out, "*", "*partial*")) == []
        assert glob.glob(os.path.join(out, "*", "CHGCAR")) == []
        assert glob.glob(os.path.join(out, "*", "EXTCAR")) == []

    def test_the_temporary_keeps_the_suffix_a_store_is_recognised_by(
            self, bulk, tmp_path, monkeypatch):
        """
        ``fields.partial-1234.h5``, never ``fields.h5.partial-1234``.

        The suffix is load-bearing rather than cosmetic:
        :meth:`~poraque.fields.base.ScalarField.write` chooses between the text
        writer and the HDF5 one by looking at it, so a temporary that loses it
        would write a `CHGCAR` into a file the cache then renames to
        ``fields.h5``.
        """
        from poraque.data import cache as cache_module

        seen, replace = [], os.replace

        def record(source, destination):
            seen.append(os.path.basename(str(source)))
            replace(source, destination)

        monkeypatch.setattr(cache_module.os, "replace", record)
        build_field_cache(bulk, tmp_path / "out", resolution=8,
                          charges=CHARGES, storage="hdf5")

        assert seen and all(name.endswith(".h5") for name in seen), seen
