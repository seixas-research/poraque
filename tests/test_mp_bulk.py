# -*- coding: utf-8 -*-
# file: test_mp_bulk.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Sizing and fetching the Materials Project in bulk, with the API mocked out.

**Nothing here touches the network.** Every test drives
:class:`~poraque.data.materials_project.MPDataFetcher` against a fake rester
and fake S3 sizes, which is the only way to test a bulk downloader at all: the
behaviours worth pinning are precisely the ones that only appear at scale —
what a resumed run skips, what a rate limit costs, what an estimate says when
it measured twenty objects and is reporting about forty thousand.

Three claims:

**An estimate says which method produced it.** Measuring every object exactly
is right for a chemical space and impractical for the whole database, so
``--sample`` measures a subset and extrapolates. A number without its method is
a number nobody can plan against, so the report carries it and so does
:attr:`~poraque.data.materials_project.Estimate.method`.

**A resumed run does not re-fetch.** The manifest is the resume point, and it
is rewritten after every single file so an interrupted run leaves one. An entry
marked ``downloaded`` or ``cached`` is settled; a ``failed`` one is retried,
because that is the only way a run over thousands of objects ever reaches
completeness.

**A transient failure is retried and a permanent one is not.** Backing off from
a rate limit is what lets the run finish; backing off from a missing object
just makes the same error slower.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from poraque.data.materials_project import (
    MANIFEST_JSON,
    Estimate,
    MPDataFetcher,
    poscar_from_pymatgen,
    retrieval_provenance,
    with_retries,
)
from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.hdf5 import field_names
from poraque.fields.vasp.poscar import Poscar


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def document(identifier, formula="SiO2", nsites=6, size=None):
    """A stand-in for a summary document, with the attributes used downstream."""
    return SimpleNamespace(
        material_id=identifier, formula_pretty=formula, chemsys="O-Si",
        nelements=2, nsites=nsites, volume=100.0, density=2.6,
        density_atomic=16.0, energy_per_atom=-6.0,
        formation_energy_per_atom=-3.0, energy_above_hull=0.0, is_stable=True,
        band_gap=5.0, is_metal=False, is_magnetic=False, ordering="NM",
        total_magnetization=0.0, theoretical=False, deprecated=False,
        has_props={"charge_density": True},
        symmetry=SimpleNamespace(symbol="P1", number=1, crystal_system="Cubic"),
        structure=None,
    )


class FakeSite:
    def __init__(self, symbol, coords):
        self.specie = SimpleNamespace(symbol=symbol)
        self.frac_coords = np.asarray(coords, dtype=float)


class FakeStructure:
    """Shaped like a pymatgen ``Structure``, which is what the client returns."""

    def __init__(self, matrix, sites, formula="Si"):
        self.lattice = SimpleNamespace(matrix=np.asarray(matrix, dtype=float))
        self._sites = list(sites)
        self.composition = SimpleNamespace(reduced_formula=formula)

    def __iter__(self):
        return iter(self._sites)

    def __len__(self):
        return len(self._sites)

    def __getitem__(self, index):
        return self._sites[index]


class FakeChgcar:
    """
    Enough of a pymatgen ``Chgcar`` for the download path.

    Deliberately pymatgen-shaped rather than Poraquê-shaped. An earlier version
    of this double carried a :class:`Poscar` and so passed while the real
    conversion raised ``AttributeError: 'Structure' object has no attribute
    'cell'`` on the first real material — a test double that is easier than the
    thing it stands for tests nothing.
    """

    def __init__(self, shape=(8, 8, 8), lattice=4.0):
        self.dim = shape
        self.structure = FakeStructure(
            np.eye(3) * lattice,
            [FakeSite("Si", [0.0, 0.0, 0.0]), FakeSite("Si", [0.5, 0.5, 0.5])])
        volume = float(lattice) ** 3
        self.data = {"total": np.full(shape, 8.0 / volume) * volume}

    def write_file(self, path):
        from poraque.data.materials_project import poscar_from_pymatgen

        structure = poscar_from_pymatgen(self.structure)
        grid = FieldGrid(self.dim, structure.cell)
        ChargeDensity(self.data["total"] / grid.volume, grid,
                      structure).write(path)


def fetcher_with(identifiers, tmp_path, monkeypatch, sizes=None,
                 chgcar=None, **options):
    """A fetcher wired to fakes, with no rester and no S3 behind it."""
    monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
    fetcher = MPDataFetcher(["Si", "O"], outdir=str(tmp_path), **options)
    documents = [document(i) for i in identifiers]
    fetcher._documents = documents

    # Stubbed rather than pre-set, so the sampled path -- which resolves only
    # the materials it picked -- goes through the same call the exact path does.
    keys = {i: f"chgcars/{i}.json.gz" for i in identifiers}
    monkeypatch.setattr(fetcher, "_resolve_keys",
                        lambda refresh=False, identifiers=None: (
                            keys if identifiers is None
                            else {i: keys[i] for i in identifiers}))
    fetcher._rester = SimpleNamespace(
        get_charge_density_from_material_id=lambda _: (chgcar or FakeChgcar()),
        session=SimpleNamespace(close=lambda: None),
    )
    sizes = sizes or {i: 1_000_000 for i in identifiers}
    monkeypatch.setattr(fetcher, "_head_sizes", lambda keys: {
        i: sizes[i] for i in keys})
    return fetcher


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    ids = [f"mp-{n}" for n in range(1, 21)]
    return fetcher_with(ids, tmp_path, monkeypatch=monkeypatch)


# ===================================================================== #
# Selecting the whole database
# ===================================================================== #
class TestSelectingEveryMaterialWithADensity:
    def test_the_query_asks_the_server_not_the_client(self, tmp_path,
                                                      monkeypatch):
        """
        ``has_props`` is what makes "every material with a charge density" a
        question the API answers. Without it the only way to ask would be to
        pull every summary document in the database and discard most of them.
        """
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        seen = {}

        def search(**kwargs):
            seen.update(kwargs)
            return [document("mp-1")]

        fetcher = MPDataFetcher(None, outdir=str(tmp_path))
        fetcher._rester = SimpleNamespace(
            materials=SimpleNamespace(summary=SimpleNamespace(search=search)),
            session=SimpleNamespace(close=lambda: None))
        fetcher.search()

        assert seen["has_props"] == ["charge_density"]
        assert "chemsys" not in seen

    def test_filters_still_narrow_the_whole_database(self, tmp_path,
                                                     monkeypatch):
        """``--all`` is not "everything": the filters are how a subset is taken."""
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        seen = {}
        fetcher = MPDataFetcher(None, outdir=str(tmp_path), num_sites=(1, 8),
                                band_gap=(0.0, 0.0))
        fetcher._rester = SimpleNamespace(
            materials=SimpleNamespace(summary=SimpleNamespace(
                search=lambda **kw: (seen.update(kw), [document("mp-1")])[1])),
            session=SimpleNamespace(close=lambda: None))
        fetcher.search()

        assert seen["num_sites"] == (1, 8)
        assert seen["band_gap"] == (0.0, 0.0)

    def test_a_chemical_space_still_queries_by_chemsys(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        seen = {}
        fetcher = MPDataFetcher(["Ag", "Pt"], outdir=str(tmp_path))
        fetcher._rester = SimpleNamespace(
            materials=SimpleNamespace(summary=SimpleNamespace(
                search=lambda **kw: (seen.update(kw), [document("mp-1")])[1])),
            session=SimpleNamespace(close=lambda: None))
        fetcher.search()

        assert sorted(seen["chemsys"]) == ["Ag", "Ag-Pt", "Pt"]
        assert "has_props" not in seen

    def test_the_label_names_what_was_selected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        assert MPDataFetcher(None, outdir=str(tmp_path)).label == "all elements"
        assert MPDataFetcher(["Pt"], outdir=str(tmp_path)).label == "Pt"


# ===================================================================== #
# Estimating
# ===================================================================== #
class TestTheExactEstimate:
    def test_it_measures_every_object(self, fetcher):
        estimate = fetcher.estimate()
        assert estimate.method == "exact"
        assert estimate.sampled == 20
        assert estimate.n_available == 20
        assert estimate.download_bytes == 20_000_000

    def test_it_transfers_no_payload(self, tmp_path, monkeypatch):
        """HEAD, not GET: the whole point of sizing before downloading."""
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        fetched = []
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f._rester.get_charge_density_from_material_id = (
            lambda i: fetched.append(i))
        f.estimate()
        assert fetched == []

    def test_the_report_names_its_method(self, fetcher):
        assert "every object measured exactly" in str(fetcher.estimate())

    def test_an_advertised_but_absent_object_is_counted_apart(self, tmp_path,
                                                              monkeypatch):
        sizes = {"mp-1": 1_000_000, "mp-2": -1, "mp-3": 2_000_000}
        f = fetcher_with(list(sizes), tmp_path, sizes=sizes,
                         monkeypatch=monkeypatch)
        estimate = f.estimate()
        assert estimate.n_available == 2
        assert estimate.download_bytes == 3_000_000

    def test_storage_projections_follow_the_measured_ratios(self, fetcher):
        estimate = fetcher.estimate()
        assert estimate.gz_disk_bytes < estimate.download_bytes
        assert estimate.unzipped_bytes > estimate.gz_disk_bytes


class TestBatchingIdFilteredQueries:
    """
    The API refuses an id filter beyond a few thousand — *"List of
    material/molecule IDs provided is too long"* — which is exactly what the
    whole-database estimate walked into on its first live run. The suggested
    alternative, pulling every document and filtering locally, is the thing
    :meth:`_resolve_keys` exists to avoid.
    """

    def test_a_long_id_list_is_split(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        f = MPDataFetcher(None, outdir=str(tmp_path))
        batches = []

        def search(**kwargs):
            batches.append(len(kwargs["material_ids"]))
            return []

        f._batched_search(search, "material_ids",
                          [f"mp-{n}" for n in range(2500)])
        assert len(batches) == 3
        assert batches == [1000, 1000, 500]

    def test_a_short_one_is_a_single_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        f = MPDataFetcher(None, outdir=str(tmp_path))
        calls = []
        f._batched_search(lambda **kw: calls.append(kw) or [],
                          "task_ids", ["mp-1", "mp-2"])
        assert len(calls) == 1

    def test_every_result_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        f = MPDataFetcher(None, outdir=str(tmp_path))
        found = f._batched_search(lambda **kw: list(kw["material_ids"]),
                                  "material_ids",
                                  [f"mp-{n}" for n in range(2500)])
        assert len(found) == 2500

    def test_a_batch_is_retried(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MP_API_KEY", "not-a-real-key")
        f = MPDataFetcher(None, outdir=str(tmp_path), retries=3,
                          retry_delay=0.0)
        monkeypatch.setattr("poraque.data.materials_project.time.sleep",
                            lambda *_: None)
        calls = []

        def flaky(**kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise TimeoutError("rate limited")
            return ["ok"]

        assert f._batched_search(flaky, "task_ids", ["mp-1"]) == ["ok"]
        assert len(calls) == 2


class TestTheSampledEstimate:
    def test_it_resolves_only_the_sample(self, tmp_path, monkeypatch):
        """
        Resolving everything and then sampling would make a 20-object estimate
        cost the same queries as an exact one over the whole database — which
        is the cost ``--sample`` exists to avoid.
        """
        ids = [f"mp-{n}" for n in range(1, 51)]
        f = fetcher_with(ids, tmp_path, monkeypatch=monkeypatch)
        asked = []
        keys = {i: f"chgcars/{i}.json.gz" for i in ids}
        monkeypatch.setattr(f, "_resolve_keys",
                            lambda refresh=False, identifiers=None: (
                                asked.append(identifiers),
                                {i: keys[i] for i in (identifiers or ids)})[1])
        f.estimate(sample=7)

        assert asked and asked[0] is not None
        assert len(asked[0]) == 7

    def test_it_measures_only_the_sample(self, fetcher):
        estimate = fetcher.estimate(sample=5)
        assert estimate.method == "sampled"
        assert estimate.sampled == 5

    def test_the_total_is_the_mean_times_the_count(self, tmp_path,
                                                   monkeypatch):
        sizes = {f"mp-{n}": n * 1_000_000 for n in range(1, 21)}
        f = fetcher_with(list(sizes), tmp_path, sizes=sizes,
                         monkeypatch=monkeypatch)
        estimate = f.estimate(sample=8)
        assert estimate.download_bytes == pytest.approx(
            estimate.mean_bytes * 20)

    def test_the_same_seed_gives_the_same_answer(self, tmp_path, monkeypatch):
        sizes = {f"mp-{n}": n * 1_000_000 for n in range(1, 41)}
        first = fetcher_with(list(sizes), tmp_path, sizes=sizes,
                             monkeypatch=monkeypatch).estimate(sample=6, seed=3)
        second = fetcher_with(list(sizes), tmp_path, sizes=sizes,
                              monkeypatch=monkeypatch).estimate(sample=6, seed=3)
        assert first.download_bytes == second.download_bytes

    def test_a_different_seed_may_give_a_different_one(self, tmp_path,
                                                       monkeypatch):
        """
        Charge-density sizes are strongly right-tailed, which is exactly why
        the seed is fixed by default: an estimate that moved every time it was
        asked would be useless for planning.
        """
        sizes = {f"mp-{n}": n * 1_000_000 for n in range(1, 41)}
        totals = {
            fetcher_with(list(sizes), tmp_path, sizes=sizes,
                         monkeypatch=monkeypatch).estimate(sample=5, seed=s
                                                           ).download_bytes
            for s in range(6)}
        assert len(totals) > 1

    def test_the_report_says_it_extrapolated(self, fetcher):
        text = str(fetcher.estimate(sample=5))
        assert "ESTIMATED" in text
        assert "random sample" in text

    def test_asking_for_more_than_there_is_measures_everything(self, fetcher):
        estimate = fetcher.estimate(sample=500)
        assert estimate.method == "exact"

    def test_the_size_caps_are_scaled_to_the_whole_set(self, tmp_path,
                                                       monkeypatch):
        """A cap row must mean the same thing in both modes."""
        sizes = {f"mp-{n}": 1_000_000 for n in range(1, 41)}
        f = fetcher_with(list(sizes), tmp_path, sizes=sizes,
                         monkeypatch=monkeypatch)
        count, total = f.estimate(sample=10).files_under(5)
        assert count == 40
        assert total == pytest.approx(40_000_000)

    def test_a_dry_run_writes_nothing(self, tmp_path, fetcher):
        before = set(os.listdir(tmp_path))
        fetcher.dry_run(sample=5)
        assert set(os.listdir(tmp_path)) == before


class TestTheEstimateObjectOnItsOwn:
    def test_an_empty_estimate_says_so(self):
        assert "no charge densities" in str(
            Estimate(elements=["Pt"], n_materials=0, n_advertised=0,
                     n_available=0))

    def test_the_label_falls_back_to_the_database(self):
        assert "all elements" in Estimate(
            elements=[], n_materials=1, n_advertised=1, n_available=1).label


# ===================================================================== #
# Retrying
# ===================================================================== #
class TestBackingOff:
    def test_a_transient_failure_is_retried(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("rate limited")
            return "ok"

        result, attempts = with_retries(flaky, attempts=5, base_delay=0.0,
                                        log=lambda *_: None,
                                        sleep=lambda *_: None)
        assert result == "ok" and attempts == 3

    def test_a_permanent_failure_is_not(self):
        """Retrying a missing object makes the same error slower, not rarer."""
        calls = []

        def broken():
            calls.append(1)
            raise ValueError("the API returned no charge density")

        with pytest.raises(ValueError):
            with_retries(broken, attempts=5, base_delay=0.0,
                         log=lambda *_: None, sleep=lambda *_: None)
        assert len(calls) == 1

    def test_it_gives_up_eventually(self):
        with pytest.raises(TimeoutError):
            with_retries(lambda: (_ for _ in ()).throw(TimeoutError("nope")),
                         attempts=3, base_delay=0.0, log=lambda *_: None,
                         sleep=lambda *_: None)

    def test_the_delay_doubles_within_its_jitter(self):
        """
        The backoff is ``base * 2**(n-1)`` scaled by a jitter in [0.5, 1.5], so
        consecutive delays are *not* strictly increasing and asserting that they
        are is a flaky test of a property the code does not have. The jitter is
        deliberate: workers throttled together must not all come back together.
        What is guaranteed is the window each delay falls in.
        """
        delays = []

        def flaky():
            if len(delays) < 4:
                raise TimeoutError("rate limited")
            return "ok"

        with_retries(flaky, attempts=6, base_delay=1.0, log=lambda *_: None,
                     sleep=delays.append)

        assert len(delays) == 4
        for index, delay in enumerate(delays):
            base = 1.0 * (2 ** index)
            assert 0.5 * base <= delay <= 1.5 * base

    def test_the_delays_are_not_all_the_same(self):
        """Jitter, not a fixed schedule: that is what spreads the retries."""
        delays = []

        def flaky():
            if len(delays) < 5:
                raise TimeoutError("rate limited")
            return "ok"

        with_retries(flaky, attempts=8, base_delay=0.0 + 1.0,
                     log=lambda *_: None, sleep=delays.append)
        scaled = [d / (2 ** i) for i, d in enumerate(delays)]
        assert len(set(scaled)) > 1

    def test_one_attempt_disables_retrying(self):
        calls = []
        with pytest.raises(TimeoutError):
            with_retries(lambda: (calls.append(1),
                                  (_ for _ in ()).throw(TimeoutError("x")))[1],
                         attempts=1, base_delay=0.0, log=lambda *_: None,
                         sleep=lambda *_: None)
        assert len(calls) == 1


# ===================================================================== #
# The manifest, and resuming from it
# ===================================================================== #
class TestTheManifest:
    def test_it_records_what_a_reader_needs(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        row = f.download()[0]

        assert row["material_id"] == "mp-1"
        assert row["formula_pretty"] == "SiO2"
        assert row["nsites"] == 6
        assert row["grid"] == [8, 8, 8]
        assert row["size_mb"] > 0
        assert row["status"] == "downloaded"

    def test_it_records_where_the_bytes_came_from(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        row = f.download()[0]

        assert row["bucket"] == "materialsproject-parsed"
        assert row["mp_api_version"]
        assert row["retrieved"].startswith("20")

    def test_both_manifests_are_written(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.download()
        assert (f.outdir / MANIFEST_JSON).exists()
        assert (f.outdir / "manifest.csv").exists()

    def test_provenance_is_stamped_per_run(self):
        stamp = retrieval_provenance()
        assert set(stamp) >= {"retrieved", "mp_api_version", "bucket"}

    def test_a_failure_is_recorded_rather_than_hidden(self, tmp_path,
                                                      monkeypatch):
        """A hole in a dataset that nothing records is a hole nobody finds."""
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f._rester.get_charge_density_from_material_id = lambda _: None
        rows = f.download()

        assert rows[0]["status"] == "failed"
        assert "no charge density" in rows[0]["error"]

    def test_one_failure_does_not_abort_the_batch(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2", "mp-3"], tmp_path,
                         monkeypatch=monkeypatch)
        seen = []

        def flaky(identifier):
            seen.append(identifier)
            if identifier == "mp-2":
                raise ValueError("gone")
            return FakeChgcar()

        f._rester.get_charge_density_from_material_id = flaky
        rows = f.download()
        assert len(seen) == 3
        assert sorted(r["status"] for r in rows) == ["downloaded", "downloaded",
                                                     "failed"]


class TestResuming:
    def _manifest(self, fetcher, rows, legacy=False):
        directory = fetcher.legacy_chgcar_dir if legacy else fetcher.outdir
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / MANIFEST_JSON).open("w") as handle:
            json.dump(rows, handle)

    def test_a_settled_entry_is_not_fetched_again(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2", "mp-3"], tmp_path,
                         monkeypatch=monkeypatch)
        self._manifest(f, [
            {"material_id": "mp-1", "status": "downloaded", "size_mb": 1.0},
            {"material_id": "mp-2", "status": "cached", "size_mb": 1.0},
        ])
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == ["mp-3"]

    def test_a_failed_entry_is_tried_again(self, tmp_path, monkeypatch):
        """
        Retrying the failures is the only way a run over thousands of objects
        ever reaches completeness.
        """
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        self._manifest(f, [
            {"material_id": "mp-1", "status": "failed", "error": "timeout"},
            {"material_id": "mp-2", "status": "downloaded", "size_mb": 1.0},
        ])
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == ["mp-1"]

    def test_restarting_ignores_the_manifest(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        self._manifest(f, [
            {"material_id": "mp-1", "status": "downloaded", "size_mb": 1.0},
            {"material_id": "mp-2", "status": "downloaded", "size_mb": 1.0},
        ])
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download(skip_existing=False)

        assert sorted(fetched) == ["mp-1", "mp-2"]

    def test_a_file_on_disk_counts_even_with_no_manifest(self, tmp_path,
                                                         monkeypatch):
        """The two mechanisms overlap; neither depends on the other."""
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        path = f._chgcar_path("mp-1", compress=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        FakeChgcar().write_file(str(path))
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == ["mp-2"]

    def test_an_unreadable_manifest_costs_a_refetch_not_the_run(self, tmp_path,
                                                                monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f.outdir.mkdir(parents=True, exist_ok=True)
        (f.outdir / MANIFEST_JSON).write_text("{ not json")

        assert f.load_manifest() == {}
        assert f.download()[0]["status"] == "downloaded"

    def test_the_manifest_survives_the_round_trip(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.download()
        assert sorted(f.load_manifest()) == ["mp-1", "mp-2"]

    def test_a_resumed_run_keeps_the_earlier_rows(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        self._manifest(f, [{"material_id": "mp-1", "status": "downloaded",
                            "size_mb": 1.0}])
        rows = f.download()
        assert sorted(r["material_id"] for r in rows) == ["mp-1", "mp-2"]


# ===================================================================== #
# Storing what came back
# ===================================================================== #
class TestConvertingAPymatgenStructure:
    """
    ``Poscar.from_structure`` reads ``.cell``/``.symbols``/``.counts`` off
    Poraquê's own structure class. A pymatgen structure has a ``lattice`` and a
    list of sites and none of those attributes, so the HDF5 download path threw
    ``AttributeError: 'Structure' object has no attribute 'cell'`` on the first
    real material it met — after the object had already been downloaded, which
    is the worst place to discover a conversion bug.
    """

    FakeSite = FakeSite
    FakeStructure = FakeStructure

    def test_the_lattice_and_positions_survive(self):
        structure = self.FakeStructure(
            np.eye(3) * 5.0,
            [self.FakeSite("Si", [0.0, 0.0, 0.0]),
             self.FakeSite("Si", [0.5, 0.5, 0.5])], "Si")
        poscar = poscar_from_pymatgen(structure)

        assert np.allclose(poscar.cell, np.eye(3) * 5.0)
        assert poscar.symbols == ["Si"]
        assert list(poscar.counts) == [2]
        assert np.allclose(poscar.scaled_positions,
                           [[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_species_are_grouped_as_vasp_requires(self):
        structure = self.FakeStructure(
            np.eye(3) * 5.0,
            [self.FakeSite("Si", [0.0, 0.0, 0.0]),
             self.FakeSite("O", [0.5, 0.0, 0.0]),
             self.FakeSite("O", [0.0, 0.5, 0.0])])
        poscar = poscar_from_pymatgen(structure)
        assert poscar.symbols == ["Si", "O"]
        assert list(poscar.counts) == [1, 2]

    def test_interleaved_species_are_regrouped(self):
        """
        A pymatgen structure does not guarantee grouping, and VASP's format
        requires it. Regrouping is safe here because the grid is indexed by
        position and not by site order — reordering the list cannot move a
        value.
        """
        structure = self.FakeStructure(
            np.eye(3) * 5.0,
            [self.FakeSite("Si", [0.0, 0.0, 0.0]),
             self.FakeSite("O", [0.5, 0.0, 0.0]),
             self.FakeSite("Si", [0.0, 0.5, 0.0])])
        poscar = poscar_from_pymatgen(structure)

        assert poscar.symbols == ["Si", "O"]
        assert list(poscar.counts) == [2, 1]
        assert sum(poscar.counts) == len(poscar.scaled_positions)
        # The O position is still the O position.
        assert np.allclose(poscar.scaled_positions[2], [0.5, 0.0, 0.0])

    def test_it_survives_the_writer_it_feeds(self):
        structure = self.FakeStructure(
            np.eye(3) * 5.0, [self.FakeSite("Pt", [0.0, 0.0, 0.0])], "Pt")
        assert "Pt" in poscar_from_pymatgen(structure).to_string()

    def test_one_of_ours_passes_straight_through(self):
        poscar = Poscar(np.eye(3) * 5.0, ["Si"], [1], [[0.0, 0.0, 0.0]])
        assert poscar_from_pymatgen(poscar) is poscar


class TestTheDownloadLayout:
    r"""
    One directory per material, named by its id: ``data/MP/mp-124/CHGCAR.gz``.

    It replaces a flat ``chgcar/CHGCAR_mp-124.gz``, where the id lived in the
    filename and every density in the archive shared one directory. What that
    cost was not tidiness — nothing could be placed *beside* a density without
    inventing a second naming convention for it.

    Two things have to hold for the change to be safe rather than merely
    tidier, and both are asserted here: an archive already on disk in the old
    layout must still be **resumed** rather than re-fetched, and the manifest
    must still locate its files now that every one of them is called
    ``CHGCAR``.
    """

    def test_each_density_gets_its_own_directory(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.download()

        for identifier in ("mp-1", "mp-2"):
            assert (f.outdir / identifier / "CHGCAR.gz").exists()

    def test_the_filename_no_longer_carries_the_id(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f.download(compress=False)

        assert (f.outdir / "mp-1" / "CHGCAR").exists()
        assert not list(f.outdir.glob("CHGCAR_*"))

    def test_the_manifest_sits_at_the_top_not_beside_the_densities(
            self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f.download()

        assert (f.outdir / MANIFEST_JSON).exists()
        assert not (f.outdir / "chgcar").exists()

    def test_a_manifest_row_locates_its_file(self, tmp_path, monkeypatch):
        """
        Every density is now called ``CHGCAR``, so a bare filename would name
        all of them at once. The row records the path relative to ``outdir``.
        """
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        row = f.download()[0]

        assert row["file"] == os.path.join("mp-1", "CHGCAR.gz")
        assert (f.outdir / row["file"]).exists()

    def test_an_hdf5_store_goes_in_the_same_place(self, tmp_path, monkeypatch):
        pytest.importorskip("h5py")
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        row = f.download(hdf5=True)[0]

        assert (f.outdir / "mp-1" / "fields.h5").exists()
        assert row["file"] == os.path.join("mp-1", "fields.h5")

    def test_a_failure_leaves_no_empty_directory_behind(self, tmp_path,
                                                        monkeypatch):
        """
        An empty ``mp-2/`` would be discovered later as a material with no
        density — a hole that looks like a directory rather than like a gap.
        """
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)

        def flaky(identifier):
            if identifier == "mp-2":
                raise ValueError("no such object")
            return FakeChgcar()

        f._rester.get_charge_density_from_material_id = flaky
        f.download()

        assert (f.outdir / "mp-1").is_dir()
        assert not (f.outdir / "mp-2").exists()


class TestAnArchiveInTheOldLayoutIsResumedNotRefetched:
    """
    The layout moved; the objects did not. A resume that only recognised the
    new spelling would re-download an entire archive in order to rename it,
    which is the one outcome this module exists to prevent.
    """

    def _legacy(self, fetcher, identifier, suffix=".gz"):
        directory = fetcher.legacy_chgcar_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"CHGCAR_{identifier}{suffix}"
        path.write_bytes(b"not really a density, but it is on disk")
        return path

    def test_a_flat_file_counts_as_already_downloaded(self, tmp_path,
                                                      monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        self._legacy(f, "mp-1")
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == ["mp-2"]

    def test_an_uncompressed_flat_file_counts_too(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        self._legacy(f, "mp-1", suffix="")
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == []

    def test_a_flat_hdf5_store_counts_too(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        directory = f.legacy_chgcar_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mp-1.h5").write_bytes(b"a store")
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download()

        assert fetched == []

    def test_a_manifest_written_beside_the_old_densities_is_read(
            self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.legacy_chgcar_dir.mkdir(parents=True, exist_ok=True)
        with (f.legacy_chgcar_dir / MANIFEST_JSON).open("w") as handle:
            json.dump([{"material_id": "mp-1", "status": "downloaded",
                        "size_mb": 1.0}], handle)

        assert set(f.load_manifest()) == {"mp-1"}

    def test_the_new_manifest_wins_when_both_exist(self, tmp_path, monkeypatch):
        """A machine part way through the change has both; the current one is
        the one that was written last."""
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f.legacy_chgcar_dir.mkdir(parents=True, exist_ok=True)
        with (f.legacy_chgcar_dir / MANIFEST_JSON).open("w") as handle:
            json.dump([{"material_id": "mp-old", "status": "downloaded"}],
                      handle)
        with (f.outdir / MANIFEST_JSON).open("w") as handle:
            json.dump([{"material_id": "mp-new", "status": "downloaded"}],
                      handle)

        assert set(f.load_manifest()) == {"mp-new"}

    def test_storage_commands_see_both_layouts(self, tmp_path, monkeypatch):
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.download()                       # mp-1/CHGCAR.gz, mp-2/CHGCAR.gz
        self._legacy(f, "mp-9")            # chgcar/CHGCAR_mp-9.gz

        assert len(f._downloaded()) == 3


class TestStoringAsHDF5:
    def test_the_download_writes_a_store(self, tmp_path, monkeypatch):
        pytest.importorskip("h5py")
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        rows = f.download(hdf5=True, compression="gzip")

        store = f.material_dir("mp-1") / "fields.h5"
        assert store.exists()
        assert field_names(str(store)) == ["CHGCAR"]
        assert rows[0]["grid"] == [8, 8, 8]

    def test_the_values_match_the_chgcar_it_replaces(self, tmp_path,
                                                     monkeypatch):
        """A re-encoding, not a conversion: no value changes."""
        pytest.importorskip("h5py")
        f = fetcher_with(["mp-1"], tmp_path, monkeypatch=monkeypatch)
        f.download(hdf5=True, compression="gzip")
        g = fetcher_with(["mp-1"], tmp_path / "text", monkeypatch=monkeypatch)
        g.download(compress=False)

        binary = ChargeDensity.read(
            str(f.material_dir("mp-1") / "fields.h5") + "::CHGCAR",
            dtype="float64")
        text = ChargeDensity.read(g._chgcar_path("mp-1", compress=False),
                                  dtype="float64")
        assert np.allclose(binary.data, text.data, rtol=0, atol=1e-10)

    def test_a_store_is_resumable_like_any_other_file(self, tmp_path,
                                                      monkeypatch):
        pytest.importorskip("h5py")
        f = fetcher_with(["mp-1", "mp-2"], tmp_path, monkeypatch=monkeypatch)
        f.download(hdf5=True)
        fetched = []
        f._rester.get_charge_density_from_material_id = (
            lambda i: (fetched.append(i), FakeChgcar())[1])
        f.download(hdf5=True)

        assert fetched == []
