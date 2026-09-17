# -*- coding: utf-8 -*-
# file: test_paw_profiles.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The radial PAW core charge density, from the POTCAR into the checkpoint.

A checkpoint carries, per element and keyed by atomic number, the core density
read from the POTCAR that built :math:`V_{\rm ext}` during training, beside the
augmentation occupancies ``--add-paw`` writes. Two conventions of the POTCAR's
``PAW radial sets`` block are written nowhere in the file, and a reader that
guessed either would store a smooth, plausible and wrong array:

* the mesh is in **Å** --- the pseudized core joins the all-electron one at
  ``RPACOR`` converted to Å, not at ``RPACOR`` read as Å;
* the values are :math:`r^2\rho_{00}(r)` with :math:`\rho = \rho_{00}/\sqrt{4\pi}`,
  so :math:`\sqrt{4\pi}\int` of them is the core electron count, exactly
  :math:`Z - Z_{\rm val}`.

The tests that need a real pseudopotential skip unless
``PORAQUE_TEST_POTCAR_DIR`` names a library; everything else runs on a
synthetic dataset whose core is an analytic function with a known integral.
"""

import json
import os
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from poraque.data import discover_records, resolve_source  # noqa: E402
from poraque.data.cache import (  # noqa: E402
    PAW_PROFILES_FILENAME,
    PAW_REFERENCE_FILENAME,
    build_paw_profiles,
    load_paw_profiles,
)
from poraque.fields.vasp.augmentation import reference_from_profiles  # noqa: E402
from poraque.fields.vasp.potcar import Potcar  # noqa: E402
from test_data_sources import write_calculation  # noqa: E402

LIBRARY = os.environ.get("PORAQUE_TEST_POTCAR_DIR") or None

#: Si: Z = 14, ZVAL = 4, so ten core electrons.
CORE_ELECTRONS = 10.0
#: Decay length of the synthetic core, Å.
CORE_LENGTH = 0.12
#: Where the synthetic pseudized core stops following the all-electron one, Å.
PARTIAL_CORE_RADIUS = 0.6


def _mesh():
    """A logarithmic radial mesh, like VASP's: 300 points from 1e-4 to 3 Å."""
    return np.geomspace(1e-4, 3.0, 300)


def _core(r):
    r"""Hydrogenic-shaped core, e/Å³, integrating to :data:`CORE_ELECTRONS`."""
    return (CORE_ELECTRONS / (np.pi * CORE_LENGTH ** 3)
            * np.exp(-2.0 * r / CORE_LENGTH))


def _pseudo_core(r):
    """Flat inside the partial-core radius, the true core outside it."""
    return np.where(r < PARTIAL_CORE_RADIUS, _core(PARTIAL_CORE_RADIUS), _core(r))


def _table(values):
    return "\n".join(
        "".join(f"{value:20.12E}" for value in values[i:i + 5])
        for i in range(0, len(values), 5))


def potcar_text(radial=True, decoy=True):
    r"""
    A Si PAW dataset whose radial tables are stored the way VASP stores them.

    ``core charge-density`` holds :math:`r^2\rho_{00} = \sqrt{4\pi}\,r^2\rho`.
    The ``local part`` is a complete 1000-point table so a library accepts the
    entry, and ``decoy`` puts the reciprocal-space ``core charge-density
    (partial)`` table ahead of the radial block, where a prefix match on the
    title would read it instead.
    """
    r = _mesh()
    norm = np.sqrt(4.0 * np.pi)
    parts = [
        " PAW_PBE Si 05Jan2001",
        "   4.00000000000000",
        " parameters from PSCTR are:",
        "   TITEL  = PAW_PBE Si 05Jan2001",
        "   POMASS =   28.085; ZVAL   =    4.000    mass and valenz",
        "   RCORE  =    1.900    outmost cutoff radius",
        "   ENMAX  =  245.345; ENMIN  = 184.009 eV",
        " local part",
        "   30.0000000000000",
        _table(-np.linspace(1.0, 0.0, 1000)),
    ]
    # One p projector: a record of 6 values, (p,p) at L = 0 and L = 2 --- the
    # length `test_ext2paw` writes its synthetic records at.
    parts += [" Non local Part", "           1           1   1.50000000000000"]
    if decoy:
        parts += [" core charge-density (partial)",
                  "   8.00000000000000",
                  _table(np.full(1000, 99.0))]
    if radial:
        parts += [
            " PAW radial sets",
            f"         {len(r)}   1.20000000000000",
            "(5E20.12)",
            " grid", _table(r),
            " aepotential", _table(-1.0 / r),
            " core charge-density", _table(norm * r ** 2 * _core(r)),
            " kinetic energy-density", _table(np.zeros_like(r)),
            " core charge-density (pseudized)",
            _table(norm * r ** 2 * _pseudo_core(r)),
        ]
    parts.append(" End of Dataset")
    return "\n".join(parts) + "\n"


def _entry(**kwargs):
    return Potcar.from_string(potcar_text(**kwargs), parse_tables=True)[0]


class TestTheCoreDensityIsReadInTheUnitsItIsStoredIn:
    """
    :math:`r^2\\rho_{00}` in, :math:`\\rho` in e/Å³ out.

    The error this guards against produces no exception and no NaN. Storing
    the raw table as a density gives a curve that *rises* away from the nucleus
    and integrates, with :math:`4\\pi r^2`, to 19.7 electrons for Pt instead of
    68 --- smooth, finite and wrong, and nothing downstream would notice.
    """

    def test_the_density_is_the_function_that_was_stored(self):
        profile = _entry().core_profile
        r = np.asarray(profile["r"])
        np.testing.assert_allclose(profile["core_density"], _core(r),
                                   rtol=1e-9)
        np.testing.assert_allclose(profile["pseudo_core_density"],
                                   _pseudo_core(r), rtol=1e-9)

    def test_it_integrates_to_z_minus_zval(self):
        profile = _entry().core_profile
        assert profile["expected_core_electrons"] == CORE_ELECTRONS
        assert profile["core_electrons"] == pytest.approx(CORE_ELECTRONS,
                                                          rel=1e-4)

    def test_it_is_keyed_and_labelled_as_an_element(self):
        profile = _entry().core_profile
        assert (profile["element"], profile["atomic_number"]) == ("Si", 14)
        assert profile["titel"] == "PAW_PBE Si 05Jan2001"
        assert profile["zval"] == 4.0

    def test_the_reciprocal_space_partial_core_is_not_mistaken_for_it(self):
        """
        ``core charge-density`` is a prefix of ``core charge-density
        (partial)``, which sits earlier in the file on a q-mesh. Matching
        titles by prefix from the top of the file reads that table.
        """
        with_decoy = _entry(decoy=True).core_profile
        without = _entry(decoy=False).core_profile
        assert with_decoy["core_density"] == without["core_density"]
        assert max(with_decoy["core_density"]) != 99.0

    def test_nothing_is_read_without_the_tables_or_without_a_paw_block(self):
        header_only = Potcar.from_string(potcar_text())[0]
        assert header_only.core_profile is None
        assert _entry(radial=False).core_profile is None


@pytest.mark.skipif(LIBRARY is None,
                    reason="set PORAQUE_TEST_POTCAR_DIR to a POTCAR library")
class TestARealDatasetPinsBothConventions:
    """
    Measured on the pseudopotentials the project trains with, not assumed.

    The electron count pins the :math:`r^2\\rho_{00}` convention and cannot
    pin the unit --- the integral is dimensionless either way --- so the unit
    is pinned separately, by where the pseudized core rejoins the true one.
    """

    @pytest.fixture(params=["Pt", "Ag", "Si"])
    def dataset(self, request):
        path = os.path.join(LIBRARY, request.param, "POTCAR")
        if not os.path.exists(path):
            pytest.skip(f"no {request.param} in {LIBRARY}")
        with open(path, errors="replace") as handle:
            text = handle.read()
        return text, Potcar.from_string(text, parse_tables=True)[0]

    def test_the_core_holds_z_minus_zval_electrons(self, dataset):
        _, entry = dataset
        profile = entry.core_profile
        assert profile["core_electrons"] == pytest.approx(
            profile["expected_core_electrons"], abs=1e-3)

    def test_the_mesh_is_in_angstrom(self, dataset):
        from poraque.fields.constants import BOHR_TO_ANGSTROM

        text, entry = dataset
        rpacor = float(re.search(r"RPACOR\s*=\s*([-\d.]+)", text).group(1))
        profile = entry.core_profile
        r = np.asarray(profile["r"])
        same = np.isclose(profile["core_density"],
                          profile["pseudo_core_density"], rtol=1e-6)
        # The first radius from which the two tables agree for good.
        joined = r[np.argmax(np.cumprod(same[::-1])[::-1].astype(bool))]
        assert joined == pytest.approx(rpacor * BOHR_TO_ANGSTROM, rel=0.05)
        assert abs(joined - rpacor) > 0.5


class TestTheCacheRecordsTheCoreOfThePotcarThatBuiltVext:
    """
    One profile per element, from the file the potential was built from.

    A profile from any other POTCAR would describe a pseudopotential the model
    was never trained against, so the precedence is
    ``_external_potential``'s: the run's own ``POTCAR``, then ``potcar_dir``.
    """

    @staticmethod
    def _records(root, **options):
        return discover_records([resolve_source(str(root), **options)],
                                required=("CHGCAR",))

    @staticmethod
    def _potcar(tmp_path):
        path = tmp_path / "Si.POTCAR"
        path.write_text(potcar_text())
        return str(path)

    def test_a_runs_own_potcar_is_read_once_per_element(self, tmp_path):
        runs = tmp_path / "runs"
        potcar = self._potcar(tmp_path)
        for index in range(3):
            write_calculation(runs / f"structure_{index:04d}", shape=(8, 8, 8),
                              seed=index, potcar=potcar)
        lines = []
        profiles = build_paw_profiles(self._records(runs), str(tmp_path / "c"),
                                      lines.append)

        assert sorted(profiles) == [14]
        assert profiles[14]["core_electrons"] == pytest.approx(10.0, rel=1e-4)
        assert sum("PAW core profile: Si" in line for line in lines) == 1

    def test_a_library_stands_in_for_a_stripped_run(self, tmp_path):
        library = tmp_path / "potcars" / "Si"
        library.mkdir(parents=True)
        (library / "POTCAR").write_text(potcar_text())
        runs = tmp_path / "runs"
        write_calculation(runs / "structure_0000", shape=(8, 8, 8))

        profiles = build_paw_profiles(
            self._records(runs, potcar_dir=str(tmp_path / "potcars")),
            str(tmp_path / "c"))
        assert profiles[14]["titel"] == "PAW_PBE Si 05Jan2001"

    def test_a_gaussian_dataset_has_no_core_to_record(self, tmp_path):
        runs = tmp_path / "runs"
        write_calculation(runs / "structure_0000", shape=(8, 8, 8))
        cache = tmp_path / "c"
        assert build_paw_profiles(self._records(runs), str(cache)) == {}
        assert not (cache / PAW_PROFILES_FILENAME).exists()

    def test_a_node_without_the_potcars_keeps_what_the_cache_job_read(
            self, tmp_path):
        """
        On a cluster the cache is built in a CPU job and training runs in a
        GPU job that may not see the POTCAR library. Rebuilding there would
        find nothing and overwrite the profiles with an empty table.
        """
        runs = tmp_path / "runs"
        write_calculation(runs / "structure_0000", shape=(8, 8, 8),
                          potcar=self._potcar(tmp_path))
        cache = str(tmp_path / "c")
        first = build_paw_profiles(self._records(runs), cache)

        os.remove(runs / "structure_0000" / "POTCAR")
        lines = []
        again = build_paw_profiles(self._records(runs), cache, lines.append)
        assert again == first
        assert any("cached" in line for line in lines)


class TestTheCheckpointHoldsEverythingPawPerElement:
    """
    Core density and augmentation occupancies, one entry per atomic number.

    They are separate files in the cache because they come from different
    places --- a POTCAR and the CHGCARs --- and one entry in the checkpoint
    because they answer one question: what a VASP density of this element
    needs that no grid model predicts.
    """

    @staticmethod
    def _augmentation():
        return {"values": [0.5, 0.25], "atoms": 4, "structures": 2,
                "source": "material_average"}

    def test_the_two_tables_merge_under_the_atomic_number(self, tmp_path):
        cache = tmp_path / "c"
        cache.mkdir()
        profile = _entry().core_profile
        (cache / PAW_PROFILES_FILENAME).write_text(json.dumps({"14": profile}))
        (cache / PAW_REFERENCE_FILENAME).write_text(json.dumps(
            {"Si": self._augmentation(), "O": self._augmentation()}))

        merged = load_paw_profiles(str(cache))
        assert sorted(merged) == [8, 14]
        assert merged[14]["core_density"] == profile["core_density"]
        assert merged[14]["augmentation"] == self._augmentation()
        # O has records from its CHGCARs and no POTCAR: augmentation only.
        assert "core_density" not in merged[8]
        assert merged[8]["element"] == "O"

    def test_inference_reads_the_augmentation_back_by_symbol(self, tmp_path):
        profiles = {14: {"element": "Si", "atomic_number": 14,
                         "augmentation": self._augmentation()},
                    8: {"element": "O", "atomic_number": 8}}
        assert reference_from_profiles(profiles) == {
            "Si": self._augmentation()}
        assert reference_from_profiles({}) == {}

    def test_a_trained_checkpoint_carries_the_profiles(self, tmp_path):
        """End to end through ``save_bundle`` and ``read_bundle``."""
        import torch

        from poraque.ml import FieldOperator, read_bundle, save_bundle

        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 device="cpu")
        profiles = {14: dict(_entry().core_profile,
                             augmentation=self._augmentation())}
        path = save_bundle(str(tmp_path / "m.poraque"), {"ext2chg": operator},
                           paw_profiles=profiles)

        stored = read_bundle(path)["paw_profiles"][14]
        r = np.asarray(stored["r"])
        np.testing.assert_allclose(stored["core_density"], _core(r),
                                   rtol=1e-9)
        assert stored["augmentation"]["atoms"] == 4
        assert isinstance(torch.load(path, weights_only=False), dict)


class TestATrainingRunWritesTheWholeCheckpoint:
    """
    ``poraque-train``, end to end, writes the three keys with the core inside.

    Nothing else drives the save: the pieces above are tested one at a time,
    and a run that built the profiles into the cache and then saved a
    checkpoint without them would pass every one of those tests.
    """

    def test_a_one_epoch_run_on_potcar_bearing_data(self, tmp_path):
        import yaml

        sys.path.insert(0, os.path.join(_ROOT, "scripts"))
        import poraque_train

        from poraque.ml import CHECKPOINT_KEYS, load_bundle, read_bundle

        potcar = tmp_path / "Si.POTCAR"
        potcar.write_text(potcar_text())
        runs = tmp_path / "runs"
        for index in range(2):
            write_calculation(runs / f"structure_{index:04d}", shape=(8, 8, 8),
                              seed=index, potcar=str(potcar))
        config = tmp_path / "train.yaml"
        config.write_text(yaml.safe_dump({
            "task": {"type": "all", "name": "paw_smoke"},
            "data": {"data_paths": [str(runs)], "cache": str(tmp_path / "c"),
                     "resolution": 8, "delta_density": False,
                     "paw_source": "material", "spin": False},
            "model": {"width": 4, "modes": 2, "n_layers": 1,
                      "projection_channels": 4},
            "training": {"epochs": 1, "valid_fraction": 0.0, "eval_epoch": 1,
                         "early_stopping": 0, "device": "cpu"},
            "output": {"root": str(tmp_path / "models"),
                       "plot_figures": False, "write_pdf_report": False},
        }))

        poraque_train.run(["--config", str(config)])

        path = tmp_path / "models" / "paw_smoke" / "paw_smoke.poraque"
        payload = read_bundle(str(path))
        assert tuple(sorted(payload)) == tuple(sorted(CHECKPOINT_KEYS))
        assert sorted(payload["model_state_dict"]) == ["chg2tau", "ext2chg"]
        assert payload["config"]["task"]["name"] == "paw_smoke"
        assert payload["config"]["training"]["epochs"] == 1

        silicon = payload["paw_profiles"][14]
        assert silicon["titel"] == "PAW_PBE Si 05Jan2001"
        assert silicon["core_electrons"] == pytest.approx(10.0, rel=1e-4)
        np.testing.assert_allclose(silicon["core_density"],
                                   _core(np.asarray(silicon["r"])), rtol=1e-9)
        for task in ("ext2chg", "chg2tau"):
            assert load_bundle(str(path), task, device="cpu").task.name == task
