# -*- coding: utf-8 -*-
# file: test_tau_validation.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The kinetic-energy-density ingestion gate.

These tests exist because a whole dataset of invalid :math:`\tau` was ingested,
cached, trained on and reported before anyone noticed. The
regression they pin is therefore not a bug in a function — it is the absence of
any check at all.

The synthetic system is a normalised isotropic Gaussian,

.. math::

    \rho(\mathbf r) = \frac{N}{(2\pi\sigma^2)^{3/2}}
                      e^{-r^2/2\sigma^2},

chosen because **both** anchors of the gate are known in closed form for it:

.. math::

    \int \tau_{\rm vW}\,d^3r = \frac{3N}{8\sigma^2},
    \qquad
    \int \tau_{\rm TF}\,d^3r = C_{\rm TF}\,
        \Big[\frac{N}{(2\pi\sigma^2)^{3/2}}\Big]^{5/3}
        \Big(\frac{6\pi\sigma^2}{5}\Big)^{3/2}

(atomic units, and the second follows from
:math:`\int e^{-5r^2/6\sigma^2} = (6\pi\sigma^2/5)^{3/2}`). So the first class
below checks Poraquê's own :math:`\tau_{\rm TF}`/:math:`\tau_{\rm vW}` against
mathematics rather than against themselves, and every later class can then use
them as ground truth.

The headline test is
:class:`TestTheThousandFoldScalingIsRejected`: the exact failure mode that
destroyed the platinum τ dataset, reduced to a fixture.
"""

import json
import os

import numpy as np
import pytest

from poraque.data.validation import (
    TauValidationConfig,
    TauValidationError,
    TauValidationManifest,
    code_version,
    file_hash,
    read_tau_provenance,
    thomas_fermi_scale,
    validate_tau,
    version_at_least,
    von_weizsacker_violations,
)
from poraque.fields import FieldGrid
from poraque.fields.constants import BOHR_TO_ANGSTROM, C_TF, HARTREE_TO_EV
from poraque.fields.density import thomas_fermi_tau, von_weizsacker_tau

ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def gaussian_density(grid, n_electrons=8.0, sigma=0.8):
    r"""A normalised isotropic Gaussian centred in the cell, in e/Å³."""
    coords = grid.cartesian_coordinates()
    center = 0.5 * np.asarray(grid.cell, dtype=float).sum(axis=0)
    r2 = ((coords - center) ** 2).sum(axis=-1)
    amplitude = n_electrons / (2 * np.pi * sigma ** 2) ** 1.5
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def analytic_vw_integral(n_electrons, sigma):
    """:math:`3N/8\\sigma^2` in eV, from a σ given in Å."""
    sigma_bohr = sigma * ANGSTROM_TO_BOHR
    return 3.0 * n_electrons / (8.0 * sigma_bohr ** 2) * HARTREE_TO_EV


def analytic_tf_integral(n_electrons, sigma):
    """:math:`C_{\\rm TF}A^{5/3}(6\\pi\\sigma^2/5)^{3/2}` in eV."""
    sigma_bohr = sigma * ANGSTROM_TO_BOHR
    amplitude = n_electrons / (2 * np.pi * sigma_bohr ** 2) ** 1.5
    return (C_TF * amplitude ** (5.0 / 3.0)
            * (6 * np.pi * sigma_bohr ** 2 / 5.0) ** 1.5 * HARTREE_TO_EV)


@pytest.fixture
def system():
    """``(grid, rho, tau)`` for a physically consistent synthetic sample."""
    grid = FieldGrid((64, 64, 64), np.eye(3) * 10.0)
    rho = gaussian_density(grid)
    # tau_TF + tau_vW: >= tau_vW everywhere by construction, since both terms
    # are non-negative. This is the well-formed sample every rejection test
    # below perturbs.
    tau = thomas_fermi_tau(rho) + von_weizsacker_tau(rho, grid)
    return grid, rho, tau


def good_provenance():
    """A provenance record that satisfies every requirement."""
    return {"code": "vasp", "version": "6.6.1", "tau_tag": "LTAU",
            "tau_tag_value": ".TRUE.", "tau_tag_set": True,
            "incar_sha256": "0" * 64, "incar_path": "/tmp/INCAR",
            "other_tags": {}, "source": "calculation"}


def write_run(directory, incar_text, outcar_version="6.6.1"):
    """A minimal calculation directory carrying only what provenance reads."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "INCAR"), "w") as handle:
        handle.write(incar_text)
    if outcar_version:
        with open(os.path.join(directory, "OUTCAR"), "w") as handle:
            handle.write(f" vasp.{outcar_version} 18Jan21 "
                         f"(build Jul 30 2026 13:44:49) complex\n")
    return str(directory)


# ===================================================================== #
# The anchors, against closed form
# ===================================================================== #
class TestTheAnchorsMatchTheirClosedForms:
    """
    Both bounds are checked against mathematics, not against each other.

    If :func:`thomas_fermi_tau` or :func:`von_weizsacker_tau` ever drifts — a
    unit conversion, a factor of two in the gradient — every test below would
    keep passing while measuring the wrong thing. These two assertions are what
    stop that.
    """

    def test_von_weizsacker_integrates_to_three_n_over_eight_sigma_squared(
            self, system):
        grid, rho, _ = system
        numeric = grid.integrate(von_weizsacker_tau(rho, grid))
        assert numeric == pytest.approx(analytic_vw_integral(8.0, 0.8),
                                        rel=1e-6)

    def test_thomas_fermi_integrates_to_the_gaussian_closed_form(self, system):
        grid, rho, _ = system
        numeric = grid.integrate(thomas_fermi_tau(rho))
        assert numeric == pytest.approx(analytic_tf_integral(8.0, 0.8),
                                        rel=1e-10)

    def test_the_density_carries_the_electron_count_it_was_built_with(
            self, system):
        grid, rho, _ = system
        # 1e-6, not machine precision: the Gaussian is cut off by the cell at
        # 6.25 sigma, and the midpoint sum over a 64^3 mesh is what carries the
        # remaining ~1e-8. Both are properties of the fixture, not of the code.
        assert grid.integrate(rho) == pytest.approx(8.0, rel=1e-6)


# ===================================================================== #
# The scale check
# ===================================================================== #
class TestScaleAgainstThomasFermi:
    def test_a_physical_tau_is_within_the_accepted_band(self, system):
        grid, rho, tau = system
        scale = thomas_fermi_scale(tau, rho, grid)
        # tau_TF + tau_vW is 1 + vW/TF times TF, comfortably inside [0.2, 5].
        assert 1.0 < scale["ratio"] < 2.0
        assert scale["electrons"] == pytest.approx(8.0, rel=1e-6)

    def test_the_ratio_is_the_two_integrals(self, system):
        grid, rho, tau = system
        scale = thomas_fermi_scale(tau, rho, grid)
        assert scale["ratio"] == pytest.approx(
            scale["tau_integral"] / scale["tf_integral"], rel=1e-12)

    def test_mismatched_shapes_are_an_error_not_a_broadcast(self, system):
        grid, rho, _ = system
        with pytest.raises(ValueError, match="not a pair"):
            thomas_fermi_scale(np.ones((4, 4, 4)), rho, grid)

    def test_a_zero_density_gives_no_ratio_rather_than_a_division(self, system):
        grid, _, tau = system
        scale = thomas_fermi_scale(tau, np.zeros(grid.shape), grid)
        assert scale["ratio"] is None


# ===================================================================== #
# The headline regression
# ===================================================================== #
class TestTheThousandFoldScalingIsRejected:
    r"""
    The bug that produced the post-mortem, as a fixture.

    A :math:`\tau` scaled by :math:`10^3` is finite, positive, smooth, on the
    right grid, and satisfies the von Weizsäcker bound *more* comfortably than
    the correct field does. Every structural check passes it. Only its
    **magnitude** gives it away, which is the whole reason the scale check is
    the first thing the gate does.
    """

    def test_a_thousandfold_tau_fails_the_gate(self, system):
        grid, rho, tau = system
        with pytest.raises(TauValidationError) as excinfo:
            validate_tau(tau * 1000.0, rho, grid,
                         provenance=good_provenance(), identifier="bad_tau")
        message = str(excinfo.value)
        assert "scale check" in message
        assert "bad_tau" in message
        assert excinfo.value.record["scale"]["ratio"] > 1000.0

    def test_the_same_field_unscaled_passes(self, system):
        grid, rho, tau = system
        record = validate_tau(tau, rho, grid, provenance=good_provenance())
        assert record["passed"] is True
        assert record["failures"] == []

    def test_a_thousandfold_tau_still_satisfies_the_local_bound(self, system):
        """
        Which is exactly why the scale check cannot be dropped.

        Multiplying by a large positive constant can only move τ *further*
        above τ_vW, so the one check that is a theorem is silent here.
        """
        grid, rho, tau = system
        bound = von_weizsacker_violations(tau * 1000.0, rho, grid)
        assert bound["violations"] == 0

    def test_a_thousandfold_reduction_is_rejected_too(self, system):
        """The error is symmetric; a missing volume factor points the other way."""
        grid, rho, tau = system
        with pytest.raises(TauValidationError, match="scale check"):
            validate_tau(tau / 1000.0, rho, grid,
                         provenance=good_provenance())

    def test_the_message_says_what_to_look_at(self, system):
        grid, rho, tau = system
        with pytest.raises(TauValidationError) as excinfo:
            validate_tau(tau * 1000.0, rho, grid,
                         provenance=good_provenance())
        message = str(excinfo.value)
        assert "tau*Omega" in message and "Hartree" in message
        assert "relax data.tau_validation" in message


# ===================================================================== #
# The von Weizsaecker bound
# ===================================================================== #
class TestTheVonWeizsackerBound:
    def test_tau_equal_to_the_bound_is_accepted(self, system):
        """A one-orbital density has tau = tau_vW exactly; it must not fail."""
        grid, rho, _ = system
        bound = von_weizsacker_violations(von_weizsacker_tau(rho, grid), rho,
                                          grid)
        assert bound["violations"] == 0

    def test_half_the_bound_is_refused(self, system):
        grid, rho, _ = system
        half = 0.5 * von_weizsacker_tau(rho, grid)
        bound = von_weizsacker_violations(half, rho, grid)
        assert bound["violation_fraction"] > 0.5
        assert 0.0 < bound["worst_ratio"] < 1.0

    def test_the_gate_refuses_a_field_below_the_bound(self, system):
        grid, rho, _ = system
        with pytest.raises(TauValidationError) as excinfo:
            validate_tau(0.5 * von_weizsacker_tau(rho, grid), rho, grid,
                         provenance=good_provenance())
        assert "von Weizsaecker" in str(excinfo.value)

    def test_a_few_ringing_points_are_tolerated(self, system):
        """
        Band-limited fields ring. The bound is exact for the continuum field,
        not for its truncation, so a handful of points must not fail a dataset.
        """
        grid, rho, tau = system
        spoiled = np.array(tau)
        peak = rho.max()
        significant = np.argwhere(rho > 1e-3 * peak)
        for index in significant[:5]:
            spoiled[tuple(index)] = 0.0
        record = validate_tau(spoiled, rho, grid,
                              provenance=good_provenance())
        assert record["passed"] is True
        assert record["bound"]["violations"] > 0

    def test_the_vacuum_tail_is_exempt(self, system):
        """
        Where rho is numerical noise, tau_vW is a ratio of two noises. Testing
        the bound there measures the FFT, not the physics.
        """
        grid, rho, _ = system
        bound = von_weizsacker_violations(np.zeros(grid.shape), rho, grid,
                                          density_threshold=1e-3)
        assert bound["significant_points"] < bound["total_points"]

    def test_the_bound_can_be_switched_off_on_its_own(self, system):
        grid, rho, _ = system
        record = validate_tau(
            0.5 * von_weizsacker_tau(rho, grid), rho, grid,
            provenance=good_provenance(),
            config={"check_von_weizsacker": False, "tf_ratio_range": [0.01, 5.0]})
        assert record["passed"] is True
        assert "bound" not in record


# ===================================================================== #
# Provenance
# ===================================================================== #
class TestProvenance:
    def test_a_correct_run_is_read_completely(self, tmp_path):
        run = write_run(tmp_path / "good",
                        "ENCUT = 450\nLTAU = .TRUE.\nLCHARG = .TRUE.\n")
        provenance = read_tau_provenance(run)
        assert provenance["version"] == "6.6.1"
        assert provenance["tau_tag_set"] is True
        assert len(provenance["incar_sha256"]) == 64

    def test_the_real_bad_data_is_refused(self, tmp_path, system):
        """
        The deleted dataset, reconstructed: vasp 6.2.0 and ``TAUCAR = .TRUE.``.

        Every one of the 18 purged files came from exactly this combination.
        Both halves of it are named in the failure message, because "LTAU
        missing" alone would not tell the next person what went wrong.
        """
        grid, rho, tau = system
        run = write_run(tmp_path / "patched",
                        "ENCUT = 450\nTAUCAR = .TRUE.\nEXTCAR = .TRUE.\n",
                        outcar_version="6.2.0")
        provenance = read_tau_provenance(run)

        with pytest.raises(TauValidationError) as excinfo:
            validate_tau(tau, rho, grid, provenance=provenance)
        message = str(excinfo.value)
        assert "6.2.0" in message and "6.6.1" in message
        assert "LTAU is not set" in message
        assert "not a VASP tag" in message

    def test_ltau_false_is_refused(self, tmp_path, system):
        grid, rho, tau = system
        run = write_run(tmp_path / "off", "LTAU = .FALSE.\n")
        with pytest.raises(TauValidationError, match="LTAU is not set"):
            validate_tau(tau, rho, grid,
                         provenance=read_tau_provenance(run))

    def test_no_provenance_at_all_is_refused(self, system):
        grid, rho, tau = system
        with pytest.raises(TauValidationError, match="nothing recorded"):
            validate_tau(tau, rho, grid, provenance=None)

    def test_a_run_with_no_output_file_passes_and_says_so(self, tmp_path,
                                                          system):
        """
        The version lives in ``OUTCAR``/``vasprun.xml``, which are outputs. A
        dataset is not obliged to ship them, and refusing an otherwise valid
        and physically consistent tau for want of one made a complete set
        untrainable over a file that says nothing about the field. The gap is
        recorded instead, so a run that could not name its version stays
        distinguishable from one that did.
        """
        grid, rho, tau = system
        run = write_run(tmp_path / "stripped", "LTAU = .TRUE.\n",
                        outcar_version=None)
        record = validate_tau(tau, rho, grid,
                              provenance=read_tau_provenance(run))

        assert record["passed"] is True
        assert any("no code version" in note for note in record["warnings"])
        assert record["failures"] == []

    def test_the_missing_version_can_still_be_made_fatal(self, tmp_path,
                                                         system):
        """The old behaviour, available on request rather than by default."""
        grid, rho, tau = system
        run = write_run(tmp_path / "stripped", "LTAU = .TRUE.\n",
                        outcar_version=None)
        with pytest.raises(TauValidationError, match="no code version"):
            validate_tau(tau, rho, grid,
                         provenance=read_tau_provenance(run),
                         config={"require_code_version": True})

    def test_a_recorded_version_is_still_checked(self, tmp_path, system):
        """
        Dropping the *requirement* must not drop the *comparison*. A run that
        does name a version older than 6.6.1 is still refused -- that check
        costs nothing and is the one with teeth.
        """
        grid, rho, tau = system
        run = write_run(tmp_path / "old", "LTAU = .TRUE.\n",
                        outcar_version="6.2.0")
        with pytest.raises(TauValidationError, match="below the required"):
            validate_tau(tau, rho, grid,
                         provenance=read_tau_provenance(run))

    def test_a_cached_record_is_preferred_over_re_deriving_it(self, tmp_path):
        """
        The chain of custody has to survive a re-cache.

        A prepared cache has no ``INCAR`` to re-read, so the provenance written
        beside the cached τ is the only thing that can answer the question three
        copies downstream.
        """
        material = tmp_path / "cached_material"
        material.mkdir()
        (material / "tau_provenance.json").write_text(
            json.dumps(good_provenance()))
        provenance = read_tau_provenance(str(material))
        assert provenance["source"] == "cache"
        assert provenance["version"] == "6.6.1"

    def test_provenance_can_be_waived_deliberately(self, system):
        grid, rho, tau = system
        record = validate_tau(tau, rho, grid, provenance=None,
                              config={"require_provenance": False})
        assert record["passed"] is True

    def test_version_comparison_is_component_wise(self):
        assert version_at_least("6.6.1", "6.6.1") is True
        assert version_at_least("6.10.0", "6.9.0") is True
        assert version_at_least("6.2.0", "6.6.1") is False
        assert version_at_least("6.6", "6.6.1") is False
        assert version_at_least(None, "6.6.1") is None

    def test_the_version_is_read_off_the_outcar_first_line(self, tmp_path):
        run = write_run(tmp_path / "v", "LTAU = .TRUE.\n",
                        outcar_version="6.4.2")
        assert code_version(run) == "6.4.2"

    def test_a_missing_file_hashes_to_nothing_rather_than_raising(self):
        assert file_hash("/nonexistent/INCAR") is None


# ===================================================================== #
# Configuration
# ===================================================================== #
class TestConfiguration:
    def test_an_unknown_key_raises_rather_than_being_ignored(self):
        with pytest.raises(ValueError, match="Unknown tau_validation key"):
            TauValidationConfig.from_mapping({"tf_ratio_rang": [0.5, 2.0]})

    def test_a_widened_band_accepts_what_the_default_refuses(self, system):
        grid, rho, tau = system
        with pytest.raises(TauValidationError):
            validate_tau(tau * 10.0, rho, grid, provenance=good_provenance())
        record = validate_tau(tau * 10.0, rho, grid,
                              provenance=good_provenance(),
                              config={"tf_ratio_range": [0.1, 100.0]})
        assert record["passed"] is True

    def test_disabling_the_gate_is_recorded_rather_than_silent(self, system):
        """
        A cache built with the gate off must stay distinguishable afterwards
        from one that passed it. ``passed`` is None, not True.
        """
        grid, rho, tau = system
        record = validate_tau(tau * 1000.0, rho, grid, config={"enabled": False})
        assert record["passed"] is None
        assert record["enabled"] is False

    def test_the_settings_used_are_recorded_with_the_verdict(self, system):
        grid, rho, tau = system
        record = validate_tau(tau, rho, grid, provenance=good_provenance())
        assert record["settings"]["tf_ratio_range"] == [0.2, 5.0]
        assert record["settings"]["minimum_version"] == "6.6.1"

    def test_a_spin_pair_is_reduced_to_its_total_channel(self, system):
        """
        A spin-polarised CHGCAR is (rho, m). Only the first is a density in the
        sense both bounds are written for; feeding the pair in must not compare
        tau against a magnetisation.
        """
        grid, rho, tau = system
        pair = np.stack([rho, 0.1 * rho])
        record = validate_tau(tau, pair, grid, provenance=good_provenance())
        assert record["passed"] is True


# ===================================================================== #
# The manifest
# ===================================================================== #
class TestManifest:
    def test_it_round_trips_through_disk(self, tmp_path, system):
        grid, rho, tau = system
        manifest = TauValidationManifest()
        manifest.add("m1", validate_tau(tau, rho, grid,
                                        provenance=good_provenance()))
        manifest.write(str(tmp_path))

        again = TauValidationManifest.load(str(tmp_path))
        assert again.entries["m1"]["passed"] is True
        assert again.summary() == (1, 0, 0)

    def test_a_missing_manifest_loads_empty_rather_than_raising(self, tmp_path):
        assert TauValidationManifest.load(str(tmp_path)).entries == {}

    def test_a_corrupt_manifest_does_not_stop_a_build(self, tmp_path):
        (tmp_path / "tau_validation.json").write_text("{not json")
        assert TauValidationManifest.load(str(tmp_path)).entries == {}

    def test_the_summary_counts_all_three_verdicts(self):
        manifest = TauValidationManifest({
            "a": {"passed": True}, "b": {"passed": False},
            "c": {"passed": None}, "d": {"passed": True}})
        assert manifest.summary() == (2, 1, 1)


# ===================================================================== #
# Through the real ingestion path
# ===================================================================== #
class TestTheGateStopsACacheBuild:
    """
    The gate where it actually matters: inside :func:`build_field_cache`.

    Every check above operates on arrays. This one goes through the route the
    invalid data actually took — a calculation directory, discovered, read,
    downsampled and written — and asserts that it now stops there.
    """

    @staticmethod
    def _run(directory, scale=1.0, incar=None, version="6.6.1"):
        """A one-material calculation directory with a controllable tau."""
        from poraque.fields import ChargeDensity, KineticEnergyDensity
        from poraque.fields.vasp.poscar import Poscar

        os.makedirs(directory, exist_ok=True)
        grid = FieldGrid((16, 16, 16), np.eye(3) * 6.0)
        structure = Poscar(np.eye(3) * 6.0, ["Si"], [1], [[0.5, 0.5, 0.5]])
        rho = gaussian_density(grid, n_electrons=4.0, sigma=0.9)
        tau = (thomas_fermi_tau(rho) + von_weizsacker_tau(rho, grid)) * scale

        structure.write(os.path.join(directory, "POSCAR"))
        with open(os.path.join(directory, "INCAR"), "w") as handle:
            handle.write("ENCUT = 300\nPREC = Accurate\n"
                         + (incar if incar is not None
                            else "LTAU = .TRUE.\nLCHARG = .TRUE.\n"))
        if version:
            with open(os.path.join(directory, "OUTCAR"), "w") as handle:
                handle.write(f" vasp.{version} 18Jan21 (build ...) complex\n")
        ChargeDensity(rho, grid, structure).write(
            os.path.join(directory, "CHGCAR"))
        KineticEnergyDensity(tau, grid, structure).write(
            os.path.join(directory, "TAUCAR"))
        return str(directory)

    def test_a_well_formed_run_is_cached_and_recorded(self, tmp_path):
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000")
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR", "TAUCAR"),
                                  charges={"Si": 4.0})

        assert os.path.exists(os.path.join(cache, "struct_000", "TAUCAR"))
        manifest = TauValidationManifest.load(cache)
        assert manifest.summary() == (1, 0, 0)
        assert os.path.exists(
            os.path.join(cache, "struct_000", "tau_provenance.json"))

    def test_a_thousandfold_run_never_reaches_the_cache(self, tmp_path):
        """
        The regression, end to end.

        The build stops, and — the part that matters — no ``TAUCAR`` is left
        behind for a later run to pick up and train on.
        """
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000", scale=1000.0)
        with pytest.raises(TauValidationError, match="scale check"):
            build_field_cache(root, tmp_path / "cache", resolution=8,
                              fields=("CHGCAR", "TAUCAR"),
                              charges={"Si": 4.0})

        cached_tau = os.path.join(str(tmp_path / "cache"), "struct_000",
                                  "TAUCAR")
        assert not os.path.exists(cached_tau)

    def test_the_verdict_survives_the_failure(self, tmp_path):
        """
        A build that aborts still writes its manifest.

        Losing the record to the exception would mean re-running the whole
        ingestion to find out which material failed and by how much.
        """
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000", scale=1000.0)
        cache = str(tmp_path / "cache")
        with pytest.raises(TauValidationError):
            build_field_cache(root, cache, resolution=8,
                              fields=("CHGCAR", "TAUCAR"),
                              charges={"Si": 4.0})

        manifest = TauValidationManifest.load(cache)
        assert manifest.entries["struct_000"]["passed"] is False
        assert manifest.entries["struct_000"]["scale"]["ratio"] > 100.0

    def test_the_patched_build_provenance_is_refused_end_to_end(self, tmp_path):
        """The platinum dataset's own INCAR and version, through the real path."""
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000",
                  incar="TAUCAR = .TRUE.\nEXTCAR = .TRUE.\n", version="6.2.0")
        with pytest.raises(TauValidationError, match="LTAU is not set"):
            build_field_cache(root, tmp_path / "cache", resolution=8,
                              fields=("CHGCAR", "TAUCAR"),
                              charges={"Si": 4.0})

    def test_the_gate_can_be_switched_off_for_a_build(self, tmp_path):
        """
        And the resulting cache says so, rather than looking validated.
        """
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000", scale=1000.0)
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR", "TAUCAR"),
                                  charges={"Si": 4.0},
                                  tau_validation={"enabled": False})

        assert os.path.exists(os.path.join(cache, "struct_000", "TAUCAR"))
        manifest = TauValidationManifest.load(cache)
        assert manifest.entries["struct_000"]["passed"] is None

    def test_a_density_only_build_is_untouched_by_the_gate(self, tmp_path):
        """``ext2chg`` never reads tau, so nothing here may affect it."""
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        self._run(root / "struct_000", scale=1000.0)
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR",), charges={"Si": 4.0})

        assert os.path.exists(os.path.join(cache, "struct_000", "CHGCAR"))
        assert TauValidationManifest.load(cache).entries == {}
