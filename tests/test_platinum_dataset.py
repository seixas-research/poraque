# -*- coding: utf-8 -*-
# file: test_platinum_dataset.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The Pt dataset, held to the conventions it was read wrong under.

Two of these tests are the only place in the suite where the ``TAUCAR``
convention is checked against a file **VASP actually wrote**, rather than one
Poraquê wrote and read back. That distinction is the entire lesson of
the post-mortem: a round trip is symmetric under a wrong convention and
proves nothing. What proves something is a *pair* — τ against the ρ it was
computed with — because the von Weizsäcker bound relates the two and is a
theorem rather than a convention.

The third thing checked here is that the runs pass the ingestion gate **as
delivered** — with no ``OUTCAR`` and no ``vasprun.xml``, because those are
outputs and this dataset does not ship them.

Everything here skips when ``data/vasp`` is absent, which it is in a fresh
checkout: the dataset is gitignored. That is the same bargain
``test_energy_differences.py`` makes, and it means these tests protect the
machine the data lives on and stay silent elsewhere.
"""

import glob
import os

import numpy as np
import pytest

from poraque.data.validation import (
    REQUIRED_VASP_VERSION,
    read_tau_provenance,
    thomas_fermi_scale,
    validate_tau,
    version_at_least,
    von_weizsacker_violations,
)
from poraque.fields import ChargeDensity, KineticEnergyDensity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VASP = os.path.join(REPO, "data", "vasp")
STRUCTURES = os.path.join(VASP, "structures")
ATOMS = os.path.join(VASP, "isolated_atoms")


def _has_output(directory):
    """Whether a run carries a file the code version can be read from."""
    return any(os.path.exists(os.path.join(directory, name))
               for name in ("OUTCAR", "vasprun.xml"))


def _runs():
    """Every calculation directory in the dataset, structures and atoms."""
    found = sorted(glob.glob(os.path.join(STRUCTURES, "structure_*")))
    found += sorted(path for path in glob.glob(os.path.join(ATOMS, "*"))
                    if os.path.isdir(path))
    return [path for path in found
            if os.path.exists(os.path.join(path, "CHGCAR"))
            and os.path.exists(os.path.join(path, "TAUCAR"))]


RUNS = _runs()
needs_dataset = pytest.mark.skipif(
    not RUNS, reason="data/vasp holds no Pt runs in this checkout")


# One representative of each kind. Reading a 108³ pair costs a few seconds, so
# the pointwise checks run on a sample and the cheap header checks run on all.
SAMPLE = ([RUNS[0]] + ([RUNS[-1]] if len(RUNS) > 1 else [])) if RUNS else []


def _pair(directory):
    density = ChargeDensity.read(os.path.join(directory, "CHGCAR"))
    tau = KineticEnergyDensity.read(os.path.join(directory, "TAUCAR"),
                                    grid=density.grid)
    return density, tau


@needs_dataset
class TestTheKineticEnergyDensityIsReadInVaspsOwnConvention:
    """
    Regression: τ was read as τ·Ω and as its first spin block alone.

    Together those two errors returned :math:`\\tau/(2\\Omega)` — off by
    :math:`10^3` in a 500 Å³ cell, and undetectable in anything but a paired
    check. Both are pinned here against real ``LTAU`` output.
    """

    @pytest.mark.parametrize("directory", SAMPLE)
    def test_tau_is_within_an_order_of_magnitude_of_thomas_fermi(self,
                                                                 directory):
        density, tau = _pair(directory)
        scale = thomas_fermi_scale(tau.data, density.data, density.grid)
        # Measured 0.921-0.926 on the 31 bulk cells and 0.972 on the atom.
        # The window is wide because the claim is "no convention error", not
        # "tau_TF is accurate" -- it is the uniform-gas limit, not a target.
        assert 0.5 < scale["ratio"] < 2.0, (
            f"{os.path.basename(directory)}: int(tau) is "
            f"{scale['ratio']:.4g}x the Thomas-Fermi estimate. A factor near "
            f"the cell volume means the volume scaling came back; a factor "
            f"near two means only one spin block was read.")

    @pytest.mark.parametrize("directory", SAMPLE)
    def test_tau_never_falls_below_the_von_weizsacker_bound(self, directory):
        """
        The decisive test, because it is a theorem rather than a tolerance:
        τ ≥ |∇ρ|²/(8ρ) everywhere, for any density. Read wrong, the isolated
        atom violated it at 71 % of its significant points.
        """
        density, tau = _pair(directory)
        bound = von_weizsacker_violations(tau.data, density.data, density.grid)
        assert bound["violation_fraction"] == 0.0, (
            f"{os.path.basename(directory)}: tau < tau_vW at "
            f"{100 * bound['violation_fraction']:.2f}% of significant points "
            f"(worst tau/tau_vW = {bound['worst_ratio']}). rho and tau are "
            f"not the pair they claim to be.")

    @pytest.mark.parametrize("directory", SAMPLE)
    def test_the_two_blocks_are_both_positive_so_neither_is_a_magnetisation(
            self, directory):
        """
        What identified the layout. A magnetisation block is signed and small;
        these are non-negative and comparable, which is what τ_up and τ_down
        are.
        """
        from poraque.fields.vasp.volumetric import read_volumetric

        _, first, extra = read_volumetric(os.path.join(directory, "TAUCAR"),
                                          read_all=True)
        assert len(extra) == 1, "these runs are ISPIN = 2; expected two blocks"
        second = np.asarray(extra[0], dtype=float)
        assert second.min() > -1e-9
        assert np.asarray(first, dtype=float).min() > -1e-9
        assert 0.5 < second.mean() / np.asarray(first, dtype=float).mean() < 2.0


@needs_dataset
class TestTauIsNotMultipliedByTheCellVolume:
    r"""
    The volume question, settled without appealing to any model of tau.

    A ``CHGCAR`` holds :math:`\rho\Omega` and a ``TAUCAR`` does not, which is
    the kind of asymmetry that is invisible in a round trip and wrong by three
    orders of magnitude in an integral. The anchor is the density: its
    convention is *verified*, not assumed, because the electron count comes out
    at exactly ``n_atoms x ZVAL``. Everything here hangs off that.

    The strongest argument needs no physics at all. The kinetic energy per
    valence electron is a property of the electrons; it cannot depend on how
    large a box the calculation was run in. The isolated atom's cell is twice
    the bulk cell's volume, so a spurious :math:`1/\Omega` shows up directly
    as a factor of two between two numbers that must agree.
    """

    def _per_electron(self, directory, volume_scaled):
        from poraque.fields.vasp.volumetric import read_volumetric

        density = ChargeDensity.read(os.path.join(directory, "CHGCAR"))
        grid = density.grid
        _, first, extra = read_volumetric(os.path.join(directory, "TAUCAR"),
                                          read_all=True)
        tau = np.asarray(first, dtype=float)
        for block in extra:
            tau = tau + np.asarray(block, dtype=float)
        if volume_scaled:
            tau = tau / grid.volume
        electrons = grid.integrate(np.asarray(density.data, dtype=float))
        return float(grid.integrate(tau) / electrons), float(grid.volume)

    def test_the_energy_per_electron_does_not_depend_on_the_box(self):
        """
        Two systems, two box sizes, one physical quantity.

        Read as written the two agree to a few percent. Read as volume-scaled
        they differ by exactly the ratio of the cells -- which is the signature
        being tested for, not a coincidence.
        """
        atoms = sorted(path for path in glob.glob(os.path.join(ATOMS, "*"))
                       if os.path.isdir(path))
        bulk = sorted(glob.glob(os.path.join(STRUCTURES, "structure_*")))
        if not atoms or not bulk:
            pytest.skip("need both an isolated atom and a bulk cell")

        written_a, volume_a = self._per_electron(atoms[0], False)
        written_b, volume_b = self._per_electron(bulk[0], False)
        scaled_a, _ = self._per_electron(atoms[0], True)
        scaled_b, _ = self._per_electron(bulk[0], True)

        ratio = volume_a / volume_b
        assert ratio > 1.5, (
            "this test needs two cells of genuinely different volume; "
            f"they differ by only {ratio:.2f}x")

        assert written_a / written_b == pytest.approx(1.0, abs=0.15), (
            f"read as written, tau per electron is {written_a:.3f} eV for the "
            f"atom and {written_b:.3f} eV for the bulk. These are different "
            f"systems, so they need not match exactly, but a kinetic energy "
            f"per electron cannot differ by more than a few tens of percent "
            f"between two metallic Pt environments.")

        assert scaled_a / scaled_b == pytest.approx(1.0 / ratio, rel=0.05), (
            "read as volume-scaled the two should track 1/Omega exactly -- if "
            "they no longer do, this test has stopped measuring what it says.")

    @pytest.mark.parametrize("directory", SAMPLE)
    def test_the_integral_is_near_the_thomas_fermi_estimate(self, directory):
        """
        The magnitude itself, since it is large and invites suspicion.

        int(tau) is of order 1e4 eV for a 32-atom cell. That is not evidence of
        a volume factor: *every* term in a plane-wave total energy is of that
        order and the total is small only because they cancel. What pins the
        scale is the ratio to Thomas-Fermi, which is dimensionless.
        """
        density, tau = _pair(directory)
        scale = thomas_fermi_scale(tau.data, density.data, density.grid)
        assert 0.8 < scale["ratio"] < 1.2, (
            f"{os.path.basename(directory)}: int(tau) is {scale['ratio']:.4g}x "
            f"the Thomas-Fermi estimate. A factor near the cell volume "
            f"(~500) means the volume scaling is back; a factor near two "
            f"means one spin block was dropped.")


@needs_dataset
class TestEveryRunPassesTheGateAsDelivered:
    """
    The provenance half of the ingestion gate, checked run by run.

    What the gate requires is what the run's *inputs* say: ``LTAU = .TRUE.``
    and a hash of the ``INCAR`` that says it. Both are present in every
    directory here, and neither depends on an output file.

    These runs ship no ``OUTCAR`` and no ``vasprun.xml``, so they record no
    code version -- and that must not refuse them. The version was only ever
    guarding against a build too old to honour ``LTAU``, which such a build
    ignores anyway, writing no tau at all. The gap is recorded as a warning
    and the sample proceeds on its physics.
    """

    @pytest.mark.parametrize("directory", RUNS, ids=os.path.basename)
    def test_ltau_is_set(self, directory):
        provenance = read_tau_provenance(directory)
        assert provenance["tau_tag_set"] is True, (
            f"{os.path.basename(directory)}: LTAU is "
            f"{provenance['tau_tag_value']!r}, so this TAUCAR was not written "
            f"by the documented VASP path.")

    def test_a_run_that_does_ship_its_version_is_checked_against_the_minimum(
            self):
        """
        Dropping the *requirement* must not drop the *comparison*. A run that
        does record a version has it read and compared; only its absence is
        tolerated.
        """
        recorded = [directory for directory in RUNS if _has_output(directory)]
        if not recorded:
            pytest.skip("no run in this checkout ships an output file")

        provenance = read_tau_provenance(recorded[0])
        assert provenance["version"], (
            f"{os.path.basename(recorded[0])} has an output file but no "
            f"version was read from it")
        assert version_at_least(provenance["version"], REQUIRED_VASP_VERSION), (
            f"{os.path.basename(recorded[0])} records version "
            f"{provenance['version']}, below the required "
            f"{REQUIRED_VASP_VERSION}")

    @pytest.mark.parametrize("directory", RUNS, ids=os.path.basename)
    def test_the_incar_is_hashed(self, directory):
        assert read_tau_provenance(directory)["incar_sha256"], (
            f"{os.path.basename(directory)}: no INCAR to pin the settings to.")

    def test_the_whole_gate_passes_without_any_output_file(self):
        """
        End to end, on a directory exactly as delivered.

        Regression: the gate briefly required a code version, which made this
        complete and physically consistent dataset untrainable for want of a
        file that says nothing about the field.

        The run is chosen for *not* having an ``OUTCAR`` rather than fixed by
        name: some of these directories have since acquired one, and a test
        that kept pointing at such a run would still pass while no longer
        testing anything.
        """
        bare = [directory for directory in RUNS
                if not _has_output(directory)]
        if not bare:
            pytest.skip("every run now ships an output file")
        directory = bare[0]

        density, tau = _pair(directory)
        record = validate_tau(tau.data, density.data, density.grid,
                              provenance=read_tau_provenance(directory),
                              identifier=os.path.basename(directory))
        assert record["passed"] is True
        assert any("no code version" in note for note in record["warnings"]), (
            "the missing version must still be recorded -- a gap nobody sees "
            "is the condition this gate exists to prevent")
