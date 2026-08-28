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

Everything here skips when ``data/vasp`` is absent, which it is in a fresh
checkout: the dataset is gitignored. That is the same bargain
``test_energy_differences.py`` makes, and it means these tests protect the
machine the data lives on and stay silent elsewhere.
"""

import glob
import os

import numpy as np
import pytest

from poraque.fields import ChargeDensity, KineticEnergyDensity
from poraque.fields.density import thomas_fermi_tau, von_weizsacker_tau

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VASP = os.path.join(REPO, "data", "vasp")
STRUCTURES = os.path.join(VASP, "structures")
ATOMS = os.path.join(VASP, "isolated_atoms")


def thomas_fermi_scale(tau, density, grid):
    r"""
    :math:`\int\tau` against the Thomas-Fermi estimate built from ρ.

    Dimensionless, which is the point: it is the one comparison that survives
    not knowing what convention a ``TAUCAR`` was written in.
    """
    tau = np.asarray(tau, dtype=float)
    density = np.asarray(density, dtype=float)
    assert tau.shape == density.shape, "tau and rho are not a pair on one grid"

    tau_integral = float(grid.integrate(tau))
    tf_integral = float(grid.integrate(thomas_fermi_tau(density)))
    return {"tau_integral": tau_integral, "tf_integral": tf_integral,
            "ratio": tau_integral / tf_integral,
            "electrons": float(grid.integrate(density))}


def von_weizsacker_violations(tau, density, grid, density_threshold=1e-3,
                              relative_tolerance=0.05,
                              absolute_tolerance=1e-6):
    r"""
    Where, and how badly, :math:`\tau \ge \tau_{\rm vW}` fails.

    A theorem rather than a tolerance, so any violation on a point carrying
    real density says ρ and τ are not the pair they claim to be. The vacuum
    tail is exempt -- there :math:`\tau_{\rm vW}` is a ratio of two numerical
    noises -- and a little slack absorbs the ringing of a band-limited field.
    """
    tau = np.asarray(tau, dtype=float)
    density = np.asarray(density, dtype=float)
    vw = von_weizsacker_tau(density, grid)

    peak = float(np.max(density)) if density.size else 0.0
    significant = density > density_threshold * peak
    n_significant = int(np.count_nonzero(significant))

    allowed = vw * (1.0 - relative_tolerance) - absolute_tolerance
    violating = significant & (tau < allowed)
    n_violating = int(np.count_nonzero(violating))

    worst_ratio = None
    if n_violating:
        index = int(np.argmax((vw - tau)[violating]))
        vw_here = float(vw[violating][index])
        if vw_here > 0:
            worst_ratio = float(tau[violating][index] / vw_here)

    return {"significant_points": n_significant,
            "violations": n_violating,
            "violation_fraction": (n_violating / n_significant
                                   if n_significant else 0.0),
            "worst_ratio": worst_ratio}


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
