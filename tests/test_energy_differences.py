# -*- coding: utf-8 -*-
# file: test_energy_differences.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Validate energy **differences** against VASP.

Absolute agreement is not the target and is not achievable: the fields Poraquê
integrates are pseudo-valence quantities, so the PAW one-centre terms and
VASP's atomic reference (``EATOM``) are absent and the total sits about
:math:`10^3` eV per atom away from ``TOTEN``. What must survive is
:math:`\Delta E`.

The suite is in three layers, weakest assumption first.

**1. The terms Poraquê and VASP compute the same way must agree.**
:math:`E_{\alpha Z}`, :math:`E_{\rm Ewald}` and :math:`E_{\rm H}` have exact
counterparts in the ``OUTCAR`` decomposition (``PSCENC``, ``TEWEN``, ``DENC``).
These are not approximations of each other — they are the same quantity — so
they are compared on a tight tolerance. Two structures in the shipped dataset
carry an ``OUTCAR``, which is what makes this possible.

**2. Energy differences from VASP's own fields.**
Feeding the reference ``CHGCAR``/``TAUCAR`` in removes all model error and
leaves only the energy expression. This is the honest measure of the *ceiling*
on :math:`\Delta E` accuracy, and it is recorded here as a characterization
test rather than a pass/fail physics claim: the residual is the neglected PAW
one-centre and non-local energy, which is not a per-atom constant.

**3. Molecular and cohesive references** — :math:`N_2` against isolated
nitrogen, :math:`CO`, and the cohesive energy of diamond, as the task asks.

.. note::

   Layer 3 needs reference calculations this repository does not ship: the
   dataset is 17 gold structures. Those tests **skip** until the data exists.
   Point ``PORAQUE_REFERENCE_DATA`` at a directory laid out as

   .. code-block:: text

       <root>/n2/        POSCAR POTCAR OSZICAR   (N2 molecule in a box)
       <root>/n_atom/    POSCAR POTCAR OSZICAR   (one N atom, same box)
       <root>/co/        ...
       <root>/c_atom/    ...
       <root>/o_atom/    ...
       <root>/diamond/   ...                     (bulk C)

   and they run. They additionally need **operators trained on the relevant
   chemistry**: the shipped checkpoint saw only gold, and an FNO asked for the
   density of a nitrogen molecule is extrapolating far outside its training
   distribution. Passing tolerances are therefore stated per system and are
   the point of the exercise, not a formality.
"""

import os
import re

import numpy as np
import pytest

from poraque.fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from poraque.fields.vasp.potcar import Potcar
from poraque.physics import EnergyCalculator
from poraque.physics.energy import alpha_z_energy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VASP_DIR = os.path.join(_ROOT, "data", "vasp")
CACHE_DIR = os.path.join(_ROOT, "data", "cache", "res32")

#: Where the molecular / cohesive references live, if anywhere.
REFERENCE_DIR = os.environ.get("PORAQUE_REFERENCE_DATA",
                               os.path.join(_ROOT, "data", "reference"))

#: The two structures whose full VASP energy decomposition was kept.
WITH_OUTCAR = ("struct_015", "struct_016")


# ===================================================================== #
# Reading the reference
# ===================================================================== #
def vasp_total_energy(directory):
    """Free energy ``F`` of the last ionic step, from ``OSZICAR``."""
    text = open(os.path.join(directory, "OSZICAR")).read()
    matches = re.findall(r"F=\s*([-.\dE+]+)", text)
    if not matches:
        raise ValueError(f"No 'F=' line in {directory}/OSZICAR.")
    return float(matches[-1])


def outcar_terms(path):
    """
    The ``Free energy of the ion-electron system`` block, as a dict.

    Only the terms with a Poraquê counterpart are pulled out, plus ``TOTEN``.
    """
    block = open(path).read().split("Free energy of the ion-electron system")[-1]
    terms = {}
    for key in ("PSCENC", "TEWEN", "DENC", "XCENC", "EBANDS", "EATOM"):
        match = re.search(rf"{key}\s*=\s*([-\d.]+)", block)
        if match:
            terms[key] = float(match.group(1))
    terms["TOTEN"] = float(
        re.search(r"free energy\s+TOTEN\s*=\s*([-\d.]+)", block).group(1))
    return terms


def reference_components(name, functional="pbe"):
    """
    Energy decomposition built from VASP's *own* fields for ``name``.

    Uses the spectrally downsampled cache rather than the native grid: the
    three terms compared against the ``OUTCAR`` are either grid-independent
    (:math:`E_{\\alpha Z}`, Ewald) or converged well before 32³ (Hartree), and
    reading seventeen 128³ files would dominate the suite's runtime.
    """
    cache = os.path.join(CACHE_DIR, name)
    vasp = os.path.join(VASP_DIR, name)

    potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
    density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                 grid=potential.grid)
    tau = KineticEnergyDensity.read(os.path.join(cache, "TAUCAR"),
                                    grid=potential.grid)

    potcar = Potcar.from_file(os.path.join(vasp, "POTCAR"), parse_tables=True)
    calculator = EnergyCalculator(
        grid=potential.grid,
        structure=potential.structure,
        charges={entry.element: entry.zval for entry in potcar},
        pscore={entry.element: entry.pscore for entry in potcar
                if entry.pscore is not None},
        functional=functional,
    )
    return calculator.compute(density, tau, potential), potential.structure


def _dataset_available():
    return os.path.isdir(CACHE_DIR) and os.path.isdir(VASP_DIR)


needs_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="the shipped VASP dataset is not present in this checkout")


# ===================================================================== #
# Layer 1: terms that must agree with VASP outright
# ===================================================================== #
@needs_dataset
class TestTermsSharedWithVasp:
    """
    :math:`E_{\\alpha Z}`, Ewald and Hartree are the *same* quantity in both
    codes, so a disagreement is a bug rather than an approximation.
    """

    @pytest.fixture(params=WITH_OUTCAR)
    def case(self, request):
        name = request.param
        outcar = os.path.join(VASP_DIR, name, "OUTCAR")
        if not os.path.isfile(outcar):
            pytest.skip(f"{name} carries no OUTCAR")
        components, _ = reference_components(name)
        return components, outcar_terms(outcar)

    def test_alpha_z_matches_pscenc(self, case):
        """
        The G = 0 remainder is VASP's ``alpha Z``.

        It depends only on the cell volume, the species counts, ``PSCORE`` and
        the nominal electron count — no grid, no density — so there is nothing
        to converge and the two numbers should agree to round-off.
        """
        components, vasp = case
        assert components.alpha_z == pytest.approx(vasp["PSCENC"], abs=1e-2)

    def test_ewald_matches_tewen(self, case):
        """Ion-ion electrostatics, likewise grid-independent."""
        components, vasp = case
        assert components.ewald == pytest.approx(vasp["TEWEN"], rel=1e-4)

    def test_hartree_magnitude_matches_denc(self, case):
        r"""
        VASP prints :math:`-E_{\rm H}` as ``DENC``, the double-counting
        correction to its band-energy expression. Poraquê builds the total from
        the direct expression instead, where the same quantity enters with the
        opposite sign, so the *magnitudes* are what correspond.
        """
        components, vasp = case
        assert components.hartree == pytest.approx(-vasp["DENC"], rel=1e-3)

    def test_the_shared_terms_do_not_explain_the_total(self, case):
        """
        The residual is the physics Poraquê cannot see.

        Everything above matches, yet the total is ~1e3 eV per atom from
        ``TOTEN``. Pinning that here keeps the caveat honest: the gap is the
        PAW one-centre and non-local energy plus ``EATOM``, not a slipped
        factor in a term that is already verified.
        """
        components, vasp = case
        assert abs(components.total - vasp["TOTEN"]) > 1e4


# ===================================================================== #
# Layer 2: energy differences from VASP's own fields
# ===================================================================== #
@pytest.fixture(scope="module")
def measured():
    """Poraquê and VASP totals for every structure, from reference fields."""
    rows = []
    for name in sorted(os.listdir(CACHE_DIR)):
        if not name.startswith("struct_"):
            continue
        if not os.path.isdir(os.path.join(VASP_DIR, name)):
            continue
        components, structure = reference_components(name)
        rows.append({
            "name": name,
            "natoms": structure.natoms,
            "poraque": components.total,
            "vasp": vasp_total_energy(os.path.join(VASP_DIR, name)),
        })
    if len(rows) < 2:
        pytest.skip("need at least two reference structures")
    return rows


@needs_dataset
class TestDifferencesFromReferenceFields:
    """
    The ceiling on :math:`\\Delta E`, with model error removed entirely.

    Every number here comes from VASP's own ``CHGCAR`` and ``TAUCAR``, so what
    is left is the energy expression and nothing else.
    """

    def test_the_offset_is_almost_a_per_atom_constant(self, measured):
        """
        What makes differences meaningful at all.

        The missing one-centre terms are dominated by a per-atom constant. If
        that were not so, no subtraction would rescue anything. It holds to
        about a part in 1e4 — which is the good news and, at 1e3 eV/atom, also
        the bad news measured by the next test.
        """
        offsets = np.array([(r["poraque"] - r["vasp"]) / r["natoms"]
                            for r in measured])
        assert offsets.std() / abs(offsets.mean()) < 1e-3

    def test_delta_e_error_is_bounded(self, measured):
        """
        Characterization, not a physics claim.

        With exact fields the residual error on :math:`\\Delta E` is a few
        tenths of an eV per atom. That is the neglected PAW one-centre and
        non-local energy, which varies with the environment. The bound is set
        just above the measured value so that a regression in the energy
        expression — which would move it by orders of magnitude — fails here,
        while the known physics limitation does not.
        """
        for natoms in sorted({r["natoms"] for r in measured}):
            group = [r for r in measured if r["natoms"] == natoms]
            if len(group) < 2:
                continue
            reference = group[0]
            errors = np.array([
                (r["poraque"] - reference["poraque"])
                - (r["vasp"] - reference["vasp"])
                for r in group[1:]
            ])
            per_atom = np.abs(errors) / natoms
            assert per_atom.mean() < 0.5, (
                f"{natoms}-atom group: mean |dE| error "
                f"{per_atom.mean():.4f} eV/atom")

    def test_electron_count_is_exact_for_reference_densities(self, measured):
        """A VASP density holds the count its POTCARs fix; nothing to repair."""
        for row in measured:
            components, _ = reference_components(row["name"])
            assert components.electron_drift == pytest.approx(0.0, abs=1e-6)


# ===================================================================== #
# The G = 0 prefactor
# ===================================================================== #
@needs_dataset
class TestAlphaZUsesTheNominalCount:
    r"""
    :math:`E_{\alpha Z}` must be scaled by ``NELECT``, not by
    :math:`\int\rho`.

    The two coincide for a reference density, which is exactly why the bug
    this guards against is invisible until a *predicted* density is used. The
    prefactor multiplies a quantity of order 1e3 eV, so a 1 % drift moves the
    total by more than any energy difference being sought.
    """

    def test_prefactor_is_independent_of_the_predicted_density(self):
        components, _ = reference_components(WITH_OUTCAR[0])
        scaled = _components_with_scaled_density(WITH_OUTCAR[0], 1.02)
        assert scaled.alpha_z == pytest.approx(components.alpha_z, rel=1e-12)

    def test_drift_is_reported_rather_than_absorbed(self):
        scaled = _components_with_scaled_density(WITH_OUTCAR[0], 1.02)
        assert scaled.electron_drift == pytest.approx(0.02, rel=1e-6)

    def test_alpha_z_still_scales_with_an_explicit_count(self):
        """The function itself is unchanged; only its caller's choice is."""
        _, structure = reference_components(WITH_OUTCAR[0])
        one = alpha_z_energy(structure, {"Au": 100.0}, 10.0)
        two = alpha_z_energy(structure, {"Au": 100.0}, 20.0)
        assert two == pytest.approx(2.0 * one)


def _components_with_scaled_density(name, factor):
    """Recompute ``name`` with its density multiplied by ``factor``."""
    cache = os.path.join(CACHE_DIR, name)
    vasp = os.path.join(VASP_DIR, name)

    potential = ExternalPotential.read(os.path.join(cache, "EXTCAR"))
    density = ChargeDensity.read(os.path.join(cache, "CHGCAR"),
                                 grid=potential.grid)
    tau = KineticEnergyDensity.read(os.path.join(cache, "TAUCAR"),
                                    grid=potential.grid)
    potcar = Potcar.from_file(os.path.join(vasp, "POTCAR"), parse_tables=True)
    calculator = EnergyCalculator(
        grid=potential.grid,
        structure=potential.structure,
        charges={entry.element: entry.zval for entry in potcar},
        pscore={entry.element: entry.pscore for entry in potcar
                if entry.pscore is not None},
    )
    return calculator.compute(density.data * factor, tau, potential)


# ===================================================================== #
# Layer 3: molecular and cohesive references
# ===================================================================== #
#: ``label -> (directory, formula-unit atom count)`` for the reference set.
MOLECULAR_CASES = {
    "n2": ("n2", "n_atom", 2),
    "co": ("co", "c_atom", 1),
}


def _reference_case(*names):
    """Skip unless every named reference directory is present and complete."""
    directories = []
    for name in names:
        path = os.path.join(REFERENCE_DIR, name)
        required = [os.path.join(path, f) for f in ("POSCAR", "POTCAR", "OSZICAR")]
        if not all(os.path.isfile(f) for f in required):
            pytest.skip(
                f"no reference calculation under {path}; see this module's "
                f"docstring for the expected layout")
        directories.append(path)
    return directories


def _poraque_energy(directory):
    """Single-point Poraquê energy for a VASP input directory."""
    from ase.io import read as ase_read

    from poraque.calculator import Poraque
    from poraque.ml import BUNDLE_FILENAME, resolve_bundle_path

    bundle = resolve_bundle_path(os.path.join(_ROOT, "models", BUNDLE_FILENAME))
    if not os.path.isfile(bundle):
        pytest.skip("no trained model bundle to run inference with")

    atoms = ase_read(os.path.join(directory, "POSCAR"))
    atoms.pbc = True
    atoms.calc = Poraque(models=bundle,
                         potcar=os.path.join(directory, "POTCAR"),
                         device="cpu")
    return atoms.get_potential_energy()


class TestMolecularBindingEnergies:
    """
    :math:`N_2` and :math:`CO` against their isolated atoms.

    The binding energy is a difference between systems of *different*
    composition, so the per-atom offset that cancels within a composition does
    **not** cancel here. This is the hardest thing asked of the calculator and
    the tolerance says so.
    """

    @pytest.mark.parametrize("molecule,atom,multiplicity",
                             [("n2", "n_atom", 2), ("co", "c_atom", 1)])
    def test_binding_energy(self, molecule, atom, multiplicity):
        directories = _reference_case(molecule, atom)
        if molecule == "co":
            directories = _reference_case("co", "c_atom", "o_atom")

        vasp = vasp_total_energy(directories[0]) - sum(
            multiplicity * vasp_total_energy(d) for d in directories[1:])
        poraque = _poraque_energy(directories[0]) - sum(
            multiplicity * _poraque_energy(d) for d in directories[1:])

        assert poraque == pytest.approx(vasp, abs=1.0), (
            f"{molecule}: binding energy {poraque:.3f} eV against VASP "
            f"{vasp:.3f} eV")


class TestDiamondCohesiveEnergy:
    """Bulk carbon against the isolated atom, per atom."""

    def test_cohesive_energy(self):
        bulk_dir, atom_dir = _reference_case("diamond", "c_atom")

        from ase.io import read as ase_read

        natoms = len(ase_read(os.path.join(bulk_dir, "POSCAR")))
        vasp = (vasp_total_energy(bulk_dir) / natoms
                - vasp_total_energy(atom_dir))
        poraque = (_poraque_energy(bulk_dir) / natoms
                   - _poraque_energy(atom_dir))

        assert poraque == pytest.approx(vasp, abs=0.5), (
            f"diamond cohesive energy {poraque:.3f} eV/atom against VASP "
            f"{vasp:.3f} eV/atom")
