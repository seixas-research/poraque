# -*- coding: utf-8 -*-
# file: paw_budget.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
How accurate must each PAW input be for the total energy to be worth having?

The grid-only chain's answer is already recorded in this directory's README:
the energy error tracks :math:`{\rm relL^2}(\tau)\int\tau`, and with
:math:`\int\tau \approx 7000` eV on a 32-atom platinum cell, chemical accuracy
needs :math:`{\rm relL^2}(\tau) \le 2.3\times10^{-4}`. That answer was derived
for a chain whose only outputs were two grids.

The PAW route has **three** model outputs, not two: ``ext2chg`` predicts the
pseudo-density grid *and* every atom's augmentation occupancies, and
``chg2tau`` predicts :math:`\tau`. A target accuracy for the whole is useless
without knowing how the three divide it --- an occupancy head held to the
density's tolerance may be wasted effort, or may be the binding constraint, and
nothing so far says which.

So this measures the derivative. Start from the energy evaluated on VASP's own
fields, which reproduces ``TOTEN`` to 1.2 meV on this cell, perturb **one**
input at a time to a known relative error, and read off :math:`|\Delta E|`.
Every omitted term cancels exactly between the two evaluations, so what remains
is that input's sensitivity and nothing else.

Three perturbation shapes are reported for each input, because a relative
:math:`L^2` does not determine :math:`\Delta E` on its own --- it is a norm,
and the energy is a *functional*, sensitive to which part of the error it
integrates.

``smooth`` and ``white`` are random: band-limited to the band the platinum runs
retain (:math:`g_{\max} = 16` Å⁻¹, about a third of this grid's Nyquist), and
unrestricted. Both are **incoherent** --- an error that oscillates in sign
largely cancels in an integral, so these measure the *floor* of what a given
relative error can cost.

``coherent`` scales the input by :math:`1 + \epsilon`, a systematic error of
the same relative size. That is the **ceiling**: every voxel's error adds with
the same sign, nothing cancels, and for the kinetic term the answer is exactly
:math:`\epsilon\int\tau`. A trained operator sits between the two, and this
directory's README records that it sits near the *coherent* end ---
:math:`|\Delta E|` correlated at 0.97 with
:math:`{\rm relL^2}(\tau)\int\tau`, which is the coherent bound. Quoting the
incoherent number alone would understate every tolerance by two orders of
magnitude, which is exactly the mistake this column exists to prevent.

Usage
-----
``python experiments/total_energy/paw_budget.py --potcar <POTCAR> [--json out]``
"""

import argparse
import json
import os
import time

import numpy as np

#: ``structure_0000``'s, used only when the directory has no ``OUTCAR``.
REFERENCE_TOTEN = -195.34322336
REFERENCE_ENTROPY = -0.58227919


def read_outcar_reference(directory):
    """
    ``(TOTEN, EENTRO)`` from an ``OUTCAR``, or the module defaults.

    Only ``structure_0000`` of this project's platinum set ships an
    ``OUTCAR``; for any other directory the two numbers have to come from
    somewhere, and quoting ``structure_0000``'s against a different cell would
    make the "expression error" line a fabrication.
    """
    path = os.path.join(directory, "OUTCAR")
    if not os.path.exists(path):
        return REFERENCE_TOTEN, REFERENCE_ENTROPY, False
    toten = entropy = None
    with open(path, errors="ignore") as handle:
        for line in handle:
            if "TOTEN" in line and "=" in line:
                toten = float(line.split("=")[1].split()[0])
            elif "EENTRO" in line and "=" in line:
                entropy = float(line.split("=")[1].split()[0])
    if toten is None:
        return REFERENCE_TOTEN, REFERENCE_ENTROPY, False
    return toten, (0.0 if entropy is None else entropy), True

#: The band the platinum runs retain (``model.equivariant.g_basis``, Å⁻¹).
RETAINED_BAND = 16.0


def read_inputs(directory, potcar):
    """Everything :func:`paw_total_energy` needs, read once."""
    from poraque.fields import KineticEnergyDensity, SpinDensity
    from poraque.fields.vasp.augmentation import occupancy_arrays
    from poraque.fields.vasp.paw_setup import PawSetup
    from poraque.fields.vasp.volumetric import read_augmentation_blocks

    chgcar = os.path.join(directory, "CHGCAR")
    density = SpinDensity.read(chgcar)
    charge = np.asarray(density.data[0])
    magnetization = np.asarray(density.data[1])
    tau = np.asarray(KineticEnergyDensity.read(
        os.path.join(directory, "TAUCAR"), grid=density.grid).data)
    setups = {setup.element: setup for setup in PawSetup.from_file(potcar)}
    _, blocks = read_augmentation_blocks(chgcar)
    occupancies, lengths = occupancy_arrays(blocks, channels=2)
    records = [[occupancies[atom, 0, :lengths[atom]],
                occupancies[atom, 1, :lengths[atom]]]
               for atom in range(len(lengths))]
    return {"structure": density.structure, "grid": density.grid,
            "density": charge, "magnetization": magnetization, "tau": tau,
            "records": records, "setups": setups}


def band_limited_noise(shape, cell, cutoff, generator):
    r"""
    A real random field with no content above ``cutoff`` in Å⁻¹.

    White noise truncated in reciprocal space rather than a smoothed real-space
    field: the point is a perturbation whose *band* is controlled exactly, so
    the two spectra reported are two statements about :math:`|\mathbf G|` and
    not about a kernel width.
    """
    noise = generator.normal(size=shape)
    if cutoff is None:
        return noise
    spectrum = np.fft.fftn(noise)
    # Reciprocal lattice of the cell, rows b_i, so |G| of mode (h, k, l) is
    # |h b_1 + k b_2 + l b_3|. Built here rather than borrowed from FieldGrid
    # because this needs the full fftn grid, not the rfftn half.
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(cell)).T
    waves = [np.fft.fftfreq(n) * n for n in shape]
    mesh = np.meshgrid(*waves, indexing="ij")
    squared = np.zeros(shape)
    for axis in range(3):
        component = sum(mesh[i] * reciprocal[i, axis] for i in range(3))
        squared = squared + component ** 2
    spectrum[np.sqrt(squared) > cutoff] = 0.0
    return np.fft.ifftn(spectrum).real


def scale_field(field, relative):
    r"""
    ``field`` with a **coherent** error: every voxel scaled by the same
    :math:`1 + \epsilon`, so the relative :math:`L^2` is exactly ``relative``
    and nothing cancels under an integral.
    """
    return field * (1.0 + relative)


def perturb_field(field, relative, cell, cutoff, generator, against=None):
    """
    ``field`` with a perturbation of exactly ``relative`` relative L².

    ``against`` sets the norm the perturbation is measured against, for a
    channel whose own norm is not the meaningful scale. The magnetization of a
    magnetically dead cell is the case: scaling a perturbation to 1 % of a
    field that is itself ~1e-6 of the density says nothing, whereas 1 % of the
    *density's* norm is both a real question and the denominator
    ``training.magnetization_weight`` divides by.
    """
    noise = band_limited_noise(field.shape, cell, cutoff, generator)
    reference = field if against is None else against
    scale = np.linalg.norm(reference) / max(np.linalg.norm(noise), 1e-300)
    return field + relative * scale * noise


def scale_records(records, relative):
    """Every occupancy scaled by the same ``1 + relative``: coherent."""
    return [[values * (1.0 + relative) for values in atom]
            for atom in records]


def perturb_records(records, relative, generator):
    """
    Every occupancy scaled by ``1 + relative * g``, ``g`` standard normal.

    Multiplicative, so the perturbation follows the record's own structure
    rather than swamping the small L > 0 blocks: 99 % of the norm is the L = 0
    block, and an additive perturbation scaled to the whole would be a
    many-thousand-fold relative error on the rest.
    """
    perturbed = []
    for atom in records:
        perturbed.append([
            values * (1.0 + relative * generator.normal(size=values.shape))
            for values in atom])
    return perturbed


def relative_error_of_records(reference, perturbed):
    """Realized relative error over every atom and set, for the record."""
    error = sum(float(np.sum((b - a) ** 2))
                for atom_a, atom_b in zip(reference, perturbed)
                for a, b in zip(atom_a, atom_b))
    norm = sum(float(np.sum(a ** 2))
               for atom_a in reference for a in atom_a)
    return (error / norm) ** 0.5


def energy(inputs, one_center, encut, lasph, entropy=0.0, **overrides):
    """:func:`paw_total_energy` on ``inputs``, with named fields replaced."""
    from poraque.physics.paw import paw_total_energy

    fields = {key: overrides.get(key, inputs[key])
              for key in ("density", "magnetization", "tau", "records")}
    return paw_total_energy(
        inputs["structure"], inputs["grid"], fields["density"],
        fields["magnetization"], fields["tau"], fields["records"],
        inputs["setups"], encut=encut, lasph=lasph,
        entropy=entropy, one_center=one_center)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",
                        default="data/vasp/structures/structure_0000")
    parser.add_argument("--potcar",
                        default="/Users/leseixas/Simulations/vasp/POTCARs/"
                                "Pt/POTCAR")
    parser.add_argument("--encut", type=float, default=400.0)
    parser.add_argument("--lasph", action="store_true")
    parser.add_argument("--errors", type=float, nargs="+",
                        default=[1e-3, 1e-2])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    started = time.perf_counter()
    toten, entropy, measured = read_outcar_reference(arguments.directory)
    inputs = read_inputs(arguments.directory, arguments.potcar)
    atoms = len(inputs["records"])
    cell = inputs["grid"].cell
    one_center = {}

    reference = energy(inputs, one_center, arguments.encut, arguments.lasph,
                       entropy=entropy)
    print(f"[REFERENCE] {arguments.directory}")
    print(f"  atoms                {atoms}")
    print(f"  grid                 {inputs['density'].shape}")
    print(f"  total                {reference['total']:.6f} eV")
    print(f"  VASP TOTEN           {toten:.6f} eV"
          f"{'' if measured else '   (no OUTCAR here: structure_0000 default)'}")
    if measured:
        print(f"  expression error     "
              f"{1000 * (reference['total'] - toten):.3f} meV "
              f"({1000 * (reference['total'] - toten) / atoms:.3f} meV/atom)")
    else:
        print("  expression error     not comparable without this "
              "directory's OUTCAR")
    print(f"  int tau              {reference['kinetic']:.1f} eV")
    print(f"  one-center total     "
          f"{reference['one_center_linear'] + reference['one_center_hartree'] + reference['one_center_xc']:.1f} eV")

    nyquist = np.pi * min(n / length for n, length in zip(
        inputs["density"].shape, np.linalg.norm(cell, axis=1)))
    polarization = float(np.linalg.norm(inputs["magnetization"])
                         / np.linalg.norm(inputs["density"]))
    print(f"  ||m|| / ||rho||      {polarization:.3e}   "
          f"(this cell is magnetically dead; the magnetization row is "
          f"perturbed against ||rho||)")
    print(f"  Nyquist |G|          {nyquist:.1f} 1/Ang")
    print(f"  retained band        {RETAINED_BAND:.1f} 1/Ang "
          f"({RETAINED_BAND / nyquist:.2f} of Nyquist)")

    cases = [("density", "smooth"), ("density", "white"),
             ("density", "coherent"),
             ("tau", "smooth"), ("tau", "white"), ("tau", "coherent"),
             ("magnetization", "smooth vs rho"),
             ("magnetization", "coherent"),
             ("occupancies", "multiplicative"),
             ("occupancies", "coherent")]
    results = []
    for case_index, (name, spectrum) in enumerate(cases):
        for relative in arguments.errors:
            shifts = []
            realized = []
            terms = {}
            # A coherent perturbation is deterministic, so one evaluation is
            # the whole answer rather than a sample of one.
            draws = 1 if spectrum == "coherent" else arguments.seeds
            for seed in range(draws):
                # Seeded from the case index rather than hash(), which is
                # salted per process: the run was not reproducible and the
                # JSON could not be regenerated.
                generator = np.random.default_rng(
                    [case_index, int(round(relative * 1e6)), seed])
                if spectrum == "coherent":
                    if name == "occupancies":
                        perturbed = scale_records(inputs["records"], relative)
                        realized.append(relative_error_of_records(
                            inputs["records"], perturbed))
                        override = {"records": perturbed}
                    else:
                        perturbed = scale_field(inputs[name], relative)
                        denominator = (inputs["density"]
                                       if name == "magnetization"
                                       else inputs[name])
                        realized.append(float(
                            np.linalg.norm(perturbed - inputs[name])
                            / np.linalg.norm(denominator)))
                        override = {name: perturbed}
                elif name == "occupancies":
                    perturbed = perturb_records(inputs["records"], relative,
                                                generator)
                    realized.append(relative_error_of_records(
                        inputs["records"], perturbed))
                    override = {"records": perturbed}
                else:
                    cutoff = None if spectrum == "white" else RETAINED_BAND
                    against = (inputs["density"]
                               if spectrum == "smooth vs rho" else None)
                    perturbed = perturb_field(inputs[name], relative, cell,
                                              cutoff, generator,
                                              against=against)
                    denominator = (inputs["density"] if against is not None
                                   else inputs[name])
                    realized.append(float(
                        np.linalg.norm(perturbed - inputs[name])
                        / np.linalg.norm(denominator)))
                    override = {name: perturbed}
                result = energy(inputs, one_center, arguments.encut,
                                arguments.lasph, entropy=entropy, **override)
                shifts.append(result["total"] - reference["total"])
                for key, value in result.items():
                    if key in ("total", "entropy", "atomic"):
                        continue
                    terms[key] = terms.get(key, 0.0) + abs(
                        value - reference[key]) / draws
            magnitude = np.abs(shifts)
            dominant = max(terms.items(), key=lambda item: item[1])
            results.append({
                "input": name, "spectrum": spectrum,
                "requested_relative_error": relative,
                "realized_relative_error": float(np.mean(realized)),
                "mean_abs_dE_eV": float(magnitude.mean()),
                "std_abs_dE_eV": float(magnitude.std()),
                "max_abs_dE_eV": float(magnitude.max()),
                "mean_abs_dE_meV_per_atom":
                    float(1000.0 * magnitude.mean() / atoms),
                "dominant_term": dominant[0],
                "dominant_term_mean_abs_dE_eV": float(dominant[1]),
                "seeds": draws,
            })

    print("\n[SENSITIVITY]")
    header = (f"  {'input':14s} {'spectrum':15s} {'rel err':>9s} "
              f"{'mean |dE|':>12s} {'sd':>9s} {'max |dE|':>12s} "
              f"{'meV/atom':>10s} {'dominant term':>20s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results:
        print(f"  {row['input']:14s} {row['spectrum']:15s} "
              f"{row['realized_relative_error']:9.2e} "
              f"{row['mean_abs_dE_eV']:12.4f} {row['std_abs_dE_eV']:9.4f} "
              f"{row['max_abs_dE_eV']:12.4f} "
              f"{row['mean_abs_dE_meV_per_atom']:10.1f} "
              f"{row['dominant_term']:>20s}")

    print("\n[LINEARITY]  mean |dE| at 1e-2 over mean |dE| at 1e-3; 10 is "
          "first order, 100 second")
    for name, spectrum in cases:
        pair = [row for row in results
                if row["input"] == name and row["spectrum"] == spectrum]
        if len(pair) == 2 and pair[0]["mean_abs_dE_eV"] > 0:
            ratio = pair[1]["mean_abs_dE_eV"] / pair[0]["mean_abs_dE_eV"]
            print(f"  {name:14s} {spectrum:15s} {ratio:6.1f}")

    print("\n[TOLERANCE]  relative error each input may carry alone, for "
          "1.6 eV (50 meV/atom x 32)")
    for row in results:
        if row["requested_relative_error"] != 1e-2:
            continue
        slope = row["mean_abs_dE_eV"] / row["realized_relative_error"]
        print(f"  {row['input']:14s} {row['spectrum']:15s} "
              f"{1.6 / slope:9.2e}")

    print(f"\n  elapsed {time.perf_counter() - started:.1f} s")

    if arguments.json:
        payload = {"directory": arguments.directory,
                   "reference": {k: float(v) for k, v
                                 in reference.items()},
                   "vasp_toten": toten,
                   "vasp_toten_measured_here": measured,
                   "entropy": entropy,
                   "atoms": atoms,
                   "nyquist_inverse_angstrom": float(nyquist),
                   "polarization": polarization,
                   "retained_band_inverse_angstrom": RETAINED_BAND,
                   "sensitivity": results}
        with open(arguments.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"  wrote {arguments.json}")


if __name__ == "__main__":
    main()
