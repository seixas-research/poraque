# ext2chg + Harris–Foulkes

`poraque.physics.harris.HarrisEvaluator` builds the fixed-density Hamiltonian
and diagonalizes it once. `poraque.ml.harris.Ext2ChgHarris` bridges ASE atomic
coordinates, species and periodic cell through **only ext2chg** to that evaluator.
`scripts/poraque_harris.py` is the standalone CHON test/benchmark driver.

```bash
python scripts/poraque_harris.py \
  --models models/chon/chon_ext2paw_charge.poraque \
  --potcar-dir data/CHON/isolated_atoms \
  --shape 16 16 16 --repair-density --spin-average \
  --scf --mixing 0.05 --max-scf-steps 600 --output harris_chon.json
pytest -q tests/test_harris.py
```

The default molecules are methane, ammonia, water and formaldehyde (ASE names
CH4, NH3, H2O, H2CO). Use `--molecules` to select systems. The script records
per-system failures, continues to the remaining systems, saves JSON and exits
nonzero if any evaluation or reference fails. `--scf` is optional.

## Mathematical and numerical contract

The spin-degenerate occupations sum to the pseudopotential valence count.
All terms in eV are evaluated on one uniform periodic grid, with lengths in
Angstrom and positive electron density in electrons/Angstrom^3:

```
H_in = -hbar^2/(2m) Laplacian + V_ext + V_H[n_in] + V_xc[n_in]
E_band = sum_i f_i epsilon_i
E_H = 1/2 integral(n_in V_H)
E_Harris = E_band - integral(n_in V_H) + E_H
                     - integral(n_in V_xc) + E_xc + E_ion-ion
```

There is no separate electron/external-potential energy to add: it is already
in the band energy. Hartree uses the existing FFT Poisson solver, G=0 set to
zero; the ionic Ewald term uses the matching neutralizing-background convention.
XC is unpolarized LDA (Slater exchange + PW92 correlation); its potential is
the derivative of the same energy implementation, evaluated with float64 torch
autograd. Integrals use float64 compensated summation (`math.fsum`). Eigenpairs
are checked by their absolute residual in eV; ARPACK failure propagates.
The matrix-free FFT Hamiltonian uses the full mesh collocation basis. Very small
meshes use dense diagonalization for a reliable reference. Dense near-full
occupations above 2048 basis functions are rejected to bound memory use.

The orbital output density is a diagnostic. **It never replaces n_in in the
Harris double-counting corrections.** The low-level evaluator rejects invalid,
negative or incorrectly normalized densities. The sanity check and the repair
live in one module, `poraque.physics.density_repair`, shared with the PAW
Bloch solver and the VASP benchmark's SCF diagnostic: `inspect_density`
measures the defects (negative charge and voxel fraction, minimum, integral
against the POTCAR valence count, signed defect) without touching the array;
`require_valid_density` raises naming every one of them; `repair_density`
floors the density at zero (a `floor` argument admits a positive ε) and
rescales by one global factor to the valence count. With `--repair-density`
the bridge calls the latter and keeps the `DensityRepair` record
(`HarrisPrediction.repair`, `as_dict()` for a report) beside the raw array;
the global factor is reported because it shifts the eigenvalues and the Fermi
level of the Hamiltonian built from the repaired density. This is an
explicitly transformed ext2chg prediction, with no reference density or
atomic-density fallback. Without that option, unphysical predictions fail
with the same diagnostics in the message. A predicted magnetization requires
explicit `--spin-average` for this unpolarized backend.

## Scope and interpretation

This is a **Gamma-only local-pseudopotential LDA** implementation. POTCAR supplies
the same local input potential the network expects, but PAW nonlocal projectors,
overlap operator, augmentation, partial core XC, and PAW reference-energy offsets
are absent. Predicted PAW occupancies are not used. Absolute energies are not
VASP PAW total energies. PBE-trained densities evaluated with LDA introduce an
additional functional mismatch. Molecular cells are periodic, with image
interactions; vary both mesh and vacuum to establish convergence. General
k-points, finite-temperature entropy, magnetism and forces are not implemented.

The optional reference is a genuine self-consistent solution of **the same local
LDA Hamiltonian**, starting from an independent uniform density, with simple
linear mixing. It is not a production PAW SCF benchmark. Only a converged
reference generates an energy error or speedup. The SCF timer covers its solves;
the Harris timer includes geometry/potential/Ewald construction and inference,
while model loading is separately reported. Single-run timings, especially
large iteration counts with simple mixing, do not establish production speedup.
The input model was trained on periodic CHON structures; these isolated molecules
and changed mesh may be outside its training distribution.

## Observed smoke evaluation

`chon_16_smoke.json` records the command above on the available CPU/checkpoint.
`chon_16_pipeline.json` records a separate successful run without SCF references.
All four Harris calculations returned finite energies and eigenpair residuals
below 2.5e-8 eV. The model required substantial corrections: 1.64–2.76 electrons
of negative density were clipped. These predictions are not chemically accurate.

| Molecule | Harris (eV) | local SCF reference | Harris minus SCF (eV) |
|---|---:|---|---:|
| CH4 | -217.456608 | failed (600 iterations, residual 1.87e-5 e) | — |
| NH3 | -312.056419 | converged (332 iterations) | -4.665898 |
| H2O | -433.056633 | converged (332 iterations) | +3.493064 |
| H2CO | -609.715797 | failed (600 iterations, residual 0.359 e) | — |

Nominal timing ratios for NH3/H2O were about 226/100, dominated by this simple
SCF scheme. They are diagnostics only. No chemical-accuracy or production-speed
claim follows. The tests separately check the uniform analytic limit, gauge
shifts, Hartree double counting, XC derivatives, density validation, preservation
of the input, dense/iterative eigensolver agreement, strict ext2chg selection,
explicit density processing, and legacy checkpoint-name compatibility.

The input-density convention follows the
[VASP Harris–Foulkes documentation](https://vasp.at/wiki/Harris-Foulkes_functional).
