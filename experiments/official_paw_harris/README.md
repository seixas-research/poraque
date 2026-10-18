# Official VASP PAW spin/PBE/SOC Harris solver

A valid VASP license is required. `VaspBlochHarris` and `poraque-bloch` require
explicit user-provided POTCAR files at runtime. Poraque does not bundle,
generate, download, copy, or locate potential data automatically. Earlier
fixed-root behavior has been removed. The UPF/NC low-level reference is not a
fallback for this PAW pipeline. Reports contain only results and source hashes.

## Entry points

- `poraque.fields.vasp.user_paw.UserPawDataset`: explicit runtime file loader,
  license warning, functional matching, structural validation and source SHA256.
- `poraque.fields.vasp.paw_setup.PawSetup`: local tables, DION, reciprocal/real
  projectors, partial waves, AE/pseudo cores, kinetic tables, Q_ij and reference
  atomic occupancy data. Optional pseudized kinetic tables remain absent when
  not provided (e.g. the installed H dataset); no table is invented.
- `poraque.physics.spin_xc`: Cartesian FFT gradients and variational spin LDA/PBE.
- `poraque.physics.paw_variational`: PAW Gaunt maps, one-center functional
  derivatives, compensation fields and on-site radial SOC.
- `poraque.physics.magnetic_bloch`: spinor matrices and generalized eigensolves.
- `poraque.physics.paw_bloch`: official PAW H/S and Harris double counting.
- `poraque.ml.bloch.VaspBlochHarris`: mandatory ext2chg charge, magnetization
  and **predicted** on-site occupancy records. Atomic reference records are
  never used as an inference fallback.

## Run

Place POSCAR, INCAR, KPOINTS and optionally a Line-Mode KPATH in `calculation`.
For collinear PBE:

```text
ENCUT = 400
NBANDS = 24
GGA = PE
ISPIN = 2
LASPH = .TRUE.
ISMEAR = -1 ; SIGMA = 0.05
EDIFF = 1D-8
```

For SOC, set `LSORBIT=.TRUE.` and `SAXIS=0 0 1`; SOC implies spinors.
`LNONCOLLINEAR=.TRUE.` also selects spinors without requiring SOC. `ISPIN=1`
with neither spinor flag uses scalar, spin-degenerate orbitals. `GGA=PE` selects
PBE; omitting GGA selects LDA and requires matching official LDA datasets.

```bash
python scripts/poraque_bloch.py calculation --models models/ext2chg.poraque \
  --potcar /licensed/inputs/POTCAR --bands calculation/KPATH --output electronic_structure.json
```

The installed entry point is `poraque-bloch`. `--potcar` is an explicit
user-supplied concatenated file matching POSCAR species order. The Python API
also accepts a species-to-external-file mapping. Matching LDA or PBE datasets
must be supplied by the licensed user; PBE potentials are never relabeled LDA.
`OfficialPawDataset` remains a compatibility alias, without a built-in root or
authenticity/license certification. Keep runtime POTCAR files out of source
control and distributions. See [benchmark and compliance documentation](../vasp_benchmark/README.md).

```python
from poraque.ml.bloch import VaspBlochHarris

solver = VaspBlochHarris("calculation", "model.poraque", "/licensed/inputs/POTCAR")
result = solver.total_energy()
bands = solver.bands("calculation/KPATH")
print(result.total, result.free_energy, result.spin_mode)
print(solver.evaluator.last_terms)  # includes both one-center energy and its subtraction
```

The model must cover the species and exactly match the official projector
channels. Density-only checkpoints fail, because a PAW Hamiltonian requires
on-site records as well. MAGMOM does not replace model magnetization. A
charge-only checkpoint needs explicit `--unpolarized-input` for a magnetic job;
this embeds its density/records with zero magnetization. A collinear model can
supply SOC input by embedding its predicted m_z and on-site magnetic records
along SAXIS. That does not constitute a prediction of transverse magnetization.
Four-channel inference decodes `(n,mx,my,mz)` in fixed Cartesian coordinates.
Existing collinear checkpoints do not gain noncollinear training accuracy.

`--repair-density` explicitly clips negative eigen-densities `(n±|m|)/2`,
normalizes charge and magnetization together, and scales on-site records by the
same global factor (`poraque.physics.density_repair.repair_density`, the one
repair shared with the Gamma evaluator and the benchmark's SCF diagnostic;
`VaspBlochHarris.repair` holds the record, `density_diagnostics` its
dictionary: `raw_electrons`, `charge_defect`, `negative_electrons`,
`negative_voxel_fraction`, `minimum_raw`, `clipped_electrons`,
`normalization_factor`, `input_electrons`, `repaired`). The raw density,
magnetization and on-site records remain available. Local on-site record
positivity is not inferred from grid repair. Without the flag a defective
prediction fails before the Hamiltonian is built, with the same diagnostics
in the message. `--spin-average` is permitted only for a scalar calculation;
it cannot discard magnetization in a magnetic job.

## XC gradients and spin-density matrices

Use `R(r)=(n I+m.sigma)/2`, in electrons/Angstrom^3. FFT differentiation on the
periodic cell uses `grad n = IFFT(i G FFT(n))`, with Cartesian G for skew cells.
The real even-grid derivative uses a skew-adjoint Nyquist convention. LDA/PBE
pointwise energies are the existing Slater/PW92 and spin-PBE implementations.

For GGA the derivative is

```
v_sigma = partial e_xc/partial n_sigma
          - div(partial e_xc/partial grad n_sigma).
```

Float64 autograd differentiates through the complete FFT/radial/angular gradient
operators, including the divergence. Grid XC includes the official partial
pseudo-core density; only the valence n and m enter double counting, since the
core is fixed. Native VASP length/energy/electrostatic constants are used in the
PAW path. The exact discrete contraction is

```
int Tr(R V_xc) = int (n v_xc + m.B_xc).
```

There is no extra standalone GGA subtraction: gradient derivatives are already
inside v_xc and B_xc. For noncollinear PBE, use local eigen-densities
`n_±=(n±|m|)/2` and their gradients. This is the **locally collinear extension**;
it omits transverse spin stiffness/spin-current functionals and is not a unique
general noncollinear GGA. The same approximation is used on grid and spheres.

## Hamiltonian, overlap and SOC

At each k use cutoff plane waves and official tabulated projectors P:

```
S = I + sum_a P_a Q_a P_a†
H = T + V_local + V_H + V_xc + sum_a P_a D_a P_a†
H c = epsilon S c,       c† S c = 1.
```

Q is the AE-minus-pseudo partial-wave overlap. All radial/angular channels and
complex atomic phases are retained. The density-dependent strengths include
both the derivative of the PAW one-center functional and the derivative of the
compensation density integrated against the effective grid potential. Bare
POTCAR DION alone is insufficient. The compensation G=0 monopole moment is
retained exactly, preserving agreement of electron count with the overlap trace.

Collinear up/down problems are solved separately with a shared chemical
potential. Spinors use spin-major ordering `[all up G, all down G]`:

```
H_spinor = [[H0+Bz, Bx-iBy], [Bx+iBy, H0-Bz]]
           + diag(P,P) [[D0+Dz,Dx-iDy],[Dx+iDy,D0-Dz]] diag(P,P)†
           + diag(P,P) D_SOC diag(P,P)†
S_spinor = diag(S,S).
```

For each same-l partial-wave pair, the spherical on-site SOC matrix is
`D_SOC,ij = integral dr u_i(r) xi(r) u_j(r) * L.S`, where

```
L.S = 1/2 [[Lz, L_-], [L_+, -Lz]]
xi(r) = hbar²/[2(m_e c)²] * (1-V(r)/(2 m_e c²))^-2 * (dV/dr)/r.
```

V is the reconstructed absolute spherical AE potential (nucleus, frozen core,
input valence and XC), not the local pseudo-potential alone. The angular operator
is transformed to the official real-harmonic basis. This includes explicit
spin-flip blocks and uses AE partial waves from the official POTCAR. SOC is
frozen at the input density for the one-shot calculation. This is the spherical
on-site ZORA PAW approximation, not a full four-component Dirac calculation.

Explicit spin states have occupations <=1; scalar states have degeneracy two.
Gaussian/Fermi-Dirac smearing and internal/free energies remain distinct.
Collinear bands have shape `(Nk,2,Nbands)` and channel labels; spinor bands have
shape `(Nk,Nbands)`. Physical symmetries determine whether SOC bands split.

## PAW Harris energy

The model grid is the CHGCAR compensated density `n_T=n_tilde+n_hat`, not the AE
density. Let E1[R] be the full one-center correction evaluated from predicted
charge/magnetic LM records. The implemented band-energy expression is

```
E_Harris = E_band - E_H[n_T] - int(n_T v_xc + m_T.B_xc) + E_xc[n_T+n_core]
           + sum_a (E1_a[R_in] - sum_K R_in,K * partial E1_a/partial R_K)
           + E_alphaZ + E_Ewald + E_atomic.
F_Harris = E_Harris - entropy_term.
```

E1 includes Dbar, AE-minus-pseudo Hartree, compensation charges, full angular
XC, frozen-core XC subtraction and the reference terms already defined by
Poraque's PAW implementation. Its derivative is taken from the **same discrete
functional**, not a guessed projector strength. All nonlinear corrections use
the ext2chg input records/density, never the orbital output. SOC is in the band
term and is not added a second time. The diagnostic output reconstructs smooth
spinor density plus orbital-derived PAW compensation and checks its integral.

## Validation and limits

```bash
pytest -q tests/test_spinor_paw.py tests/test_bloch.py tests/test_harris.py \
  tests/test_vasp_inputs.py tests/test_fields.py tests/test_paw_energy.py tests/test_paw.py \
  tests/test_hartree.py tests/test_xc_potential.py tests/test_paw_operator.py tests/test_ml.py
python experiments/harris_bloch/demo.py \
  --models models/chon/chon_ext2paw_charge.poraque \
  --potcar /licensed/inputs/H/POTCAR \
  --output-dir /tmp/poraque-paw-demo --mode spinor
```

The regression run passed 390 tests with 16 skips (unavailable reference data or
accelerators). Spin tests check PBE/LDA functional derivatives, skew-cell
analytic gradients, vacuum, spin rotations, L.S spectra, Gaunt contractions,
one-center derivatives, SOC Hermiticity/spin flips (including official Bi),
Kramers pairing, S-normalization, charge conservation, scalar/collinear/spinor
zero-SOC equivalence and the full PAW Harris identity.

`collinear_smoke.json` and `spinor_smoke.json` contain real checkpoint runs with
the official H potential. They copy only results/provenance, no potential tables.
They use deliberately unconverged ENCUT=20 eV and a 9^3 mesh for software smoke
checks. No chemical accuracy or production performance conclusion follows.

This is an experimental dense reference solver. Exact **discrete** one-center
terms/derivatives are included; agreement with VASP SOC total energies has not
been established. Required remaining validation includes converged matched
VASP reference calculations and magnetic/heavy-element-trained ext2chg models.
Numerical projector interpolation, radial/angular/FFT quadrature and cutoffs
remain approximations. LASPH=False, hybrid/meta-GGA/U, fixed NUPDOWN, spin spirals
and forces are unsupported. Dense matrices are bounded and compensation caches
over 1 GB are rejected. ENCUT outside the official basis-error table is rejected
instead of extrapolating atomic energy corrections.

The spin-density convention follows the [VASP noncollinear documentation](https://vasp.at/wiki/LNONCOLLINEAR),
and the spherical on-site SOC convention is described in the [VASP SOC documentation](https://vasp.at/wiki/LSORBIT).
