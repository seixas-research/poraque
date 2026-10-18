> Historical NC prototype documentation. The current `VaspBlochHarris` and
> `poraque-bloch` interfaces are official-VASP-PAW-only; the UPF commands below
> describe the earlier interface. See [current PAW/spin/PBE/SOC usage](../official_paw_harris/README.md).
> `demo.py` now exercises the official PAW pipeline.

# k-dependent Harris–Foulkes and bands

This extends the local Gamma prototype with cutoff plane waves, full (unreduced)
Brillouin-zone meshes and **explicit scalar norm-conserving nonlocal projectors**.
The density is generated once by `ext2chg` and held fixed for both energy and
band calculations. There is no SCF loop or reference-density fallback.

## Modules

- `poraque.fields.vasp.incar`: quoted values, comment/semicolon handling,
  backslash continuation, Fortran exponents/logicals and repetition lists;
  typed ENCUT, ISMEAR, SIGMA, EDIFF, NELM and FFT mesh controls.
- `poraque.fields.vasp.kpoints`: Gamma/Monkhorst-Pack automatic meshes, shifts,
  explicit weighted points, and paired Line-Mode segments. Cartesian coordinates
  require the POSCAR length scale; returned coordinates are fractional reciprocal.
- `poraque.physics.pseudopotentials`: XML UPF v2 scalar NC radial tables, local
  Fourier transforms and separable projector matrices.
- `poraque.physics.bloch`: cutoff basis, Bloch Hamiltonians, global occupations,
  Harris internal/free energies and band paths.
- `poraque.ml.bloch.VaspBlochHarris`: POSCAR/INCAR/KPOINTS + UPFs + ext2chg pipeline.
- `scripts/poraque_bloch.py`: standalone CLI with structured JSON output.

## Inputs and usage

Place POSCAR, INCAR and KPOINTS in a calculation directory. Provide one matching
UPF v2 NC dataset per species. Its local potential supplies the **ext2chg input**;
the same dataset supplies the nonlocal projectors. Do not pair a POTCAR local
potential with unrelated NC projectors.

Example INCAR (energies in eV):

```text
ENCUT = 40
NBANDS = 8
ISMEAR = -1 ; SIGMA = 0.05
EDIFF = 1D-8 ; NELM = 60
ISPIN = 1
```

The default mesh is sized conservatively from ENCUT with the accurate FFT rule.
An explicit density mesh requires all of NGXF, NGYF, NGZF. An undersampled mesh
fails instead of aliasing plane-wave potential matrix elements. Changing density
resolution relative to training can degrade model predictions.

KPOINTS for energy:

```text
Full Monkhorst-Pack mesh
0
Monkhorst-Pack
2 2 2
0 0 0
```

An additional KPATH file for bands:

```text
Gamma-X-M
20
Line-mode
Reciprocal
0.0 0.0 0.0 ! Gamma
0.5 0.0 0.0 ! X

0.5 0.0 0.0 ! X
0.5 0.5 0.0 ! M
```

```python
import json
from poraque.ml.bloch import VaspBlochHarris

solver = VaspBlochHarris(
    "calculation", "models/nc_ext2chg.poraque",
    {"C": "pseudos/C.upf", "H": "pseudos/H.upf"},
)
energy = solver.total_energy()
print(energy.total, energy.free_energy, energy.fermi)  # eV
bands = solver.bands("calculation/KPATH")
bands["reference_fermi_ev"] = energy.fermi
with open("bands.json", "w") as handle:
    json.dump(bands, handle, indent=2)
```

```bash
python scripts/poraque_bloch.py calculation \
  --models models/nc_ext2chg.poraque \
  --upf C=pseudos/C.upf H=pseudos/H.upf \
  --bands calculation/KPATH --output electronic_structure.json
```

A Line-Mode KPOINTS as the primary input produces bands only. It is rejected
for energy integration. Band JSON contains eigenvalues (eV), fractional k points,
path distances (inverse Angstrom), segment starts, endpoint labels and solver
residuals. Disconnected segments do not add artificial jump distances. Energies
remain unshifted; use the mesh Fermi energy for plotting relative to a reference.

`--repair-density` explicitly clips negative predicted voxels and normalizes to
the valence electron count (or NELECT); raw count, clipped electrons and scale
are reported. Without this flag invalid predictions fail. Raw and processed
arrays remain accessible on the Python object. `--spin-average` explicitly
permits discarding predicted magnetization for this scalar solver.

## Hamiltonian and nonlocal matrix elements

For each fractional k, form a sphere of integer reciprocal vectors G satisfying
`hbar²|G+k|²/(2m) <= ENCUT`, with cell-normalized waves
`<r|G+k> = exp(i(G+k).r)/sqrt(Omega)`. Then

```text
H_GG'(k) = delta_GG' hbar²|G+k|²/(2m)
           + V_eff(G-G') + sum_a B_a(G,k) D_a B_a(G',k)*
V_eff = V_local + V_H[rho_in] + V_xc[rho_in]
B_a(G,im,k) = 4pi (-i)^l exp[-i(G+k).R_a] Y_lm(qhat)/sqrt(Omega)
               * integral r² dr j_l(|G+k|r) beta_i(r)
```

Every magnetic channel `m=-l..l` is expanded. Same-l radial channels retain the
full off-diagonal D matrix; different atoms have separate D blocks. For arbitrary
wavefunction coefficients c, its nonlocal expectation is
`(B†c)† D (B†c)`, including complex phases. No fitted scalar energy shift is
substituted for these matrix elements.

UPF r is in Bohr, Vlocal and D in Ry, and beta tables store `u(r)=r*beta(r)`.
The implementation integrates `r*u(r)*j_l(qr)` and retains the coupled beta/D
normalization. It converts the resulting Fourier coefficients to the common
Angstrom/eV basis. Radial transforms use Simpson quadrature in bounded batches;
converge radial tables/cutoff in production. The local transform separates the
Coulomb tail, keeps the finite short-range G=0 mean and uses ion Ewald with the
matching omitted Coulomb zero mode.

The dense Hermitian solver checks eigenpair residuals (1e-7 eV absolute limit).
The default maximum is 2048 plane waves per k, with preallocation checks. This
is a small-system reference implementation using NumPy/SciPy BLAS/LAPACK;
large production calculations need an iterative, distributed projector backend.
No spatial symmetry reduction is applied to the mesh.

## Energy and occupations

One global chemical potential enforces `sum_kn w_k*f_kn = NELECT`, with weights
normalized to one and occupations between zero and two. SIGMA=0 uses exact
weighted filling with equal fractional occupations for degenerate states.
ISMEAR=-1 uses Fermi-Dirac; ISMEAR=0 uses Gaussian smearing. Finite smearing
requires negligible highest-band occupation and otherwise requests more bands.
VASP's default ISMEAR=1, higher Methfessel-Paxton and tetrahedra are rejected.

```text
E_band = sum_kn w_k*f_kn*epsilon_kn
E_Harris = E_band - E_H[rho_in]
                     - integral rho_in*V_xc[rho_in] + E_xc[rho_in] + E_ion-ion
F_Harris = E_Harris - entropy_term
```

The Hartree correction is `-integral rho_in*V_H + E_H = -E_H`. All corrections
use **rho_in**, never the diagnostic orbital output density. NC Vnl is
independent of density, already included in the eigenvalues and needs no extra
Hartree/XC subtraction. Fermi-Dirac uses physical `T*S`; Gaussian uses its
corresponding broadening correction, not a physical electronic temperature.
Internal and free energies are reported separately. EDIFF/NELM are parsed and
reported as SCF metadata; they do not cause the fixed density to be iterated.

## Supported physics and validation

Only spin-unpolarized LDA/PW92 XC is currently implemented. PBE/meta-GGA/hybrid,
Hubbard U, spin-orbit, noncollinear/spin-polarized Hamiltonians, nonlinear core
corrections, PAW and ultrasoft datasets are explicitly unsupported. PAW/USPP
cannot be reduced to this NC Hamiltonian: overlap and augmentation terms are
required. UPFs should be generated with a compatible XC functional; the UPF
functional identifier is retained on each dataset for inspection. No automatic
POTCAR-to-NC conversion is attempted. Existing CHON PAW-trained checkpoints
are outside the demonstrated NC training domain; their energy accuracy remains
unvalidated here. Finite k/plane-wave/radial meshes still require convergence.

```bash
pytest -q tests/test_bloch.py tests/test_harris.py tests/test_vasp_inputs.py \
  tests/test_fields.py tests/test_hartree.py tests/test_xc_potential.py
python experiments/harris_bloch/demo.py \
  --models models/chon/chon_ext2paw_charge.poraque \
  --output-dir /tmp/poraque-bloch-demo
```

The test command passed 205 tests. It covers analytic Fourier/projector
normalization, angular channels, coupled-projector expectations, translation
phases, nonlocal eigenvalue shifts, Hermiticity, time reversal, reciprocal
periodicity, FFT alias rejection, weighted smearing/charge, double counting,
free energy and strict ext2chg input-deck integration.

`synthetic_smoke.json` records a real ext2chg invocation on a synthetic Gaussian
NC test potential (not a fitted hydrogen pseudopotential). The 2-point mesh and
5-point band path produced finite energies, with maximum residuals below
6e-15 eV; predicted charge was explicitly clipped and normalized. This is
software validation, not an accuracy or production performance benchmark.

Input conventions follow [VASP KPOINTS](https://vasp.at/wiki/index.php/KPOINTS).
Radial-data conventions follow the [QE UPF specification](https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format).
