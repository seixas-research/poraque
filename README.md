<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/poraque?color=red&style=for-the-badge)](https://pypi.org/project/poraque/)

# Poraquê

Poraquê is intended to grow into a research code for orbital-free density functional
theory (OF-DFT), Kohn-Sham DFT (KS-DFT), and frozen-density embedding (FDE).

At the moment, the repository is still mostly a scaffold, so the roadmap below is
organized as an implementation sequence rather than only a list of methods.

# Roadmap

## 1. Numerical Core and Data Model

Build the numerical foundation first. Everything else depends on this being stable.

* [ ] Define the core data structures
  - [ ] `Grid`: shape, spacing, cell vectors, volume element, periodic boundary conditions
  - [ ] `System`: ions, charges, electron count, spin setting, boundary conditions
  - [ ] `Density`: storage, normalization, positivity checks, integration utilities
  - [ ] Units and conventions: Hartree atomic units, sign conventions, energy decomposition
* [ ] Implement differential operators
  - [ ] Finite-difference gradient and Laplacian on the real-space grid
  - [ ] FFT-based reciprocal-space operators for periodic systems
  - [ ] Poisson solver for the Hartree potential
* [ ] Define a clean solver API
  - [ ] Energy functional interface
  - [ ] Functional derivative / potential interface
  - [ ] Minimization / SCF driver interface
  - [ ] Result object with energies, density, convergence history, diagnostics

Tests to add:

* [ ] Grid indexing and inverse indexing
* [ ] Integration of constant density gives the correct electron number
* [ ] Finite-difference and FFT Laplacians reproduce analytic results for plane waves
* [ ] Poisson solver reproduces known simple charge distributions
* [ ] All energies and potentials use consistent units and array shapes

## 2. Minimal OF-DFT Solver

Start with the simplest end-to-end OF-DFT calculation and make it robust.

* [ ] External potential
  - [ ] Point-charge potential for toy problems
  - [ ] Regularized ionic potential for numerical stability
* [ ] Energy terms
  - [ ] Thomas-Fermi kinetic functional
  - [ ] Hartree energy
  - [ ] Dirac exchange (LDA exchange)
* [ ] Total energy assembly
  - [ ] `E[n] = T_s[n] + E_H[n] + E_xc[n] + E_ext[n]`
  - [ ] Separate reporting of each contribution
* [ ] OF-DFT minimizer
  - [ ] Optimize with respect to `sqrt(n)` or another positivity-preserving variable
  - [ ] Enforce electron-number normalization
  - [ ] Add line search or damping for stability
  - [ ] Store convergence metrics: energy change, density residual, chemical potential

Tests to add:

* [ ] Functional derivative matches finite-difference energy derivatives
* [ ] Density remains non-negative during minimization
* [ ] Electron number is conserved after every iteration
* [ ] Total energy decreases or stabilizes under controlled minimization steps
* [ ] Uniform-density limit reproduces the expected Thomas-Fermi behavior

## 3. Improved OF-DFT Functionals

Once the minimal solver works, add the first useful physical corrections.

* [ ] Thomas-Fermi-von Weizsäcker (TFvW)
  - [ ] Full von Weizsäcker term
  - [ ] Mixing parameter support
* [ ] Better exchange-correlation support
  - [ ] LDA correlation
  - [ ] Shared XC interface usable by OF-DFT and KS-DFT
* [ ] Pauli enhancement factor models
  - [ ] Base class for generalized kinetic energy density functionals
  - [ ] Local and semilocal enhancement-factor implementations
* [ ] Local pseudopotentials (LPP)
  - [ ] Library of simple analytic local pseudopotentials
  - [ ] Input format for tabulated local pseudopotentials

Tests to add:

* [ ] TFvW reduces to TF when `lambda_vW = 0`
* [ ] von Weizsäcker term gives the expected behavior for one-orbital densities
* [ ] Functional derivatives of all added terms pass finite-difference checks
* [ ] Energies converge with grid refinement
* [ ] Reference calculations for simple atoms or jellium-like model systems

## 4. Nonlocal OF-DFT and Periodic Infrastructure

This is where OF-DFT becomes more useful for realistic condensed-phase systems.

* [ ] Reciprocal-space infrastructure
  - [ ] FFT wrappers
  - [ ] Reciprocal lattice vectors and kinetic cutoffs
  - [ ] Convolution operators
* [ ] Nonlocal kinetic energy functionals
  - [ ] Kernel-based functional framework
  - [ ] At least one nonlocal KEDF implementation
  - [ ] Efficient evaluation in reciprocal space
* [ ] Periodic solids workflow
  - [ ] Cell optimization hooks
  - [ ] Bravais lattice helpers
  - [ ] Structure input/output

Tests to add:

* [ ] Reciprocal-space operators are consistent with real-space operators
* [ ] Nonlocal kernels are translationally invariant
* [ ] Nonlocal energy and potential pass numerical derivative checks
* [ ] Convergence with respect to grid density and FFT cutoff
* [ ] Benchmark on a simple metallic solid or jellium reference

## 5. KS-DFT Infrastructure

Do not start KS-DFT until the numerical core, Hartree solver, XC interface, and
minimization/SCF diagnostics are already reliable.

Recommended strategy: reuse the real-space grid first, then add a planewave basis
later if periodic materials become the main target.

* [ ] Orbital representation
  - [ ] Real-space orbitals on the existing grid
  - [ ] Occupations, spin channels, and density reconstruction
  - [ ] Orthonormalization utilities
* [ ] Kohn-Sham Hamiltonian
  - [ ] Kinetic operator
  - [ ] External potential
  - [ ] Hartree potential
  - [ ] XC potential
* [ ] SCF machinery
  - [ ] Fixed-point SCF loop
  - [ ] Density or potential mixing
  - [ ] Convergence criteria for energy, density, and eigenvalues
  - [ ] Subspace diagonalization / eigensolver interface
* [ ] Total KS energy
  - [ ] Band energy bookkeeping
  - [ ] Double-counting corrections
  - [ ] Consistent total-energy decomposition

Tests to add:

* [ ] Orbitals remain orthonormal after each update
* [ ] Density integrates to the correct electron number
* [ ] KS total energy is internally consistent with its components
* [ ] One-electron test problem reproduces the expected exact limit
* [ ] Small-system benchmarks against trusted reference data

## 6. Pseudopotentials and Basis Extensions for KS-DFT

Only add this after basic KS-DFT is working for toy systems.

* [ ] Pseudopotentials for KS-DFT
  - [ ] Norm-conserving pseudopotentials
  - [ ] Local + nonlocal projector structure
  - [ ] Parser for standard pseudopotential formats
* [ ] Optional planewave basis
  - [ ] Reciprocal-space orbital representation
  - [ ] Kinetic cutoff handling
  - [ ] FFT transfer between real and reciprocal space
* [ ] PAW method
  - [ ] Treat this as a later-generation milestone, not a first implementation target

Tests to add:

* [ ] Pseudopotential normalization and projector consistency checks
* [ ] No obvious ghost-state pathologies in basic benchmarks
* [ ] Total energies converge with basis/grid cutoff
* [ ] Agreement with published reference values for small atoms or solids

## 7. Frozen-Density Embedding (FDE)

Introduce FDE only after both subsystem density handling and KS/OF total-energy
machinery are already dependable.

* [ ] Subsystem partitioning
  - [ ] Define subsystem objects with their own ions, densities, and solvers
  - [ ] Support active and frozen subsystems
* [ ] Embedding potential
  - [ ] Electrostatic contribution
  - [ ] Nonadditive exchange-correlation contribution
  - [ ] Nonadditive kinetic contribution
* [ ] FDE workflows
  - [ ] OF-in-OF embedding
  - [ ] KS-in-KS embedding
  - [ ] KS-in-OF or OF-in-KS mixed embedding as an advanced target
* [ ] Freeze-and-thaw cycles
  - [ ] Alternating subsystem relaxation
  - [ ] Convergence criteria for subsystem densities and total embedded energy

Tests to add:

* [ ] Subsystem densities sum to the total density
* [ ] Embedding contributions vanish in appropriate noninteracting limits
* [ ] Freeze-and-thaw lowers or stabilizes the embedded energy
* [ ] Numerical derivatives of nonadditive terms match embedding potentials
* [ ] Small dimer or weakly interacting benchmark systems against literature data

## 8. Validation, Performance, and Research Readiness

These should evolve in parallel with the physics, not only at the end.

* [ ] Validation suite
  - [ ] Regression tests for energies, densities, and convergence history
  - [ ] Reference-data folder for trusted benchmarks
  - [ ] Grid-convergence and box-size studies
* [ ] Performance
  - [ ] Sparse operators where appropriate
  - [ ] FFT acceleration
  - [ ] Profiling of Hartree, nonlocal kernels, and eigensolvers
* [ ] Usability
  - [ ] Input file format or Python API examples
  - [ ] Reproducible examples for OF-DFT, KS-DFT, and FDE
  - [ ] Error messages for invalid densities, missing parameters, and nonconvergence
* [ ] Documentation
  - [ ] Theory notes for each functional and approximation
  - [ ] Developer notes describing the code architecture
  - [ ] Benchmark notebook or script collection

## Suggested Implementation Order in This Repository

Map the roadmap to the current package structure so the code grows coherently.

* [ ] `src/poraque/core/grid.py`
  - [ ] grid geometry, spacing, integration weights, Laplacian/gradient, FFT helpers
* [ ] `src/poraque/core/system.py`
  - [ ] ions, electron counts, spin, cell, pseudopotential references
* [ ] `src/poraque/functionals/`
  - [ ] base functional API
  - [ ] Hartree, TF, vW, XC, nonlocal KEDF implementations
* [ ] `src/poraque/potentials/`
  - [ ] external potentials and pseudopotential library
* [ ] `src/poraque/engine.py`
  - [ ] OF minimizer, KS SCF driver, convergence control
* [ ] `src/poraque/calculator.py`
  - [ ] high-level user-facing API
* [ ] `tests/`
  - [ ] unit tests for operators and functionals
  - [ ] regression tests for total energies
  - [ ] integration tests for complete workflows

## Practical Milestones

If the goal is to reach working science quickly, a good milestone sequence is:

* [ ] Milestone 1: 3D grid + external potential + Hartree + Thomas-Fermi + minimizer
* [ ] Milestone 2: TFvW + Dirac exchange + stable OF-DFT examples
* [ ] Milestone 3: local pseudopotentials + periodic real-space OF-DFT
* [ ] Milestone 4: nonlocal KEDFs
* [ ] Milestone 5: minimal KS-DFT on the same grid
* [ ] Milestone 6: norm-conserving pseudopotentials
* [ ] Milestone 7: frozen-density embedding

This order keeps the hardest abstractions until the shared numerical core is already
tested and reusable.

# License

This is an open source code under [MIT License](LICENSE).
