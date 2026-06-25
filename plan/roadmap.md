
## Main layers

* [x] Frontend and workflow layer
  - [x] user-facing calculator API
  - [x] ASE integration
  - [x] examples and workflow helpers (benchmarks pending)
* [x] Scientific engine layer
  - [x] OF-DFT minimization
  - [x] KS-DFT SCF drivers later
  - [x] FDE drivers later
  - [x] convergence control and diagnostics
* [x] Physics model layer
  - [x] kinetic, Hartree, exchange-correlation, and embedding functionals
  - [x] external and pseudopotential models
* [x] Numerical backend layer
  - [x] finite-difference and FFT operators
  - [x] Poisson solvers
  - [x] Hartree and nonlocal KEDF kernels
  - [ ] future MPI/OpenMPI domain decomposition and halo exchange

## Package direction

The codebase should evolve toward the following responsibilities:

* [x] `src/poraque/core/`
  - [x] `grid.py`: grid geometry, reciprocal-space data, indexing, domain metadata
  - [x] `system.py`: atomic structure, electrons, spin, boundary conditions, ASE conversion
  - [x] future density, result, units, and validation objects
* [x] `src/poraque/functionals/`
  - [x] common functional interface
  - [x] kinetic, Hartree, XC, nonlocal KEDF, and ML-KEDF models
* [x] `src/poraque/potentials/`
  - [x] ionic, external, local pseudopotential, and embedding-related potentials
* [x] `src/poraque/engine.py`
  - [x] method drivers and convergence logic
* [x] `src/poraque/calculator.py`
  - [x] main public API and ASE bridge
* [x] future `src/poraque/backends/`
  - [x] NumPy reference backend
  - [ ] native accelerated kernels
  - [ ] MPI-aware distributed backend
* [ ] future `src/poraque/ml/`
  - [ ] datasets, preprocessing, symbolic regression, CNN models, and inference wrappers

## Near-term architectural priorities

* [x] Define stable `Grid`, `System`, `Density`, and `Result` objects
* [x] Introduce a `NumpyBackend` as the reference implementation
* [x] Standardize functional interfaces: energy, potential, and later forces/stress
* [x] Keep ASE logic isolated in a dedicated namespace
* [ ] Design MPI around domain decomposition, not ad hoc communication calls
* [ ] Replace hotspots with compiled kernels only after profiling the reference path

# Roadmap

## 1. Numerical Core and Data Model

Build the numerical foundation first. Everything else depends on this being stable.

* [x] Define the core data structures
  - [x] `Grid`: shape, spacing, cell vectors, volume element, periodic boundary conditions
  - [x] `System`: ions, charges, electron count, spin setting, boundary conditions
  - [x] `Density`: storage, normalization, positivity checks, integration utilities
  - [x] Units and conventions: Hartree atomic units, sign conventions, energy decomposition
  - [x] Conversion helpers between internal objects and ASE `Atoms`
* [x] Implement differential operators
  - [x] Finite-difference gradient and Laplacian on the real-space grid
  - [x] FFT-based reciprocal-space operators for periodic systems
  - [x] Poisson solver for the Hartree potential
* [x] Define a clean solver API
  - [x] Energy functional interface
  - [x] Functional derivative / potential interface
  - [x] Minimization / SCF driver interface
  - [x] Result object with energies, density, convergence history, diagnostics

Tests to add (with pytest):

* [x] Grid indexing and inverse indexing
* [x] Integration of constant density gives the correct electron number
* [x] Finite-difference and FFT Laplacians reproduce analytic results for plane waves
* [x] Poisson solver reproduces known simple charge distributions
* [x] All energies and potentials use consistent units and array shapes

## 2. ASE Integration Layer

Add ASE early so structures, workflows, and future geometry optimization can reuse a
standard ecosystem instead of a custom one.

* [x] ASE interoperability
  - [x] Read atomic structures from ASE `Atoms`
  - [x] Export internal structures back to ASE `Atoms`
  - [x] Preserve cell, periodic boundary conditions, positions, species, and charges
* [x] ASE calculator interface
  - [x] Implement a Poraquê ASE `Calculator`
  - [x] Return total energy, forces, and stress when available
  - [x] Expose OF-DFT and later KS-DFT through the same interface
* [x] ASE workflows
  - [x] Single-point energy calculations
  - [x] Geometry optimization hooks
  - [ ] Molecular dynamics compatibility as a future extension

Tests to add (with pytest):

* [x] Round-trip conversion between internal `System` objects and ASE `Atoms`
* [x] Correct handling of periodic and nonperiodic boundary conditions
* [x] ASE single-point calls return energies in expected units
* [x] Basic force-consistency checks when forces are implemented

## 3. Minimal OF-DFT Solver

Start with the simplest end-to-end OF-DFT calculation and make it robust.

* [x] External potential
  - [x] Point-charge potential for toy problems
  - [x] Regularized ionic potential for numerical stability
* [x] Energy terms
  - [x] Thomas-Fermi kinetic functional
  - [x] Hartree energy
  - [x] Dirac exchange (LDA exchange)
* [x] Total energy assembly
  - [x] `E[n] = T_s[n] + E_H[n] + E_xc[n] + E_ext[n]`
  - [x] Separate reporting of each contribution
* [x] OF-DFT minimizer
  - [x] Optimize with respect to `sqrt(n)` or another positivity-preserving variable
  - [x] Enforce electron-number normalization
  - [x] Add line search or damping for stability
  - [x] Store convergence metrics: energy change, density residual, chemical potential

Tests to add (with pytest):

* [x] Functional derivative matches finite-difference energy derivatives
* [x] Density remains non-negative during minimization
* [x] Electron number is conserved after every iteration
* [x] Total energy decreases or stabilizes under controlled minimization steps
* [x] Uniform-density limit reproduces the expected Thomas-Fermi behavior

## 4. Improved OF-DFT Functionals

Once the minimal solver works, add the first useful physical corrections.

* [x] Thomas-Fermi-von Weizsäcker (TFvW)
  - [x] Full von Weizsäcker term
  - [x] Mixing parameter support
* [x] Better exchange-correlation support
  - [x] LDA correlation
  - [x] Shared XC interface usable by OF-DFT and KS-DFT
* [ ] Pauli enhancement factor models
  - [ ] Base class for generalized kinetic energy density functionals
  - [ ] Local and semilocal enhancement-factor implementations
* [ ] Local pseudopotentials (LPP)
  - [ ] Library of simple analytic local pseudopotentials
  - [ ] Input format for tabulated local pseudopotentials

Tests to add (with pytest):

* [x] TFvW reduces to TF when `lambda_vW = 0`
* [x] von Weizsäcker term gives the expected behavior for one-orbital densities
* [x] Functional derivatives of all added terms pass finite-difference checks
* [ ] Energies converge with grid refinement
* [ ] Reference calculations for simple atoms or jellium-like model systems

## 5. Nonlocal OF-DFT and Periodic Infrastructure

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

Tests to add (with pytest):

* [ ] Reciprocal-space operators are consistent with real-space operators
* [ ] Nonlocal kernels are translationally invariant
* [ ] Nonlocal energy and potential pass numerical derivative checks
* [ ] Convergence with respect to grid density and FFT cutoff
* [ ] Benchmark on a simple metallic solid or jellium reference

## 6. Machine-Learned KEDFs (ML-KEDFs)

Use ML-KEDF as a parallel research track after the basic OF-DFT infrastructure is
stable enough to generate descriptors, evaluate energies, and compare against
reference data.

* [ ] Dataset pipeline
  - [ ] Define a dataset format for molecules, geometries, densities, and reference kinetic energies
  - [ ] Collect electron-density data for many molecules
  - [ ] Store metadata: composition, charge, spin, geometry, grid, reference method
  - [ ] Split data into training, validation, and test sets without leakage
* [ ] Density preprocessing
  - [ ] Normalize densities and align grids or resample onto a standard representation
  - [ ] Build local descriptors based on density, gradients, and Laplacians
  - [ ] Generate 2D slices of the 3D electron density for image-like CNN inputs
  - [ ] Evaluate whether multi-slice, orthogonal-slice, or full 3D tensor inputs are best
* [ ] Symbolic-regression ML-KEDF
  - [ ] Fit interpretable formulas for kinetic energy density or enhancement factors
  - [ ] Constrain candidate expressions to respect positivity, scaling, and known limits
  - [ ] Compare learned formulas against TF, TFvW, and other baseline KEDFs
* [ ] CNN-based ML-KEDF
  - [ ] Train CNN models on density slices treated as images
  - [ ] Predict local kinetic energy density, nonlocal corrections, or total kinetic energy
  - [ ] Study transferability across molecule sizes and chemical compositions
* [ ] Physics-informed constraints
  - [ ] Enforce electron-number consistency where relevant
  - [ ] Penalize violations of exact constraints and asymptotic behavior
  - [ ] Ensure the model produces a usable functional derivative or a differentiable surrogate
* [ ] Integration into the OF-DFT engine
  - [ ] Wrap symbolic-regression models as analytic KEDFs
  - [ ] Wrap CNN models as differentiable learned functionals
  - [ ] Support inference inside self-consistent minimization loops

Tests to add (with pytest):

* [ ] Dataset loading is deterministic and reproducible
* [ ] No train/validation/test leakage across related molecular geometries
* [ ] Learned models outperform simple baselines on held-out data
* [ ] Predictions are smooth enough for stable minimization
* [ ] Functional derivatives from the ML-KEDF are numerically consistent
* [ ] OF-DFT calculations with ML-KEDF remain stable on small benchmark molecules

## 7. KS-DFT Infrastructure

Do not start KS-DFT until the numerical core, Hartree solver, XC interface, and
minimization/SCF diagnostics are already reliable.

Recommended strategy: reuse the real-space grid first, then add a planewave basis
later if periodic materials become the main target.

* [x] Orbital representation
  - [x] Real-space orbitals on the existing grid
  - [x] Occupations, spin channels, and density reconstruction
  - [x] Orthonormalization utilities
* [x] Kohn-Sham Hamiltonian
  - [x] Kinetic operator
  - [x] External potential
  - [x] Hartree potential
  - [x] XC potential
* [x] SCF machinery
  - [x] Fixed-point SCF loop
  - [x] Density or potential mixing
  - [x] Convergence criteria for energy, density, and eigenvalues
  - [x] Subspace diagonalization / eigensolver interface
* [x] Total KS energy
  - [x] Band energy bookkeeping
  - [x] Double-counting corrections
  - [x] Consistent total-energy decomposition

Tests to add (with pytest):

* [x] Orbitals remain orthonormal after each update
* [x] Density integrates to the correct electron number
* [x] KS total energy is internally consistent with its components
* [x] One-electron test problem reproduces the expected exact limit
* [ ] Small-system benchmarks against trusted reference data

## 8. Pseudopotentials and Basis Extensions for KS-DFT

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

Tests to add (with pytest):

* [ ] Pseudopotential normalization and projector consistency checks
* [ ] No obvious ghost-state pathologies in basic benchmarks
* [ ] Total energies converge with basis/grid cutoff
* [ ] Agreement with published reference values for small atoms or solids

## 9. Frozen-Density Embedding (FDE)

Introduce FDE only after both subsystem density handling and KS/OF total-energy
machinery are already dependable.

* [x] Subsystem partitioning
  - [x] Define subsystem objects with their own ions, densities, and solvers
  - [x] Support active and frozen subsystems
* [x] Embedding potential
  - [x] Electrostatic contribution
  - [x] Nonadditive exchange-correlation contribution
  - [x] Nonadditive kinetic contribution
* [x] FDE workflows
  - [x] OF-in-OF embedding
  - [x] KS-in-KS embedding
  - [x] KS-in-OF or OF-in-KS mixed embedding as an advanced target
* [x] Freeze-and-thaw cycles
  - [x] Alternating subsystem relaxation
  - [x] Convergence criteria for subsystem densities and total embedded energy

Tests to add (with pytest):

* [x] Subsystem densities sum to the total density
* [x] Embedding contributions vanish in appropriate noninteracting limits
* [x] Freeze-and-thaw lowers or stabilizes the embedded energy
* [x] Numerical derivatives of nonadditive terms match embedding potentials
* [ ] Small dimer or weakly interacting benchmark systems against literature data

## 10. Validation, Performance, and Research Readiness

These should evolve in parallel with the physics, not only at the end.

* [ ] Validation suite
  - [x] Regression tests for energies, densities, and convergence history
  - [ ] Reference-data folder for trusted benchmarks
  - [ ] Grid-convergence and box-size studies
* [ ] Performance
  - [ ] Sparse operators where appropriate
  - [ ] FFT acceleration
  - [ ] Profiling of Hartree, nonlocal kernels, and eigensolvers
* [ ] Usability
  - [x] Input file format or Python API examples
  - [x] Reproducible examples for OF-DFT, KS-DFT, FDE, and ASE workflows (ML-KEDF pending)
  - [ ] Error messages for invalid densities, missing parameters, and nonconvergence
* [ ] Documentation
  - [x] Theory notes for each functional and approximation
  - [x] Developer notes describing the code architecture
  - [ ] Benchmark notebook or script collection
  - [ ] Training notes and model cards for ML-KEDF experiments

## Suggested Implementation Order in This Repository

Map the roadmap to the current package structure so the code grows coherently.

* [x] `src/poraque/core/grid.py`
  - [x] grid geometry, spacing, integration weights, Laplacian/gradient, FFT helpers
* [x] `src/poraque/core/system.py`
  - [x] ions, electron counts, spin, cell, pseudopotential references, ASE conversion hooks
* [x] `src/poraque/functionals/`
  - [x] base functional API
  - [x] Hartree, TF, vW, XC implementations (nonlocal KEDF and ML-KEDF pending)
* [x] `src/poraque/potentials/`
  - [x] external potentials (pseudopotential library pending)
* [x] `src/poraque/engine.py`
  - [x] OF minimizer, KS SCF driver, convergence control
* [x] `src/poraque/calculator.py`
  - [x] high-level user-facing API and ASE calculator bridge
* [ ] `src/poraque/ml/`
  - [ ] dataset loading, preprocessing, symbolic regression, CNN training, inference wrappers
* [x] `examples/`
  - [x] ASE single-point and geometry-optimization examples
  - [ ] ML-KEDF data-preparation and training examples
* [x] `tests/`
  - [x] unit tests for operators and functionals
  - [x] regression tests for total energies and ASE calculator behavior
  - [x] integration tests for complete workflows

## Practical Milestones

If the goal is to reach working science quickly, a good milestone sequence is:

* [x] Milestone 1: 3D grid + external potential + Hartree + Thomas-Fermi + minimizer
* [x] Milestone 2: ASE structure I/O + ASE calculator for OF-DFT single points
* [ ] Milestone 3: TFvW + Dirac exchange + stable OF-DFT examples
* [ ] Milestone 4: local pseudopotentials + periodic real-space OF-DFT
* [ ] Milestone 5: nonlocal KEDFs
* [ ] Milestone 6: ML-KEDF dataset pipeline + symbolic-regression baseline
* [ ] Milestone 7: CNN-based ML-KEDF on electron-density slices
* [x] Milestone 8: minimal KS-DFT on the same grid
* [ ] Milestone 9: norm-conserving pseudopotentials
* [x] Milestone 10: frozen-density embedding

This order keeps the hardest abstractions until the shared numerical core is already
tested and reusable.
