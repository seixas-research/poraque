# Poraquê Strategy

## Purpose

This document describes the strategic direction for Poraquê before major development
begins. Its goal is to keep the project focused, scientifically credible, and
architecturally ready for performance-oriented growth.

Poraquê should be developed as a research electronic-structure platform centered on:

- orbital-free density functional theory (OF-DFT)
- frozen-density embedding (FDE)
- machine-learned kinetic energy density functionals (ML-KEDFs)
- periodic materials workflows
- interoperability with the Atomic Simulation Environment (ASE)

The project should not aim to become a feature-complete general-purpose code in its
first phase. It should instead become excellent in a narrower, high-value domain.

## Positioning

The clearest identity for Poraquê is:

> A modern, research-oriented electronic-structure code for scalable real-space
> OF-DFT, subsystem embedding, and learned kinetic functionals, with clean Python
> workflows and a backend designed for high-performance computing.

This identity gives the project a coherent center:

- OF-DFT provides scale and method-development focus.
- FDE gives a strong subsystem-methods story.
- ML-KEDF gives a distinctive research direction.
- ASE makes the code easier to use in practical workflows.
- HPC-aware architecture makes future scaling realistic.

## Core Principles

The project should be guided by a few explicit rules.

### 1. Correctness before breadth

Early scientific credibility should come from:

- consistent units and conventions
- tested numerical operators
- verified functional derivatives
- reproducible benchmark cases
- documented convergence behavior

A small set of reliable methods is more valuable than a large set of incomplete ones.

### 2. Architecture before optimization hacks

Performance matters, but it should be addressed structurally:

- stable data layout
- clean backend interfaces
- isolated numerical kernels
- explicit MPI boundaries
- early profiling support

The project should be designed so native kernels and distributed execution can be
introduced without rewriting the whole codebase.

### 3. Python for orchestration, compiled code for kernels

Python should remain the top-level language for:

- workflows
- APIs
- ASE integration
- examples
- testing
- data processing
- ML experimentation

Compiled code should be planned for:

- stencil operations
- Poisson solver internals
- FFT-based convolutions
- nonlocal KEDF kernels
- distributed grid communication hot paths
- future high-cost force and stress kernels

### 4. Verification is part of implementation

Every major feature should arrive with:

- unit tests
- numerical derivative checks
- regression tests
- small reference benchmarks
- convergence studies where appropriate

## Scientific Scope

### Primary scope

The first scientific focus should be:

- real-space OF-DFT
- local and nonlocal KEDFs
- periodic systems
- local pseudopotentials
- subsystem embedding

### Secondary scope

These should be developed later, after the OF-DFT core is trustworthy:

- minimal KS-DFT infrastructure
- broader pseudopotential support
- advanced basis expansions
- more general electronic-structure workflows

### Differentiating scope

Poraquê can stand out by combining:

- OF-DFT for larger systems
- embedding methods for subsystems and environments
- ML-KEDFs with physical constraints
- clean integration with Python-driven workflows

## Scientific Priorities

### 1. Build a robust OF-DFT core

The first serious target is a dependable OF-DFT engine with:

- structured `Grid`, `System`, and density objects
- Hartree solver
- Thomas-Fermi and Thomas-Fermi-von Weizsacker functionals
- basic exchange-correlation support
- stable constrained minimization
- energy decomposition and diagnostics

### 2. Treat periodic materials as a first-class use case

To align the project with quantum materials work, periodic systems should be built in
early:

- lattice vectors and cell handling
- periodic boundary conditions
- reciprocal-space operators
- FFT infrastructure
- bulk-material workflows
- later support for stress and equation-of-state studies

### 3. Develop embedding as a flagship capability

FDE should not be an afterthought. It should become one of the defining strengths of
the project:

- subsystem partitioning
- nonadditive kinetic contributions
- nonadditive exchange-correlation terms
- freeze-and-thaw cycles
- mixed OF/KS subsystem strategies later

### 4. Make ML-KEDF scientifically useful, not only novel

ML-KEDF development should focus on practical utility:

- physically informed descriptors and losses
- exact-constraint awareness
- smooth differentiable inference
- stable use inside minimization loops
- held-out transfer tests across chemistry and system size

## Materials Benchmark Strategy

Poraquê should be validated on materials that match its early strengths.

### Early benchmarks

These are good first systems:

- uniform electron gas / jellium
- aluminum
- sodium
- lithium
- magnesium

These systems are especially useful for early OF-DFT and periodic infrastructure.

### Intermediate benchmarks

After the periodic core is stable:

- silicon
- diamond carbon
- sodium chloride
- magnesium oxide
- graphene
- hexagonal boron nitride

These broaden the code's credibility across metallic, covalent, ionic, and 2D systems.

### Later benchmarks

After spin, stronger KS support, and more complete materials workflows:

- iron
- nickel
- titanium dioxide
- strontium titanate
- molybdenum disulfide
- representative defect supercells

### Properties to benchmark

For each benchmark family, aim to measure:

- total energy
- equilibrium lattice constant
- bulk modulus
- cohesive energy where applicable
- density and grid convergence
- force consistency
- runtime scaling

## Software Architecture Strategy

The project should be developed as a layered system.

### Layer 1: Python frontend

Responsibilities:

- user-facing API
- ASE calculator integration
- structure input and output
- workflow composition
- examples and benchmark scripts
- ML data handling and experiments

Likely modules:

- `calculator.py`
- Python workflow helpers
- `examples/`
- future CLI or notebook-facing utilities

### Layer 2: Scientific engine

Responsibilities:

- functional assembly
- minimization and SCF control
- convergence logic
- diagnostics
- backend dispatch

Likely modules:

- `engine.py`
- `core/solver.py`
- `functionals/`

### Layer 3: Numerical kernels

Responsibilities:

- Laplacian and gradient evaluation
- Hartree and Poisson operations
- FFT-based convolutions
- nonlocal KEDF kernels
- distributed communication helpers

This layer should be backend-oriented from the start, even if the initial
implementation is pure NumPy.

## Performance and HPC Strategy

Performance should be planned before major feature growth.

### Hybrid-language plan

Recommended long-term model:

- Python for orchestration and frontend workflows
- NumPy/SciPy for the first correct reference implementation
- compiled extensions for performance-critical kernels
- MPI-aware distributed execution for larger runs

This avoids the trap of building the entire system around slow assumptions.

### Candidate compiled languages

The project can support multiple possible backend directions.

#### C or C++

Strengths:

- mature HPC ecosystem
- straightforward binding options
- easy interoperability with optimized libraries

Suggested binding route:

- `pybind11`

#### Fortran

Strengths:

- natural fit for numerical kernels
- mature scientific computing tradition
- practical for PDE, stencil, and linear algebra kernels

Suggested binding route:

- `f2py` for straightforward kernels
- ISO C binding for cleaner interop where needed

#### Rust

Strengths:

- safer systems programming
- good long-term maintainability
- attractive for robust backend services and kernels

Suggested binding route:

- `pyo3`
- `maturin`

### Recommendation

For the first accelerated backend, the most practical strategy is:

- keep Python as the frontend
- use NumPy as the serial reference layer
- add compiled kernels through C/C++ or Fortran
- keep Rust as an optional later backend direction if the team wants it

The exact low-level language matters less than the interface discipline and data
layout discipline established now.

## MPI and Distributed Execution Strategy

MPI should be treated as a design concern from the beginning, even if the first
working implementation is serial.

### Initial MPI model

Start with domain decomposition for real-space grids:

- each rank owns a local grid subdomain
- halo regions are exchanged for stencil operations
- global reductions are used for norms, energies, and convergence metrics

This is the most natural parallel strategy for OF-DFT on grids.

### Python-level MPI

Use `mpi4py` initially for:

- rank management
- high-level communication
- reductions
- distributed workflow control
- MPI-aware logging and diagnostics

### Native MPI

Move to native MPI/OpenMPI in compiled kernels when:

- halo exchange becomes a bottleneck
- distributed FFTs are needed
- communication volume dominates runtime
- Python-level orchestration becomes too expensive

### MPI questions that must be answered early

Before implementation grows, decide:

- what the local grid storage layout is
- how ghost cells are represented
- how local and global indexing are mapped
- which operations require synchronization every iteration
- whether FFTs are initially local-only or distributed
- what convergence values are reduced globally

These choices should be documented before the first distributed kernels are written.

## Data Layout Strategy

Data layout will strongly influence performance and interoperability.

The project should standardize:

- `float64` as the default numerical dtype
- explicit array shape conventions
- contiguous NumPy arrays for buffer exchange
- consistent real-space and reciprocal-space storage layouts
- local versus global indexing rules
- ghost-cell conventions for distributed grids

Avoid ad hoc copies, reshapes, and layout changes in inner loops. They will become
expensive and make native bindings harder.

## Backend Strategy

Poraquê should support interchangeable backends for heavy operations.

Examples:

- `NumpyLaplacian`
- `NativeLaplacian`
- `NumpyPoissonSolver`
- `NativePoissonSolver`
- `NumpyHartreeKernel`
- `NativeHartreeKernel`

Benefits:

- serial reference and accelerated backend can be compared directly
- optimization can happen incrementally
- tests can validate correctness across implementations
- future GPU or alternative CPU backends remain possible

The backend abstraction should be introduced early, even if only one backend exists
at first.

## External Library Strategy

The project should rely on battle-tested numerical libraries whenever possible.

Likely external dependencies or future integrations:

- BLAS and LAPACK
- FFT libraries
- sparse linear algebra tools
- OpenMPI
- ASE
- NumPy and SciPy

The goal should be to implement the science, not to rebuild every numerical
primitive from scratch.

## ML-KEDF Strategy

ML-KEDF should be developed as a structured research program rather than as an
isolated model experiment.

### Dataset strategy

Plan for:

- many-molecule density datasets
- later extension to bulk materials, strained cells, and surfaces
- metadata for geometry, charge, spin, grid, and reference method
- leakage-resistant train/validation/test splitting

### Model strategy

Two early directions are promising:

- symbolic regression for interpretable kinetic functionals
- CNN models on 2D density slices or multi-slice inputs

### Scientific constraints

Prioritize:

- smooth predictions
- usable functional derivatives
- known scaling behavior
- physically sensible asymptotics
- stability inside OF-DFT minimization

### Integration strategy

Learned models should be treated as functionals that can be evaluated and
differentiated within the same solver framework used by analytic KEDFs.

## ASE Strategy

ASE integration should be more than a file-format convenience layer.

Poraquê should support:

- conversion between internal systems and ASE `Atoms`
- ASE calculator interface
- single-point calculations
- geometry optimization workflows
- periodic cells, slabs, and supercells
- future force and stress exposure through ASE conventions

ASE should remain in the Python layer. It is part of the workflow interface, not the
performance kernel layer.

## KS-DFT Strategy

KS-DFT is important, but it should not dominate the first phase of development.

Recommended role for KS-DFT in the early project:

- provide reference calculations and validation hooks
- support future embedding workflows
- build a minimal correct implementation on the same real-space grid

Recommended non-goals for the first phase:

- broad method coverage
- advanced all-purpose basis support
- immediate production-scale scope

KS should initially serve the core research direction rather than define it.

## Development Phases

### Phase 1: Reference core

Build:

- `Grid`, `System`, and density objects
- Laplacian, gradient, and integration utilities
- Hartree solver
- minimal OF-DFT with TF and TFvW
- stable tests and diagnostics

Outcome:

- correct serial reference implementation

### Phase 2: Workflow usability

Build:

- ASE interoperability
- calculator interface
- basic benchmark scripts
- simple periodic materials examples

Outcome:

- usable research workflows for small systems

### Phase 3: Advanced OF-DFT

Build:

- local pseudopotentials
- reciprocal-space infrastructure
- nonlocal KEDFs
- improved convergence control

Outcome:

- scientifically useful OF-DFT for larger and more realistic systems

### Phase 4: Performance foundation

Build:

- profiling tools
- backend abstraction
- first native accelerated kernels
- early MPI-aware design decisions

Outcome:

- architecture ready for serious scaling

### Phase 5: Flagship research features

Build:

- FDE subsystem workflows
- symbolic-regression ML-KEDF baselines
- CNN-based ML-KEDF experiments
- stable integration of learned functionals into minimization loops

Outcome:

- a distinctive research platform rather than only a numerical prototype

### Phase 6: Distributed and production-oriented growth

Build:

- MPI-enabled real-space domain decomposition
- stronger periodic materials workflows
- forces, stress, and scaling studies
- more complete documentation and reproducibility tools

Outcome:

- credible path toward high-performance scientific usage

## Risks to Avoid

The main strategic risks are:

- trying to implement too many methods before the numerical core is trusted
- postponing data layout and backend decisions until after APIs harden
- scattering MPI logic throughout the codebase
- relying on ML models that are accurate on paper but unstable in self-consistent use
- building performance features without a trusted serial reference
- underinvesting in benchmarks, regression tests, and documentation

## Success Criteria

Poraquê will be on a strong path if it can demonstrate:

- a trustworthy OF-DFT core
- reproducible results on a well-chosen materials benchmark set
- practical ASE-based workflows
- at least one distinctive flagship capability, such as FDE or ML-KEDF
- a backend architecture that supports native kernels and MPI growth
- clear documentation that collaborators can build on

## Immediate Planning Decisions

Before heavy development begins, the project should explicitly decide:

1. the internal data layout for densities, potentials, and grids
2. the functional and solver interfaces
3. the backend abstraction boundary
4. the domain decomposition strategy for MPI
5. the benchmark materials set
6. the validation protocol for every new method
7. the role of KS-DFT in the first development phase
8. the initial compiled-language direction for accelerated kernels

These decisions do not need to lock the whole future, but they should be written down
early enough to keep the project coherent.
