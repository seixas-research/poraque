# Poraquê Software Architecture

## Purpose

This document defines the concrete software architecture for Poraquê. It translates
the project strategy into package boundaries, responsibilities, interfaces, and
implementation constraints.

The architecture is designed for:

- scientific correctness first
- gradual performance growth
- clean ASE interoperability
- future MPI/OpenMPI support
- optional compiled kernels without rewriting the Python frontend

## Architectural Principles

The codebase should follow these rules.

### 1. Keep physics, numerics, and workflows separate

- `core/` defines mathematical objects and numerical infrastructure
- `functionals/` defines energy terms and functional derivatives
- `potentials/` defines external ionic and embedding-related potentials
- `engine.py` orchestrates minimization and SCF workflows
- `calculator.py` exposes user-facing and ASE-facing entry points

No physics should be hardcoded inside the frontend layer, and no workflow logic
should be buried inside low-level kernels.

### 2. Maintain a serial reference path

Every high-performance or distributed implementation must have a trusted serial
reference path in Python/NumPy. The reference path is the correctness baseline for:

- unit tests
- regression tests
- derivative checks
- backend-to-backend comparisons

### 3. Optimize by replacing kernels, not redesigning the API

The public API and engine interfaces should remain stable while implementations of:

- Laplacians
- Poisson solvers
- FFT operators
- Hartree kernels
- nonlocal KEDF kernels

can be swapped between NumPy, compiled CPU, and MPI-enabled backends.

### 4. Keep array conventions explicit

All numerical interfaces must document:

- dtype
- shape
- ordering
- local versus global indexing
- periodic boundary assumptions
- ghost-cell rules when distributed

This is required before MPI and native kernels can stay sane.

## Target Package Layout

The current repository is still a scaffold. The target layout below keeps the
existing structure but makes future growth explicit.

```text
src/poraque/
  __init__.py
  calculator.py
  engine.py
  density_matrix.py
  ofdft.py
  version.py

  core/
    grid.py
    system.py
    density.py
    state.py
    solver.py
    results.py
    units.py
    validation.py

  functionals/
    base.py
    kinetic.py
    hartree.py
    xc.py
    nonlocal.py
    embedding.py
    registry.py

  potentials/
    external.py
    library.py
    pseudopotential.py
    embedding.py

  ase/
    atoms.py
    calculator.py
    builders.py

  io/
    structures.py
    densities.py
    checkpoints.py
    pseudopotentials.py

  backends/
    base.py
    numpy/
      operators.py
      poisson.py
      fft.py
      linear_algebra.py
    native/
      operators.py
      poisson.py
      fft.py
    mpi/
      decomposition.py
      communicators.py
      halo.py
      distributed_fft.py

  ml/
    datasets.py
    preprocessing.py
    descriptors.py
    symbolic.py
    cnn.py
    inference.py

  workflows/
    ofdft.py
    ksdft.py
    embedding.py
    benchmarks.py
```

Not every file needs to exist immediately, but new work should move toward this
structure rather than growing ad hoc.

## Layered Architecture

Poraquê should operate in four layers.

### Layer 1: Frontend and workflow layer

Primary files:

- `calculator.py`
- `ase/`
- `workflows/`

Responsibilities:

- expose high-level APIs
- create systems from user input or ASE objects
- configure solver settings
- launch calculations
- collect results into user-friendly objects

This layer should remain Python.

### Layer 2: Scientific engine layer

Primary files:

- `engine.py`
- `core/solver.py`

Responsibilities:

- drive OF minimization
- drive future KS SCF loops
- build effective potentials from functionals and external terms
- handle convergence checks, normalization, and diagnostics
- dispatch heavy numerical work to a selected backend

This layer should know about physics objects and backend interfaces, but not about
how the kernels are implemented internally.

### Layer 3: Physics model layer

Primary files:

- `functionals/`
- `potentials/`

Responsibilities:

- compute energy contributions
- compute functional derivatives and local potentials
- assemble total energies and potentials
- define reusable physics components for OF-DFT, KS-DFT, and FDE

This layer should be independent of MPI and mostly independent of ASE.

### Layer 4: Numerical backend layer

Primary files:

- `backends/`

Responsibilities:

- finite-difference operators
- FFT wrappers
- Poisson solver kernels
- Hartree kernels
- distributed domain decomposition
- halo exchange and collective reductions later

This is the performance-critical layer and the main place where compiled code may be
introduced.

## Current Files and Their Intended Roles

The existing files can already be assigned stable responsibilities.

### `src/poraque/core/grid.py`

Owns grid topology and geometry:

- global shape
- spacing
- simulation cell
- volume element
- reciprocal vectors
- index mapping
- local-domain metadata later for MPI

It should not directly implement full solver workflows.

### `src/poraque/core/system.py`

Owns physical system metadata:

- atomic positions
- species
- charges
- total electron count
- spin configuration
- periodic boundary flags
- pseudopotential references

It should also provide conversion helpers to and from ASE `Atoms`.

### `src/poraque/core/solver.py`

Owns generic solver configuration and utilities:

- iteration limits
- tolerances
- line-search parameters
- mixing parameters
- shared convergence logic

This file should remain method-agnostic where possible.

### `src/poraque/functionals/base.py`

Defines the functional interface. Every functional should expose a consistent API.

Minimum interface:

```python
class Functional:
    name: str

    def energy(self, density, system, grid, backend):
        ...

    def potential(self, density, system, grid, backend):
        ...
```

Optional later extensions:

- `stress(...)`
- `forces(...)`
- `metadata()`

### `src/poraque/functionals/kinetic.py`

Owns kinetic functionals:

- Thomas-Fermi
- von Weizsacker
- Pauli enhancement models
- nonlocal KEDF wrappers later
- ML-KEDF wrappers later

### `src/poraque/functionals/hartree.py`

Owns:

- Hartree energy evaluation
- Hartree potential evaluation
- interface to Poisson solvers and FFT backends

### `src/poraque/potentials/external.py`

Owns ionic and other external scalar potentials:

- point-charge toy models
- local pseudopotentials
- regularized nuclear potentials

### `src/poraque/potentials/library.py`

Owns reusable potential parameter sets and lookup logic. It should not contain solver
logic.

### `src/poraque/engine.py`

Owns calculation drivers:

- OF-DFT minimizer
- future KS SCF driver
- future FDE driver

It is responsible for composing:

- the physical system
- a backend
- a list of functionals
- convergence settings

### `src/poraque/calculator.py`

Owns the user-facing calculator object and, later, the bridge to ASE.

This should become the primary public entry point for:

- single-point calculations
- optional structure relaxation setup
- backend selection
- workflow-level configuration

### `src/poraque/density_matrix.py`

This should be reserved for orbital-based data structures needed by KS-DFT. It should
not accumulate OF-DFT logic.

### `src/poraque/ofdft.py`

This file can initially hold OF-specific convenience wrappers, but long term the main
solver behavior should live in `engine.py` plus `workflows/ofdft.py`.

## Core Domain Objects

The architecture should standardize a small set of domain objects.

### `Grid`

Represents the simulation grid and cell.

Expected fields:

- `shape`
- `spacing`
- `cell`
- `volume_element`
- `pbc`
- `reciprocal_vectors`

### `System`

Represents the physical system.

Expected fields:

- `positions`
- `atomic_numbers`
- `charges`
- `electrons`
- `spin_polarized`
- `cell`
- `pbc`

### `Density`

Represents electron density on the grid.

Expected fields and methods:

- array storage
- integral
- normalization
- positivity check
- copy and reshape helpers

### `Result`

Represents the output of a calculation.

Expected contents:

- total energy
- component energies
- converged density
- chemical potential
- iteration count
- residual history
- backend name
- wall-time diagnostics later

## Calculation Flow

The end-to-end OF-DFT flow should be:

1. Build `System` and `Grid`
2. Select backend
3. Build external potential
4. Instantiate functionals
5. Build solver settings
6. Initialize density
7. Call engine driver
8. Return `Result`

Conceptually:

```python
system = System(...)
grid = Grid(...)
backend = NumpyBackend()
functionals = [ThomasFermi(), VonWeizsaecker(), Hartree(), LDAExchange()]

result = OFDFTEngine(
    system=system,
    grid=grid,
    functionals=functionals,
    backend=backend,
    settings=settings,
).run(initial_density)
```

The same pattern should extend later to:

- KS-DFT
- FDE
- ML-KEDF-augmented OF-DFT

## Backend Interface

Backends should expose a stable set of numerical operations. The exact class design
can evolve, but the responsibilities should be fixed.

Minimum backend capabilities:

- `integrate(field, grid)`
- `gradient(field, grid)`
- `laplacian(field, grid)`
- `poisson(charge_density, grid)`
- `fft(field, grid)`
- `ifft(field, grid)`
- `dot(a, b)`
- `norm(a)`

The first implementation can be a pure NumPy backend:

- `NumpyBackend`

Later backends may include:

- `NativeCPUBackend`
- `MPIBackend`

The engine should depend on the interface, not on concrete backend internals.

## Data Layout and Memory Rules

These conventions should be fixed early.

### Array rules

- default dtype: `float64`
- real-space scalar fields use shape `(Nx, Ny, Nz)`
- spin-resolved fields may use `(nspin, Nx, Ny, Nz)`
- reciprocal-space fields may use complex dtype but must keep documented shapes

### Memory rules

- arrays passed into backend kernels should be C-contiguous unless documented otherwise
- avoid hidden copies in inner loops
- do not silently transpose arrays between layers
- functions should state whether they allocate or write into provided buffers

### Distributed rules for later MPI support

- each rank owns a local subdomain
- local arrays may include ghost layers
- local shapes and owned slices must be explicit
- global reductions happen only through defined backend/communicator calls

## MPI/OpenMPI Architecture

MPI should be designed into the architecture before distributed kernels are written.

### First parallel model

Use real-space domain decomposition:

- split the global grid into rank-local subdomains
- maintain halo cells for finite-difference stencils
- perform reductions for normalization, energies, and convergence metrics

### MPI boundary

MPI logic should be isolated in:

- `backends/mpi/decomposition.py`
- `backends/mpi/communicators.py`
- `backends/mpi/halo.py`

The higher layers should never manually perform ad hoc MPI calls.

### Python-level MPI

Use `mpi4py` first for:

- communicator setup
- reductions
- distributed orchestration
- rank-aware diagnostics

### Native MPI

Move communication-heavy kernels to native MPI/OpenMPI when profiling shows need:

- halo exchange
- distributed FFTs
- stencil-heavy loops
- collective-intensive Hartree or nonlocal kernels

## Compiled Kernel Strategy

The project should assume that some kernels will eventually move out of pure Python.

### Good early candidates

- Laplacian
- gradient
- Poisson iterations
- FFT-based convolution kernels
- nonlocal KEDF kernels

### Binding strategy

Allowed long-term backend directions:

- C/C++ via `pybind11`
- Fortran via `f2py` or ISO C binding
- Rust via `pyo3`

The recommended first step is still:

- write the correct NumPy reference
- profile it
- replace only the measured hotspots

## ASE Architecture

ASE support should live in a dedicated namespace rather than leaking through the whole
project.

Recommended files:

- `ase/atoms.py`: conversion between `System` and ASE `Atoms`
- `ase/calculator.py`: ASE `Calculator` implementation
- `ase/builders.py`: helpers for periodic cells, slabs, and supercells later

This keeps ASE interoperability visible and contained.

## ML Architecture

ML-KEDF work should also be isolated from the solver core.

Recommended files:

- `ml/datasets.py`
- `ml/preprocessing.py`
- `ml/descriptors.py`
- `ml/symbolic.py`
- `ml/cnn.py`
- `ml/inference.py`

Rules:

- training logic should not live in `engine.py`
- learned models should be wrapped as functionals before entering solver workflows
- inference should present the same `energy` and `potential` behavior expected from
  analytic KEDFs

## Testing Architecture

Tests should mirror the layer boundaries.

### Unit tests

Cover:

- grid indexing
- integration
- finite-difference operators
- Poisson solver behavior
- each functional's energy and potential

### Numerical verification tests

Cover:

- finite-difference derivative checks
- normalization conservation
- symmetry checks
- translational invariance where applicable

### Integration tests

Cover:

- small end-to-end OF-DFT runs
- ASE single-point workflows
- periodic benchmark systems
- later MPI smoke tests

### Backend parity tests

Every accelerated backend should be compared against the NumPy reference backend on
the same problems.

## Recommended Immediate Refactors

Before implementing many features, the following structural changes would help:

1. give `core/system.py`, `calculator.py`, `functionals/base.py`, and
   `functionals/kinetic.py` real implementations
2. introduce a `Density` object instead of passing raw arrays everywhere
3. define a `Result` object for solver outputs
4. create a `backends/` package with a first `NumpyBackend`
5. keep OF-DFT orchestration in `engine.py`, not scattered across convenience files
6. move ASE-specific logic into a dedicated `ase/` package
7. create `docs/` references for architecture and strategy from the main README

## Near-Term Build Order

The architecture suggests this implementation order:

1. `Grid`, `System`, `Density`, and `Result`
2. `NumpyBackend` with integration, Laplacian, gradient, and Poisson support
3. functional base class and first OF-DFT functionals
4. OF-DFT engine with convergence diagnostics
5. user-facing calculator API
6. ASE bridge
7. backend abstraction hardening
8. native accelerated kernels
9. MPI domain decomposition
10. ML and embedding extensions on top of the stable engine

## Non-Goals for the First Phase

To keep the architecture coherent, the first phase should not aim for:

- many unrelated solver families
- multiple overlapping public APIs
- distributed logic spread across all modules
- backend-specific branches inside every functional
- large feature additions without matching tests

The architecture should stay narrow enough that each new subsystem earns its place.
