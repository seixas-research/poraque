# Python versus hybrid C profiling

The suite measures performance; speedup is an outcome, not a prerequisite.
SciPy and NumPy already execute compiled LAPACK/BLAS and FFT kernels. A C
wrapper alone does not imply a speedup. The primary opportunities are parallel
k tasks, distributed eigensolves large enough to amortize communication, and
reduced data movement.

## Run

Build the optional backend as described in `src/poraque/hpc/README.md`, then:

```bash
python scripts/poraque_profile.py \
  --library /absolute/build/libporaque_hpc.so \
  --sizes 64 256 512 --nk 4 --ranks 1 2 4 --threads 1 2 \
  --ranks-per-k 1 2 --mode both --repeats 5 --warmups 1 \
  --output results.csv
MPLCONFIGDIR=/tmp/poraque-mpl python scripts/poraque_plot_profile.py \
  results.csv --output scaling.png
```

Installed entry points are `poraque-profile` and `poraque-plot-profile`.
Default launcher is `mpiexec --bind-to none`, allowing threaded ranks to use
multiple cores. On a cluster use `--launcher 'mpiexec <site placement flags>'`
and allocate the requested resources. The runner appends `-n <ranks>`; use an
MPI launcher that accepts that flag. Set placement explicitly for reproducible
HPC comparisons. A single workstation run is not a multi-node scaling claim.
`--python-threads 1` controls the SciPy baseline; the native sweep independently
sets BLAS/OpenMP/FFTW thread counts. Repeat with a resource-matched Python BLAS
thread count to distinguish resource gains from implementation gains.

CSV rows contain medians, minima and sample standard deviations; speedup is
**median(T_Python) / median(T_C)**. JSON retains raw repetitions, command lines,
process times, loaded BLAS/MPI/FFTW libraries, threadpool metadata when
`threadpoolctl` is installed, affinity, requested threads and library SHA256.
The requested OpenMP count does not prove ScaLAPACK's BLAS is threaded: inspect
the loaded libraries. The local build tested here links sequential MKL.

## Numerical gates and workload scope

Before any timed samples, every complex eigenvalue and the Harris total energy
must match within an **absolute 1e-8 eV per cell**, without relative tolerance.
FFT results and Hamiltonian elements are checked too. Native validation also
checks sampled eigenvector residuals and S-orthogonality. Every timed native
repetition is checked again after stopping timers. A failure exits the runner;
no CSV is written for that invocation. Existing output files from older runs
are not removed. Root accuracy failures propagate to all k groups collectively.

The default workload is reproducible **synthetic kernel data**. It uses a
positive periodic analytic density, float64 Hartree/LDA input corrections,
complex Hermitian nonlocal projector couplings and a positive-definite overlap.
No pseudopotential data is generated or distributed. It evaluates
`E = sum(w_k f_nk epsilon_nk) - E_H - integral(rho V_xc) + E_xc`, with uniform k
weights and global zero-temperature filling; the input density is fixed.
The synthetic overlap/strength matrices are an algebraic test, not a complete
PAW material model. This mode is intentionally independent of licensed inputs
and does **not** substitute for an ext2chg/PAW accuracy regression.

For the actual solver, supply your external inputs explicitly:

```bash
python scripts/poraque_profile.py \
  --library /absolute/build/libporaque_hpc.so --production-only \
  --production-deck /private/material --models /private/ext2chg.poraque \
  --potcar /licensed/POTCAR --threads 1 --repeats 5 \
  --output production.csv
```

A valid VASP license is required for the supplied POTCAR. The suite neither
copies nor serializes its contents. It checks actual ext2chg-derived density,
predicted on-site records, eigenvalues, internal Harris energy and free energy
through the production PAW evaluator. It fails rather than substituting an
analytic or SCF density. Optional `--repair-density`, `--spin-average`, and
`--unpolarized-input` are explicit transformations, recorded in the report;
never enable them silently. `--production-only` writes JSON, without an empty
CSV. Line-mode paths cannot define a total-energy quadrature.

Production mode is currently single-process: the production runtime adapter
supports a native eigensolve but still constructs H/S in Python. Cold model
initialization is separate from warm energy evaluation, which includes
functional and one-center terms, construction, solve and output diagnostics.
Backend order alternates each repetition. Production SciPy may request a subset
of states whereas ScaLAPACK computes the full spectrum; this is an operational
comparison, explicitly reported. The synthetic kernel comparison requests all
eigenvectors in both backends and compares the same amount of spectral work.
MPI process elapsed times include validation/warmup and must not be interpreted
as cold single-energy latency.

## Timing definitions

All Python intervals use `perf_counter`; C intervals use `MPI_Wtime`. No
subtraction of timestamps from different ranks is used. A barrier aligns the
start of each native repetition and MAX over rank-local elapsed durations gives
the critical path for total evaluation. Construction/FFT/k-loop and solve-API
stages use group-root observed durations before taking the MAX across groups;
this avoids counting a worker's wait for root assembly as diagonalization.
Native internal intervals still use MAX across all ranks. No per-stage barriers are inserted into the k loop.

* `total`: effective-potential FFT, matrix lookup/assembly, full Hc=epsilon Sc
  eigensolves, global spectrum reduction and Harris band sum. Input preparation,
  projector generation, density inference, Hartree/XC evaluation and fixed input
  corrections are outside the synthetic timing scope.
* `k_loop`: construction and eigensolves for all assigned k points, plus local
  loop overhead. Idle groups can have zero local work; global maxima remain
  well-defined. In the native runner, FFTs are local threaded FFTW calls on each
  group root. This is not a distributed FFTW-MPI benchmark.
* `fft`: wrapper wall time, including conversion, allocation, FFTW_ESTIMATE
  planning, execution, normalization and output copies, compared with NumPy
  `fftn` plus the same normalization. FFTW planning/execution are also exposed
  separately via C timers. No plan cache is implemented.
* `construction`: Fourier lookup and `Vlocal + diag(T) + B D B†` assembly,
  including ctypes conversion/allocation on the native root. Shared prepared
  projectors and overlap are outside this scope. The assembly kernel is root
  local OpenMP, not distributed matrix generation.
* `diagonalization`: complete solve API latency, including NumPy finite/shape
  checks, copies, native allocation, packing, scatter, ScaLAPACK generalized
  reduction/solve/backtransform and eigenvector gather. Only residual diagnostic
  products are skipped during repetitions after the mandatory validated solve.

`c_internal_*` fields use MAX of cumulative C intervals across ranks. They are
**inclusive**: `solve_root` contains `solve_local`, which contains `pzheevd`;
`fft_total` contains `fft_plan` and `fft_execute`. `pzheevd` excludes workspace
query/allocation, overlap reduction and backtransform. `assemble` excludes
Python lookup and conversion. Parent/child timings must not be added. Stage
maxima may come from different ranks and must not be summed to reconstruct the
critical path. Small intervals are noisy; use larger sizes/more repeats before
interpreting modest differences.

## Scaling and expected limits

Strong scaling fixes basis dimension N and total k count. Independent k groups
receive cyclic tasks; each group has `ranks_per_k` ScaLAPACK ranks. Strong
native speedup is `T_C(1 rank,1 thread)/T_C(P ranks,t threads)`; efficiency is
that speedup divided by `P*t`. Missing single-resource reference rows leave
empty efficiency cells. Python-relative speedup is separately reported.

Weak scaling keeps N fixed and sets `nk = base_nk * MPI_ranks`, maintaining
approximately constant cubic eigensolve work per rank. For each fixed thread
count, efficiency is `T_C(1 rank,t threads)/T_C(P ranks,t threads)`. This is
**k-task weak scaling**, not a claim about increasing a single distributed
matrix at fixed memory per rank. Varying N in the size sweep is a separate
complexity experiment. Native single-rank references must be included for
computed efficiency curves. Keep N, base k count, thread count, placement and
numerical settings consistent across compared rows.

Full dense diagonalization costs O(N^3) arithmetic and O(N^2) storage; projector
assembly costs O(N^2 M + N M^2); a G-point FFT costs O(G log G). SOC doubles N,
increasing dense arithmetic by roughly eightfold before considering symmetry.
For a measured serial fraction f and ideal parallel compute speedup p,
Amdahl's law gives `S(p) = 1/[f + (1-f)/p]`, so the limit is `1/f`. Root matrix
construction, scatter/gather, Python dispatch, inference and one-center terms
all contribute to that limit in this architecture. Actual communication and
synchronization increase runtime beyond this ideal bound. Small matrices often
slow down in distributed ScaLAPACK; k parallelism should help first when enough
independent k points exist. Increasing OpenMP threads cannot accelerate a
sequential BLAS eigensolver. Profile the complete production workflow before
extrapolating isolated eigensolve speedups to total Harris evaluation.

Primary reference for the full-eigenpair contract:
[ScaLAPACK PZHEEVD](https://www.netlib.org/scalapack/explore-html/da/d75/pzheevd_8f_a54ca0fd8007440fdf34013f71414588b.html).

Production JSON also contains scoped Python stage hooks for forward/inverse
NumPy FFTs, H construction, overlap construction, solve latency, the k loop,
and input functional/one-center preparation. These hooks are active only
inside an otherwise idle worker and restore the original functions on exit.
Forward FFT totals include Hartree/XC/local-potential transforms; PyTorch's
internal GGA transforms are not intercepted. Production FFTs remain NumPy;
FFTW comparisons apply to the explicit kernel workload. These scopes are also
inclusive and must not be added together.

CSV also reports the measured Python diagonalization fraction and the ideal
Amdahl limit if only that fraction were infinitely accelerated. This is a
conditional upper bound, not a prediction for k-parallel execution (which also
parallelizes assembly across independent groups).

The runner also records `dladdr` symbol providers where available. It rejects
MKL's FFTW compatibility symbols as a replacement for an FFTW measurement.
CMake places the real FFTW libraries before MKL in the link order; otherwise
MKL can satisfy the FFTW ABI and `--as-needed` can silently discard FFTW. This
linkage issue was caught during development and the affected preliminary FFT
measurements were replaced after fixing and validating the build.

The synthetic size sweep holds the projector count fixed unless you change
`--projectors`. Scaling to larger physical systems also changes atom/projector
counts, density grids and inference costs; those effects require representative
production inputs. MPI context initialization occurs during untimed validation
and is excluded from warm speedup ratios. These are not cold-launch speedups.
