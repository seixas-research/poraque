# Measured performance

Host: AMD Ryzen 9 5900X 12-Core Processor; 24 logical CPUs available. Single-node measurements; no multi-node claim.

SciPy/NumPy use one-thread OpenBLAS. Native ScaLAPACK uses sequential oneMKL, with OpenMP assembly/copies and actual FFTW3. One warmup and five measured repetitions per case. The full sweep has 60 native configurations (N=64/256/512; MPI=1/2/4; threads=1/2; ranks/k=1/2 where divisible; strong and k-task weak scaling).

| N | k points | Python (s) | Native (s) | Python/native speedup | Native strong efficiency |
|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 0.002454 | 0.001253 | 1.958× | 79.4% |
| 256 | 4 | 0.068276 | 0.027774 | 2.458× | 81.0% |
| 512 | 4 | 0.436903 | 0.173867 | 2.513× | 76.2% |

These rows use four MPI k groups and one thread per rank against the one-thread SciPy baseline. Native efficiency references the single-rank, single-thread native run. Different allocations are compared explicitly; this is not evidence of an intrinsic C-language gain over resource-matched Python.

Largest absolute eigenvalue difference across the sweep: 6.395e-14 eV; largest Harris total-energy difference: 6.821e-13 eV per cell. Every timed repetition was checked; all passed the absolute 1e-8 eV gate.

Full ext2chg/PAW H₂ regression: 5 Å cube, 80 eV cutoff, 16³ grid, Gamma point, existing CHON model and an explicitly supplied licensed H potential. Density repair and spin averaging were explicitly requested and their diagnostics are in the JSON; no SCF density is used.

Warm production medians: Python 0.124611 s, native 0.131593 s, speedup 0.947×. Eigenvalues differ by 3.553e-14 eV and total energies by 3.553e-15 eV. Initialization/inference takes 1.354 s and is excluded from warm ratios. Production native mode offloads eigensolves only, and requests a full spectrum while SciPy requests a subset.

Input functional/PAW preparation accounts for approximately 92.3% of Python production time; diagonalization accounts for 3.4%. With all other work fixed, infinitely fast diagonalization would therefore give only approximately 1.035× speedup by Amdahl’s law. Larger dense problems can gain more as O(N³) eigensolve work grows relative to FFT O(G log G) and root staging. These synthetic gains do not establish a general production speedup.

Validation: eight profiling/backend tests passed, including real MPI k groups, dense generalized eigenproblems, accuracy rejection, timer reset, complex projector assembly, FFT round trips and report generation. The existing worker also passed a four-rank 2×2 BLACS grid.

[Raw CSV](results.csv) · [Raw repetitions and environment](results.json) · [Production regression](production.json) · [Scaling figure](scaling.png) · [Methodology and usage](README.md)
