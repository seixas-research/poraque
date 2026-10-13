#!/bin/bash
# file: poraque_cache.sbatch
#
# Build the dataset cache on a CPU node, and stop.
#
# The cache is a text codec, not NumPy work. Measured per material on the
# 97-cell platinum set at resolution 48: the Fortran E17.11 reader and writer
# are 86 % of the clock, computing V_ext is 8 %, and the spectral downsampling
# -- the only FFT in the path -- is 2.2 %. There is no BLAS call anywhere in
# it and np.fft is single-threaded, so a thread count buys nothing. On a cold
# parallel filesystem the whole thing is minutes to hours during which a GPU
# allocation would sit idle, so the two halves of a training run are two jobs:
# this one on a CPU partition, then scripts/slurm/poraque_ddp.sbatch on the GPU
# partition with the SAME config, which finds the cache and rebuilds nothing.
#
#   cache=$(sbatch --parsable scripts/slurm/poraque_cache.sbatch configs/train.yaml)
#   sbatch --dependency=afterok:${cache} scripts/slurm/poraque_ddp.sbatch configs/train.yaml
#
# ONE TASK, several PROCESSES. What scales is materials: build_field_cache
# splits them over a pool, one material per worker, and the writes were already
# atomic (.partial-<pid> + os.replace) so the result is byte-identical to the
# single-process build. `data.cache_workers: auto` -- the default -- reads
# SLURM_CPUS_PER_TASK, so it uses exactly the --cpus-per-task below, and sizes
# itself against 80 % of THIS JOB'S --mem (SLURM_MEM_PER_NODE, or
# SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK) divided by an estimate taken from
# the LARGEST native grid in the set. So both numbers below are real knobs:
# raise --mem together with --cpus-per-task or the memory bound wins and the
# extra cores idle. A 288^3 density needs ~1.6 GiB per worker, so 8 workers on
# the platinum set wants --mem=20G or more. The startup line says which of the
# three bounds it hit:
#
#   workers: 8 (auto: 8 usable core(s); 12 within the 25.6 GiB Slurm
#   allocation at ~1995 MiB each, sized on the largest grid at 23.89 M
#   voxels; 97 material(s))
#
# Several *tasks* would still each build the whole cache, which is why this is
# one task. Across nodes, give each a different --ntasks and a stride; nothing
# here needs a communicator.
#
#SBATCH --job-name=poraque-cache
#SBATCH --partition=sequana_cpu     # a CPU partition; adjust to the cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one process. Not four.
#SBATCH --cpus-per-task=8           # = the worker count, if --mem allows it
#SBATCH --mem=32G                   # >= 2 GiB x cpus-per-task for 288^3 grids
#SBATCH --time=06:00:00
#SBATCH --output=slurm-cache-%j.out
#SBATCH --error=slurm-cache-%j.err

set -euo pipefail

CONFIG="${1:-configs/train.yaml}"

# Same environment discipline as the GPU job: the run banner prints where
# torch was imported from, and this makes the answer be the environment.
export PYTHONNOUSERSITE=1
module load python/3.11 2>/dev/null || true
source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate poraque

# One thread per worker process, deliberately. The cache path has no BLAS call
# and np.fft is single-threaded, so there is nothing for OMP_NUM_THREADS to
# parallelize -- but leaving it at the core count makes every one of the
# `cache_workers` processes think it owns all of them, and oversubscription is
# the one thing a thread count can still cost here.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --cache-only forces the CPU and forms no process group whatever the config
# says, so the config can carry `training.device: cuda` for the GPU job that
# follows. It exits 0 after a summary naming the cache, the material count per
# task, the grid shapes, and the size on disk and decoded in RAM -- read the
# last one before sizing the GPU job's memory request.
poraque-train --config "${CONFIG}" --cache-only
