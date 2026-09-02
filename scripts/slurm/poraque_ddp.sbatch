#!/bin/bash
# file: poraque_ddp.sbatch
#
# Four-GPU DistributedDataParallel training on Santos Dumont (LNCC).
#
# The one thing to get right is that --ntasks-per-node must equal the number
# of GPUs. Poraque reads the topology from Slurm and cannot invent ranks: a
# job that requests four GPUs and launches one task leaves SLURM_NTASKS at 1,
# runs single-GPU on cuda:0, and looks from inside the process exactly like the
# single-GPU run somebody asked for. The run log prints the Slurm variables it
# saw for precisely this reason -- read the "ranks :" line before reading a
# scaling number.
#
#   sbatch scripts/slurm/poraque_ddp.sbatch configs/train.yaml
#
#SBATCH --job-name=poraque-ddp
#SBATCH --partition=sequana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU. Not optional.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONFIG="${1:-configs/train.yaml}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# A torch in ~/.local outranks the one in the activated environment, and on a
# cluster that is how a +cu130 wheel ends up on a Volta node reporting
# `torch.cuda.is_available() == True` and aborting at the first kernel launch.
# The run banner prints where torch was imported from; this makes the answer be
# the environment.
export PYTHONNOUSERSITE=1

module load python/3.11 2>/dev/null || true
source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate poraque

# NCCL over the node's own fabric. Set explicitly rather than left to NCCL's
# interface discovery, which on a multi-homed node can select a management
# interface and produce a rendezvous that times out rather than fails.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"

# One BLAS thread per task. Four ranks each spawning as many OpenMP threads as
# there are cores oversubscribes the node by 4x, and the symptom is a data
# pipeline that gets *slower* as GPUs are added.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---------------------------------------------------------------------------
# Pre-flight: three seconds against a place in the GPU queue
# ---------------------------------------------------------------------------
# Reports the torch build, its CUDA runtime, where it was imported from, the
# architectures it carries kernels for, and one line per GPU. Exits 1 when the
# requested device is unusable -- which is the case `torch.cuda.is_available()`
# cannot detect, since CUDA 13 dropped Volta and a +cu130 wheel on a V100
# reports available and then aborts.
python -m poraque.ml.device --check || exit 1

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# srun starts one process per task; each reads SLURM_PROCID / SLURM_LOCALID /
# SLURM_NTASKS and binds itself to cuda:<local_rank>. MASTER_ADDR comes from
# the first host of SLURM_STEP_NODELIST and MASTER_PORT is derived from
# SLURM_JOB_ID, so nothing here is hard-coded and two jobs on one node cannot
# collide.
#
# strict_device is in every batch job for the reason the CUDA work list gives:
# without it the fallback is not graceful, it is a day of queue time spent
# training on the CPU inside a GPU allocation.
srun --kill-on-bad-exit=1 \
     poraque-train --config "${CONFIG}" \
                   --device cuda \
                   --strict-device \
                   --distributed auto

# To bisect a scaling result, run the same allocation single-GPU:
#
#   srun --ntasks=1 poraque-train --config "${CONFIG}" --device cuda \
#        --strict-device --distributed off
#
# and compare seconds_per_epoch in the metrics JSON. Remember that the
# effective batch size is training.batch_size x world_size, so the four-rank
# run is not the same optimiser -- halve the learning rate or quarter the batch
# size if the comparison is meant to be of the same optimisation.
