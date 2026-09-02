# -*- coding: utf-8 -*-
# file: distributed.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Multi-GPU training: ``DistributedDataParallel`` over NCCL, launched by Slurm.

Poraquê trained on one device for its whole life, and :func:`train`'s docstring
said so and said why. This module is what changed: Santos Dumont's
``sequana_gpu`` nodes carry four Tesla V100s apiece, an allocation is granted
per *node*, and a single-device run leaves three of the four idle for the whole
wall clock. That is not a throughput argument — it is that the queue charges
for four either way.

What it does **not** change is the reason single-GPU was defensible: the FNO is
small (a few megabytes) and the constraint measured on Santos Dumont was
getting data to it, not the arithmetic. Data-parallel replication helps exactly
when the per-rank data pipeline can keep up, which after the in-memory cache
landed it can. Expect the useful scaling to end somewhere well short of linear,
and read ``seconds_per_epoch`` rather than assuming.

Three things had to be built
----------------------------
**1. A sampler that survives.** The blocker recorded in ``train()``'s docstring
was real: :class:`~poraque.ml.data.ShapeBucketSampler` is a ``batch_sampler``
that groups materials by grid shape so no padding ever reaches the FFT, and
``DataLoader`` accepts ``sampler`` *or* ``batch_sampler``, never both. A plain
:class:`~torch.utils.data.distributed.DistributedSampler` handed to a loader
would therefore have to *replace* the bucketing, and a batch mixing :math:`32^3`
with :math:`40^3` does not train badly — it raises in ``collate_fields``.

The resolution is to distribute the **batches** rather than the samples:
:class:`~poraque.ml.data.DistributedShapeBucketSampler` builds the identical
shape-bucketed batch list on every rank (it is a pure function of ``seed`` and
``epoch``) and then hands the *indices of those batches* to a real
``DistributedSampler``. Each rank receives a unique, non-overlapping subset;
padding to an equal count comes for free, and that padding is not a nicety —
see below.

**2. Equal batch counts, or a hang.** DDP synchronises gradients inside each
``backward()``. A rank that runs out of batches while another still has one
leaves that other waiting on an all-reduce that will never be joined, and the
job burns its wall clock in a collective rather than failing. ``DistributedSampler``
pads its index list to a multiple of the world size by wrapping around, and
that is precisely the property being borrowed.

**3. Slurm, without hard-coding anything.** :func:`discover` reads
``SLURM_PROCID``, ``SLURM_LOCALID`` and ``SLURM_NTASKS`` for the topology and
expands ``SLURM_STEP_NODELIST`` — ``sdumont[1234-1236,1240]`` and friends — to
take its first host as ``MASTER_ADDR``. The port is derived from
``SLURM_JOB_ID``, so two jobs sharing a node cannot collide on it and a
resubmission does not need a new number written into a script.

Everything degrades to nothing
------------------------------
:func:`discover` returns a disabled :class:`DistributedContext` when there is
no Slurm allocation, when the allocation has one task, when ``torchrun`` did
not set its variables either, or when CUDA is absent. A disabled context is
falsy, :func:`initialize` on one is a no-op, and every call site is written so
that the single-GPU path is what runs when it is. There is deliberately **no
``DataParallel`` fallback**: single-process multi-GPU is slower than one GPU
for a model this small, it silently changes the effective batch size, and
offering it would mean a laptop run and a cluster run differed in a way nothing
recorded.

NCCL only
---------
The backend is ``nccl`` and the module refuses to initialise without CUDA. Gloo
would work on CPU and would be a way to *test* the plumbing, but it would also
let a misconfigured cluster job quietly run distributed on 96 CPU cores at a
fraction of one GPU's speed, which is the same failure ``strict_device`` exists
to prevent. Untested-on-CPU is the honest state of this module: it has been
written and syntax-checked but never executed, because the machine it was
written on has no CUDA at all.
"""

import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field

#: Slurm variables :func:`discover` consults, in the order they are needed.
#: Listed as data rather than only read inline so :func:`describe` can report
#: what was actually present when a launch does not do what was expected.
SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_STEP_NODELIST",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_GPUS_ON_NODE",
)

#: Where a derived port lands. Above the registered range and below the
#: ephemeral one Linux hands out by default (32768 upwards), so a derived port
#: cannot collide with a socket the kernel assigned to something else on the
#: same node.
PORT_RANGE = (20000, 29999)

#: Environment variables ``torch.distributed`` reads. Set from the Slurm
#: topology rather than expected to be exported by the submission script: a
#: variable the user has to remember is a variable that is wrong on the run
#: that matters.
RENDEZVOUS_VARIABLES = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
                        "WORLD_SIZE")


def _int(name, default=None):
    """One environment variable as an ``int``, or ``default`` if unusable."""
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def expand_nodelist(nodelist):
    """
    Expand a Slurm hostlist into the host names it stands for.

    ``scontrol show hostnames`` is asked first, because it is Slurm's own
    answer and is right by definition — including for the syntaxes this
    function's own parser does not cover. The parser is the fallback for a
    login node, a container, or a test where ``scontrol`` is not on ``PATH``,
    which is most of the places this code gets read.

    Parameters
    ----------
    nodelist : str
        A Slurm hostlist: ``sdumont1234``, ``sdumont[1234-1236]``,
        ``sdumont[1234,1240-1242]``, or several of those comma-separated.

    Returns
    -------
    list of str
        The hosts, in the order Slurm lists them — the first is the one every
        rank must agree to call the master, so order is load-bearing and not
        cosmetic.

    Examples
    --------
    >>> expand_nodelist("sdumont[1234-1236,1240]")
    ['sdumont1234', 'sdumont1235', 'sdumont1236', 'sdumont1240']
    >>> expand_nodelist("node01,node02")
    ['node01', 'node02']
    """
    text = str(nodelist or "").strip()
    if not text:
        return []

    try:
        output = subprocess.run(
            ["scontrol", "show", "hostnames", text],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        hosts = [line.strip() for line in output.splitlines() if line.strip()]
        if hosts:
            return hosts
    except (OSError, subprocess.SubprocessError):
        pass

    return _parse_nodelist(text)


def _parse_nodelist(text):
    """
    Expand a hostlist without Slurm's help.

    Handles the forms Santos Dumont actually produces: a bare host, a bracketed
    range, a bracketed comma list, zero-padded numbers, and any comma-separated
    mixture of those. The subtlety is that the separating commas are *outside*
    brackets and the ones inside are not, so the string cannot simply be split.
    """
    entries, depth, current = [], 0, []
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        if character == "," and depth == 0:
            entries.append("".join(current))
            current = []
        else:
            current.append(character)
    entries.append("".join(current))

    hosts = []
    for entry in (e.strip() for e in entries if e.strip()):
        match = re.match(r"^(.*?)\[(.+)\](.*)$", entry)
        if match is None:
            hosts.append(entry)
            continue
        prefix, body, suffix = match.groups()
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            bounds = part.split("-")
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                low, high = bounds
                # The zero padding is part of the host name, and Slurm pads to
                # the width it was given rather than to a fixed one.
                width = len(low)
                for number in range(int(low), int(high) + 1):
                    hosts.append(f"{prefix}{number:0{width}d}{suffix}")
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return hosts


def default_master_port(job_id=None):
    """
    A rendezvous port derived from the job id.

    Two jobs can land on one node — Santos Dumont's ``sequana_gpu`` partition
    allocates by node but a shared queue does not guarantee exclusivity for
    every account — and two runs agreeing on a hard-coded 29500 is a bind
    failure that looks like a network problem. Deriving the port from
    ``SLURM_JOB_ID`` makes it distinct per job, identical across the ranks of
    one job (which is the only thing that has to be true), and stable across a
    requeue of the same job.

    Parameters
    ----------
    job_id : int or str, optional
        ``SLURM_JOB_ID`` when omitted; an arbitrary constant when there is
        none, since without a job there is nothing to collide with.

    Returns
    -------
    int
        A port inside :data:`PORT_RANGE`.
    """
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "0")
    digits = "".join(ch for ch in str(job_id) if ch.isdigit()) or "0"
    low, high = PORT_RANGE
    return low + (int(digits) % (high - low + 1))


@dataclass
class DistributedContext:
    """
    What every rank needs to know about the group it is part of.

    A *disabled* context is the single-device case and is falsy, so the call
    sites read ``if context:`` rather than testing a world size — the same
    object is passed around whether or not there is a group, and there is no
    ``None`` to guard against separately.

    Attributes
    ----------
    enabled : bool
        Whether a process group should exist. ``False`` on a workstation, in a
        one-task allocation, and whenever CUDA is absent.
    rank : int
        Global rank, ``0`` when disabled.
    local_rank : int
        Rank within the node, and therefore the **CUDA ordinal this process
        owns**. On a four-GPU node the four tasks take ``cuda:0`` to ``cuda:3``
        and each must set its own; leaving them all on ``cuda:0`` is the
        classic way to get four processes contending for one device and a
        fourfold slowdown reported as a scaling failure.
    world_size : int
        Total ranks.
    backend : str
        Always ``"nccl"`` here.
    master_addr, master_port : str, int
        Rendezvous endpoint, derived from Slurm.
    launcher : {"slurm", "torchrun", "none"}
        Where the topology came from. Recorded because the two launchers fail
        differently and the log should say which one was believed.
    initialized : bool
        Whether :func:`initialize` has run. Set on the context rather than
        asked of ``torch.distributed`` so a context can be built, inspected and
        printed without importing torch's distributed machinery.
    present : dict
        The Slurm variables that were set, for :func:`describe`.
    """

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    launcher: str = "none"
    initialized: bool = False
    present: dict = field(default_factory=dict)

    def __bool__(self):
        return bool(self.enabled)

    @property
    def is_main(self):
        """
        Whether this process writes.

        Rank 0 alone writes the checkpoint, the metrics JSON, the figures and
        the PDF, and alone prints. Four ranks opening one log file truncate
        each other's; four ranks calling ``save_bundle`` on one path race on
        the same inode and can leave a half-written checkpoint that loads
        without error and holds nonsense. A disabled context is main, which is
        what makes every call site unconditional.
        """
        return (not self.enabled) or self.rank == 0

    @property
    def device(self):
        """
        The device string this rank must use: ``cuda:<local_rank>``.

        ``"auto"`` when disabled, so the value can be handed straight to
        :func:`~poraque.ml.device.resolve_device` in both cases.
        """
        return f"cuda:{self.local_rank}" if self.enabled else "auto"

    def describe(self):
        """One line per rank, for the run log."""
        if not self.enabled:
            return "single device (no distributed group)"
        return (f"{self.backend} rank {self.rank}/{self.world_size} "
                f"local_rank {self.local_rank} on {self.master_addr}:"
                f"{self.master_port} (launched by {self.launcher})")


def _torchrun_context():
    """A context from ``torchrun``/``torch.distributed.run``'s variables."""
    world_size = _int("WORLD_SIZE")
    rank = _int("RANK")
    if world_size is None or rank is None or world_size < 2:
        return None
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=_int("LOCAL_RANK", 0),
        world_size=world_size,
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=_int("MASTER_PORT", 29500),
        launcher="torchrun",
    )


def _slurm_context():
    """A context from a Slurm step's variables, or ``None`` outside one."""
    rank = _int("SLURM_PROCID")
    world_size = _int("SLURM_NTASKS")
    if rank is None or world_size is None:
        return None
    if world_size < 2:
        # One task is not a group. Initialising a world of one costs a NCCL
        # bootstrap and buys nothing, and every collective below would then be
        # a no-op reached through a wrapper.
        return None

    # SLURM_LOCALID is the rank within the node, which is the CUDA ordinal on a
    # one-task-per-GPU launch. Its absence is not fatal -- a single-node step
    # can be numbered globally -- so fall back to the global rank modulo the
    # tasks per node, and to the rank itself if even that is unknown.
    per_node = _int("SLURM_NTASKS_PER_NODE") or _int("SLURM_GPUS_ON_NODE")
    local_rank = _int("SLURM_LOCALID")
    if local_rank is None:
        local_rank = rank % per_node if per_node else rank

    # SLURM_STEP_NODELIST is the nodes of *this step*; SLURM_JOB_NODELIST is
    # the whole allocation. They differ when a script runs several steps over
    # subsets of its nodes, and the step is the group being formed here.
    nodelist = (os.environ.get("SLURM_STEP_NODELIST")
                or os.environ.get("SLURM_JOB_NODELIST")
                or os.environ.get("SLURM_NODELIST") or "")
    hosts = expand_nodelist(nodelist)
    master_addr = hosts[0] if hosts else "127.0.0.1"

    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=int(local_rank),
        world_size=world_size,
        master_addr=master_addr,
        master_port=default_master_port(),
        launcher="slurm",
        present={name: os.environ[name] for name in SLURM_VARIABLES
                 if name in os.environ},
    )


def discover(requested="auto"):
    """
    Work out whether this process is one rank of a group, and which.

    Slurm is consulted first and ``torchrun`` second, because on Santos Dumont
    a job launched with ``srun`` has the Slurm variables and no others, while a
    ``torchrun`` launch *inside* an allocation has both and its own are the
    ones that describe the group it actually formed. Reversing the order would
    make the second case read the allocation's task count instead of the
    launcher's.

    Parameters
    ----------
    requested : {"auto", "off"} or bool, optional
        ``"auto"`` (the default) uses a group when the environment describes
        one. ``"off"`` — or ``False`` — refuses to, which is how a four-task
        allocation is made to run four independent single-GPU jobs, or how a
        distributed run is bisected against a single-device one.

    Returns
    -------
    DistributedContext
        Disabled whenever there is no group to join, no CUDA to join it with,
        or ``requested`` says not to. Never raises: a workstation must reach
        the single-device path by falling through, not by catching something.
    """
    if isinstance(requested, str):
        wanted = requested.strip().lower() not in ("off", "false", "no", "0",
                                                   "none", "disabled")
    else:
        wanted = bool(requested)
    if not wanted:
        return DistributedContext()

    context = _slurm_context() or _torchrun_context()
    if context is None:
        return DistributedContext()

    # NCCL is a CUDA library. Without CUDA there is nothing to distribute over,
    # and falling back to Gloo would let a misconfigured job train distributed
    # across CPU cores at a fraction of one GPU's speed -- the same silent
    # waste `training.strict_device` exists to stop.
    try:
        import torch
    except ImportError:                             # pragma: no cover
        return DistributedContext()
    if not torch.cuda.is_available():
        warnings.warn(
            f"{context.launcher} describes a {context.world_size}-rank group "
            f"but this process has no CUDA device, and Poraque distributes "
            f"only over NCCL. Running single-device instead; check the module "
            f"environment and the --gres request.",
            RuntimeWarning, stacklevel=2,
        )
        return DistributedContext()

    visible = torch.cuda.device_count()
    if context.local_rank >= visible:
        # One task per GPU is the launch this is written for. More tasks than
        # devices means every device is shared and the ordinals wrap, which is
        # a real configuration and not one to enter by accident.
        warnings.warn(
            f"local rank {context.local_rank} exceeds the {visible} CUDA "
            f"device(s) this process can see; ranks will share devices. "
            f"Launch one task per GPU (--ntasks-per-node equal to the number "
            f"of GPUs requested).",
            RuntimeWarning, stacklevel=2,
        )
        context.local_rank %= max(visible, 1)
    return context


def initialize(context, timeout_minutes=30):
    """
    Join the process group this context describes.

    Sets the rendezvous variables from the context rather than requiring the
    submission script to export them, calls
    :func:`torch.distributed.init_process_group` on NCCL, and binds this
    process to ``cuda:<local_rank>``. The device binding is done here, before
    any allocation: NCCL takes the current device at group creation, and a rank
    that sets it afterwards has already put its communicator on the wrong one.

    Parameters
    ----------
    context : DistributedContext
        A disabled context is a no-op, which is what lets the caller invoke
        this unconditionally.
    timeout_minutes : float, optional
        Collective timeout. The default is long because the first collective
        happens after every rank has built or read its cache, and a cold
        Lustre read of a few hundred densities is minutes, not seconds. A
        timeout that fires there reports itself as a NCCL error and sends the
        reader looking at the network.

    Returns
    -------
    DistributedContext
        The same object, with ``initialized`` set.

    Raises
    ------
    RuntimeError
        If the group cannot be formed. Deliberately *not* softened into a
        fallback: a run that asked for four GPUs and silently took one would
        report a scaling result that is a measurement of nothing.
    """
    if not context:
        return context
    if context.initialized:
        return context

    import datetime

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", str(context.master_addr))
    os.environ.setdefault("MASTER_PORT", str(context.master_port))
    os.environ["RANK"] = str(context.rank)
    os.environ["LOCAL_RANK"] = str(context.local_rank)
    os.environ["WORLD_SIZE"] = str(context.world_size)

    torch.cuda.set_device(context.local_rank)
    dist.init_process_group(
        backend=context.backend,
        init_method="env://",
        world_size=context.world_size,
        rank=context.rank,
        timeout=datetime.timedelta(minutes=float(timeout_minutes)),
    )
    context.initialized = True
    return context


def shutdown(context):
    """
    Leave the process group, if this process is in one.

    Called from a ``finally``: a rank that exits without destroying its group
    can leave the others in a collective until the step's wall clock ends,
    which turns one process's exception into an hour of billed silence.
    """
    if not context or not context.initialized:
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:                               # pragma: no cover
        # Nothing useful can be done at this point, and raising here would
        # replace whatever exception sent us into the `finally`.
        pass
    context.initialized = False


def barrier(context):
    """
    Wait for every rank, or return at once when there is no group.

    Used where one rank does work the others must not race: building the
    prepared cache, and creating the run directory.
    """
    if not context or not context.initialized:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        # device_ids pins the barrier to this rank's own GPU. Without it NCCL
        # guesses from the current device, and a guess that differs between
        # ranks is a hang rather than an error.
        dist.barrier(device_ids=[context.local_rank])


def all_reduce_mean(value, context):
    """
    The mean of a scalar over every rank.

    The training loss must be reduced or the log is a lie: with the batches
    split four ways, each rank's mean is over a quarter of the data and the
    four numbers differ. Reducing also keeps the ranks *in agreement*, which
    matters more than the log — early stopping is decided from this number, and
    ranks that disagreed about whether to stop would leave the ones still
    training waiting on a collective nobody will join.

    Parameters
    ----------
    value : torch.Tensor
        A zero-dimensional tensor on this rank's device.
    context : DistributedContext

    Returns
    -------
    torch.Tensor
        The mean across ranks, or ``value`` unchanged without a group.
    """
    if not context or not context.initialized:
        return value
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / float(context.world_size)


def unwrap(model):
    """
    The module inside a ``DistributedDataParallel``, or the model itself.

    DDP prefixes every ``state_dict`` key with ``module.``, exactly as
    :func:`torch.compile` prefixes them with ``_orig_mod.``. A checkpoint
    written from the wrapper reloads nowhere else, so nothing in Poraquê ever
    stores the wrapper on the operator — but a caller handed one anyway should
    still be able to reach the real module.
    """
    return getattr(model, "module", model)


def describe(context):
    """
    Several lines about the launch, for the run log.

    Prints the Slurm variables that were *present* rather than the ones that
    were expected. The usual failure is a submission script that requests four
    GPUs and launches one task, which leaves ``SLURM_NTASKS`` at 1 and looks
    from inside the process exactly like a single-GPU run that was asked for.

    Returns
    -------
    list of str
    """
    lines = [context.describe()]
    if context.enabled and context.launcher == "slurm":
        for name in SLURM_VARIABLES:
            if name in context.present:
                lines.append(f"  {name} = {context.present[name]}")
    elif not context.enabled:
        seen = [name for name in SLURM_VARIABLES if name in os.environ]
        if seen:
            lines.append(
                "  a Slurm allocation is present but no group was formed; "
                "check --ntasks-per-node")
            for name in seen:
                lines.append(f"  {name} = {os.environ[name]}")
    return lines
