# -*- coding: utf-8 -*-
# file: profiling.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
What a run cost, stage by stage: wall time, peak host memory, peak device memory.

An HPC allocation is sized from three numbers --- how long, how much RAM, how
much GPU memory --- and before this they could only be had from outside the
process (``sacct``, ``nvidia-smi`` sampling), for the whole job rather than for
the part of it that needed them. :class:`ResourceProfile` records them per
stage from inside, and :meth:`ResourceProfile.summary` prints the table a job
script can be sized from.

Three choices, each measured against the alternative:

**Peak host memory is the kernel's high-water mark**, from
:func:`resource.getrusage`, not a sampled resident set. ``psutil`` reports the
*current* RSS, and a sampler would have to run fast enough to catch a spike ---
the spectral downsampling of an 800 MB ``CHGCAR`` allocates and frees its peak
within one call. ``ru_maxrss`` cannot miss it, needs no dependency, and is also
available for the process's children, which is where DataLoader workers live.
It is the peak **of the process so far**, so a stage's value is the high-water
mark at its end, never below the previous stage's.

**Peak device memory is per stage** on CUDA: the allocator's peak is reset when a
stage begins and read when it ends, so each row says what that stage needed.
Both ``max_memory_allocated`` (tensors) and ``max_memory_reserved`` (what the
caching allocator held, which is what ``nvidia-smi`` shows) are kept. Apple MPS
exposes no peak; the driver's allocation at the end of the stage is recorded
instead, and labelled as such.

**Stage boundaries synchronise the device**, as the training loop's own clock
does: CUDA and MPS dispatch asynchronously, and an unsynchronised clock times
how fast work was queued.
"""

import resource
import sys
import time

import torch

from .device import synchronize

#: ``ru_maxrss`` is bytes on macOS and kibibytes on Linux.
_RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def peak_rss_bytes(children=False):
    """
    The process's peak resident set so far, in bytes.

    Parameters
    ----------
    children : bool, optional
        The largest peak among **terminated** child processes instead ---
        DataLoader workers, once their loader has finished.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return int(resource.getrusage(who).ru_maxrss) * _RSS_UNIT


def format_bytes(count):
    """``1.23 GiB``, or an em dash for ``None``."""
    if count is None:
        return "—"
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0


def format_seconds(seconds):
    """``42.3 s``, ``13m 02.1s`` or ``2h 05m 03s``."""
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {int(rest):02d}s"


class ResourceProfile:
    """
    Wall time and peak memory per named stage of a run.

    Stages are marked imperatively, so instrumenting a long function does not
    mean re-indenting it: :meth:`begin` closes whatever stage is open and opens
    the next, :meth:`end` closes the last. A stage still open when
    :meth:`summary` is called --- because the run raised inside it --- is
    closed there and marked ``interrupted``, which is the row a failed job most
    needs.

    Parameters
    ----------
    device : torch.device or str, optional
        Where the run computes; settable later through :attr:`device`, since a
        run resolves its device after it has started the clock.

    Examples
    --------
    >>> profile = ResourceProfile("cpu")
    >>> profile.begin("cache")
    >>> profile.begin("training")        # closes "cache"
    >>> profile.end()
    >>> [stage["name"] for stage in profile.stages]
    ['cache', 'training']
    """

    def __init__(self, device=None):
        self.device = None if device is None else torch.device(device)
        self.started = time.perf_counter()
        self.stages = []
        self._open = None

    # ------------------------------------------------------------------ #
    def _cuda(self):
        return (self.device is not None and self.device.type == "cuda"
                and torch.cuda.is_available())

    def begin(self, name):
        """Close the open stage, if any, and start ``name``."""
        self.end()
        if self.device is not None:
            synchronize(self.device)
        if self._cuda():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._open = {"name": str(name), "start": time.perf_counter()}

    def end(self, status="ok"):
        """Close the open stage; a no-op when none is open."""
        if self._open is None:
            return
        if self.device is not None:
            try:
                synchronize(self.device)
            except RuntimeError:
                # A device that faulted is exactly when the row is wanted.
                status = "interrupted"
        stage = self._open
        self._open = None
        stage["seconds"] = time.perf_counter() - stage.pop("start")
        stage["status"] = status
        stage["peak_rss_bytes"] = peak_rss_bytes()
        stage["peak_vram_bytes"] = None
        stage["peak_vram_reserved_bytes"] = None
        stage["mps_allocated_bytes"] = None
        if self._cuda():
            stage["peak_vram_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device))
            stage["peak_vram_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device))
        elif (self.device is not None and self.device.type == "mps"
              and hasattr(torch, "mps")):
            try:
                stage["mps_allocated_bytes"] = int(
                    torch.mps.driver_allocated_memory())
            except (AttributeError, RuntimeError):
                pass
        self.stages.append(stage)

    # ------------------------------------------------------------------ #
    @property
    def elapsed(self):
        """Seconds since the profile was created."""
        return time.perf_counter() - self.started

    def as_dict(self):
        """Every stage and the totals, JSON-ready."""
        vram = [s["peak_vram_bytes"] for s in self.stages
                if s["peak_vram_bytes"] is not None]
        reserved = [s["peak_vram_reserved_bytes"] for s in self.stages
                    if s["peak_vram_reserved_bytes"] is not None]
        return {
            "device": None if self.device is None else str(self.device),
            "stages": [dict(stage) for stage in self.stages],
            "total_seconds": self.elapsed,
            "staged_seconds": sum(s["seconds"] for s in self.stages),
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_worker_rss_bytes": peak_rss_bytes(children=True) or None,
            "peak_vram_bytes": max(vram) if vram else None,
            "peak_vram_reserved_bytes": max(reserved) if reserved else None,
        }

    def summary(self):
        """
        The "Resource Profiling Summary" table, as lines.

        Closes a stage left open, marking it ``interrupted``.
        """
        if self._open is not None:
            self.end(status="interrupted")
        totals = self.as_dict()
        cuda = totals["peak_vram_bytes"] is not None
        mps = any(s["mps_allocated_bytes"] is not None for s in self.stages)
        memory_heading = ("peak VRAM" if cuda else
                          "MPS at end" if mps else "")

        width = max([len(s["name"]) + (15 if s["status"] != "ok" else 0)
                     for s in self.stages] + [len("total (wall clock)")])
        rule = "=" * 78
        lines = [rule, "RESOURCE PROFILING SUMMARY", rule]
        header = f"  {'stage':<{width}s}  {'wall time':>12s}  {'peak RSS':>11s}"
        if memory_heading:
            header += f"  {memory_heading:>11s}"
        lines += [header, "  " + "-" * (len(header) - 2)]

        for stage in self.stages:
            label = stage["name"] + ("  (interrupted)"
                                     if stage["status"] != "ok" else "")
            row = (f"  {label:<{width}s}  {format_seconds(stage['seconds']):>12s}"
                   f"  {format_bytes(stage['peak_rss_bytes']):>11s}")
            if cuda:
                row += f"  {format_bytes(stage['peak_vram_bytes']):>11s}"
            elif mps:
                row += f"  {format_bytes(stage['mps_allocated_bytes']):>11s}"
            lines.append(row)

        lines.append("  " + "-" * (len(header) - 2))
        row = (f"  {'total (wall clock)':<{width}s}"
               f"  {format_seconds(totals['total_seconds']):>12s}"
               f"  {format_bytes(totals['peak_rss_bytes']):>11s}")
        if cuda:
            row += f"  {format_bytes(totals['peak_vram_bytes']):>11s}"
        elif mps:
            row += f"  {'':>11s}"
        lines.append(row)

        untracked = totals["total_seconds"] - totals["staged_seconds"]
        lines.append(f"  device: {totals['device'] or 'not resolved'}   "
                     f"outside the stages above: {format_seconds(untracked)}")
        lines.append("  peak RSS: the process's high-water resident set so "
                     "far (getrusage), so it never falls between rows")
        if totals["peak_worker_rss_bytes"]:
            lines.append(f"  DataLoader workers peaked at "
                         f"{format_bytes(totals['peak_worker_rss_bytes'])} each")
        if cuda:
            lines.append(
                f"  peak VRAM: torch.cuda.max_memory_allocated within each "
                f"stage; the caching allocator reserved up to "
                f"{format_bytes(totals['peak_vram_reserved_bytes'])}, which "
                f"is what nvidia-smi reports")
        elif mps:
            lines.append("  MPS at end: the Metal driver's allocation when the "
                         "stage ended; MPS exposes no peak")
        return lines
