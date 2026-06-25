# -*- coding: utf-8 -*-
# file: profiling.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Lightweight execution timing / profiling for Poraquê.

This module provides a tiny, dependency-free profiler that records the
wall-clock time spent in each named component (grid construction, the external
potential, the SCF loop, the exchange-correlation evaluation, the Ewald sum,
...) and renders an aggregated summary at the end of a run.

Two interchangeable instrumentation styles are offered:

* a context manager, for ad-hoc blocks::

      from poraque.profiling import profiler
      with profiler.timer("SCF"):
          engine.run(...)

* a decorator, for whole functions::

      from poraque.profiling import timed

      @timed("XC.potential")
      def potential(...):
          ...

Both feed the same per-name accumulator (total time and call count), so a
component called many times (e.g. the XC potential, evaluated once per SCF
iteration) is reported as a single aggregated row with its hit count. The
profiler is process-global by default (:data:`profiler`) but additional
isolated instances can be created for nested or per-calculation reports.
"""

import functools
import time
from contextlib import contextmanager

__all__ = ["Profiler", "profiler", "timed", "profile_summary"]


class Profiler:
    """
    Accumulate wall-clock time per named code region.

    The profiler keeps, for every name, the total elapsed time and the number
    of times the region was entered. It is intentionally minimal (a plain dict
    of counters) so that instrumenting a hot path adds negligible overhead.

    Attributes
    ----------
    enabled : bool
        When ``False``, :meth:`timer` and :meth:`timed` become near-zero-cost
        pass-throughs and nothing is recorded.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        # name -> [total_seconds, n_calls]
        self._records = {}
        # Preserve first-seen insertion order for a stable, readable summary.
        self._order = []

    def reset(self):
        """Clear all recorded timings."""
        self._records.clear()
        self._order.clear()

    def record(self, name, dt):
        """
        Add a single timing sample of ``dt`` seconds to ``name``.

        Parameters
        ----------
        name : str
            Component label.
        dt : float
            Elapsed wall-clock seconds to attribute to ``name``.
        """
        rec = self._records.get(name)
        if rec is None:
            self._records[name] = [float(dt), 1]
            self._order.append(name)
        else:
            rec[0] += float(dt)
            rec[1] += 1

    @contextmanager
    def timer(self, name):
        """
        Context manager timing the enclosed block and attributing it to ``name``.

        Parameters
        ----------
        name : str
            Component label under which to accumulate the elapsed time.
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0)

    def timed(self, name=None):
        """
        Decorator form of :meth:`timer`.

        Parameters
        ----------
        name : str, optional
            Component label. Defaults to the wrapped function's qualified name.
        """
        def decorator(func):
            label = name or getattr(func, "__qualname__", func.__name__)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                t0 = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    self.record(label, time.perf_counter() - t0)

            return wrapper

        return decorator

    def records(self):
        """
        Return the raw timings as ``{name: (total_seconds, n_calls)}``.

        Returns
        -------
        dict
            A copy of the accumulated records in first-seen order.
        """
        return {name: tuple(self._records[name]) for name in self._order}

    def total(self):
        """Total wall-clock time across all recorded components (seconds)."""
        return sum(rec[0] for rec in self._records.values())

    def summary(self, title="Execution timing summary"):
        """
        Render an aligned, human-readable timing table.

        Components are listed in descending order of total time, each with its
        cumulative time, call count, mean time per call, and percentage of the
        summed profiled time.

        Parameters
        ----------
        title : str, optional
            Heading printed above the table.

        Returns
        -------
        str
            A multi-line report (empty string when nothing was recorded).
        """
        if not self._records:
            return ""

        rule = "=" * 70
        thin = "-" * 70
        total = self.total()
        lines = [rule, title, thin,
                 f"  {'component':<32s} {'time (s)':>10s} {'calls':>7s} "
                 f"{'ms/call':>9s} {'%':>6s}",
                 thin]
        ordered = sorted(self._order, key=lambda n: self._records[n][0],
                         reverse=True)
        for name in ordered:
            tsum, ncalls = self._records[name]
            pct = 100.0 * tsum / total if total > 0 else 0.0
            ms = 1e3 * tsum / ncalls if ncalls else 0.0
            label = name if len(name) <= 32 else name[:29] + "..."
            lines.append(f"  {label:<32s} {tsum:>10.4f} {ncalls:>7d} "
                         f"{ms:>9.3f} {pct:>5.1f}%")
        lines.append(thin)
        lines.append(f"  {'TOTAL (profiled)':<32s} {total:>10.4f}")
        lines.append(rule)
        return "\n".join(lines)


# Process-global profiler used by the engines and the ASE calculator.
profiler = Profiler()


def timed(name=None):
    """
    Decorator that times a function with the process-global :data:`profiler`.

    Parameters
    ----------
    name : str, optional
        Component label (defaults to the function's qualified name).
    """
    return profiler.timed(name)


def profile_summary(title="Execution timing summary"):
    """Return the global profiler's summary table (see :meth:`Profiler.summary`)."""
    return profiler.summary(title=title)
