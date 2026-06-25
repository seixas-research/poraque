# -*- coding: utf-8 -*-
# file: test_profiling.py
"""Tests for the execution timing / profiling module."""

import time

from poraque.profiling import Profiler


class TestProfiler:
    def test_timer_records_time_and_calls(self):
        prof = Profiler()
        for _ in range(3):
            with prof.timer("block"):
                time.sleep(0.001)
        total, calls = prof.records()["block"]
        assert calls == 3
        assert total > 0.0

    def test_decorator_records_under_custom_name(self):
        prof = Profiler()

        @prof.timed("my-func")
        def work():
            return 42

        assert work() == 42
        assert "my-func" in prof.records()

    def test_decorator_defaults_to_qualname(self):
        prof = Profiler()

        @prof.timed()
        def labelled():
            return None

        labelled()
        assert any("labelled" in name for name in prof.records())

    def test_reset_clears_records(self):
        prof = Profiler()
        with prof.timer("x"):
            pass
        prof.reset()
        assert prof.records() == {}
        assert prof.summary() == ""

    def test_summary_lists_components_and_total(self):
        prof = Profiler()
        with prof.timer("alpha"):
            time.sleep(0.001)
        with prof.timer("beta"):
            time.sleep(0.001)
        summary = prof.summary(title="Timing")
        assert "Timing" in summary
        assert "alpha" in summary
        assert "beta" in summary
        assert "TOTAL" in summary

    def test_disabled_profiler_is_passthrough(self):
        prof = Profiler(enabled=False)
        with prof.timer("ignored"):
            pass
        assert prof.records() == {}

    def test_total_sums_components(self):
        prof = Profiler()
        prof.record("a", 1.0)
        prof.record("b", 2.5)
        assert prof.total() == 3.5
