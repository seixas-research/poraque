# -*- coding: utf-8 -*-
# file: conftest.py
"""Shared pytest configuration for the Poraquê test-suite."""

import pytest


@pytest.fixture(autouse=True)
def _keep_working_tree_clean(tmp_path, monkeypatch):
    """Run every test from a scratch directory.

    Tests must never write into the repository. Any code that defaults to a
    relative output path — a log, a checkpoint, a figure — would otherwise
    deposit it wherever pytest happens to be invoked from.
    """
    monkeypatch.chdir(tmp_path)
