# -*- coding: utf-8 -*-
# file: test_environment.py

"""
The start-up banner, and the cost of asking for it.

``poraque/__init__.py`` used to build the banner *at import time*. Printing was
the visible half of the problem; the invisible half was that reporting the
versions of ase, matplotlib, numpy, pytest, scipy, torch and yaml meant
importing all seven — so ``import poraque`` cost 0.7 s and pulled in torch
before the caller had asked for anything at all. A ``--help`` paid for it. A
tab-completion paid for it. The package docstring claimed ASE and PyTorch were
not imported, three lines below the imports that did it.

The banner now lives in :mod:`poraque.environment`, its dependency imports sit
*inside* :func:`~poraque.environment.banner_lines`, and ``__init__`` resolves
the two names lazily through ``__getattr__``. Both halves of that need a test:
the cheapness is silent when it regresses, and so is the layout.
"""

import subprocess
import sys

import pytest

import poraque


#: Imported only to name versions in the banner, never by the package itself.
REPORTED_ONLY = ("torch", "matplotlib", "pytest", "ase", "scipy")


class TestImportIsCheap:
    def test_importing_the_package_does_not_import_its_dependencies(self):
        """
        The regression this refactor exists to prevent.

        Run in a subprocess: this interpreter has already imported torch to run
        the rest of the suite, so ``sys.modules`` here proves nothing.
        """
        probe = (
            "import sys; import poraque; "
            f"print([m for m in {REPORTED_ONLY!r} if m in sys.modules])"
        )
        out = subprocess.run([sys.executable, "-c", probe],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "[]", (
            f"import poraque pulled in {out.stdout.strip()}; the banner's "
            f"dependency imports have escaped banner_lines() again."
        )

    def test_the_package_does_not_print_on_import(self):
        """
        Importing a library is not a request to be told about it.

        The banner is emitted by the console commands, which is where a user
        asked for one.
        """
        out = subprocess.run([sys.executable, "-c", "import poraque"],
                             capture_output=True, text=True, check=True)
        assert out.stdout == ""


class TestBannerContent:
    def test_lines_are_strings_without_trailing_newlines(self):
        """
        ``banner_lines`` returns lines, not a blob: ``emit`` decides how they
        are terminated, which is what lets the same banner go to ``print`` and
        to a log file.
        """
        lines = poraque.banner_lines()
        assert lines and all(isinstance(line, str) for line in lines)
        assert not any(line.endswith("\n") for line in lines)

    def test_it_reports_the_package_version(self):
        assert any(poraque.__version__ in line
                   for line in poraque.banner_lines())

    def test_it_reports_every_dependency_it_promises(self):
        text = "\n".join(poraque.banner_lines())
        for name in ("torch", "numpy", "scipy", "ase", "matplotlib"):
            assert name in text.lower(), f"{name} dropped out of the banner"

    def test_emit_defaults_to_print(self, capsys):
        poraque.banner()
        printed = capsys.readouterr().out.splitlines()
        assert printed == poraque.banner_lines()

    def test_emit_redirects_every_line(self):
        """
        The hook a :class:`Tee` uses, so a training log opens with the exact
        environment the run happened in.
        """
        captured = []
        poraque.banner(emit=captured.append)
        assert captured == poraque.banner_lines()


class TestLazyAttributes:
    def test_both_names_resolve(self):
        assert callable(poraque.banner)
        assert callable(poraque.banner_lines)

    def test_an_unknown_name_still_raises_attribute_error(self):
        """``__getattr__`` must not turn typos into ``ImportError``."""
        with pytest.raises(AttributeError, match="no attribute"):
            poraque.nonexistent_attribute

    def test_dir_advertises_them(self):
        """So ``from poraque import <tab>`` finds them despite the laziness."""
        assert {"banner", "banner_lines"} <= set(dir(poraque))
