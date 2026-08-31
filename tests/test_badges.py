# -*- coding: utf-8 -*-
# file: test_badges.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
``README.md`` and the Sphinx landing page must carry the **same** badge row.

Two files, one row, kept in step by hand — which is exactly the kind of pair
that drifts, and on the sibling Calango project it already had. Before the row
was added here the Sphinx landing page had no badges at all while the README
had four, so the manual's own front page said nothing about the licence, the
Python version or where the package lives.

What is asserted
----------------
* both files list the same shields.io image URLs, **in the same order**;
* every badge uses the ``for-the-badge`` style, so the row reads as one row;
* the version-bearing badges tell the truth — the PyTorch badge against the
  ``torch`` pin in ``pyproject.toml``, read out of the file rather than
  trusted, and the Python badge is required to be the *dynamic*
  ``pypi/pyversions`` kind rather than a hard-coded number that would drift
  from ``requires-python`` the first time it moved;
* the documentation badge is labelled "Manual" and the word "sphinx" appears
  nowhere in either row. That is a house rule shared with the website
  (``poraque.seixas.dev``'s own notes say the same): the manual is a manual,
  and which generator built it is an implementation detail;
* every badge naming a dependency names one ``pyproject.toml`` actually
  declares, and every badge is a link. A badge for a package the project does
  not depend on is a claim about the stack that nothing would ever correct —
  which is why there is no ``pymatgen`` badge: it arrives transitively through
  ``mp-api`` and is not declared here.

Pure text: no network, no build, so it runs everywhere and in milliseconds.
Whether the URLs *resolve* is a separate question, checked by hand when the row
changes — a test that fetched them would fail on an outage rather than on a
defect.
"""

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(relative):
    with open(os.path.join(ROOT, relative), encoding="utf-8") as handle:
        return handle.read()


def readme_badges(text):
    """shields.io image URLs from the README's markdown badge row."""
    return [match.group(1) for match in re.finditer(
        r"\[!\[[^\]]*\]\((https://img\.shields\.io/[^)]+)\)\]", text)]


def sphinx_badges(text):
    """The same, from the landing page's raw-HTML badge row."""
    return [match.group(1).replace("&amp;", "&") for match in re.finditer(
        r'<img src="(https://img\.shields\.io/[^"]+)"', text)]


@pytest.fixture(scope="module")
def rows():
    """``(readme, sphinx)`` badge lists, read once."""
    return (readme_badges(_read("README.md")),
            sphinx_badges(_read("docs/source/index.rst")))


@pytest.fixture(scope="module")
def declared():
    """The distributions ``pyproject.toml`` lists under ``dependencies``."""
    block = re.search(r"dependencies = \[(.*?)\]", _read("pyproject.toml"),
                      re.S)
    assert block, "pyproject.toml no longer lists dependencies"
    return {re.split(r"[<>=!~\[]", name)[0].strip().lower()
            for name in re.findall(r"'([^']+)'", block.group(1))}


class TestTheTwoBadgeRowsAgree:
    """A reader arriving at the manual should see what a reader of the
    repository sees. They are the same project."""

    def test_the_readme_has_a_badge_row(self, rows):
        assert len(rows[0]) >= 4

    def test_the_landing_page_has_one(self, rows):
        assert len(rows[1]) >= 4

    def test_they_are_identical_in_content_and_order(self, rows):
        readme, sphinx = rows
        assert readme == sphinx, (
            f"the two rows have drifted\n  README: {readme}\n  Sphinx: {sphinx}")

    def test_every_badge_uses_the_same_style(self, rows):
        assert all("style=for-the-badge" in url for url in rows[0])


class TestTheVersionBearingBadgesAreTruthful:
    """A badge stating a version is a claim, and claims here are checked
    against the file that decides them rather than against memory."""

    def test_the_pytorch_badge_matches_the_dependency_pin(self, rows):
        pin = re.search(r"['\"]torch>=([\d.]+)['\"]", _read("pyproject.toml"))
        assert pin, "pyproject.toml no longer pins torch"
        badges = [url for url in rows[0] if "/PyTorch-" in url]
        assert len(badges) == 1
        assert f"-{pin.group(1)}%2B-" in badges[0], (
            f"the badge should say PyTorch {pin.group(1)}+, "
            f"which is what pyproject.toml requires")

    def test_the_python_badge_is_read_from_pypi_not_typed_in(self, rows):
        """
        `requires-python` moves; a hand-written "3.11+" badge would not move
        with it. The dynamic shield reads the classifiers instead, so there is
        no second copy of the number to keep in step.
        """
        badges = [url for url in rows[0] if "logo=python" in url]
        assert len(badges) == 1
        assert badges[0].startswith("https://img.shields.io/pypi/pyversions/")

    def test_the_licence_badge_is_read_from_the_repository(self, rows):
        badges = [url for url in rows[0] if "license" in url]
        assert len(badges) == 1
        assert "/github/license/seixas-research/poraque" in badges[0]
        assert "MIT" in _read("LICENSE")


class TestTheDependencyBadgesNameRealDependencies:
    """
    The row doubles as a statement of what the package is built on, so each
    entry has to be backed by ``pyproject.toml``. The regression this guards
    against is a badge added for a library that is merely *used somewhere* —
    `pymatgen` is the live example: it reaches the package through ``mp-api``
    and is not a declared dependency, so it gets no badge.
    """

    #: Badge label -> the distribution `pyproject.toml` must declare.
    STACK = {"PyTorch": "torch", "ASE": "ase", "h5py": "h5py",
             "mp--api": "mp-api"}

    @pytest.mark.parametrize("label,distribution", sorted(STACK.items()))
    def test_the_badge_names_a_declared_dependency(self, label, distribution,
                                                   declared):
        assert distribution in declared

    @pytest.mark.parametrize("label", sorted(STACK))
    def test_the_badge_is_in_the_row(self, label, rows):
        assert len([url for url in rows[0] if f"/badge/{label}-" in url]) == 1

    def test_h5py_is_there_because_the_cache_format_is(self, rows):
        """
        `data.storage: hdf5` is a first-class cache layout, not an extra, and
        `h5py` is a hard dependency because of it. The badge says so.
        """
        badge = [url for url in rows[0] if "/badge/h5py-" in url]
        assert len(badge) == 1
        assert "style=for-the-badge" in badge[0]

    def test_no_badge_claims_pymatgen(self, rows):
        joined = " ".join(rows[0] + rows[1]).lower()
        assert "pymatgen" not in joined


class TestEveryBadgeIsALink:
    """
    A badge that is only an image is a dead end: the reader has been told a
    library is used and given nowhere to go. Both rows are checked, because
    they are written in different markup and only one of them is read by the
    people who write it.
    """

    def test_the_readme_row_has_one_target_per_badge(self, rows):
        targets = re.findall(
            r"\[!\[[^\]]*\]\(https://img\.shields\.io/[^)]+\)\]\(([^)]+)\)",
            _read("README.md"))
        assert len(targets) == len(rows[0])

    def test_the_landing_page_row_has_one_target_per_badge(self, rows):
        text = _read("docs/source/index.rst")
        block = text[:text.index("Poraquê\n=======")]
        assert len(re.findall(r'<a href="([^"]+)">', block)) == len(rows[1])

    @pytest.mark.parametrize("label,home", (
        ("PyTorch", "https://pytorch.org/"),
        ("ASE", "https://wiki.fysik.dtu.dk/ase/"),
        ("h5py", "https://www.h5py.org/"),
        ("mp--api", "https://pypi.org/project/mp-api/"),
    ))
    def test_a_stack_badge_points_at_that_project(self, label, home):
        """Not at Poraquê's own PyPI page, which is what a copied badge does."""
        pattern = (r"\[!\[[^\]]*\]\(https://img\.shields\.io/badge/"
                   + re.escape(label) + r"-[^)]+\)\]\(([^)]+)\)")
        found = re.search(pattern, _read("README.md"))
        assert found, f"no {label} badge in README.md"
        assert found.group(1) == home

    def test_the_python_badge_points_at_python(self):
        """
        It reads Poraquê's classifiers, but it is a badge *about Python*, and
        a reader clicking it wants the language, not this package's PyPI page.
        """
        found = re.search(
            r"\[!\[[^\]]*\]\(https://img\.shields\.io/pypi/pyversions/"
            r"[^)]+\)\]\(([^)]+)\)", _read("README.md"))
        assert found and found.group(1) == "https://www.python.org/"


class TestTheDocumentationBadge:
    def test_there_is_exactly_one(self, rows):
        assert len([url for url in rows[0] if "readthedocs/" in url]) == 1

    def test_it_is_labelled_manual(self, rows):
        badge = [url for url in rows[0] if "readthedocs/" in url][0]
        assert "label=Manual" in badge

    @pytest.mark.parametrize("index", (0, 1))
    def test_no_row_says_sphinx(self, rows, index):
        """
        The house rule the website states too: it is a manual, and the
        generator that built it is not the reader's business. `logo=readthedocs`
        names the host, not the generator, and is fine.
        """
        assert "sphinx" not in " ".join(rows[index]).lower()


class TestTheLinkTargets:
    """A relative link works from the README and 404s from the docs site, so
    anything the two rows share has to be absolute on the docs side."""

    def test_readme_targets_are_absolute_or_its_own_licence(self):
        targets = re.findall(
            r"\[!\[[^\]]*\]\(https://img\.shields\.io/[^)]+\)\]\(([^)]+)\)",
            _read("README.md"))
        assert targets
        assert all(target.startswith("http") or target == "LICENSE"
                   for target in targets)

    def test_landing_page_targets_are_all_absolute(self):
        text = _read("docs/source/index.rst")
        block = text[:text.index("Poraquê\n=======")]
        targets = re.findall(r'<a href="([^"]+)">', block)
        assert targets
        assert all(target.startswith("https://") for target in targets)

    def test_every_badge_points_at_poraque(self):
        """
        The row was written by adapting Calango's. A leftover `calango` in a
        URL would render a perfectly plausible badge for the wrong project.
        """
        text = _read("README.md") + _read("docs/source/index.rst")
        row = "\n".join(line for line in text.splitlines()
                        if "shields.io" in line or "<a href" in line)
        assert "calango" not in row.lower()
