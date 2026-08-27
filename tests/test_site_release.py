# -*- coding: utf-8 -*-
# file: test_site_release.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The website's release script, exercised against a **fake site tree**.

``poraque.seixas.dev`` publishes three things this repository owns: the version
string, the User Guide PDF and the Technical Guide PDF. Its
``deploy/release.py`` is what carries them across, and ``deploy_poraque`` is
what runs it. That script lives in the *other* repository, but the thing it can
get wrong belongs to this one — it parses ``src/poraque/version.py``, and if
that parse ever silently returns the wrong string the site publishes a version
Poraquê never had.

So the tests live here, next to the file being parsed, and every one of them
runs against a tree built in ``tmp_path``: a fake ``version.py``, a fake
``content.py``, fake PDFs. **Nothing touches the real site**, which is the
whole point — a test that reached into ``poraque.seixas.dev`` to check the
copying worked would be a deploy, not a test.

The whole module skips when the site repository is not checked out beside this
one, so the suite stays green on a machine that only has the program.
"""

import importlib.util
import os
import stat

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Same sibling rule release.py itself uses: …/pages/poraque.seixas.dev beside
#: …/seixas-research/poraque. PORAQUE_SITE overrides it, as it does there.
SITE = os.environ.get(
    "PORAQUE_SITE",
    os.path.join(os.path.dirname(os.path.dirname(ROOT)),
                 "pages", "poraque.seixas.dev"))
RELEASE_PY = os.path.join(SITE, "deploy", "release.py")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(RELEASE_PY),
    reason=f"the website repository is not checked out at {SITE}")


@pytest.fixture(scope="module")
def release():
    """``deploy/release.py`` loaded by path — it is a script, not a package."""
    spec = importlib.util.spec_from_file_location("poraque_site_release",
                                                  RELEASE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_repo(tmp_path):
    """A tree shaped like this repository, with one version and two guides."""
    repo = tmp_path / "poraque"
    (repo / "src" / "poraque").mkdir(parents=True)
    (repo / "src" / "poraque" / "version.py").write_text(
        '# -*- coding: utf-8 -*-\n__version__ ="26.8.35"\n', encoding="utf-8")
    for directory, filename in (("user_guide", "poraque_user_guide.pdf"),
                                ("technical_guide",
                                 "poraque_technical_guide.pdf")):
        target = repo / "latex" / directory
        target.mkdir(parents=True)
        (target / filename).write_bytes(b"%PDF-1.5 " + directory.encode())
    return repo


@pytest.fixture
def fake_site(tmp_path):
    """A tree shaped like the website, with a content.py and a docs directory."""
    site = tmp_path / "site"
    (site / "static" / "docs").mkdir(parents=True)
    (site / "content.py").write_text(
        'PRODUCT = {\n'
        '    "name": "Poraquê",\n'
        '    # CalVer, YY.M.D — matches __version__ in src/poraque/version.py.\n'
        '    "version": "26.8.31",\n'
        '    "license": "MIT",\n'
        '}\n', encoding="utf-8")
    return site


class TestReadingTheVersion:
    """The site never states the version; it copies this file's."""

    def test_it_reads_the_assignment(self, release, fake_repo):
        assert release.read_program_version(fake_repo) == "26.8.35"

    def test_the_real_version_py_parses(self, release):
        """
        The fixture's spelling is a copy of the real file's, down to the
        missing space before the `=`. If the real one is ever reformatted this
        catches it before the site publishes nothing.
        """
        import pathlib

        version = release.read_program_version(pathlib.Path(ROOT))
        assert version.count(".") == 2
        assert all(part.isdigit() for part in version.split("."))

    def test_it_agrees_with_the_installed_package(self, release):
        import pathlib

        from poraque.version import __version__

        assert release.read_program_version(pathlib.Path(ROOT)) == __version__

    def test_a_missing_repository_is_refused_by_name(self, release, tmp_path):
        with pytest.raises(SystemExit, match="does not exist"):
            release.read_program_version(tmp_path / "nowhere")

    def test_a_version_py_without_a_version_is_refused(self, release,
                                                       fake_repo):
        (fake_repo / "src" / "poraque" / "version.py").write_text(
            "# nothing here\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no __version__"):
            release.read_program_version(fake_repo)


class TestSyncingTheVersionIntoTheSite:
    def test_it_rewrites_the_product_version(self, release, fake_site):
        content = fake_site / "content.py"
        old = release.sync_version("26.8.35", content_py=content)
        assert old == "26.8.31"
        assert '"version": "26.8.35"' in content.read_text()

    def test_it_reports_nothing_to_do_when_already_current(self, release,
                                                           fake_site):
        content = fake_site / "content.py"
        release.sync_version("26.8.35", content_py=content)
        assert release.sync_version("26.8.35", content_py=content) is None

    def test_it_leaves_the_rest_of_the_line_alone(self, release, fake_site):
        """
        Only the captured group is replaced. The comment beside it says where
        the number comes from, and a rewrite that ate it would remove the one
        hint that the field is not hand-maintained.
        """
        content = fake_site / "content.py"
        release.sync_version("26.8.35", content_py=content)
        text = content.read_text()
        assert "CalVer, YY.M.D" in text
        assert '"name": "Poraquê"' in text and '"license": "MIT"' in text

    def test_a_dry_run_writes_nothing(self, release, fake_site):
        content = fake_site / "content.py"
        before = content.read_text()
        assert release.sync_version("26.8.35", dry_run=True,
                                    content_py=content) == "26.8.31"
        assert content.read_text() == before

    def test_a_content_py_without_the_key_is_refused(self, release, fake_site):
        content = fake_site / "content.py"
        content.write_text("PRODUCT = {}\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="PRODUCT"):
            release.sync_version("26.8.35", content_py=content)


class TestSyncingTheGuides:
    """A stale PDF on the site is invisible: it opens, it reads, it is a
    version behind. Bytes are compared, not timestamps."""

    def _docs(self, site):
        return site / "static" / "docs"

    def test_both_guides_are_copied_when_the_site_has_none(self, release,
                                                           fake_repo,
                                                           fake_site):
        updated = release.sync_guides(repo=fake_repo,
                                      docs_dir=self._docs(fake_site))
        assert set(updated) == {"User Guide", "Technical Guide"}
        assert (self._docs(fake_site) / "poraque_user_guide.pdf").is_file()
        assert (self._docs(fake_site) / "poraque_technical_guide.pdf").is_file()

    def test_an_unchanged_guide_is_not_recopied(self, release, fake_repo,
                                                fake_site):
        release.sync_guides(repo=fake_repo, docs_dir=self._docs(fake_site))
        assert release.sync_guides(repo=fake_repo,
                                   docs_dir=self._docs(fake_site)) == []

    def test_a_changed_guide_is_recopied(self, release, fake_repo, fake_site):
        release.sync_guides(repo=fake_repo, docs_dir=self._docs(fake_site))
        (fake_repo / "latex" / "user_guide" / "poraque_user_guide.pdf"
         ).write_bytes(b"%PDF-1.5 revised")
        assert release.sync_guides(repo=fake_repo,
                                   docs_dir=self._docs(fake_site)) \
            == ["User Guide"]

    def test_a_guide_that_was_never_built_is_skipped_not_fatal(self, release,
                                                               fake_repo,
                                                               fake_site):
        """
        /docs drops the card for a missing file rather than linking to a 404,
        so leaving the site alone is always better than failing the release.
        """
        (fake_repo / "latex" / "user_guide" / "poraque_user_guide.pdf").unlink()
        assert release.sync_guides(repo=fake_repo,
                                   docs_dir=self._docs(fake_site)) \
            == ["Technical Guide"]

    def test_a_dry_run_copies_nothing(self, release, fake_repo, fake_site):
        updated = release.sync_guides(repo=fake_repo, dry_run=True,
                                      docs_dir=self._docs(fake_site))
        assert set(updated) == {"User Guide", "Technical Guide"}
        assert not any(self._docs(fake_site).iterdir())

    def test_a_copied_guide_is_world_readable(self, release, fake_repo,
                                              fake_site):
        """
        nginx serves static/ off disk as its own user. A PDF that arrives mode
        600 from a stricter umask reads fine here and 403s for every visitor,
        and `rsync -az` preserves the mode all the way to the server.
        """
        source = fake_repo / "latex" / "user_guide" / "poraque_user_guide.pdf"
        source.chmod(0o600)
        release.sync_guides(repo=fake_repo, docs_dir=self._docs(fake_site))
        mode = (self._docs(fake_site) / "poraque_user_guide.pdf").stat().st_mode
        assert stat.S_IMODE(mode) == 0o644


class TestTheBuildStepsDegradeRatherThanFail:
    """Neither build is allowed to take the release down with it."""

    def test_a_tree_without_makefiles_does_not_raise(self, release, tmp_path):
        release.build_guides(repo=tmp_path)

    def test_a_tree_without_sphinx_sources_is_reported_as_fine(self, release,
                                                               tmp_path):
        assert release.check_sphinx_builds(repo=tmp_path) is True

    def test_a_dry_run_builds_nothing(self, release, fake_repo):
        assert release.check_sphinx_builds(repo=fake_repo, dry_run=True) is True


class TestTheTwoToolsDoNotOverlap:
    """
    `versioning_poraque` sets the version; `deploy_poraque` propagates it.
    There is one place the number is decided, and the release script is not it.
    """

    def test_release_py_never_writes_version_py(self, release):
        source = open(RELEASE_PY, encoding="utf-8").read()
        body = source.split("def main(")[0]
        assert "version.py" in body, "it does read it"
        for forbidden in ("write_text(", "open(", "chmod("):
            block = body[body.index("def read_program_version"):
                         body.index("def read_site_version")]
            assert forbidden not in block, (
                f"read_program_version must only read: found {forbidden}")

    def test_it_names_the_source_of_truth(self, release):
        assert release.VERSION_PY == "src/poraque/version.py"
