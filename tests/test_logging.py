# -*- coding: utf-8 -*-
# file: test_logging.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
:class:`~poraque.utils.logging.Logger`: one layout for every command's output.

Before it, each console command drew its own frame. ``poraque-inference`` ruled
78 ``=`` signs and indented its stages by eight spaces, ``poraque-mp`` ruled 66,
the C backend 70; ``poraque-committee`` labeled its parts in lowercase
("calibration (committee spread vs its own error):") and ``poraque-vasp`` not
at all. A person could read each of them, and no script could read any two
with the same code. The format is now a title between two rules and a sequence
of blocks, each opened by a ``[NAME]`` header at column 0 with its body
indented under it --- and :func:`~poraque.utils.logging.split_blocks` is the
reader that has to understand all of them.

So the module is pinned twice over. The first half holds the Logger to its
contract: the header, the aligned fields that skip ``None``, the table, the
title, a file that says exactly what the terminal said, a silent rank that
writes nothing at all, and a reader that gets back what a writer wrote. The
second half runs one cheap path through every migrated command and asks the
reader what it found: a command that slipped back to a hand-made frame shows
up as a block that is missing.
"""

import importlib.util
import io
import os
import sys

import numpy as np
import pytest

from poraque.utils.logging import (
    INDENT,
    RULE,
    Logger,
    field_lines,
    header,
    split_blocks,
    table_lines,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")


def _script(name):
    """Import ``scripts/<name>.py`` as a module, without installing it."""
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    spec = importlib.util.spec_from_file_location(
        f"_logging_{name}", os.path.join(_SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_logging_{name}"] = module
    spec.loader.exec_module(module)
    return module


def _capture():
    """A Logger writing into a string, and the string."""
    stream = io.StringIO()
    return Logger(stream=stream), stream


def _names(text):
    """The block headers of ``text``, in order, as :func:`split_blocks` sees them."""
    return [name for name, _ in split_blocks(text) if name]


def _body(text, name):
    """The body of the first block called ``name``."""
    return dict(split_blocks(text))[name]


# ===================================================================== #
# The format
# ===================================================================== #
class TestEveryBlockOpensWithItsHeader:
    """
    A block is found by searching for its header, so the header has exactly
    one spelling: upper case, in brackets, at column 0, after a blank line.
    """

    def test_a_name_becomes_an_upper_case_bracketed_header(self):
        assert header("model") == "[MODEL]"
        assert header("paw records") == "[PAW RECORDS]"

    def test_brackets_already_given_are_not_doubled(self):
        """A caller who writes ``"[cache]"`` should not get ``[[CACHE]]``."""
        assert header("[cache]") == "[CACHE]"

    def test_a_section_is_a_blank_line_then_the_header(self):
        log, stream = _capture()
        log.section("grid")
        assert stream.getvalue() == "\n[GRID]\n"
        assert log.current == "[GRID]"


class TestFieldsLineUpAndSkipWhatIsMissing:
    """
    ``key: value`` lines are read by eye down the column of values, which only
    works when the colons align; and an optional value that is absent is not
    a line reading ``None``.
    """

    def test_the_values_start_in_one_column(self):
        lines = field_lines({"width": 64, "modes": 24, "n_layers": 4})
        assert lines == ["  width:    64", "  modes:    24", "  n_layers: 4"]
        starts = {line.index(line.split()[-1]) for line in lines}
        assert len(starts) == 1

    def test_none_is_skipped_rather_than_printed(self):
        lines = field_lines({"cache": "data/cache", "json": None})
        assert lines == ["  cache: data/cache"]
        assert not any("None" in line for line in lines)

    def test_nothing_to_say_is_no_lines(self):
        assert field_lines({"json": None}) == []

    def test_the_body_is_indented_under_its_header(self):
        log, stream = _capture()
        log.section("model")
        log.fields({"width": 16})
        assert stream.getvalue().splitlines()[-1] == INDENT + "width: 16"


class TestATableIsAHeadingARuleAndItsRows:
    """
    A table is laid out from its widest cell, so a long label cannot push a
    column out of line, and the first column reads left while numbers read
    right.
    """

    def test_layout(self):
        lines = table_lines(("path", "time"),
                            [("torch.einsum", "14.9 ms"), ("C", "0.9 ms")])
        assert lines[0] == "  path             time"
        assert lines[1] == "  " + "-" * len("path             time")
        assert lines[2] == "  torch.einsum  14.9 ms"
        assert lines[3] == "  C              0.9 ms"

    def test_an_explicit_alignment_is_honored(self):
        lines = table_lines(("a", "b"), [("x", "y")], align="<<")
        assert lines[2] == "  x  y"

    def test_every_row_is_as_wide_as_the_rule_or_narrower(self):
        lines = table_lines(("element", "Zval"), [("Pt", "10.000"),
                                                  ("Ag", "11.000")])
        assert all(len(line) <= len(lines[1]) for line in lines)


class TestTheTitleIsFramedByTwoRules:
    """The one line that says what a command is doing, set off from the banner."""

    def test_framing(self):
        log, stream = _capture()
        log.title("FNO inference")
        assert stream.getvalue().splitlines() == ["", RULE, INDENT + "FNO inference",
                                                  RULE]

    def test_a_title_is_not_a_block(self):
        """Lines before the first header belong to the unnamed block ``""``."""
        log, stream = _capture()
        log.title("Poraque C backend")
        log.section("status")
        log.text("loaded")
        blocks = split_blocks(stream.getvalue())
        assert blocks[0][0] == ""
        assert _names(stream.getvalue()) == ["STATUS"]


class TestTheFileSaysWhatTheTerminalSaid:
    """
    A log file is read after the run to find out what the run printed. If the
    two ever differed, the file would be a record of a different run.
    """

    def test_file_and_terminal_are_identical(self, tmp_path):
        stream = io.StringIO()
        path = tmp_path / "logs" / "run.log"
        with Logger(str(path), stream=stream) as log:
            log.title("a run")
            log.section("input")
            log.fields({"directory": "Pt2", "json": None})
            log.table(("field", "r"), [("CHGCAR", "0.99")])
            log("  a verbatim line")
        assert path.read_text(encoding="utf-8") == stream.getvalue()

    def test_the_directory_is_created(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.log"
        with Logger(str(path), terminal=False) as log:
            log("x")
        assert path.read_text() == "x\n"

    def test_terminal_off_writes_the_file_alone(self, tmp_path, capsys):
        with Logger(str(tmp_path / "quiet.log"), terminal=False) as log:
            log.section("model")
        assert capsys.readouterr().out == ""
        assert "[MODEL]" in (tmp_path / "quiet.log").read_text()


class TestASilentLoggerWritesNothing:
    """
    Every rank but the first gets a silent Logger. Four ranks opening one file
    truncate and interleave each other, so silent has to mean *no file at
    all*, not an empty one.
    """

    def test_no_output_and_no_file(self, tmp_path, capsys):
        path = tmp_path / "rank1" / "run.log"
        log = Logger(str(path), silent=True)
        log.title("training")
        log.section("model")
        log.fields({"width": 16})
        log("verbatim")
        log.close()
        assert capsys.readouterr().out == ""
        assert not path.exists()
        assert not path.parent.exists()


class TestTheReaderGetsBackWhatTheWriterWrote:
    """
    :func:`split_blocks` is the reference reader. Whatever a Logger writes,
    the reader has to return block for block: the names in order, the bodies
    without their indentation, and nothing lost to the blank separators.
    """

    def test_round_trip(self):
        log, stream = _capture()
        log.title("a command")
        log.section("input")
        log.fields({"structure": "Pt2", "atoms": 2})
        log.section("comparison")
        log.table(("field", "relative L2"), [("CHGCAR", "0.0100")])
        log.block("output files", {"CHGCAR": "out/CHGCAR"}, ["done."])

        blocks = split_blocks(stream.getvalue())
        assert [name for name, _ in blocks] == ["", "INPUT", "COMPARISON",
                                                "OUTPUT FILES"]
        assert blocks[0][1] == ["", RULE, "a command", RULE]
        assert blocks[1][1] == ["structure: Pt2", "atoms:     2"]
        assert blocks[2][1][0].split() == ["field", "relative", "L2"]
        assert blocks[2][1][2].split() == ["CHGCAR", "0.0100"]
        assert blocks[3][1] == ["CHGCAR: out/CHGCAR", "done."]

    def test_an_indented_bracket_is_body_not_a_header(self):
        """
        ``[1/3] EXTCAR`` inside a block is a line of that block. Only a
        bracketed line at column 0 opens a new one.
        """
        log, stream = _capture()
        log.section("potential")
        log.text("[1/3] EXTCAR  (analytic, no network)")
        assert _names(stream.getvalue()) == ["POTENTIAL"]

    def test_a_logger_is_still_a_plain_callable(self):
        """
        Every library function that reports progress takes ``log=print``. A
        Logger has to be a drop-in for that, writing its argument verbatim.
        """
        def report(log=print):
            log("  scored 4/4 candidates")

        log, stream = _capture()
        report(log)
        assert stream.getvalue() == "  scored 4/4 candidates\n"


# ===================================================================== #
# The commands
# ===================================================================== #
@pytest.fixture
def single_atom(tmp_path):
    """One silicon atom in a 6 Ang box: an isolated-atom reference calculation."""
    from poraque.fields import ChargeDensity, FieldGrid
    from poraque.fields.vasp.poscar import Poscar

    cell = np.eye(3) * 6.0
    grid = FieldGrid((16, 16, 16), cell)
    structure = Poscar(cell, ["Si"], [1], [[0.0, 0.0, 0.0]])
    axis = np.minimum(np.arange(16), 16 - np.arange(16)) * 6.0 / 16
    r2 = axis[:, None, None] ** 2 + axis[None, :, None] ** 2 \
        + axis[None, None, :] ** 2
    values = np.exp(-r2 / 0.5)
    values *= 4.0 / (values.sum() * grid.volume / values.size)
    directory = tmp_path / "atoms" / "Si"
    directory.mkdir(parents=True)
    ChargeDensity(values, grid, structure).write(directory / "CHGCAR")
    return str(directory.parent)


@pytest.fixture
def density(tmp_path):
    """A two-atom CHGCAR on a small cubic grid, as ``poraque-vasp`` reads one."""
    from poraque.fields import ChargeDensity, FieldGrid
    from poraque.fields.vasp.poscar import Poscar

    cell = np.eye(3) * 6.0
    grid = FieldGrid((12, 12, 12), cell)
    structure = Poscar(cell, ["Si"], [2], [[0.0, 0.0, 0.0],
                                           [0.25, 0.25, 0.25]])
    values = 1.0 + 0.1 * np.cos(2 * np.pi * np.arange(12)[:, None, None] / 12)
    path = tmp_path / "CHGCAR"
    ChargeDensity(np.broadcast_to(values, grid.shape).copy(), grid,
                  structure).write(path)
    return str(path)


class TestEveryCommandReportsInBlocks:
    """
    One cheap path through each migrated command, read back with the reference
    reader. Each test names the blocks that path has to produce, so a command
    that drifts back to a frame of its own fails by name.
    """

    def test_poraque_atoms_build_and_show(self, single_atom, tmp_path, capsys):
        atoms = _script("poraque_atoms")
        output = str(tmp_path / "atomic_reference.json")

        assert atoms.main([single_atom, "--output", output]) == 0
        built = capsys.readouterr().out
        assert {"SYSTEM", "PYTHON", "DEPENDENCIES"} <= set(_names(built))
        assert _names(built)[-4:] == ["INPUT", "ATOMS", "OUTPUT FILES", "USAGE"]
        assert any(line.startswith("Si") for line in _body(built, "ATOMS"))

        assert atoms.main(["--show", output]) == 0
        shown = capsys.readouterr().out
        assert _names(shown)[-3:] == ["DATABASE", "ENTRIES", "NOTES"]
        rows = _body(shown, "ENTRIES")
        assert rows[0].split()[:2] == ["element", "Zval"]
        assert rows[2].split()[0] == "Si"

    def test_poraque_atoms_on_an_empty_database(self, tmp_path, capsys):
        atoms = _script("poraque_atoms")
        assert atoms.main(["--show", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert _names(out)[-1] == "DATABASE"
        assert "no isolated-atom references" in "\n".join(
            _body(out, "DATABASE"))

    def test_poraque_vasp(self, density, tmp_path, capsys):
        vasp = _script("poraque_vasp")
        assert vasp.main(["dos", density, "--encut", "300", "--mesh", "4",
                          "4", "4", "--output", str(tmp_path / "dos")]) == 0
        out = capsys.readouterr().out
        assert _names(out)[-4:] == ["DENSITY", "SETTINGS", "OUTPUT FILES",
                                    "NEXT STEPS"]
        settings = _body(out, "SETTINGS")
        assert settings[0].startswith("ENCUT")
        assert any(line.startswith("k-mesh") for line in settings)

    def test_poraque_vasp_with_a_plain_callable(self, density, tmp_path):
        """
        ``run`` keeps taking any callable. Handed ``list.append`` it still
        writes the headers, as plain lines.
        """
        vasp = _script("poraque_vasp")
        args = vasp.build_parser().parse_args(
            ["energy", density, "--encut", "300", "--output",
             str(tmp_path / "energy")])
        lines = []
        vasp.run(args, log=lines.append)
        assert [line for line in lines if line.startswith("[")] == [
            "[DENSITY]", "[SETTINGS]", "[OUTPUT FILES]", "[NEXT STEPS]"]

    def test_poraque_committee(self):
        committee = _script("poraque_committee")
        records = [{"material": f"m{i}", "jsd": 0.1 * (i + 1),
                    "jsd_normalized": 0.05 * (i + 1),
                    "relative": 0.2 * (i + 1),
                    "integral_relative": 0.01, "error": 0.3 * (i + 1),
                    "ratio": 1.0} for i in range(4)]
        records.append({"material": "signed", "jsd": None,
                        "jsd_normalized": None, "relative": 0.9,
                        "integral_relative": 0.1, "error": 0.5,
                        "ratio": 1.0})
        log, stream = _capture()
        committee.report_ranking(records, log)
        committee.report_calibration(
            records, log, truth={f"m{i}": 0.1 * i for i in range(4)},
            against="kfold.json")
        out = stream.getvalue()
        assert _names(out) == ["RANKING", "CALIBRATION", "MEASURED ERRORS"]
        assert "1 structure(s) have no JSD" in "\n".join(_body(out, "RANKING"))
        assert any("Spearman" in line for line in _body(out, "CALIBRATION"))

    def test_poraque_active_learning(self):
        active = _script("poraque_active_learning")
        from poraque.ml.active_learning import jsd_statistics, select_top_k

        records = [{"material": f"mp-{i}", "jsd": 0.01 * (i + 1),
                    "jsd_normalized": 0.02 * (i + 1),
                    "relative": 0.1, "integral_relative": 0.01}
                   for i in range(5)]
        result = {"statistics": jsd_statistics(records), "records": records,
                  "selection": select_top_k(records, 2)}
        log, stream = _capture()
        active.report_round(result, log, round_number=3)
        out = stream.getvalue()
        assert _names(out) == ["STATISTICS", "RANKING", "SELECTION"]
        assert "round 3" in _body(out, "STATISTICS")[0]
        assert _body(out, "SELECTION")[0] == "selected 2 structure(s):"

    def test_poraque_inference(self, tmp_path, capsys):
        """
        The one command with a file of its own: ``inference.log`` beside the
        prediction, which has to say exactly what the terminal said.
        """
        import torch

        from poraque.ml import BUNDLE_FILENAME, FieldOperator, save_bundle

        inference = _script("poraque_inference")
        directory = tmp_path / "Pt2"
        directory.mkdir()
        (directory / "POSCAR").write_text(
            "Pt2\n1.0\n 4.08 0 0\n 0 4.08 0\n 0 0 4.08\n Pt\n 2\nDirect\n"
            " 0 0 0\n 0.5 0.5 0.5\n")
        built = {task: FieldOperator(task, width=4, modes=2, n_layers=1,
                                     projection_channels=8, device="cpu",
                                     training_resolution=8)
                 for task in ("ext2chg", "chg2tau")}
        with torch.no_grad():
            built["ext2chg"].model.project[-1].bias.fill_(4.0)
        bundle = save_bundle(str(tmp_path / BUNDLE_FILENAME), built)

        output = tmp_path / "predictions"
        inference.predict([str(directory), "--models", bundle,
                           "--output", str(output), "--grid", "8", "8", "8",
                           "--zval", "Pt=10", "--functional", "skip",
                           "--device", "cpu", "--compare"])
        out = capsys.readouterr().out
        names = _names(out)
        for name in ("RUNTIME", "INPUT", "GRID", "POTENTIAL", "DENSITY",
                     "KINETIC ENERGY DENSITY", "COMPARISON", "OUTPUT FILES"):
            assert name in names, f"[{name}] missing from poraque-inference"
        assert names[-1] == "OUTPUT FILES"
        assert (output / "inference.log").read_text(encoding="utf-8") == out

    def test_poraque_mp_estimate(self):
        from poraque.data.materials_project import Estimate

        sizes = {f"mp-{i}": int(2e6 * (i + 1)) for i in range(6)}
        rows = [{"material_id": key, "formula_pretty": "Si", "nsites": 2,
                 "size_mb": value / 1e6} for key, value in sizes.items()]
        text = str(Estimate(elements=["Si"], n_materials=8, n_advertised=60,
                            n_available=6, sizes=sizes, rows=rows,
                            method="sampled", sampled=6, seed=3))
        assert _names(text) == ["ESTIMATE", "PROJECTION", "STORAGE",
                                "FILE SIZES", "SIZE CAPS", "LARGEST FILES"]
        assert "SAMPLED AT RANDOM:" in "\n".join(_body(text, "ESTIMATE"))

    def test_poraque_mp_fetcher_reports_through_the_logger(self, tmp_path):
        from poraque.data.materials_project import MPDataFetcher

        log, stream = _capture()
        fetcher = MPDataFetcher(["Si"], api_key="unused",
                                outdir=str(tmp_path), log=log)
        fetcher.decompress()
        fetcher.recompress()
        out = stream.getvalue()
        assert _names(out) == ["DECOMPRESS", "RECOMPRESS"]
        assert _body(out, "DECOMPRESS") == [
            f"No gzipped CHGCARs under {fetcher.outdir}"]

    def test_the_c_backend_report(self, capsys):
        from poraque.ml import backend

        status = backend.main([])
        out = capsys.readouterr().out
        assert _names(out)[:2] == ["CONFIGURATION", "STATUS"]
        assert _names(out)[-1] == "NEXT STEPS"
        assert status in (0, 1)

    def test_the_c_backend_report_is_silent_when_quiet(self, capsys):
        from poraque.ml import backend

        backend.main(["--quiet"])
        assert capsys.readouterr().out == ""
