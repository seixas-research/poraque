# -*- coding: utf-8 -*-
# file: test_vis.py

"""
Tests for the figure and PDF-report machinery.

Two classes of defect are covered, both of which produce a file that looks
fine to the code and is unreadable to a human: tick labels that overprint each
other, and a table cell that runs off the page. Neither raises, so neither is
caught by asserting that a figure was written.
"""

import csv
import os
import re
import shutil

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from poraque.vis.pdf_report import (  # noqa: E402
    _escape,
    _format_value,
    _wrappable,
)
from poraque.vis.report import (  # noqa: E402
    TrainingReport,
    _rotate_if_crowded,
    _thin_log_minor_labels,
)


@pytest.fixture
def fields():
    """A reference field and a prediction that tracks it closely."""
    rng = np.random.default_rng(0)
    reference = np.exp(rng.normal(1.0, 0.8, size=(12, 12, 12)))
    return reference, reference * (1 + 0.02 * rng.standard_normal(reference.shape))


def _panel_meshes(figure):
    """The density meshes of the data panels, excluding the colourbar's own."""
    return [collection
            for axis in figure.axes if axis.get_xlabel()
            for collection in axis.collections
            if hasattr(collection, "get_coordinates")]


@pytest.fixture
def captured(monkeypatch):
    """
    Hold on to the figure a report writes.

    `TrainingReport` closes each figure as it saves it, so the only way to
    assert on what was actually drawn is to intercept it at the save.
    """
    from poraque.vis import report as report_module

    class Captured:
        figure = None

    holder = Captured()
    original = report_module.TrainingReport._save

    def capture(self, figure, name):
        holder.figure = figure
        return original(self, figure, name)

    monkeypatch.setattr(report_module.TrainingReport, "_save", capture)
    return holder


# ===================================================================== #
# Log-axis tick decluttering
# ===================================================================== #
class TestLogTickLabels:
    """
    Over a range narrower than ~2 decades Matplotlib labels the minor log
    ticks too, and horizontally they overlap into unreadable mush.
    """

    def _labelled_minors(self, panel):
        panel.figure.draw_without_rendering()
        return [t.get_text() for t in panel.get_xticklabels(which="minor")
                if t.get_text()]

    def test_thins_the_minor_labels(self):
        figure, panel = plt.subplots()
        panel.set_xscale("log")
        panel.set_xlim(1.2, 61.0)          # the tau range that triggered it
        before = len(self._labelled_minors(panel))
        _thin_log_minor_labels(panel)
        after = len(self._labelled_minors(panel))
        assert before > after
        plt.close(figure)

    def test_leaves_a_linear_axis_alone(self):
        """A linear axis is spaced by MaxNLocator and needs no help."""
        figure, panel = plt.subplots()
        panel.set_xlim(0, 5)
        locator = panel.xaxis.get_minor_locator()
        _thin_log_minor_labels(panel)
        assert panel.xaxis.get_minor_locator() is locator
        plt.close(figure)


class TestRotateIfCrowded:
    """Rotation is a cost — tilted labels are harder to read — so it is
    applied only when the labels actually collide."""

    def test_rotates_when_labels_overlap(self):
        figure, panel = plt.subplots(figsize=(1.4, 1.4))
        panel.set_xlim(0, 1)
        panel.set_xticks(np.linspace(0, 1, 12))
        panel.set_xticklabels([f"{v:.5f}" for v in np.linspace(0, 1, 12)])
        assert _rotate_if_crowded(figure, panel) is True
        assert all(label.get_rotation() > 0
                   for label in panel.get_xticklabels() if label.get_text())
        plt.close(figure)

    def test_leaves_roomy_labels_upright(self):
        figure, panel = plt.subplots(figsize=(6.0, 4.0))
        panel.set_xlim(0, 5)
        panel.set_xticks([0, 1, 2, 3, 4, 5])
        assert _rotate_if_crowded(figure, panel) is False
        assert all(label.get_rotation() == 0
                   for label in panel.get_xticklabels() if label.get_text())
        plt.close(figure)

    def test_parity_x_labels_do_not_overlap(self, fields, tmp_path):
        """End to end: the axis that shipped the overlapping labels."""
        reference, prediction = fields
        report = TrainingReport(tmp_path)
        report.parity(reference, prediction, log=True, name="p")

        # `parity` closes its figure, so re-measure on an equivalent axis.
        figure, panel = plt.subplots(figsize=(5.6, 5.4))
        panel.set_xscale("log")
        panel.set_xlim(float(reference.min()), float(reference.max()))
        _thin_log_minor_labels(panel)
        _rotate_if_crowded(figure, panel)
        figure.draw_without_rendering()
        boxes = sorted((t.get_window_extent()
                        for t in panel.get_xticklabels(which="both")
                        if t.get_text()), key=lambda b: b.x0)
        assert all(later.x0 >= earlier.x1
                   for earlier, later in zip(boxes, boxes[1:]))
        plt.close(figure)


# ===================================================================== #
# Train vs validation on one parity plot
# ===================================================================== #
class TestParitySplits:
    def test_single_set_still_writes_a_density_map(self, fields, tmp_path):
        """The one-set path must be untouched by the two-set feature."""
        reference, prediction = fields
        path = TrainingReport(tmp_path).parity(reference, prediction, name="one")
        assert os.path.exists(path)

    def test_draws_both_sets(self, fields, tmp_path):
        reference, prediction = fields
        held_out = (reference * 1.3, reference * 1.3 * 1.08)
        path = TrainingReport(tmp_path).parity(
            reference, prediction, validation=held_out, name="two")
        assert os.path.exists(path)

    def test_uses_one_panel_per_set(self, fields, tmp_path, captured):
        """Two binned maps overlaid are unreadable, so they go side by side."""
        reference, prediction = fields
        TrainingReport(tmp_path).parity(
            reference, prediction, validation=(reference, reference * 1.1),
            name="panels")
        assert len(captured.figure.axes) >= 2       # two panels + a colourbar

    def test_a_single_set_keeps_one_panel(self, fields, tmp_path, captured):
        reference, prediction = fields
        TrainingReport(tmp_path).parity(reference, prediction, name="one_panel")
        panels = [ax for ax in captured.figure.axes if ax.get_xlabel()]
        assert len(panels) == 1

    def test_panels_share_axis_limits(self, fields, tmp_path, captured):
        """A fair comparison needs both panels on the same scale."""
        reference, prediction = fields
        TrainingReport(tmp_path).parity(
            reference, prediction,
            validation=(reference * 4.0, reference * 4.0 * 1.1), name="limits")
        panels = [ax for ax in captured.figure.axes if ax.get_xlabel()]
        assert len({ax.get_xlim() for ax in panels}) == 1
        assert len({ax.get_ylim() for ax in panels}) == 1

    def test_panels_share_one_colour_scale(self, fields, tmp_path, captured):
        """
        Separate colourbars would each autoscale, so two different densities
        could wear the same colour -- the one lie this plot must not tell.
        """
        reference, prediction = fields
        TrainingReport(tmp_path).parity(
            reference, prediction, validation=(reference, reference * 1.1),
            name="norm")
        meshes = _panel_meshes(captured.figure)
        assert len(meshes) == 2
        assert len({mesh.get_clim() for mesh in meshes}) == 1

    def test_each_panel_is_titled_with_its_metrics(self, fields, tmp_path,
                                                   captured):
        reference, prediction = fields
        TrainingReport(tmp_path).parity(
            reference, prediction, validation=(reference, reference * 1.1),
            name="titles")
        titles = [ax.get_title() for ax in captured.figure.axes if ax.get_title()]
        assert any(t.startswith("training") for t in titles)
        assert any(t.startswith("validation") for t in titles)

    def test_bins_are_shared_so_a_cell_means_the_same_thing(self, fields,
                                                            tmp_path, captured):
        reference, prediction = fields
        TrainingReport(tmp_path).parity(
            reference, prediction, validation=(reference, reference * 1.1),
            name="bins", bins=24)
        meshes = _panel_meshes(captured.figure)
        assert len(meshes) == 2
        first, second = (mesh.get_coordinates() for mesh in meshes)
        assert np.allclose(first, second)

    def test_validation_may_be_a_different_size(self, fields, tmp_path):
        """Train and validation are different structures, not a split of one."""
        reference, prediction = fields
        small = np.exp(np.random.default_rng(1).normal(1.0, 0.8, size=(8, 8, 8)))
        path = TrainingReport(tmp_path).parity(
            reference, prediction, validation=(small, small * 1.05),
            name="sizes")
        assert os.path.exists(path)

    def test_rejects_a_mismatched_validation_pair(self, fields, tmp_path):
        reference, prediction = fields
        with pytest.raises(ValueError, match="same size"):
            TrainingReport(tmp_path).parity(
                reference, prediction,
                validation=(np.ones((4, 4, 4)), np.ones((5, 5, 5))), name="x")

    def test_rejects_a_non_pair(self, fields, tmp_path):
        reference, prediction = fields
        with pytest.raises(ValueError, match="reference, prediction"):
            TrainingReport(tmp_path).parity(
                reference, prediction, validation=np.ones((4, 4, 4)), name="x")


# ===================================================================== #
# LaTeX table overflow
# ===================================================================== #
class TestWrappableValues:
    """
    `training.physics_informed` is 140 characters of nested dict. In
    an `l` column --
    a single unbreakable box -- it ran past the right margin of the PDF.
    """

    def test_short_values_are_left_alone(self):
        assert _wrappable("cosine") == "cosine"
        assert r"\allowbreak" not in _wrappable("0.002")

    def test_long_unbroken_values_gain_break_points(self):
        path = "logs/very/deeply/nested/output/directory/fno_training_run.json"
        wrapped = _wrappable(path)
        assert r"\allowbreak{}" in wrapped
        assert wrapped.count(r"\allowbreak{}") >= path.count("/")

    def test_escaping_still_applies(self):
        wrapped = _wrappable("a_" + "b" * 40)
        assert r"\_" in wrapped
        assert "_" not in wrapped.replace(r"\_", "")

    def test_break_markup_is_not_reprocessed(self):
        """`\\allowbreak{}` must not itself be split by a later separator."""
        wrapped = _wrappable("x" * 30 + "/y")
        assert r"\allowbreak{}\allowbreak{}" not in wrapped

    def test_physics_dict_can_wrap(self):
        value = ("{'enable': 'auto', 'electron_count_weight': 0.0, "
                 "'positivity_weight': 0.0, "
                 "'von_weizsacker_weight': 0.0, 'euler_lagrange_weight': 0.0}")
        wrapped = _wrappable(value)
        # Spaces already give a p-column somewhere to break.
        assert " " in wrapped
        assert _escape("{") in wrapped


class TestConfigurationKeys:
    """
    The key column is fixed-width, so a long key must be *breakable*, not just
    escaped: `fine_tuning.pretrained_checkpoint` is 33 characters with no
    space, and one unbreakable word overruns its column and prints on top of
    the value in the next one.

    The example used to be `symbolic.enable_symbolic_distillation`, which at 37
    characters was the longest key the schema produced and what
    `CONFIG_KEY_FRACTION` was sized against. It became `symbolic.enable` in
    26.9.8; the column was deliberately not retightened, because a fraction
    trimmed to today's longest key is one the next key breaks.
    """

    LONG = "fine_tuning.pretrained_checkpoint"

    def test_a_long_key_gains_break_points(self):
        rendered = _wrappable(self.LONG)
        assert r"\allowbreak{}" in rendered

    def test_no_fragment_is_too_wide_for_the_column(self):
        """Every piece between break points must fit p{4.6cm}."""
        pieces = _wrappable(self.LONG).split(r"\allowbreak{}")
        longest = max(len(piece.replace("\\_", "_")) for piece in pieces)
        assert longest <= 22          # ~4.2 cm at 11pt, inside the 4.6 cm cell

    def test_short_keys_are_left_alone(self):
        for key in ("training.epochs", "model.projection_channels",
                    "symbolic.epsilon"):
            assert r"\allowbreak" not in _wrappable(key)

    def test_decimal_values_are_never_split(self):
        """
        A break after `.` would fix keys and wreck numbers, so `.` is
        deliberately not a break character.
        """
        for value in ("0.0032", "1e-08", "0.002", "1.0"):
            assert r"\allowbreak" not in _wrappable(value)

    def test_every_real_config_key_fits(self):
        from poraque.ml.config import TrainingConfig

        sections = TrainingConfig().to_dict()
        keys = [f"{section}.{key}" for section, values in sections.items()
                if isinstance(values, dict) for key in values]
        for key in keys:
            pieces = _wrappable(key).split(r"\allowbreak{}")
            assert max(len(p.replace("\\_", "_")) for p in pieces) <= 26


class TestNestedConfigValues:
    """
    `training.physics_informed` is a dict, and since 26.9.8 so are
    `model.equivariant` and `symbolic.physics`. Printed as a repr it is one
    unreadable 140-character run; it belongs in the table as one setting per
    line.
    """

    PHYSICS = {"enable": "auto", "electron_count_weight": 0.0,
               "positivity_weight": 0.0,
               "von_weizsacker_weight": 0.0, "euler_lagrange_weight": 0.0}

    def test_each_entry_gets_its_own_line(self):
        rendered = _format_value(self.PHYSICS)
        assert rendered.count(r"\newline") == len(self.PHYSICS) - 1
        for key in self.PHYSICS:
            assert _escape(key) in rendered

    def test_breaks_the_line_not_the_table_row(self):
        r"""
        `\\` inside a p-column ends the table ROW, which would split one
        setting across two rows and desynchronise the whole table from its
        keys. Only `\newline` breaks within the cell.
        """
        rendered = _format_value(self.PHYSICS)
        assert r"\\" not in rendered

    def test_keys_are_escaped(self):
        """Underscores in the keys would otherwise be subscript markup."""
        rendered = _format_value({"von_weizsacker_weight": 0.0})
        assert r"\_" in rendered
        assert "_" not in rendered.replace(r"\_", "")

    def test_scalars_are_unaffected(self):
        assert _format_value("cosine") == _wrappable("cosine")
        assert _format_value(300) == _wrappable(300)

    def test_an_empty_dict_falls_back_to_the_scalar_path(self):
        """`{}` has nothing to list; it should read as an empty mapping."""
        assert r"\newline" not in _format_value({})

    def test_a_real_config_renders(self):
        from poraque.ml.config import TrainingConfig

        physics = TrainingConfig().training.physics_informed
        rendered = _format_value(physics)
        assert rendered.count(r"\newline") == len(physics) - 1

    def test_every_block_in_the_schema_renders(self):
        """
        Three blocks now, not one, and each is a dict in the table.

        `model.equivariant` is the interesting one: it carries a single key by
        default, so it exercises the one-entry case that produces no
        `\newline` at all.
        """
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        for block in (config.model.equivariant, config.training.physics_informed,
                      config.symbolic.physics):
            rendered = _format_value(block)
            assert rendered.count(r"\newline") == len(block) - 1
            for key in block:
                assert _escape(key) in rendered


class TestDescribeNesting:
    """The same clustering in the terminal run header."""

    def test_physics_is_listed_not_inlined(self):
        from poraque.ml.config import TrainingConfig

        described = TrainingConfig().describe()
        assert "physics_informed={" not in described
        assert "  physics_informed:" in described
        for key in TrainingConfig().training.physics_informed:
            assert any(line.strip().startswith(key)
                       for line in described.splitlines())

    def test_the_dict_no_longer_rides_on_the_training_line(self):
        """
        The scalar settings are still inlined -- that format is unchanged.
        What moved off the line is the 140-character physics repr, which is
        what made it unreadable.
        """
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        training = next(line for line in config.describe().splitlines()
                        if line.startswith("training:"))
        inlined = (len(training)
                   + len(str(config.training.physics_informed)))
        assert len(training) < inlined - 100
        assert "electron_count_weight" not in training


# ===================================================================== #
# Report style and the symbolic parity plot
# ===================================================================== #
class TestReportStyle:
    """
    The generated report must read as part of the same family as the guides.

    Checked on the LaTeX source rather than the rendered PDF: these are
    statements about which palette and furniture the document declares, and
    reading them off the source is exact where pixel comparison is not.
    """

    @pytest.fixture
    def preamble(self):
        from poraque.vis.pdf_report import ModelReport

        return ModelReport("reports", logo=None)._preamble("chg2tau", "subtitle")

    def test_uses_the_four_colour_brand_palette(self, preamble):
        """
        Four colours, and the yellow is the logo's own.

        Earlier revisions used a blue that appeared nowhere else, then a red
        that appeared only here. The palette is now a single sweep of hues --
        48 -> 77 -> 160 -> 163 degrees -- anchored on the yellow the logo is
        drawn in, with the last reserved for cover grounds.
        """
        assert r"\definecolor{poraqueyellow}{RGB}{255,204,0}" in preamble
        assert r"\definecolor{poraquelime}{RGB}{163,198,75}" in preamble
        assert r"\definecolor{poraquegreen}{RGB}{15,61,46}" in preamble
        assert r"\definecolor{poraquecover}{RGB}{6,35,27}" in preamble
        assert "poraqueblue" not in preamble

    def test_the_cover_ground_is_used_only_as_a_cover_ground(self, preamble):
        """
        ``poraquecover`` has exactly one permitted use, and the name says so.

        Counted rather than merely present: one definition and one use, the
        masthead panel. A third occurrence means it has leaked into ordinary
        furniture, where it is indistinguishable from ``poraquegreen`` at
        1.37:1 and buys nothing.
        """
        uses = [line.strip() for line in preamble.splitlines()
                if "poraquecover" in line and not line.strip().startswith("%")]
        # One \definecolor, and one panel that sets colback and colframe.
        assert len(uses) == 2, uses
        assert uses[0].startswith(r"\definecolor{poraquecover}")
        assert "colback=poraquecover" in uses[1]

    def test_the_yellow_is_never_used_for_text(self, preamble):
        """
        Yellow on white is 1.7:1 -- far below the 4.5:1 body-text floor and
        below even the 3:1 large-text one. It carries rules and bands; the
        green (12.2:1) carries anything that has to be read.
        """
        for role in (r"coltitle=poraqueyellow", r"\color{poraqueyellow}\thesection"):
            assert role not in preamble, role

    def test_the_masthead_is_the_cover_ground(self, preamble):
        """
        The report's masthead is its cover, so it takes the cover ground
        rather than the anchor green the headings use.
        """
        assert "colback=poraquecover" in preamble
        assert "colback=poraquegreen" not in preamble
        assert "poraqueyellow" in preamble

    def test_the_masthead_uses_the_dark_ground_logo(self):
        """
        The banner is dark green, so it takes the logo drawn for dark grounds.

        An absolute path because the suite runs from a scratch directory: with
        a relative one the logo simply would not exist and the branch under
        test would never be reached.
        """
        from poraque.vis.pdf_report import ModelReport

        logo = os.path.join(os.path.dirname(__file__), "..", "assets", "logo",
                            "logo_light.png")
        if not os.path.exists(logo):
            pytest.skip("logo assets not present")
        source = ModelReport("reports", logo=logo)._preamble("chg2tau", "sub")
        assert "logo_dark.png" in source

    def test_the_dark_logo_reaches_the_compile_directory(self, tmp_path):
        """
        Without the copy the banner compiles to a missing-figure box: the
        source references logo_dark.png but only logo.png was shipped.
        """
        from poraque.vis.pdf_report import ModelReport

        logo = os.path.join(os.path.dirname(__file__), "..", "assets", "logo",
                            "logo_light.png")
        if not os.path.exists(logo):
            pytest.skip("logo assets not present")

        report = ModelReport(str(tmp_path), logo=logo)
        seen = {}

        def fake_toolchain(stem):
            return None            # stop before pdflatex; we want the workdir

        report._toolchain = staticmethod(fake_toolchain)
        original = shutil.copy

        def spy(src, dst):
            seen[os.path.basename(str(src))] = True
            return original(src, dst)

        shutil.copy = spy
        try:
            report.build(task="chg2tau", per_material={}, unit="eV")
        finally:
            shutil.copy = original
        assert "logo_dark.png" in seen

    def test_shares_the_guides_geometry(self, preamble):
        assert "margin=2.5cm" in preamble
        assert "headheight=26pt" in preamble

    def test_has_the_guides_red_header_rule(self, preamble):
        assert r"\renewcommand{\headrulewidth}{0.8pt}" in preamble
        assert r"\headrule" in preamble

    def test_defines_the_guides_callout_boxes(self, preamble):
        assert r"\newtcolorbox{pwarn}" in preamble
        assert r"\newtcolorbox{pnote}" in preamble

    def test_section_numbers_take_the_accent(self, preamble):
        assert r"\textcolor{poraquered}{\thesection}" in preamble

    def test_caveats_go_in_the_warning_box(self, tmp_path):
        """
        The most skippable part of the report gets the most visible frame.

        Asserted on the source through a stubbed compile, since a caveat list
        that silently became a plain itemize would still produce a valid PDF.
        """
        from poraque.vis.pdf_report import ModelReport

        report = ModelReport(str(tmp_path), logo=None)
        captured = {}

        def fake_compile(source, figures, target):
            captured["source"] = source
            return target

        report._compile = fake_compile
        report.build(task="chg2tau", per_material={}, unit="eV",
                     caveats=["one element only"])
        assert r"\begin{pwarn}" in captured["source"]
        assert "one element only" in captured["source"]


class TestSymbolicParityPlot:
    """
    The parity plot must reach the PDF whenever distillation produced one.

    It is the only figure that shows whether the closed form tracks the data,
    so a report that quietly omits it is worse than one that says it is a
    training fit.
    """

    @pytest.fixture
    def report_source(self, tmp_path):
        from poraque.vis.pdf_report import ModelReport

        def build(symbolic):
            report = ModelReport(str(tmp_path), logo=None)
            captured = {}

            def fake_compile(source, figures, target):
                captured["source"] = source
                captured["figures"] = list(figures)
                return target

            report._compile = fake_compile
            report.build(task="chg2tau", per_material={}, unit="eV",
                         symbolic=symbolic)
            return captured

        return build

    @pytest.fixture
    def parity_file(self, tmp_path):
        path = tmp_path / "chg2tau_symbolic_parity.png"
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.savefig(path)
        plt.close()
        return str(path)

    def _symbolic(self, parity, validated):
        payload = {
            "expression": "1 - 5*p**2/3", "latex": "1", "full_latex": r"\tau = 1",
            "complexity": 7, "r2": 0.98, "relative_l2": 0.04,
            "n_samples": 1000, "scheme": "reduced", "units": "atomic",
            "target": "model", "target_name": "tau", "template": "pauli",
            "parity_plot": parity, "validation": {}, "fitted": {},
        }
        if validated:
            payload["validation"] = {"n_points": 500, "relative_l2": 0.06,
                                     "r2": 0.95}
        return payload

    def test_is_embedded_and_shipped_to_the_compiler(self, report_source,
                                                     parity_file):
        captured = report_source(self._symbolic(parity_file, validated=True))
        assert os.path.basename(parity_file) in captured["source"]
        assert parity_file in captured["figures"], (
            "the plot must also be copied into the compile directory")

    def test_caption_says_held_out_when_something_was(self, report_source,
                                                     parity_file):
        captured = report_source(self._symbolic(parity_file, validated=True))
        assert "on held-out structures" in captured["source"]

    def test_caption_admits_a_training_fit_when_nothing_was_held_out(
            self, report_source, parity_file):
        """
        The distinction the caption must not blur.

        With no validation split the plot is drawn on the voxels the search was
        fitted to, and calling that "held-out" would overstate it.
        """
        captured = report_source(self._symbolic(parity_file, validated=False))
        assert "voxels it was fitted to" in captured["source"]
        assert "on held-out structures" not in captured["source"]

    def test_a_missing_file_is_skipped_rather_than_breaking_the_build(
            self, report_source):
        captured = report_source(self._symbolic("/nonexistent/parity.png",
                                                validated=True))
        assert "nonexistent" not in captured["source"]

    def test_a_constrained_search_says_so_beside_its_front(self, report_source,
                                                           parity_file):
        """
        A constrained objective is not comparable with an unconstrained one, and
        the loss column looks identical either way. The page has to say which.
        """
        payload = self._symbolic(parity_file, validated=True)
        payload["constraints_enforced"] = ["positivity", "thomas_fermi"]
        captured = report_source(payload)
        assert "penalised inside" in captured["source"]
        assert "thomas\\_fermi" not in captured["source"], (
            "constraint names are prose here, not identifiers")
        assert "thomas fermi" in captured["source"]

    def test_an_unconstrained_search_claims_nothing(self, report_source,
                                                    parity_file):
        captured = report_source(self._symbolic(parity_file, validated=True))
        assert "penalised inside" not in captured["source"]


class TestThePerformanceTableIsBoundedBySplits:
    """
    The per-structure table became a per-split one, and the reason is length.

    It used to print one row per structure, so its length was the size of the
    dataset: 115 MP materials made six pages of seven-column numbers, and the
    two things a reader wanted from it -- how many structures, and how well did
    it do -- had to be recovered by scrolling and averaging. Both are printed
    directly now.

    The three tests this class replaces asserted that the table was a
    ``longtable`` with a repeating header, because a ``tabular`` that outgrows
    the page loses its overflow silently (88 of 140 structures, in the case
    that prompted them). That requirement has not been waived -- it has stopped
    applying, because the row count is now the number of splits plus one. A
    ``tabular`` is the right container for a table that cannot grow, and
    `test_the_table_no_longer_grows_with_the_dataset` is what holds that claim
    to the reason for it rather than to the change.
    """

    #: Every metric the split table can print, plus the ones it deliberately
    #: does not. A fixture carrying only the printed five would not notice a
    #: column quietly appearing for one of the others.
    METRICS = {"mse": 1e-6, "mae": 2e-4, "rmse": 1e-3, "max_abs": 9.9e-3,
               "relative_l2": 0.01, "relative_h1": 0.02,
               "integral_error": 0.004, "nrmse_range": 0.005, "r2": 0.999,
               "jsd": 5e-7}

    @classmethod
    def _per_material(cls, n, held_out=lambda i: i % 5 == 0, **override):
        metrics = {**cls.METRICS, **override}
        return {f"mp-{i:06d}": {"split": "validation" if held_out(i) else "train",
                                "metrics": dict(metrics)}
                for i in range(n)}

    def _table(self, per_material, unit="e/Ang^3"):
        from poraque.vis.pdf_report import ModelReport

        return ModelReport(logo=None)._metrics_table(per_material, unit)

    def test_the_table_no_longer_grows_with_the_dataset(self):
        rows = [self._table(self._per_material(n)).count(r"\\")
                for n in (8, 120, 1200)]
        assert len(set(rows)) == 1, (
            f"the row count still tracks the dataset size: {rows}")

    def test_no_structure_is_named(self):
        source = self._table(self._per_material(120))
        assert "mp-000007" not in source, (
            "the list of materials is what made the report unreadable")

    def test_it_counts_the_dataset_and_both_splits(self):
        source = self._table(self._per_material(100))
        # 20 held out by `i % 5 == 0`, 80 trained on, 100 in total.
        assert "train & 80 &" in source
        assert "validation & 20 &" in source
        assert r"\textbf{all} & \textbf{100} &" in source

    def test_a_run_with_nothing_held_out_prints_no_validation_row(self):
        """
        An empty split and a split that scored badly must not look alike.

        A ``validation`` row of dashes reads as a measurement that came out
        undefined; there was no measurement.
        """
        source = self._table(self._per_material(12, held_out=lambda i: False))
        assert "validation" not in source
        assert "train & 12 &" in source
        # And no total row either: with one split it would repeat the row
        # above it verbatim.
        assert r"\textbf{all}" not in source

    def test_the_five_columns_are_there_and_no_others(self):
        """
        Five metrics, and the table stops there because the page does.

        ``mse``, ``rmse``, ``r2``, ``nrmse_range`` and ``jsd`` are still
        written per structure into the metrics JSON, which is machine-readable
        and has no margins. What they are not is typeset: a ten-column table
        overflows the text block, and the five kept are the five that answer
        questions the others cannot.
        """
        source = self._table(self._per_material(8))
        for present in (r"rel.\ $L^2$", r"rel.\ $H^1$", "MAE",
                        r"Max.\ error", r"$|\Delta N|$"):
            assert present in source, present
        for absent in ("MSE", "RMSE", "$R^2$", "JSD"):
            assert absent not in source, absent
        # Split, count, and exactly five metrics.
        assert source.count("r") and r"@{}l" + "r" * 6 + r"@{}" in source

    def test_the_means_are_still_there(self):
        source = self._table(self._per_material(8))
        assert "0.01" in source and "0.02" in source, (
            "the means are what the per-structure rows were being read for")

    def test_the_headers_are_bold(self):
        source = self._table(self._per_material(8))
        assert r"\textbf{Split}" in source
        assert r"\textbf{Structures}" in source
        assert r"\textbf{MAE}" in source

    def test_the_train_and_validation_rows_show_the_generalisation_gap(self):
        """
        The whole reason the splits are rows: they are read against each other.

        Spread over a hundred rows sorted by material id, the same comparison
        needed the reader to average two subsets by hand.
        """
        per_material = self._per_material(10)
        for name, entry in per_material.items():
            if entry["split"] == "validation":
                entry["metrics"]["relative_l2"] = 0.04
        source = self._table(per_material)
        train = next(line for line in source.splitlines()
                     if line.startswith("train "))
        held = next(line for line in source.splitlines()
                    if line.startswith("validation "))
        assert "0.01" in train and "0.04" in held

    def test_a_metric_nothing_recorded_loses_its_column(self):
        """
        ``relative_h1`` and ``integral_error`` are ``None`` for a field whose
        shape is not the grid's. A column of dashes claims a measurement was
        attempted and failed; none was attempted.
        """
        source = self._table(self._per_material(
            8, relative_h1=None, integral_error=None))
        assert r"rel.\ $H^1$" not in source
        assert r"$|\Delta N|$" not in source
        assert r"rel.\ $L^2$" in source
        assert r"@{}l" + "r" * 4 + r"@{}" in source

    def test_nothing_at_all_says_so_rather_than_drawing_an_empty_table(self):
        assert "No per-structure metrics" in self._table({})

    def test_the_conservation_column_takes_the_integrated_unit(self):
        """
        ``e/Ang^3`` integrates to ``e``, ``eV/Ang^3`` to ``eV``.

        The number is small either way; only its unit says whether it is small
        enough, so a wrong one is worse than none.
        """
        assert "[e]" in self._table(self._per_material(4), unit="e/Ang^3")
        assert "[eV]" in self._table(self._per_material(4), unit="eV/Ang^3")

    def test_an_unrecognised_unit_is_not_guessed_at(self):
        source = self._table(self._per_material(4), unit="furlongs")
        assert "[furlongs]" in source, "the field's own unit is still printed"
        assert "per cell" in source, "but its integral's is not invented"

    def test_a_split_nobody_anticipated_is_printed_rather_than_dropped(self):
        """A fold label, a third split: unrecognised is not a reason to lose it."""
        source = self._table({
            "a": {"split": "fold 3", "metrics": {"mse": 1.0, "mae": 1.0,
                                                 "rmse": 1.0,
                                                 "relative_l2": 1.0, "r2": 1.0}},
            **self._per_material(4)})
        assert "fold 3 & 1 &" in source

    @pytest.mark.skipif(
        not (shutil.which("pdflatex") and shutil.which("pdftotext")),
        reason="needs a LaTeX toolchain and poppler's pdftotext")
    def test_the_counts_reach_the_page_and_the_material_list_does_not(
            self, tmp_path):
        """
        The end-to-end statement: compile the PDF and read the table back out.

        Its ancestor asserted the opposite -- that all 140 structure names
        reached the page -- and was the only check that would have caught the
        original overflow, because the source was valid LaTeX and the build
        reported success. The same reasoning applies to the replacement: a
        summary that is computed correctly and then typeset off the edge of the
        paper is indistinguishable from one that was never computed.
        """
        import subprocess

        from poraque.vis.pdf_report import ModelReport

        per_material = self._per_material(140)
        pdf = ModelReport(str(tmp_path), logo=None).build(
            task="chg2tau", per_material=per_material, unit="eV")
        assert pdf.endswith(".pdf"), "the LaTeX fallback path was taken"

        text = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                              capture_output=True, text=True,
                              check=True).stdout
        assert "mp-000007" not in text
        # 28 held out by `i % 5 == 0`, 112 trained on, 140 in total.
        for label, count in (("train", 112), ("validation", 28), ("all", 140)):
            assert re.search(rf"{label}\s+{count}\b", text), (
                f"the {label} count never reached the page")


class TestLongTableBreaking:
    r"""
    Every table whose length is set by the data must be able to turn the page.

    A ``tabular`` is one unbreakable box. LaTeX does not truncate it and does
    not fail: it prints as much as fits and lets the rest run off the bottom of
    the page, where it is simply not on the paper. The per-structure metrics
    table used to be the worst case and is no longer a case at all (see
    :class:`TestThePerformanceTableIsBoundedBySplits`); the Pareto front of a
    symbolic distillation still has as many rows as the search kept.
    """

    def test_the_pareto_front_header_repeats(self):
        from poraque.vis.pdf_report import ModelReport

        source = ModelReport(logo=None)._symbolic_block({
            "latex": "1", "full_latex": r"F = 1", "complexity": 3,
            "r2": 0.9, "relative_l2": 0.1, "n_samples": 10,
            "scheme": "reduced", "units": "atomic", "target": "model",
            "target_name": "F", "template": "pauli",
            "pareto": [{"complexity": c, "loss": 1e-3,
                        "limits": {"badge": "TF/vW"}, "expression": "x0"}
                       for c in range(1, 60)],
        })
        assert r"\endhead" in source



class TestRawPlotDataIsWrittenBesideTheFigure:
    r"""
    ``output.save_raw_plot_data`` writes the numbers behind each figure, so a
    plot can be redrawn for publication without re-running the model.

    Three properties matter more than the file existing. The sidecar shares
    the **stem** of the figure it belongs to, so the pairing is visible in a
    listing rather than reconstructed from a convention. It carries what was
    *drawn*, not a second computation of it — a file that disagreed with its
    own image would be worse than no file. And it is off unless asked for: a
    run that never wanted it should write nothing extra.
    """

    def _report(self, tmp_path, save_data=True):
        return TrainingReport(str(tmp_path), prefix="ext2chg",
                              save_data=save_data)

    def _history(self):
        return {"train_loss": [0.9, 0.5, 0.3, 0.21],
                "val_error": [0.8, 0.25], "val_epoch": [2, 4],
                "val_metric": "rel L2"}

    # -------------------------- off by default -------------------------- #
    def test_nothing_extra_is_written_when_it_is_off(self, tmp_path, fields):
        report = self._report(tmp_path, save_data=False)
        report.loss_curves(self._history())
        report.parity(*fields, bins=16)
        report.field_comparison(*fields)

        assert report.data_files == []
        assert not [n for n in os.listdir(tmp_path)
                    if n.endswith((".csv", ".npz"))]

    def test_the_default_is_off(self):
        assert TrainingReport("anywhere").save_data is False

    # ------------------------- the training curve ------------------------ #
    def test_the_curve_lands_beside_its_own_figure(self, tmp_path):
        figure = self._report(tmp_path).loss_curves(self._history())
        assert os.path.exists(os.path.splitext(figure)[0] + ".csv")

    def test_every_epoch_is_a_row_and_the_gaps_stay_empty(self, tmp_path):
        """
        `eval_epoch > 1` means validation exists only on some epochs. Carrying
        the last value forward, or writing a zero, would put points in the file
        that the run never measured.
        """
        self._report(tmp_path).loss_curves(self._history())
        rows = list(csv.DictReader(
            open(tmp_path / "ext2chg_loss_curves.csv", encoding="utf-8")))

        assert [row["epoch"] for row in rows] == ["1", "2", "3", "4"]
        assert [row["train_loss"] for row in rows] == ["0.9", "0.5", "0.3",
                                                       "0.21"]
        assert [row["val_rel_l2"] for row in rows] == ["", "0.8", "", "0.25"]

    def test_the_column_names_the_norm_it_holds(self, tmp_path):
        """A `loss: sobolev` run measures an H1; calling both columns
        "validation loss" would make the file unreadable a month later."""
        history = dict(self._history(), val_metric="rel H1")
        self._report(tmp_path).loss_curves(history)
        header = open(tmp_path / "ext2chg_loss_curves.csv",
                      encoding="utf-8").readline().strip()

        assert header == "epoch,train_loss,val_rel_h1"

    def test_a_run_without_validation_writes_one_column(self, tmp_path):
        self._report(tmp_path).loss_curves({"train_loss": [1.0, 0.5]})
        rows = list(csv.reader(
            open(tmp_path / "ext2chg_loss_curves.csv", encoding="utf-8")))

        assert rows[0] == ["epoch", "train_loss"]
        assert len(rows) == 3

    def test_there_is_no_physics_column(self, tmp_path):
        """
        `train_loss` is the total the optimiser stepped on, and the per-term
        breakdown was removed from `history` in 2026-08-28 because the
        components are reported unweighted and never summed to it. A
        `physics_loss` column here would be an invented number.
        """
        self._report(tmp_path).loss_curves(self._history())
        header = open(tmp_path / "ext2chg_loss_curves.csv",
                      encoding="utf-8").readline()

        assert "physics" not in header

    # ----------------------------- the parity ---------------------------- #
    def test_the_bins_account_for_every_voxel(self, tmp_path, fields):
        """
        The counts are the histogram's, so they must add up to the field --
        which is the check that says the file holds the drawn data rather than
        a re-binning of it.
        """
        reference, prediction = fields
        self._report(tmp_path).parity(reference, prediction, bins=16)
        rows = list(csv.DictReader(
            open(tmp_path / "ext2chg_parity.csv", encoding="utf-8")))

        assert sum(int(row["count"]) for row in rows) == reference.size
        assert sum(float(row["density"]) for row in rows) == pytest.approx(1.0)

    def test_each_split_is_normalised_to_its_own_voxels(self, tmp_path, fields):
        """
        Two structures need not have the same grid size, so the figure colours
        by each split's own share. The file has to say the same thing.
        """
        reference, prediction = fields
        held_out = (reference[:6], prediction[:6])
        self._report(tmp_path).parity(reference, prediction,
                                      validation=held_out, bins=16)
        rows = list(csv.DictReader(
            open(tmp_path / "ext2chg_parity.csv", encoding="utf-8")))

        for split, expected in (("training", reference.size),
                                ("validation", held_out[0].size)):
            selected = [row for row in rows if row["split"] == split]
            assert selected
            assert sum(int(row["count"]) for row in selected) == expected
            assert sum(float(row["density"])
                       for row in selected) == pytest.approx(1.0)

    def test_empty_bins_are_dropped_and_the_edges_kept(self, tmp_path, fields):
        """
        A 200-bin grid is 40 000 cells of which a few thousand are occupied.
        The zeros carry nothing -- but the edges do, and a log axis makes them
        unrecoverable from the centres, so they get their own file.
        """
        self._report(tmp_path).parity(*fields, bins=32)
        rows = list(csv.DictReader(
            open(tmp_path / "ext2chg_parity.csv", encoding="utf-8")))
        edges = list(csv.DictReader(
            open(tmp_path / "ext2chg_parity_bin_edges.csv", encoding="utf-8")))

        assert 0 < len(rows) < 32 * 32
        assert len(edges) == 33
        assert all(float(row["count"]) > 0 for row in rows)

    def test_a_scatter_parity_writes_the_points_it_drew(self, tmp_path, fields):
        """With `scatter` there are no bins; the points are the data."""
        self._report(tmp_path).parity(*fields, scatter=True, bins=16)
        rows = list(csv.DictReader(
            open(tmp_path / "ext2chg_parity.csv", encoding="utf-8")))

        assert len(rows) == fields[0].size
        assert set(rows[0]) == {"split", "reference", "prediction"}

    # ----------------------------- the slices ---------------------------- #
    def test_the_three_panels_are_stored_as_arrays(self, tmp_path, fields):
        self._report(tmp_path).field_comparison(*fields, label="rho",
                                                unit="e/Ang^3")
        stored = np.load(tmp_path / "ext2chg_field_slice.npz")

        assert stored["reference"].shape == (12, 12)
        assert stored["prediction"].shape == (12, 12)
        assert str(stored["label"]) == "rho"

    def test_the_error_panel_is_stored_not_left_to_arithmetic(self, tmp_path,
                                                              fields):
        """
        A reader who reconstructs it and gets something else then knows the
        file is inconsistent, instead of trusting their own subtraction.
        """
        self._report(tmp_path).field_comparison(*fields)
        stored = np.load(tmp_path / "ext2chg_field_slice.npz")

        assert np.allclose(stored["error"],
                           stored["prediction"] - stored["reference"])

    def test_the_arrays_are_the_slice_that_was_drawn(self, tmp_path, fields):
        reference, prediction = fields
        self._report(tmp_path).field_comparison(reference, prediction,
                                                axis=0, index=3)
        stored = np.load(tmp_path / "ext2chg_field_slice.npz")

        assert np.array_equal(stored["reference"], reference[3])
        assert int(stored["axis"]) == 0 and int(stored["index"]) == 3

    def test_the_file_records_what_it_takes_to_redraw_it(self, tmp_path, fields):
        """Colour limits included: they were chosen from a quantile of the
        error, so a redraw that recomputed them would not match the figure."""
        self._report(tmp_path).field_comparison(*fields, unit="eV")
        stored = np.load(tmp_path / "ext2chg_field_slice.npz")

        assert set(stored.files) >= {"reference", "prediction", "error",
                                     "axis", "index", "colour_limits",
                                     "error_limits", "label", "unit",
                                     "log_scale"}
        assert stored["error_limits"][0] < 0 < stored["error_limits"][1]

    # ------------------------------ bookkeeping -------------------------- #
    def test_written_files_are_reported_back(self, tmp_path, fields):
        """So the run log can name them without re-deriving the names."""
        report = self._report(tmp_path)
        report.loss_curves(self._history())
        report.field_comparison(*fields)

        assert len(report.data_files) == 2
        assert all(os.path.exists(path) for path in report.data_files)


class TestParetoPlot:
    """
    The front drawn, with both candidates a reader might quote marked.
    """

    FRONT = [{"complexity": 1, "loss": 0.6, "expression": "1.0"},
             {"complexity": 5, "loss": 0.08, "expression": "exp(-p)"},
             {"complexity": 7, "loss": 0.021, "expression": "1/(1+p**2)"},
             {"complexity": 18, "loss": 0.0188, "expression": "exp(-p*p)"}]

    def test_it_writes_a_figure(self, tmp_path):
        from poraque.vis.report import TrainingReport

        path = TrainingReport(str(tmp_path)).pareto(self.FRONT)
        assert path and os.path.exists(path)

    def test_an_empty_front_writes_nothing(self, tmp_path):
        from poraque.vis.report import TrainingReport

        assert TrainingReport(str(tmp_path)).pareto([]) is None

    def test_a_front_with_no_positive_loss_writes_nothing(self, tmp_path):
        """The axis is logarithmic, so a zero loss has nowhere to go."""
        from poraque.vis.report import TrainingReport

        assert TrainingReport(str(tmp_path)).pareto(
            [{"complexity": 3, "loss": 0.0}]) is None

    def test_the_knee_is_computed_when_not_supplied(self, tmp_path):
        from poraque.vis.report import TrainingReport

        assert TrainingReport(str(tmp_path)).pareto(self.FRONT, knee=None)


class TestSymbolicFiguresReachTheReport:
    """
    All three symbolic figures are referenced by basename, so each has to be
    copied into the compile directory. A missing one renders as an empty box.
    """

    def test_every_figure_is_shipped(self, tmp_path):
        from poraque.vis.pdf_report import ModelReport

        names = {}
        for name in ("parity_plot", "knee_parity_plot", "pareto_plot"):
            path = tmp_path / f"{name}.png"
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.savefig(path)
            plt.close()
            names[name] = str(path)

        report = ModelReport(str(tmp_path), logo=None)
        captured = {}
        report._compile = lambda source, figures, target: (
            captured.update(source=source, figures=list(figures)), target)[1]
        report.build(task="chg2tau", per_material={}, unit="eV",
                     symbolic={"latex": "1", "full_latex": "F = 1",
                               "complexity": 3, "r2": 0.9, "relative_l2": 0.1,
                               "n_samples": 10, "scheme": "reduced",
                               "units": "atomic", "target": "model",
                               "target_name": "F", "template": "pauli",
                               "knee": {"complexity": 2, "loss": 0.2,
                                        "expression": "1 - p"},
                               **names})
        for name, path in names.items():
            assert path in captured["figures"], name

    def test_the_two_parity_plots_are_shown_side_by_side(self, tmp_path):
        """
        The question they answer is a comparison, and a comparison read across
        a page turn is not one.
        """
        from poraque.vis.pdf_report import ModelReport

        paths = {}
        for name in ("parity_plot", "knee_parity_plot"):
            path = tmp_path / f"{name}.png"
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.savefig(path)
            plt.close()
            paths[name] = str(path)

        source = ModelReport(str(tmp_path), logo=None)._symbolic_block({
            "latex": "1", "full_latex": "F = 1", "complexity": 9,
            "r2": 0.9, "relative_l2": 0.1, "n_samples": 10,
            "scheme": "reduced", "units": "atomic", "target": "model",
            "target_name": "F", "template": "pauli",
            "knee": {"complexity": 3, "loss": 0.2, "expression": "1 - p"},
            **paths})
        assert source.count("minipage") == 4        # two opened, two closed
        assert "Pareto knee, 3 nodes" in source
        assert "lowest loss, 9 nodes" in source

    def test_a_knee_equal_to_the_winner_says_so(self, tmp_path):
        from poraque.vis.pdf_report import ModelReport

        source = ModelReport(str(tmp_path), logo=None)._symbolic_block({
            "latex": "1", "full_latex": "F = 1", "complexity": 3,
            "expression": "1 - p", "r2": 0.9, "relative_l2": 0.1,
            "n_samples": 10, "scheme": "reduced", "units": "atomic",
            "target": "model", "target_name": "F", "template": "pauli",
            "knee": {"complexity": 3, "loss": 0.2, "expression": "1 - p"}})
        assert "nothing was traded away" in source

    def test_the_front_table_shows_the_distance_and_marks_the_knee(self,
                                                                   tmp_path):
        from poraque.vis.pdf_report import ModelReport

        front = [{"complexity": 3, "loss": 0.2, "expression": "1 - p",
                  "distance": 0.4, "limits": {"badge": "TF/vW"}},
                 {"complexity": 9, "loss": 0.1, "expression": "exp(-p)",
                  "distance": 0.9, "limits": {"badge": "TF/--"}}]
        source = ModelReport(str(tmp_path), logo=None)._symbolic_block({
            "latex": "1", "full_latex": "F = 1", "complexity": 9,
            "r2": 0.9, "relative_l2": 0.1, "n_samples": 10,
            "scheme": "reduced", "units": "atomic", "target": "model",
            "target_name": "F", "template": "pauli", "pareto": front,
            "knee": {"complexity": 3, "loss": 0.2, "expression": "1 - p"}})
        assert r"Nodes & Loss & $d$ & Limits & Expression" in source
        assert r"3\,$\bullet$" in source, "the knee must be marked in the table"
