# -*- coding: utf-8 -*-
# file: test_vis.py

"""
Tests for the figure and PDF-report machinery.

Two classes of defect are covered, both of which produce a file that looks
fine to the code and is unreadable to a human: tick labels that overprint each
other, and a table cell that runs off the page. Neither raises, so neither is
caught by asserting that a figure was written.
"""

import os

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
    `training.physics` is 116 characters of nested dict. In an `l` column --
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
        value = ("{'electron_count_weight': 0.0, 'positivity_weight': 0.0, "
                 "'von_weizsacker_weight': 0.0, 'euler_lagrange_weight': 0.0}")
        wrapped = _wrappable(value)
        # Spaces already give a p-column somewhere to break.
        assert " " in wrapped
        assert _escape("{") in wrapped


class TestConfigurationKeys:
    """
    The key column is fixed-width, so a long key must be *breakable*, not just
    escaped: `symbolic.enable_symbolic_distillation` is 37 characters with no
    space, and one unbreakable word overruns its column and prints on top of
    the value in the next one.
    """

    LONG = "symbolic.enable_symbolic_distillation"

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
    `training.physics` is a dict. Printed as its repr it is one unreadable
    116-character run; it belongs in the table as one setting per line.
    """

    PHYSICS = {"electron_count_weight": 0.0, "positivity_weight": 0.0,
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

        physics = TrainingConfig().training.physics
        rendered = _format_value(physics)
        assert rendered.count(r"\newline") == len(physics) - 1


class TestDescribeNesting:
    """The same clustering in the terminal run header."""

    def test_physics_is_listed_not_inlined(self):
        from poraque.ml.config import TrainingConfig

        described = TrainingConfig().describe()
        assert "physics={" not in described
        assert "  physics:" in described
        for key in TrainingConfig().training.physics:
            assert any(line.strip().startswith(key)
                       for line in described.splitlines())

    def test_the_dict_no_longer_rides_on_the_training_line(self):
        """
        The scalar settings are still inlined -- that format is unchanged.
        What moved off the line is the 116-character physics repr, which is
        what made it unreadable.
        """
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        training = next(line for line in config.describe().splitlines()
                        if line.startswith("training:"))
        inlined = len(training) + len(str(config.training.physics))
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

    def test_uses_the_brand_red_not_the_old_blue(self, preamble):
        """
        The accent is the logo's own colour.

        The report previously used a blue that appears nowhere else in the
        project, which is what made it look like a different document.
        """
        assert r"\definecolor{poraquered}{RGB}{248,65,55}" in preamble
        assert "poraqueblue" not in preamble

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
