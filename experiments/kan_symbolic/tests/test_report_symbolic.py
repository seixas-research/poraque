# -*- coding: utf-8 -*-
# file: test_report_symbolic.py
"""
The report tests of symbolic distillation, cut out of ``tests/test_vis.py``
when it was archived (2026-10-13). Verbatim; they need the fixtures and
imports of that file (``report_source``, ``ModelReport``, ``TrainingReport``)
and the methods kept in ``../src/report_symbolic.py``.
"""


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
            report.build(task="pseudo2tau", per_material={}, unit="eV",
                         symbolic=symbolic)
            return captured

        return build

    @pytest.fixture
    def parity_file(self, tmp_path):
        path = tmp_path / "pseudo2tau_symbolic_parity.png"
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
        assert "penalized inside" in captured["source"]
        assert "thomas\\_fermi" not in captured["source"], (
            "constraint names are prose here, not identifiers")
        assert "thomas fermi" in captured["source"]

    def test_an_unconstrained_search_claims_nothing(self, report_source,
                                                    parity_file):
        captured = report_source(self._symbolic(parity_file, validated=True))
        assert "penalized inside" not in captured["source"]


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
        report.build(task="pseudo2tau", per_material={}, unit="eV",
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
