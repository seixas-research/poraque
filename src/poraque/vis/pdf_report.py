# -*- coding: utf-8 -*-
# file: pdf_report.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Automated PDF reports for a trained model.

:class:`ModelReport` assembles the metrics and the figures produced by
:class:`~poraque.vis.report.TrainingReport` into a typeset document: it writes a
LaTeX source into a **temporary directory**, compiles it, moves the PDF to the
reports folder, and removes the scratch directory entirely. Nothing is left
behind — no ``.tex``, no ``.aux``, no ``.log`` — because the deliverable is the
PDF and a directory full of build residue is a liability, not a record.

Compilation is attempted with ``latexmk`` and falls back to two ``pdflatex``
passes (needed for the table of contents and any cross-references). When
neither is installed the LaTeX source is written next to the target PDF so the
report can still be produced elsewhere, and the caller is told plainly rather
than left with a silent failure.
"""

import datetime
import os
import shutil
import subprocess
import tempfile

import numpy as np


def _escape(text):
    """Escape the LaTeX special characters that appear in identifiers."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


#: Characters a long unbroken configuration value may be split after.
_BREAK_AFTER = ("/", ",", ":", ";", "-", r"\_")


def _wrappable(text, threshold=28):
    """
    Escape a configuration value and let it break inside a fixed-width column.

    A ``p`` column wraps at spaces, which is enough for a value like the
    ``training.physics`` dict. It is not enough for a long value with no
    spaces at all -- an absolute path, or a cache tag such as
    ``res32_blur0.15spec`` -- which stays one unbreakable word and overruns the
    column exactly as an ``l`` column did. Break opportunities are inserted
    after the separators such values are built from, so the line can turn
    without hyphenating a path in a misleading place.

    Short values are returned untouched: the markup is pure noise below the
    width where wrapping could ever be needed.
    """
    escaped = _escape(text)
    if len(str(text)) <= threshold:
        return escaped
    for separator in _BREAK_AFTER:
        escaped = escaped.replace(separator, separator + r"\allowbreak{}")
    return escaped


def _format_value(value):
    """
    Render one configuration value for a fixed-width table cell.

    A nested mapping -- ``training.physics`` is the only one today -- becomes
    one ``key: value`` per line rather than a single 116-character repr. The
    break is ``\\newline`` and not ``\\\\``: inside a ``p`` column ``\\\\`` ends
    the table *row*, which would split one setting across two rows and put the
    rest of the table out of step with its keys.

    The dict is taken as a dict, never parsed back out of its ``str()`` -- a
    repr is not a format, and reparsing one is how a stray comma inside a value
    turns into a silently mangled table.
    """
    if isinstance(value, dict) and value:
        # No column padding: LaTeX collapses runs of spaces, so padding the
        # keys would align in the source and not on the page.
        return r" \newline ".join(f"{_escape(key)}: {_wrappable(inner)}"
                                  for key, inner in value.items())
    return _wrappable(value)


def _number(value, digits=5):
    """Format a metric, falling back to a dash for missing values."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}g}"


class ModelReport:
    """
    Build a PDF report for one trained model.

    Parameters
    ----------
    output_dir : str, optional
        Where the finished PDF is placed.
    logo : str, optional
        Path to a logo for the title block; omitted if the file is absent.
    keep_source : bool, optional
        Keep the generated ``.tex`` beside the PDF. Off by default: the
        scratch directory is deleted wholesale, which is the only reliable way
        to guarantee no auxiliary files survive.

    Examples
    --------
    >>> report = ModelReport("reports")                       # doctest: +SKIP
    >>> report.build(task="chg2tau", metrics=..., figures=[...])  # doctest: +SKIP
    """

    def __init__(self, output_dir="reports", logo="assets/logo/logo_light.png",
                 keep_source=False):
        self.output_dir = str(output_dir)
        self.logo = logo
        self.keep_source = bool(keep_source)

    # ------------------------------------------------------------------ #
    # LaTeX generation
    # ------------------------------------------------------------------ #
    def _preamble(self, title, subtitle):
        logo_line = ""
        if self.logo and os.path.exists(self.logo):
            logo_line = (r"\includegraphics[width=0.34\textwidth]{logo.png}\par"
                         "\n" r"\vspace{0.7cm}")

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
% Defines the starred \caption* used for the unnumbered figure captions
% below. Without it LaTeX prints a literal "Figure N: *" above the text.
\usepackage{{caption}}
\usepackage{{titlesec}}
\usepackage{{fancyhdr}}
\usepackage{{parskip}}
\usepackage[colorlinks=true,linkcolor=poraqueblue,urlcolor=poraqueblue]{{hyperref}}
\usepackage{{float}}

\definecolor{{poraqueblue}}{{RGB}}{{42,120,214}}
\definecolor{{poraquedark}}{{RGB}}{{30,34,40}}
\definecolor{{shadegray}}{{RGB}}{{90,96,104}}

\titleformat{{\section}}
  {{\Large\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraqueblue}}{{\thesection}}}}{{0.7em}}{{}}
  [\vspace{{-6pt}}{{\color{{poraqueblue!35}}\rule{{\textwidth}}{{0.8pt}}}}]
\titleformat{{\subsection}}
  {{\large\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraqueblue}}{{\thesubsection}}}}{{0.7em}}{{}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyfoot[L]{{\small\color{{shadegray}}Poraqu\^e model report}}
\fancyfoot[R]{{\small\color{{shadegray}}\thepage}}
\renewcommand{{\headrulewidth}}{{0pt}}

\begin{{document}}

\begin{{center}}
{logo_line}
{{\Huge\bfseries {title}\par}}
\vspace{{0.35cm}}
{{\large\color{{shadegray}} {subtitle}\par}}
\vspace{{0.25cm}}
{{\small\color{{shadegray}} generated {stamp}\par}}
\end{{center}}
\vspace{{0.6cm}}
"""

    def _metrics_table(self, per_material, unit):
        """Per-structure metrics as a booktabs table."""
        rows = []
        for name, entry in sorted(per_material.items()):
            values = entry.get("metrics", entry)
            split = entry.get("split", "")
            rows.append(
                f"{_escape(name)} & {_escape(split)} & "
                f"{_number(values.get('mse'))} & {_number(values.get('mae'))} & "
                f"{_number(values.get('rmse'))} & "
                f"{_number(values.get('relative_l2'), 4)} & "
                f"{_number(values.get('r2'), 4)} \\\\"
            )

        aggregate = ""
        collected = [e.get("metrics", e) for e in per_material.values()]
        if collected:
            def mean(key):
                values = [c[key] for c in collected
                          if c.get(key) is not None and np.isfinite(c[key])]
                return _number(np.mean(values)) if values else "--"

            aggregate = (
                r"\midrule" "\n"
                f"\\textbf{{mean}} & & \\textbf{{{mean('mse')}}} & "
                f"\\textbf{{{mean('mae')}}} & \\textbf{{{mean('rmse')}}} & "
                f"\\textbf{{{mean('relative_l2')}}} & \\textbf{{{mean('r2')}}} \\\\"
            )

        return (
            r"\begin{center}" "\n"
            r"\begin{tabular}{@{}llrrrrr@{}}" "\n"
            r"\toprule" "\n"
            f"Structure & Split & MSE & MAE & RMSE & rel.\\ $L^2$ & $R^2$ \\\\\n"
            f"& & \\multicolumn{{3}}{{c}}{{[{_escape(unit)}]}} & & \\\\\n"
            r"\midrule" "\n"
            + "\n".join(rows) + "\n"
            + aggregate + "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{center}" "\n"
        )

    #: Maths symbols for the quantities a distillation can fit. Without this
    #: ``tau`` sets as the product t·a·u in an equation environment.
    _TARGET_SYMBOLS = {"tau": r"\tau", "rho": r"\rho", "F": "F"}

    @staticmethod
    def _math_number(value, digits=4):
        r"""
        A number safe to drop into maths mode.

        ``%g`` gives ``8.361e-07``, which inside ``$...$`` sets as *e* minus 07.
        Exponents become ``\times 10^{...}`` instead.
        """
        text = _number(value, digits)
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e")
        return rf"{mantissa} \times 10^{{{int(exponent)}}}"

    def _symbolic_block(self, result):
        r"""
        Typeset the distilled expression and its accuracy/complexity front.

        Accepts either a :class:`~poraque.ml.symbolic.SymbolicResult` or the
        plain dict it serialises to, so a report can be rebuilt from a run's
        JSON summary without importing the ML stack.
        """
        get = (result.get if isinstance(result, dict)
               else lambda key, default=None: getattr(result, key, default))

        target = ("the trained operator's predictions"
                  if get("target") == "model" else "the DFT reference data")
        name = get("target_name", "y")
        symbol = self._TARGET_SYMBOLS.get(name, _escape(name))
        body = [
            r"\clearpage" "\n" r"\section*{Symbolic distillation}" "\n",
            f"Closed-form expression fitted to {target} over "
            f"{get('n_samples', 0)} voxels, in the "
            f"{_escape(get('scheme', ''))} feature scheme "
            f"({_escape(get('units', ''))}):\n\n",
            r"\begin{equation*}" "\n",
            # The template is folded back in, so the equation on the page is
            # the whole physical formula rather than the factor that was fitted.
            f"{get('full_latex') or (symbol + ' = ' + get('latex', ''))}\n",
            r"\end{equation*}" "\n\n",
        ]
        if get("template", "none") != "none":
            body.append(
                f"Fitted under the {_escape(get('template'))} template: the "
                f"search saw {_escape(get('target_name', 'y'))} alone, with "
                f"$\\tau_{{\\mathrm{{TF}}}}$ supplied exactly.\n\n")

        r2, relative = get("r2", float("nan")), get("relative_l2", float("nan"))
        body.append(
            f"Complexity {get('complexity', 0)} nodes; "
            f"$R^2 = {self._math_number(r2)}$, "
            f"relative $L^2 = {self._math_number(relative)}$ "
            f"against the fitted target.\n\n")

        # The caveat is not optional garnish: without it a reader takes the
        # equation for a reconstruction of the operator rather than for the
        # best semi-local approximation to it.
        body.append(
            r"\begin{quote}\small\color{shadegray}" "\n"
            "The features are evaluated at a point, so the search space is "
            "the semi-local functionals. Whatever the operator learned that is "
            "non-local cannot appear in this expression, and shows up as "
            "residual: the fit quality above measures how much of the learned "
            "map is semi-local, not how well the search performed.\n"
            r"\end{quote}" "\n\n")

        held_out = get("validation") or {}
        if held_out.get("n_points"):
            body.append(
                f"On {held_out['n_points']} voxels of the held-out structures, "
                f"against the DFT reference: relative "
                f"$L^2 = {self._math_number(held_out.get('relative_l2'))}$, "
                f"$R^2 = {self._math_number(held_out.get('r2'))}$.\n\n")

        parity = get("parity_plot")
        if parity and os.path.exists(parity):
            body.append(self._figure_block(
                parity,
                "The distilled formula against the DFT reference on held-out "
                "structures. Read it beside the operator's own parity plot: "
                "the gap between them is what the closed form gives up."))

        body.append(self._asymptotic_block(get("limits") or {}))

        front = get("pareto") or []
        if front:
            body.append(r"\subsection*{Accuracy against complexity}" "\n")
            body.append(r"\begin{center}\begin{longtable}"
                        r"{@{}rrc>{\raggedright\arraybackslash}p{8.4cm}@{}}"
                        r"\toprule" "\n")
            body.append(r"Nodes & Loss & Limits & Expression \\" "\n"
                        r"\midrule" "\n")
            for entry in front:
                limits = entry.get("limits") or {}
                body.append(f"{entry['complexity']} & "
                            f"{_number(entry['loss'], 4)} & "
                            f"{_escape(limits.get('badge', '--/--'))} & "
                            f"{_wrappable(entry['expression'])} \\\\\n")
            body.append(r"\bottomrule\end{longtable}\end{center}" "\n")
            body.append(r"\emph{\small Limits: \texttt{TF} = recovers "
                        r"Thomas-Fermi as $p,q\to0$; \texttt{vW} = recovers "
                        r"von Weizs\"acker as $p\to\infty$.}" "\n\n")
        return "".join(body)

    def _asymptotic_block(self, limits):
        r"""
        Report the two asymptotic limits for the chosen expression.

        A symbolic fit is a numerical statement until it is checked against
        physics it was never shown. These two limits are the cheapest such
        check available, and a candidate that fails them is a curve through the
        data rather than a functional -- so the section is written to say that
        plainly rather than to decorate a good $R^2$.
        """
        if not limits:
            return ""

        thomas_fermi = limits.get("thomas_fermi") or {}
        von_weizsacker = limits.get("von_weizsacker") or {}
        body = [
            r"\subsection*{Physical asymptotic compliance}" "\n",
            "A kinetic functional is pinned at two ends. Where the density is "
            "uniform ($p,q \\to 0$) it must collapse to Thomas--Fermi; where it "
            "varies rapidly ($p \\to \\infty$) it must become von "
            "Weizs\\\"acker. Both are statements about the enhancement factor "
            "$F = \\tau/\\tau_{\\mathrm{TF}}$:\n\n",
            r"\begin{equation*}" "\n",
            r"F(0,0) = 1, \qquad F \to \tfrac{5}{3}p^{2} "
            r"\ \text{as}\ p \to \infty." "\n",
            r"\end{equation*}" "\n\n",
            r"\begin{center}\begin{tabular}"
            r"{@{}l c >{\raggedright\arraybackslash}p{8.2cm}@{}}\toprule" "\n",
            r"Limit & Satisfied & Finding \\" "\n" r"\midrule" "\n",
        ]
        for label, check in (("Thomas--Fermi", thomas_fermi),
                             ("von Weizs\\\"acker", von_weizsacker)):
            mark = r"\textbf{yes}" if check.get("passes") else "no"
            body.append(f"{label} & {mark} & "
                        f"{_escape(check.get('detail', 'not evaluated'))} \\\\\n")
        body.append(r"\bottomrule\end{tabular}\end{center}" "\n\n")

        score = limits.get("score", 0.0)
        body.append(f"Compliance score: {_number(score, 2)} of 1.\n\n")

        if limits.get("quadratic_scaling") and not von_weizsacker.get("passes"):
            body.append(
                r"\begin{quote}\small\color{shadegray}" "\n"
                "The expression does grow as $p^{2}$, but with the wrong "
                "coefficient, so it is not von Weizs\\\"acker. That "
                "combination is not unusual: the second-order gradient "
                "expansion has exactly the right scaling with coefficient "
                "$1/9$.\n" r"\end{quote}" "\n\n")

        if not (thomas_fermi.get("passes") and von_weizsacker.get("passes")):
            body.append(
                r"\begin{quote}\small\color{shadegray}" "\n"
                "Neither known functional satisfies both limits on its own --- "
                "Thomas--Fermi fails the second, von Weizs\\\"acker the first "
                "--- so failing one is not by itself damning. Failing both "
                "means the expression reproduces the training data without "
                "reproducing the physics that constrains it outside that "
                "range, and it should not be extrapolated.\n"
                r"\end{quote}" "\n\n")
        return "".join(body)

    def _figure_block(self, path, caption):
        return (
            r"\begin{figure}[H]" "\n"
            r"\centering" "\n"
            rf"\includegraphics[width=\textwidth]{{{os.path.basename(path)}}}" "\n"
            rf"\caption*{{\small\color{{shadegray}} {caption}}}" "\n"
            r"\end{figure}" "\n"
        )

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, task, per_material, figures=(), unit="", summary=None,
              configuration=None, caveats=(), filename=None, symbolic=None):
        """
        Generate the report and return the path to the finished PDF.

        Parameters
        ----------
        task : str
            Task name, used in the title and the default filename.
        per_material : dict
            ``{structure: {"metrics": {...}, "split": "train"|"holdout"}}``.
        figures : sequence of str, optional
            Image paths to embed, in order.
        unit : str, optional
            Physical unit of the target field, shown in the table header.
        summary : dict, optional
            Key/value facts listed above the table (model size, device,
            training time, ...).
        configuration : dict, optional
            Flat mapping rendered as a settings appendix.
        caveats : sequence of str, optional
            Statements about what the numbers do *not* support. Rendered
            verbatim, because a report that omits them invites over-reading.
        filename : str, optional
            Output name; defaults to ``<task>_report.pdf``.
        symbolic : SymbolicResult, optional
            Distilled closed-form expression, typeset as display mathematics
            with its accuracy/complexity front beneath it.

        Returns
        -------
        str or None
            Path to the PDF, or the path to the ``.tex`` fallback when no
            LaTeX toolchain is available.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or f"{task}_report.pdf"
        target = os.path.join(self.output_dir, filename)

        body = [self._preamble(f"{_escape(task)}", "Fourier neural operator "
                               "trained on DFT scalar fields")]

        if summary:
            body.append(r"\section*{Run summary}" "\n")
            body.append(r"\begin{center}\begin{tabular}{@{}ll@{}}\toprule" "\n")
            for key, value in summary.items():
                body.append(f"{_escape(key)} & {_escape(value)} \\\\\n")
            body.append(r"\bottomrule\end{tabular}\end{center}" "\n")

        body.append(r"\section*{Performance}" "\n")
        body.append(self._metrics_table(per_material, unit))

        if caveats:
            body.append(r"\subsection*{What these numbers do not show}" "\n")
            body.append(r"\begin{itemize}" "\n")
            for caveat in caveats:
                body.append(f"\\item {_escape(caveat)}\n")
            body.append(r"\end{itemize}" "\n")

        captions = {
            "loss_curves": "Training objective and validation error against epoch.",
            "field_slice": "Cross-section: DFT reference, prediction, and their "
                           "difference on a shared colour scale.",
            "parity": "Predicted against true voxel values, with the identity "
                      "line. Where a validation structure was held out it is "
                      "drawn beside the training one in its own colour: a "
                      "wider cloud about the line is the generalisation gap.",
            "error_histogram": "Distribution of the signed voxel-wise error.",
        }
        if figures:
            body.append(r"\section*{Comparison with DFT}" "\n")
            for path in figures:
                stem = os.path.splitext(os.path.basename(path))[0]
                caption = next((c for key, c in captions.items() if key in stem),
                               _escape(stem))
                body.append(self._figure_block(path, caption))

        if symbolic is not None:
            body.append(self._symbolic_block(symbolic))

        if configuration:
            # Fixed-width columns, not `ll`. An `l` column is a single
            # unbreakable box, so a long value -- `training.physics` is 116
            # characters of nested dict -- ran straight past the right margin.
            # 4.6 + 11.4 cm plus the inter-column gap fits the 16.6 cm text
            # block set by `geometry`'s 2.2 cm margins on A4.
            body.append(r"\clearpage" "\n" r"\section*{Configuration}" "\n")
            body.append(r"\begin{center}\begin{longtable}"
                        r"{@{}p{4.6cm}>{\raggedright\arraybackslash}p{11.4cm}@{}}"
                        r"\toprule" "\n")
            for key, value in sorted(configuration.items()):
                # The key needs the same break opportunities as the value.
                # `symbolic.enable_symbolic_distillation` is 37 characters with
                # no space in it, so escaping alone leaves one unbreakable word
                # that overruns its column and prints on top of the value.
                body.append(f"{_wrappable(key)} & {_format_value(value)} \\\\\n")
            body.append(r"\bottomrule\end{longtable}\end{center}" "\n")

        body.append(r"\end{document}" "\n")
        source = "".join(body)

        # The symbolic parity plot is referenced by basename like every other
        # figure, so it has to reach the compile directory too -- but it is not
        # in `figures`, which drives the "Comparison with DFT" section and
        # would render it a second time there.
        embedded = list(figures)
        extra = (symbolic or {}).get("parity_plot") if isinstance(
            symbolic, dict) else getattr(symbolic, "parity_plot", None)
        if extra and extra not in embedded:
            embedded.append(extra)

        return self._compile(source, embedded, target)

    def _compile(self, source, figures, target):
        """
        Compile in a scratch directory and move only the PDF out.

        Building inside :func:`tempfile.mkdtemp` and deleting the whole tree is
        what guarantees the cleanup requirement: there is no list of extensions
        to keep in sync with whatever LaTeX decides to emit.
        """
        workdir = tempfile.mkdtemp(prefix="poraque_report_")
        try:
            stem = "report"
            tex_path = os.path.join(workdir, f"{stem}.tex")
            with open(tex_path, "w") as handle:
                handle.write(source)

            if self.logo and os.path.exists(self.logo):
                shutil.copy(self.logo, os.path.join(workdir, "logo.png"))
            for path in figures:
                if os.path.exists(path):
                    shutil.copy(path, workdir)

            command = self._toolchain(stem)
            if command is None:
                fallback = os.path.splitext(target)[0] + ".tex"
                shutil.copy(tex_path, fallback)
                print(f"[poraque] no LaTeX toolchain found; wrote {fallback}")
                return fallback

            for argv in command:
                subprocess.run(argv, cwd=workdir, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)

            produced = os.path.join(workdir, f"{stem}.pdf")
            if not os.path.exists(produced):
                fallback = os.path.splitext(target)[0] + ".tex"
                shutil.copy(tex_path, fallback)
                print(f"[poraque] LaTeX compilation failed; wrote {fallback}")
                return fallback

            shutil.move(produced, target)
            if self.keep_source:
                shutil.copy(tex_path, os.path.splitext(target)[0] + ".tex")
            return target
        finally:
            # Removes the .tex and every auxiliary file in one step.
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _toolchain(stem):
        """Command sequence for the first available LaTeX driver."""
        if shutil.which("latexmk"):
            return [["latexmk", "-pdf", "-interaction=nonstopmode",
                     "-halt-on-error", f"{stem}.tex"]]
        if shutil.which("pdflatex"):
            # Twice, so the table of contents and references resolve.
            return [["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"]] * 2
        return None
