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

#: Share of the text block given to the *key* column of the configuration
#: appendix.
#:
#: Sized to the content rather than guessed. The longest key any config
#: produces is ``symbolic.enable_symbolic_distillation``, which sets 184.5 pt
#: wide at 11 pt Latin Modern -- 0.405 of the 455 pt text block on A4 with the
#: 2.5 cm margins this report uses. The value here leaves a little over 15 pt
#: of headroom on top of that, so every key of every section fits on **one
#: line** and is read as the identifier it is.
#:
#: A key broken across two lines is not merely untidy: ``symbolic.enable\_``
#: over ``symbolic\_distillation`` reads as two settings, and the reader has to
#: reassemble the name of the thing whose value sits beside it.
CONFIG_KEY_FRACTION = 0.44


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
        r"""
        Preamble matching the Poraquê User Guide.

        The palette, the rules, the heading treatment and the header/footer are
        taken from ``latex/user_guide/poraque_user_guide.tex`` so that a
        generated report and the hand-written guides read as one family of
        documents. The brand accent is ``poraquered``, the logo's own colour --
        the earlier blue matched nothing.

        Two things stay deliberately different from the guide. The class is
        ``article`` rather than ``report``, because a single-model report has no
        chapters; and there is no title *page*, because a two-page report with a
        full-bleed cover is mostly cover.
        """
        has_logo = bool(self.logo and os.path.exists(self.logo))
        logo_line = ""
        header_logo = r"\fancyhead[L]{}"
        if has_logo:
            logo_line = (r"\includegraphics[width=0.34\textwidth]{logo.png}\par"
                         "\n" r"\vspace{0.55cm}")
            header_logo = (r"\fancyhead[L]{\raisebox{-4pt}"
                           r"{\includegraphics[height=14pt]{logo.png}}}")

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.5cm,headheight=26pt,headsep=16pt]{{geometry}}
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{enumitem}}
% Defines the starred \caption* used for the unnumbered figure captions
% below. Without it LaTeX prints a literal "Figure N: *" above the text.
\usepackage{{caption}}
\usepackage{{titlesec}}
\usepackage{{fancyhdr}}
\usepackage{{tcolorbox}}
\usepackage{{parskip}}
\usepackage[colorlinks=true,linkcolor=poraquered,urlcolor=poraquered,
            citecolor=poraquered]{{hyperref}}
\usepackage{{float}}

\tcbuselibrary{{skins,breakable}}

% --- Brand palette, shared with the user and technical guides --------------
\definecolor{{poraquered}}{{RGB}}{{248,65,55}}
\definecolor{{poraqueamber}}{{RGB}}{{235,104,52}}
\definecolor{{poraquedark}}{{RGB}}{{30,34,40}}
\definecolor{{shadegray}}{{RGB}}{{90,96,104}}
\definecolor{{codebg}}{{RGB}}{{245,246,248}}
\definecolor{{warnamber}}{{RGB}}{{176,125,42}}

% --- Heading typography, as in the guides ----------------------------------
\titleformat{{\section}}
  {{\Large\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraquered}}{{\thesection}}}}{{0.8em}}{{}}
  [\vspace{{-6pt}}{{\color{{poraquered!35}}\rule{{\textwidth}}{{0.8pt}}}}]
\titleformat{{\subsection}}
  {{\large\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraquered}}{{\thesubsection}}}}{{0.8em}}{{}}
\titleformat{{\subsubsection}}
  {{\normalsize\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraquered}}{{\thesubsubsection}}}}{{0.7em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{18pt}}{{10pt}}
\titlespacing*{{\subsection}}{{0pt}}{{12pt}}{{6pt}}

% --- Running header / footer, with the guide's red rule --------------------
\pagestyle{{fancy}}
\fancyhf{{}}
{header_logo}
\fancyhead[R]{{\small\color{{shadegray}}\nouppercase{{\leftmark}}}}
\fancyfoot[L]{{\small\color{{shadegray}}Poraqu\^e model report}}
\fancyfoot[R]{{\small\color{{shadegray}}\thepage}}
\renewcommand{{\headrulewidth}}{{0.8pt}}
\renewcommand{{\headrule}}{{{{\color{{poraquered}}%
  \hrule width\headwidth height\headrulewidth}}}}
\renewcommand{{\footrulewidth}}{{0pt}}
\markboth{{{title}}}{{}}

% --- Callout boxes, matching the guides ------------------------------------
\newtcolorbox{{pwarn}}[1][]{{%
  breakable, enhanced, colback=warnamber!7, colframe=warnamber,
  boxrule=0pt, leftrule=3pt, arc=1pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fonttitle=\bfseries\small, title={{Caveat}}, coltitle=warnamber,
  attach title to upper=\par\vspace{{2pt}}, #1}}
\newtcolorbox{{pnote}}[1][]{{%
  breakable, enhanced, colback=poraquered!5, colframe=poraquered,
  boxrule=0pt, leftrule=3pt, arc=1pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fonttitle=\bfseries\small, title={{Note}}, coltitle=poraquered,
  attach title to upper=\par\vspace{{2pt}}, #1}}

\setlist[itemize]{{itemsep=2pt,topsep=4pt}}
\setlist[enumerate]{{itemsep=3pt,topsep=4pt}}

\begin{{document}}

\begin{{center}}
{logo_line}
{{\color{{poraquered}}\rule{{0.74\textwidth}}{{1.6pt}}}}\par
\vspace{{0.7cm}}
{{\Huge\bfseries {title}\par}}
\vspace{{0.35cm}}
{{\large\color{{shadegray}} {subtitle}\par}}
\vspace{{0.7cm}}
{{\color{{poraquered}}\rule{{0.74\textwidth}}{{1.6pt}}}}\par
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
        template = get("template", "none")
        if template != "none":
            supplied = (r"$\tau_{\mathrm{vW}}$ and $\tau_{\mathrm{TF}}$"
                        if template == "pauli" else r"$\tau_{\mathrm{TF}}$")
            body.append(
                f"Fitted under the {_escape(template)} template: the search "
                f"saw {_escape(get('target_name', 'y'))} alone, with "
                f"{supplied} supplied exactly.\n\n")

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
            # Which data the plot is drawn from depends on whether anything was
            # held out; saying "held-out" regardless would overstate a training
            # fit, which is the one thing a parity plot must not do.
            on_held_out = bool(held_out.get("n_points"))
            provenance = ("on held-out structures" if on_held_out else
                          "on the voxels it was fitted to --- a training fit, "
                          "since this run held nothing out")
            body.append(self._figure_block(
                parity,
                f"The distilled formula against the DFT reference "
                f"{provenance}. Read it beside the operator's own parity plot: "
                f"the gap between them is what the closed form gives up."))

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
            "uniform ($p,q \\to 0$) it must collapse to Thomas--Fermi; where "
            "it varies rapidly ($p \\to \\infty$) it must become von "
            "Weizs\\\"acker. Both are statements about the \\emph{Pauli} "
            "enhancement factor "
            "$F = (\\tau - \\tau_{\\mathrm{vW}})/\\tau_{\\mathrm{TF}}$:\n\n",
            r"\begin{equation*}" "\n",
            r"F(0,0) = 1, \qquad F \to 0 \ \text{as}\ p \to \infty." "\n",
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

        if (limits.get("bounded_at_infinity")
                and not von_weizsacker.get("passes")):
            body.append(
                r"\begin{quote}\small\color{shadegray}" "\n"
                "The Pauli term does settle on a finite value as "
                "$p \\to \\infty$, just not zero --- so the functional stays "
                "bounded but never reduces to von Weizs\\\"acker. That is a "
                "repairable failure, unlike one that diverges.\n"
                r"\end{quote}" "\n\n")

        if not (thomas_fermi.get("passes") and von_weizsacker.get("passes")):
            body.append(
                r"\begin{quote}\small\color{shadegray}" "\n"
                "Neither known functional satisfies both limits on its own --- "
                "as a Pauli factor Thomas--Fermi is $1 - 5p^{2}/3$, correct at "
                "the origin and divergent at infinity, while von "
                "Weizs\\\"acker is $0$, correct at infinity and wrong at the "
                "origin --- so failing one is not by itself damning. Failing "
                "both "
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
    def _charges_block(self, analysis):
        r"""
        Per-atom populations and net charges, as a table.

        Accepts a :class:`~poraque.analysis.PartialCharges` or the dict it
        serialises to, so a report can be rebuilt from a run's JSON summary
        without importing the analysis stack.
        """
        get = (analysis.get if isinstance(analysis, dict)
               else lambda key, default=None: getattr(analysis, key, default))

        # `or []` is not usable here: these are numpy arrays, whose truth value
        # is ambiguous, so the default has to be supplied explicitly.
        def column(name):
            values = get(name)
            return [] if values is None else list(values)

        symbols = column("symbols")
        populations = column("populations")
        valence = column("valence")
        charges = (column("charges") if isinstance(analysis, dict)
                   else list(analysis.charges))
        if not symbols:
            return ""

        method = _escape(str(get("method", "")))
        body = [
            r"\clearpage" "\n" r"\section*{Partial charges}" "\n",
            f"Population analysis of the predicted density by the "
            f"\\textbf{{{method}}} partitioning. The net charge is "
            f"$q_A = Z^{{\\rm val}}_A - N_A$, positive for electron-deficient."
            "\n\n",
            r"\begin{center}\begin{longtable}{@{}rlrrr@{}}\toprule" "\n",
            r"\# & Atom & Population & $Z^{\rm val}$ & $q$ \\" "\n"
            r"\midrule\endhead" "\n",
        ]
        for index, symbol in enumerate(symbols):
            body.append(f"{index} & {_escape(str(symbol))} & "
                        f"{_number(populations[index], 4)} & "
                        f"{_number(valence[index], 2)} & "
                        f"{_number(charges[index], 4)} \\\\\n")
        body.append(r"\midrule" "\n")
        body.append(f"& \\textbf{{sum}} & "
                    f"\\textbf{{{_number(sum(populations), 4)}}} & "
                    f"\\textbf{{{_number(sum(valence), 2)}}} & "
                    f"\\textbf{{{_number(sum(charges), 4)}}} \\\\\n")
        body.append(r"\bottomrule\end{longtable}\end{center}" "\n\n")

        details = get("details") or {}
        if details:
            rendered = ", ".join(f"{_escape(str(k))}: {_escape(str(v))}"
                                 for k, v in sorted(details.items()))
            body.append(f"\\emph{{\\small {rendered}}}\n\n")

        # The caveat is not decoration: a reader who takes these for
        # all-electron charges will draw conclusions the data cannot support.
        body.append(
            r"\begin{pwarn}[title={What these charges are}]" "\n"
            "A partition of the \\emph{pseudo} valence density. The PAW core "
            "is absent, so these are not all-electron populations and are "
            "systematically compressed toward zero; they also inherit whatever "
            "error the predicted density carries. Read them as comparative "
            "across a series, not as absolute numbers.\n"
            r"\end{pwarn}" "\n\n")
        return "".join(body)

    def build(self, task, per_material, figures=(), unit="", summary=None,
              configuration=None, caveats=(), filename=None, symbolic=None,
              charges=None):
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
        charges : PartialCharges or dict, optional
            Population analysis of a predicted density, rendered as a per-atom
            table. See :mod:`poraque.analysis.charges`.

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
            # The guides put this class of statement in a `pwarn` box rather
            # than a bare list, and it is the part of the report most likely to
            # be skipped -- so it gets the same amber rule here.
            body.append(r"\begin{pwarn}[title={What these numbers do not show}]"
                        "\n")
            body.append(r"\begin{itemize}" "\n")
            for caveat in caveats:
                body.append(f"\\item {_escape(caveat)}\n")
            body.append(r"\end{itemize}" "\n")
            body.append(r"\end{pwarn}" "\n")

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

        if charges is not None:
            body.append(self._charges_block(charges))

        if symbolic is not None:
            body.append(self._symbolic_block(symbolic))

        if configuration:
            # Fixed-width columns, not `ll`. An `l` column is a single
            # unbreakable box, so a long value -- `training.physics` is 116
            # characters of nested dict -- ran straight past the right margin.
            #
            # The widths are fractions of `\textwidth` rather than centimetres,
            # so they survive a change of paper size or margin, and they are
            # sized to the *content*: see CONFIG_KEY_FRACTION. The two add to
            # one, less the single inter-column gap `@{}...@{}` leaves in the
            # middle, so the table fills the text block exactly instead of
            # overhanging it by a `\tabcolsep`.
            body.append(r"\clearpage" "\n" r"\section*{Configuration}" "\n")
            body.append(
                r"\begin{center}\begin{longtable}{@{}"
                r">{\raggedright\arraybackslash}p{" f"{CONFIG_KEY_FRACTION}"
                r"\textwidth}"
                r">{\raggedright\arraybackslash}p{\dimexpr"
                f"{1.0 - CONFIG_KEY_FRACTION:.2f}"
                r"\textwidth-2\tabcolsep\relax}"
                r"@{}}\toprule" "\n")
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
