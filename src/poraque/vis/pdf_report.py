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
\usepackage{{longtable}}
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
              configuration=None, caveats=(), filename=None):
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
            "parity": "Predicted against true voxel values, with the identity line.",
            "error_histogram": "Distribution of the signed voxel-wise error.",
        }
        if figures:
            body.append(r"\section*{Comparison with DFT}" "\n")
            for path in figures:
                stem = os.path.splitext(os.path.basename(path))[0]
                caption = next((c for key, c in captions.items() if key in stem),
                               _escape(stem))
                body.append(self._figure_block(path, caption))

        if configuration:
            body.append(r"\clearpage" "\n" r"\section*{Configuration}" "\n")
            body.append(r"\begin{center}\begin{longtable}{@{}ll@{}}\toprule" "\n")
            for key, value in sorted(configuration.items()):
                body.append(f"{_escape(key)} & {_escape(value)} \\\\\n")
            body.append(r"\bottomrule\end{longtable}\end{center}" "\n")

        body.append(r"\end{document}" "\n")
        source = "".join(body)

        return self._compile(source, figures, target)

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
