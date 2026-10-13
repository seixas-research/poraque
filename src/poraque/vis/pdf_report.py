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
#: Sized to the content rather than guessed. It was set against a 37-character
#: key, 184.5 pt wide at 11 pt Latin Modern -- 0.405 of the 455 pt text block
#: on A4 with the 2.5 cm margins this report uses -- with a little over 15 pt
#: of headroom on top. The longest key is now
#: ``fine_tuning.pretrained_checkpoint`` at 33, so the headroom has grown
#: rather than shrunk. It is **not** being retightened: this column is sized so
#: that every key of every section fits on one line, and a fraction trimmed to
#: today's longest key is one the next key breaks.
#:
#: A key broken across two lines is not merely untidy: ``fine\_tuning.pre``
#: over ``trained\_checkpoint`` reads as two settings, and the reader has to
#: reassemble the name of the thing whose value sits beside it.
CONFIG_KEY_FRACTION = 0.44


def _wrappable(text, threshold=26):
    """
    Escape a configuration value and let it break inside a fixed-width column.

    A ``p`` column wraps at spaces, which is enough for a value like the
    ``training.physics_informed`` dict. It is not enough for a long
    value with no
    spaces at all -- an absolute path, or a cache tag such as
    ``res32_blur0.15spec`` -- which stays one unbreakable word and overruns the
    column exactly as an ``l`` column did. Break opportunities are inserted
    after the separators such values are built from, so the line can turn
    without hyphenating a path in a misleading place.

    Short values are returned untouched: the markup is pure noise below the
    width where wrapping could ever be needed. The threshold is the column's
    own capacity, not a round number — set above it, a value one or two
    characters too wide gets no break opportunity and overruns the cell, which
    is exactly the case a long key such as ``fine_tuning.pretrained_checkpoint``
    lands in.
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

    A nested mapping -- ``training.physics``, an operator's ``occupancies`` --
    becomes
    one ``key: value`` per line rather than a single 140-character repr. The
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
    >>> report.build(task="pseudo2tau", metrics=..., figures=[...])  # doctest: +SKIP
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
        documents. The brand accent is ``poraquered``, the logo's own color --
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
            logo_line = (r"\includegraphics[width=0.30\textwidth]"
                         r"{logo_dark.png}\par" "\n" r"\vspace{0.5cm}")
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
% The logo's yellow and a navy ramp held at one hue (~215 degrees), varied
% only in lightness. The yellow is ACCENT ONLY -- 1.7:1 on white -- so it
% carries rules and bands and never body text. poraquelime is the sky-blue
% step between; poraquegreen (name kept for the calls that use it) is the navy
% anchor, 12.3:1 on white. poraquecover is COVER GROUND ONLY -- named for the
% restriction, since a name that states it is harder to misuse than a
% comment -- and about 9 points of lightness darker than poraquegreen.
\definecolor{{poraqueyellow}}{{RGB}}{{255,204,0}}
\definecolor{{poraquelime}}{{RGB}}{{145,178,226}}
\definecolor{{poraquegreen}}{{RGB}}{{23,53,95}}
\definecolor{{poraquecover}}{{RGB}}{{10,30,59}}
\colorlet{{poraquered}}{{poraquegreen}}
\definecolor{{poraqueamber}}{{RGB}}{{176,125,42}}
\definecolor{{poraquedark}}{{RGB}}{{23,53,95}}
\definecolor{{shadegray}}{{RGB}}{{90,96,104}}
\definecolor{{codebg}}{{RGB}}{{246,249,242}}
\definecolor{{warnamber}}{{RGB}}{{176,125,42}}

% --- Heading typography, as in the guides ----------------------------------
\titleformat{{\section}}
  {{\Large\bfseries\color{{poraquedark}}}}
  {{\textcolor{{poraquered}}{{\thesection}}}}{{0.8em}}{{}}
  [\vspace{{-6pt}}{{\color{{poraquelime}}\rule{{\textwidth}}{{0.8pt}}}}]
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
\renewcommand{{\headrule}}{{{{\color{{poraqueyellow}}%
  \hrule width\headwidth height\headrulewidth}}}}
\renewcommand{{\footrulewidth}}{{0pt}}
\markboth{{{title}}}{{}}

% --- Callout boxes, matching the guides ------------------------------------
\newtcolorbox{{pwarn}}[1][]{{%
  breakable, enhanced, colback=poraqueyellow!14, colframe=poraqueyellow,
  boxrule=0pt, leftrule=3pt, arc=1pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fonttitle=\bfseries\small, title={{Caveat}}, coltitle=poraquegreen,
  attach title to upper=\par\vspace{{2pt}}, #1}}
\newtcolorbox{{pnote}}[1][]{{%
  breakable, enhanced, colback=poraquelime!14, colframe=poraquegreen,
  boxrule=0pt, leftrule=3pt, arc=1pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fonttitle=\bfseries\small, title={{Note}}, coltitle=poraquegreen,
  attach title to upper=\par\vspace{{2pt}}, #1}}

\setlist[itemize]{{itemsep=2pt,topsep=4pt}}
\setlist[enumerate]{{itemsep=3pt,topsep=4pt}}

\begin{{document}}

% The report's answer to the guides' full-bleed cover: one dark-green panel
% carrying the logo drawn for dark grounds, closed by a yellow rule. A report
% is two or three pages, so a whole cover page would be mostly cover -- this
% gives it the same identity in the space a masthead takes.
\begin{{tcolorbox}}[enhanced, colback=poraquecover, colframe=poraquecover,
  boxrule=0pt, arc=3pt, left=18pt, right=18pt, top=16pt, bottom=14pt,
  borderline south={{3pt}}{{0pt}}{{poraqueyellow}}]
\centering\sffamily\color{{white}}
{logo_line}
{{\LARGE\bfseries {title}\par}}
\vspace{{0.28cm}}
{{\normalsize\color{{poraquelime}} {subtitle}\par}}
\end{{tcolorbox}}
\vspace{{-0.15cm}}
\hfill{{\footnotesize\color{{shadegray}} generated {stamp}}}
\vspace{{0.55cm}}
"""

    #: Order the split rows are printed in. `train` before `validation`
    #: because that is the order they happen in, and anything else -- a
    #: `k`-fold's `fold 3`, a split nobody anticipated -- follows in whatever
    #: order it sorts, rather than being dropped for not being one of the two.
    _SPLIT_ORDER = {"train": 0, "validation": 1}

    #: The columns of the split table: ``(heading, key, has a unit)``.
    #:
    #: Five metrics, chosen so that each answers a question the others cannot,
    #: and no more than five because the table has to fit the text block:
    #:
    #: ``relative_l2``
    #:     the headline error, and the one every other run in this repository
    #:     is quoted in.
    #: ``relative_h1``
    #:     values *and* gradients. A prediction can match a density pointwise
    #:     and still be rough, and the von Weizsacker kinetic energy depends on
    #:     the gradient -- so an L2 that is small beside an H1 that is not is a
    #:     model whose downstream energetics will disappoint.
    #: ``mae``
    #:     the typical voxel, in the field's own units, where a squared error
    #:     is not readable as a quantity.
    #: ``max_abs``
    #:     the worst voxel. Means and RMS both hide a single catastrophic site,
    #:     and near a nucleus is exactly where one occurs.
    #: ``integral_error``
    #:     conservation. For ``ext2pseudo`` this is the electron-count error in
    #:     electrons, and it is the metric a pointwise loss controls worst: a
    #:     per-voxel MSE is nearly indifferent to a uniform 2 % error in rho,
    #:     while the electrostatics built from that density are of order 1e4 eV.
    #:
    #: ``mse``, ``rmse``, ``r2``, ``nrmse_range`` and ``jsd`` are not dropped,
    #: only not typeset: every one of them is still written per structure into
    #: the run's metrics JSON, which is machine-readable and has no margins.
    _SPLIT_COLUMNS = (
        (r"rel.\ $L^2$", "relative_l2", False),
        (r"rel.\ $H^1$", "relative_h1", False),
        ("MAE", "mae", True),
        (r"Max.\ error", "max_abs", True),
        (r"$|\Delta N|$", "integral_error", "integral"),
    )

    @staticmethod
    def _integral_unit(unit):
        """
        The unit of a field's integral over the cell.

        ``e/Ang^3`` integrates to ``e`` and ``eV/Ang^3`` to ``eV``, so the rule
        is to strip the volume. Anything else is left alone and marked, because
        a wrong unit on a conservation error is worse than none: the number is
        small either way and only its unit says whether it is small *enough*.
        """
        volume = "/Ang^3"
        return unit[:-len(volume)] if unit.endswith(volume) else ""

    def _metrics_table(self, per_material, unit):
        r"""
        Dataset counts and five metrics, one row per split.

        **This used to be one row per structure, and that is what made the
        report unreadable.** The row count was the size of the dataset, so a
        115-material run printed a hundred and fifteen lines of seven columns
        -- six pages of numbers nobody reads individually -- and the two
        questions a reader actually brings to it, *how big was the set* and
        *did it generalize*, had to be answered by scrolling and averaging.

        Both are answered directly now. The counts are the dataset summary the
        report never had; and the ``train`` and ``validation`` rows side by
        side are the generalization gap, which is the single most useful thing
        this table can say and was previously spread over a column of a hundred
        rows sorted by something else.

        Nothing measured is lost. The per-structure numbers -- and the four
        metrics not typeset here -- stay in the run's metrics JSON, and the
        parity figure draws every structure of each split (a fixed random sample
        of each one's voxels), so the *spread* the old table carried is still on
        the page, drawn rather than enumerated.

        The rows are ``train``, ``validation`` and ``all``. A run with nothing
        held out prints ``train`` alone: a ``validation`` row of dashes reads
        as a measurement that came out undefined, and there was no measurement,
        while an ``all`` row would repeat the row above it verbatim.

        A column every structure left ``None`` is dropped rather than printed
        empty -- ``relative_h1`` and ``integral_error`` are ``None`` for a
        two-channel prediction, whose shape is not the grid's.
        """
        collected = {}
        for entry in per_material.values():
            values = entry.get("metrics", entry)
            collected.setdefault(entry.get("split", "") or "unsplit",
                                 []).append(values)
        every = [entry for entries in collected.values() for entry in entries]

        def finite(entries, key):
            return [entry[key] for entry in entries
                    if entry.get(key) is not None and np.isfinite(entry[key])]

        columns = [column for column in self._SPLIT_COLUMNS
                   if finite(every, column[1])]
        if not columns:
            # No metric survived -- an empty `per_material`, or one carrying
            # nothing this table knows. Say so rather than typesetting a table
            # of one blank column.
            return (r"\begin{center}\small\color{shadegray}"
                    r"No per-structure metrics were recorded."
                    r"\end{center}" "\n")

        def row(label, entries, bold=False):
            cells = [_escape(label), f"{len(entries)}"]
            for _, key, _unit in columns:
                values = finite(entries, key)
                cells.append(_number(np.mean(values)) if values else "--")
            if bold:
                cells = [rf"\textbf{{{cell}}}" for cell in cells]
            return " & ".join(cells) + r" \\"

        rows = [row(split, entries) for split, entries in
                sorted(collected.items(),
                       key=lambda item: (self._SPLIT_ORDER.get(item[0], 2),
                                         item[0]))]
        if len(collected) > 1:
            rows.append(r"\midrule")
            rows.append(row("all", every, bold=True))

        integral_unit = self._integral_unit(unit)
        units = {True: f"[{_escape(unit)}]",
                 "integral": f"[{_escape(integral_unit)}]" if integral_unit
                             else r"{\footnotesize per cell}",
                 False: ""}
        heading = " & ".join(
            [r"\textbf{Split}", r"\textbf{Structures}"]
            + [rf"\textbf{{{title}}}" for title, _, _ in columns])
        unit_row = " & ".join(["", ""] + [units[has_unit]
                                          for _, _, has_unit in columns])

        return (
            r"\begin{center}\begin{tabular}{@{}l"
            + "r" * (len(columns) + 1) + r"@{}}" "\n"
            r"\toprule" "\n"
            + heading + r" \\" "\n"
            + unit_row + r" \\" "\n"
            r"\midrule" "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{center}" "\n"
        )

    def _pair_block(self, left, right, left_label, right_label, caption):
        """
        Two figures side by side, under one caption.

        Used where the point *is* the comparison. Two full-width figures would
        put them on separate pages at this size, and a comparison read across a
        page turn is not a comparison.
        """
        return (
            r"\begin{figure}[H]" "\n"
            r"\centering" "\n"
            r"\begin{minipage}{0.49\textwidth}\centering" "\n"
            rf"\includegraphics[width=\textwidth]{{{os.path.basename(left)}}}" "\n"
            rf"\small\color{{shadegray}} {left_label}" "\n"
            r"\end{minipage}\hfill" "\n"
            r"\begin{minipage}{0.49\textwidth}\centering" "\n"
            rf"\includegraphics[width=\textwidth]{{{os.path.basename(right)}}}" "\n"
            rf"\small\color{{shadegray}} {right_label}" "\n"
            r"\end{minipage}" "\n"
            rf"\caption*{{\small\color{{shadegray}} {caption}}}" "\n"
            r"\end{figure}" "\n"
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
    def _charges_block(self, analysis):
        r"""
        Per-atom populations and net charges, as a table.

        Accepts a :class:`~poraque.analysis.PartialCharges` or the dict it
        serializes to, so a report can be rebuilt from a run's JSON summary
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
              configuration=None, caveats=(), filename=None, charges=None):
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
                           "difference on a shared color scale.",
            "parity_magnetization": "Predicted against true magnetization "
                                    "density, over every structure of each "
                                    "split, on linear axes: m is signed.",
            "parity": "Predicted against true voxel values over every "
                      "structure of each split, from a random sample of 10 000 "
                      "voxels per structure, with the identity line. The "
                      "validation set is drawn beside the training set on the "
                      "same bins and color scale: a wider cloud about the line "
                      "is the generalization gap.",
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

        if configuration:
            # Fixed-width columns, not `ll`. An `l` column is a single
            # unbreakable box, so a long value -- `physics_informed` is 140
            # characters of nested dict -- ran straight past the right margin.
            #
            # The widths are fractions of `\textwidth` rather than centimeters,
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
                # `fine_tuning.pretrained_checkpoint` is 33 characters with
                # no space in it, so escaping alone leaves one unbreakable word
                # that overruns its column and prints on top of the value.
                body.append(f"{_wrappable(key)} & {_format_value(value)} \\\\\n")
            body.append(r"\bottomrule\end{longtable}\end{center}" "\n")

        body.append(r"\end{document}" "\n")
        source = "".join(body)

        return self._compile(source, list(figures), target)

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
                # The masthead sits on dark green, so it needs the logo drawn
                # for dark grounds. It lives beside the light one; without
                # this copy the banner compiles to a missing-figure box.
                dark = os.path.join(os.path.dirname(self.logo), "logo_dark.png")
                shutil.copy(dark if os.path.exists(dark) else self.logo,
                            os.path.join(workdir, "logo_dark.png"))
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
