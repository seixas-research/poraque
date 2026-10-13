# -*- coding: utf-8 -*-
# file: report_symbolic.py
"""
The report code of symbolic distillation, cut out of ``poraque.vis`` when it
was archived (2026-10-13). Methods, verbatim, of ``TrainingReport`` and
``ModelReport``, held in two stand-in classes; they lean on the helpers of
those modules and do not run as they stand.
"""

class TrainingReportSymbolic:
    """From ``poraque/vis/report.py``: ``TrainingReport.pareto``."""

    def pareto(self, front, knee=None, name="pareto", title=None,
               annotate=6):
        r"""
        The accuracy/complexity front of a symbolic search, with its knee.

        A search does not return *an* expression, it returns a trade: every
        candidate that was the best of its length. This is that trade drawn,
        which is the plot a reader needs to decide what to quote.

        The loss axis is logarithmic. A front spans orders of magnitude, and on
        a linear axis every candidate but the most accurate collapses onto the
        floor -- the knee would then always be the shortest expression,
        whatever it cost.

        Two points are marked: the **lowest loss** (the leftmost value on the
        y axis) and the **knee**, the candidate nearest the ideal corner once
        both axes are rescaled to :math:`[0, 1]`. Where they coincide, nothing
        was traded away and the plot says so.

        Parameters
        ----------
        front : sequence of dict
            Entries with ``complexity`` and ``loss``; ``expression`` is used
            for the annotations when present.
        knee : dict, optional
            The chosen knee, from
            :func:`~poraque.ml.symbolic.pareto_knee`. Computed here when
            omitted, so the plot is usable on a bare front.
        name : str, optional
        title : str, optional
        annotate : int, optional
            How many expressions to label. Beyond a handful the labels overlap
            and the curve is what carries the message.

        Returns
        -------
        str or None
            Path written, or ``None`` for a front with nothing plottable.
        """
        usable = [e for e in front
                  if e.get("complexity") is not None
                  and e.get("loss") is not None
                  and np.isfinite(e["loss"]) and e["loss"] > 0]
        if not usable:
            return None
        usable = sorted(usable, key=lambda e: e["complexity"])

        if knee is None:
            from ..ml.symbolic import pareto_knee

            knee = pareto_knee(usable)

        complexity = np.array([e["complexity"] for e in usable], dtype=float)
        loss = np.array([e["loss"] for e in usable], dtype=float)
        best = usable[int(np.argmin(loss))]

        with self._context():
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(figsize=(7.0, 4.4))
            axes.step(complexity, loss, where="post", color=self.ink["muted"],
                      linewidth=1.2, zorder=1)
            axes.scatter(complexity, loss, s=34, color=self.ink["secondary"],
                         zorder=2, label="front")

            def mark(entry, color, marker, text):
                if not entry:
                    return
                axes.scatter([entry["complexity"]], [entry["loss"]], s=170,
                             marker=marker, facecolor=color, zorder=4,
                             edgecolor=self.ink["surface"], linewidth=1.8,
                             label=text)

            same = (knee and knee.get("complexity") == best["complexity"]
                    and knee.get("loss") == best["loss"])
            mark(best, series_color(0), "o", "lowest loss")
            if not same:
                mark(knee, series_color(1), "D", "Pareto knee")

            # Labeled sparsely and only on the interesting end: a formula
            # printed beside every point is unreadable at any figure size.
            marked = {id(best), id(knee)}
            step = max(1, len(usable) // max(1, annotate))
            for index, entry in enumerate(usable):
                if index % step and id(entry) not in marked:
                    continue
                text = str(entry.get("expression", ""))
                if len(text) > 26:
                    text = text[:25] + "\u2026"
                axes.annotate(text, (entry["complexity"], entry["loss"]),
                              textcoords="offset points", xytext=(6, 6),
                              fontsize=7, color=self.ink["muted"])

            axes.set_yscale("log")
            axes.set_xlabel("complexity (nodes)")
            axes.set_ylabel("loss")
            axes.set_title(title or "Accuracy against complexity")
            axes.grid(True, which="both", alpha=0.3)
            axes.legend(frameon=False, loc="upper right")
            if same:
                axes.text(0.02, 0.04,
                          "the knee is the lowest-loss expression:\n"
                          "nothing was traded away",
                          transform=axes.transAxes, fontsize=8,
                          color=self.ink["muted"])
            figure.tight_layout()
            return self._save(figure, name)



class ModelReportSymbolic:
    """From ``poraque/vis/pdf_report.py``: ``ModelReport``'s symbolic blocks."""

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

    #: Maths symbols for the quantities a distillation can fit. Without this
    #: ``tau`` sets as the product t·a·u in an equation environment.
    _TARGET_SYMBOLS = {"tau": r"\tau", "rho": r"\rho", "F": "F"}



    def _symbolic_block(self, result):
        r"""
        Typeset the distilled expression and its accuracy/complexity front.

        Accepts either a :class:`~poraque.ml.symbolic.SymbolicResult` or the
        plain dict it serializes to, so a report can be rebuilt from a run's
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

        knee = get("knee") or {}
        if knee and knee.get("expression") not in (None, get("expression")):
            body.append(
                f"The front's \\textbf{{knee}} --- the candidate nearest the ideal "
                f"corner once complexity and log-loss are rescaled to "
                f"$[0,1]$ --- is at {knee.get('complexity')} nodes with loss "
                f"{_number(knee.get('loss'), 4)}, against "
                f"{get('complexity', 0)} nodes and "
                f"{_number(get('loss'), 4)} for the lowest-loss expression. "
                f"It is the one worth quoting when length is a cost: the "
                f"parity plots below show what, if anything, the shorter form "
                f"gives away.\n\n")
        elif knee:
            body.append(
                "The front's knee coincides with the lowest-loss expression, "
                "so nothing was traded away: no shorter candidate came close "
                "enough to be worth its length.\n\n")

        held_out = get("validation") or {}
        if held_out.get("n_points"):
            body.append(
                f"On {held_out['n_points']} voxels of the held-out structures, "
                f"against the DFT reference: relative "
                f"$L^2 = {self._math_number(held_out.get('relative_l2'))}$, "
                f"$R^2 = {self._math_number(held_out.get('r2'))}$.\n\n")

        # Which data the plots are drawn from depends on whether anything was
        # held out; saying "held-out" regardless would overstate a training
        # fit, which is the one thing a parity plot must not do.
        on_held_out = bool(held_out.get("n_points"))
        provenance = ("on held-out structures" if on_held_out else
                      "on the voxels it was fitted to --- a training fit, "
                      "since this run held nothing out")

        parity = get("parity_plot")
        knee_parity = get("knee_parity_plot")
        if (parity and os.path.exists(parity)
                and knee_parity and os.path.exists(knee_parity)):
            # Side by side, at half width. The question these two answer is a
            # comparison -- does the shorter formula give anything away? -- and
            # a comparison read across a page turn is not one.
            body.append(self._pair_block(
                parity, knee_parity,
                f"lowest loss, {get('complexity', 0)} nodes",
                f"Pareto knee, {(get('knee') or {}).get('complexity')} nodes",
                f"The two candidates against the DFT reference "
                f"{provenance}, on identical voxels. The left is the most "
                f"accurate expression the search found; the right is the one "
                f"the front's knee selects. Where the clouds are "
                f"indistinguishable, the extra nodes bought nothing."))
        elif parity and os.path.exists(parity):
            body.append(self._figure_block(
                parity,
                f"The distilled formula against the DFT reference "
                f"{provenance}. Read it beside the operator's own parity plot: "
                f"the gap between them is what the closed form gives up."))

        pareto_plot = get("pareto_plot")
        if pareto_plot and os.path.exists(pareto_plot):
            body.append(self._figure_block(
                pareto_plot,
                "The accuracy/complexity front, with the lowest-loss "
                "candidate and the knee marked. The loss axis is logarithmic: "
                "a front spans orders of magnitude, and on a linear axis every "
                "candidate but the most accurate collapses onto the floor."))

        # Which constraints were *fitness* rather than a post-hoc filter. It
        # has to be on the page beside the front, because it changes what the
        # loss column means: a constrained objective is not comparable with an
        # unconstrained one, and a reader has no way to tell them apart.
        enforced = get("constraints_enforced") or []
        if enforced:
            names = ", ".join(_escape(name.replace("_", " "))
                              for name in enforced)
            body.append(
                f"The search was constrained: {names} were penalized inside "
                f"the evolutionary loop rather than filtered afterwards, so "
                f"the populations could not spend their budget on forms that "
                f"had to be discarded. The loss column below is therefore that "
                f"constrained objective, and is not comparable with an "
                f"unconstrained run's; the $R^2$ and relative $L^2$ above are "
                f"computed from the expression alone and are unaffected.\n\n")

        body.append(self._asymptotic_block(get("limits") or {}))

        front = get("pareto") or []
        if front:
            body.append(r"\subsection*{Accuracy against complexity}" "\n")
            body.append(r"\begin{center}\begin{longtable}"
                        r"{@{}rrrc>{\raggedright\arraybackslash}p{7.0cm}@{}}"
                        r"\toprule" "\n")
            # `\endhead`, so a front long enough to turn the page keeps its
            # column names. Without it the second page is four unlabeled
            # columns of numbers and a formula.
            body.append(r"Nodes & Loss & $d$ & Limits & Expression \\" "\n"
                        r"\midrule" "\n" r"\endhead" "\n")
            chosen = knee.get("complexity") if knee else None
            for entry in front:
                limits = entry.get("limits") or {}
                # The knee is marked in the table it was chosen from, so the
                # figure and the front cannot be read as saying different
                # things.
                mark = r"\,$\bullet$" if entry["complexity"] == chosen else ""
                body.append(f"{entry['complexity']}{mark} & "
                            f"{_number(entry['loss'], 4)} & "
                            f"{_number(entry.get('distance'), 3)} & "
                            f"{_escape(limits.get('badge', '--/--'))} & "
                            f"{_wrappable(entry['expression'])} \\\\\n")
            body.append(r"\bottomrule\end{longtable}\end{center}" "\n")
            body.append(r"\emph{\small $d$ is the distance to the ideal "
                        r"corner; $\bullet$ marks the knee. "
                        r"Limits: \texttt{TF} = recovers "
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

