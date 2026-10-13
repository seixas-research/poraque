# -*- coding: utf-8 -*-
# file: train_symbolic.py
"""
The symbolic-distillation stage of ``scripts/poraque_train.py`` as it stood
when it was archived (2026-10-13): validation, the post-fit search, its
figures and its command-line flags. Verbatim; not importable as it stands.
It ran after ``save_task_checkpoint`` in ``run_task``::

    if config.symbolic.enable:
        profile.begin(f"{task.name}: symbolic distillation")
    symbolic = run_symbolic_distillation(task, train_set, operator, config,
                                         log, validation=validation,
                                         report=report)

and its result went to ``ModelReport.build(symbolic=...)`` and to the
metrics JSON as ``result_to_dict(symbolic)``.
"""

def validate_symbolic_settings(settings):
    """
    Check the symbolic settings before anything is trained.

    Distillation runs *after* the fit, so a typo in ``features`` would
    otherwise surface an hour in, with the search — not the training — as the
    only casualty but the feedback uselessly late. Checked here it costs a
    millisecond and fails on the command line.
    """
    if not settings.enable:
        return
    if settings.template not in TEMPLATES:
        raise SystemExit(
            f"symbolic.template={settings.template!r} is not known; expected "
            f"one of {list(TEMPLATES)}.")
    if settings.features not in FEATURE_SCHEMES:
        raise SystemExit(
            f"symbolic.features={settings.features!r} is not a known feature "
            f"scheme; expected one of {list(FEATURE_SCHEMES)}.")
    if settings.target not in ("model", "reference"):
        raise SystemExit(
            f"symbolic.target={settings.target!r} is not known; expected "
            f"'model' (distil the trained operator) or 'reference' (fit the "
            f"DFT data).")
    if settings.epsilon <= 0:
        raise SystemExit(
            f"symbolic.epsilon={settings.epsilon!r} must be positive: it is a "
            f"density floor and clamps every denominator.")
    if settings.data_loss not in DATA_LOSSES:
        raise SystemExit(
            f"symbolic.data_loss={settings.data_loss!r} is not known; expected "
            f"one of {sorted(DATA_LOSSES)}.")
    physics = symbolic_physics(settings)
    unknown = set(physics) - {"enable", "positivity_weight",
                              "thomas_fermi_weight", "von_weizsacker_weight",
                              "p_infinity"}
    if unknown:
        raise SystemExit(
            f"Unknown key(s) in symbolic.physics: {sorted(unknown)}.\n"
            f"  Note that this block constrains the symbolic *search*. The "
            f"terms that constrain the neural operator -- electron count, "
            f"Euler-Lagrange -- belong in training.physics_informed.")
    if physics["enable"] and float(physics["p_infinity"]) <= 0:
        raise SystemExit(
            f"symbolic.physics.p_infinity={physics['p_infinity']!r} must be "
            f"positive: it is the reduced gradient standing in for the von "
            f"Weizsacker limit, and p is a magnitude.")
    for key in ("positivity_weight", "thomas_fermi_weight",
                "von_weizsacker_weight"):
        if float(physics[key]) < 0:
            raise SystemExit(
                f"symbolic.physics.{key}={physics[key]!r} must not be "
                f"negative: a negative penalty rewards the violation it is "
                f"meant to forbid.")
    # `poraque.ml.gp` selects by tournaments of three, so the one hard
    # requirement is that a tournament can choose between anything at all.
    if settings.population_size < 4:
        raise SystemExit(
            f"symbolic.population_size={settings.population_size} is too "
            f"small: the search selects by tournament and cannot choose "
            f"between fewer than four individuals.")




def _figure_sink(config):
    """
    Somewhere to write the symbolic parity plot when no plot directory is set.

    Returns ``None`` when no PDF report is being built either — with neither
    output configured there is nothing the figure could appear in, and writing
    it would only litter.
    """
    if not config.report_dir():
        return None

    from poraque.vis import TrainingReport

    return TrainingReport(os.path.join(config.report_dir(),
                                       f"{model_name(config)}_figures"),
                          dpi=config.output.dpi, fmt=config.output.plot_format,
                          save_data=config.output.save_raw_plot_data)


def run_symbolic_distillation(task, dataset, operator, config, log,
                              validation=None, report=None):
    r"""
    Search for a closed-form expression reproducing the trained operator.

    Only ``pseudo2tau`` is attempted. Distillation looks for a *functional of the
    density*, and ``ext2pseudo`` maps a potential to a density — the Hohenberg-Kohn
    map, whose whole content is non-local, so a semi-local expression for it
    would be meaningless rather than merely inaccurate.

    Returns
    -------
    SymbolicResult or None
        ``None`` when the feature is off, the task is not ``pseudo2tau``, or the
        search failed. A failure is reported and swallowed: it arrives after
        the model is already trained, and losing a finished fit to an optional
        analysis would be the worse outcome.
    """
    settings = config.symbolic
    if not settings.enable:
        return None
    if task.name != "pseudo2tau":
        log(f"\n  symbolic distillation: skipped for {task.name} "
            f"(a density functional is only meaningful for pseudo2tau).")
        return None

    from poraque.ml.symbolic import distill_dataset

    section(log, f"symbolic distillation {task.name}")
    start = time.time()
    try:
        result = distill_dataset(dataset, settings, operator=operator, log=log,
                                 validation=validation)
    except ImportError as error:
        log(f"  unavailable: {error}")
        return None
    except (ValueError, RuntimeError) as error:
        log(f"  failed: {error}")
        return None

    # The formula gets the same parity plot the network does, against the same
    # DFT reference, so the two are read on one scale rather than through two
    # differently-defined summary numbers.
    #
    # It is drawn whenever distillation produced something to draw, not only
    # when figures were asked for: the PDF report is meant to carry it, and
    # `plot_figures` and `write_pdf_report` are independent settings, so a run with a
    # report but no figure directory would otherwise silently lose the one plot
    # that shows whether the formula is any good. Held-out data is used when
    # there is any, and the fitted voxels otherwise, with the caption saying
    # which.
    scored, provenance = result.validation, "held-out data"
    if scored.get("reference") is None:
        scored, provenance = result.fitted, "the fitted voxels (training fit)"

    destination = report if report is not None else _figure_sink(config)
    if destination is not None:
        label_text, unit = FIELD_LABELS[task.target_field]
        previous, destination.prefix = (destination.prefix,
                                        f"{task.name}_symbolic")
        try:
            if scored.get("reference") is not None:
                result.parity_plot = destination.parity(
                    scored["reference"], scored["predicted"],
                    name="parity", label=label_text, unit=unit, log=True,
                    prediction_label="symbolic formula",
                    title=f"{task.name} · lowest loss "
                          f"({result.complexity} nodes) on {provenance}")
                log(f"  parity plot  : {result.parity_plot}  [{provenance}]")

            # The knee gets its own parity plot, on the same voxels. Two
            # panels of one number each say less than two plots that can be
            # laid side by side: the question is whether the shorter formula
            # gives anything away, and that is visible rather than summarized.
            knee_scored = (result.knee_validation
                           if result.knee_validation.get("reference") is not None
                           else result.knee_fitted)
            if knee_scored.get("reference") is not None:
                result.knee_parity_plot = destination.parity(
                    knee_scored["reference"], knee_scored["predicted"],
                    name="parity_knee", label=label_text, unit=unit, log=True,
                    prediction_label="symbolic formula",
                    title=f"{task.name} · Pareto knee "
                          f"({result.knee.get('complexity')} nodes) on "
                          f"{provenance}")
                log(f"  knee parity  : {result.knee_parity_plot}")

            if result.pareto:
                result.pareto_plot = destination.pareto(
                    result.pareto, knee=result.knee, name="pareto",
                    title=f"{task.name} · accuracy against complexity")
                log(f"  pareto plot  : {result.pareto_plot}")
        finally:
            destination.prefix = previous

    log(f"\n{result.summary()}")
    log(f"  search time  : {time.time() - start:.1f} s")
    log("")
    log("  NOTE: the features are semi-local, so this is the best semi-local")
    log("  functional matching the operator -- not a reconstruction of it. The")
    log("  residual measures how much of the learned map is non-local.")
    return result




def add_symbolic_arguments(parser):
    """The ``symbolic distillation`` group of ``poraque-train``'s parser."""
    group = parser.add_argument_group("symbolic distillation")
    group.add_argument("--symbolic",
                       dest="symbolic.enable",
                       action="store_const", const=True, default=None,
                       help="after training, search for a closed-form "
                            "expression reproducing the pseudo2tau operator")
    group.add_argument("--symbolic-target", dest="symbolic.target",
                       choices=["model", "reference"], default=None,
                       help="fit the operator's predictions (model) or the "
                            "DFT data (reference)")
    group.add_argument("--symbolic-features", dest="symbolic.features",
                       choices=list(FEATURE_SCHEMES), default=None,
                       help="variables handed to the engine: (rho, p, q), the "
                            "enhancement factor on (p, q), or the dimensional "
                            "(rho, |grad rho|, lap rho)")
    group.add_argument("--symbolic-template", dest="symbolic.template",
                       choices=list(TEMPLATES), default=None,
                       help="factorize the target before the search: "
                            "thomas_fermi fits the enhancement factor F in "
                            "tau = tau_TF * F")
    group.add_argument("--symbolic-epsilon", dest="symbolic.epsilon",
                       type=float, default=None, metavar="RHO",
                       help="vacuum threshold in e/a0^3; denominators are "
                            "clamped at it and voxels below it dropped")
    group.add_argument("--symbolic-iterations", dest="symbolic.iterations",
                       type=int, default=None, metavar="N")

