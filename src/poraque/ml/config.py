# -*- coding: utf-8 -*-
# file: config.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
YAML configuration for neural-operator training runs.

A training run is defined by four groups of settings — where the data is, how
the operator is shaped, how it is optimised, and where results go — and
:class:`TrainingConfig` mirrors that structure exactly. Keeping the run
definition in a file rather than in a shell history makes an experiment
reproducible and diffable: the config is written next to the results, so a run
can always be repeated verbatim.

Precedence, highest first:

1. explicit command-line flags,
2. values in the YAML file,
3. the defaults declared here.

That ordering is what allows a single committed config to be swept over one
parameter from the shell (``--epochs 500``) without editing or copying it.

Unknown keys are rejected rather than ignored: a typo like ``learn_rate`` would
otherwise be silently dropped and the run would quietly use the default,
producing results that do not match the file that appears to describe them.
"""

from dataclasses import asdict, dataclass, field, fields

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None


def _require_yaml():
    if yaml is None:  # pragma: no cover
        raise ImportError(
            "Reading or writing YAML configs requires PyYAML: pip install pyyaml"
        )


@dataclass
class TaskConfig:
    """
    What is trained, and what the run's artefacts are called.

    Attributes
    ----------
    type : str
        ``"ext2chg"``, ``"chg2tau"`` or ``"all"``.
    name : str
        Identifier for **this run**, and the only thing that separates it from
        another. Everything the run writes goes in one directory named after
        it::

            models/<name>/
                <name>.pfno         the weights
                log/                training log, metrics JSON, resolved config
                plots/              loss curves, parity, field slices
                report/             the generated PDF

        Two runs that differ in anything worth keeping — a chemical space, a
        resolution, a set of physics weights — should differ in ``name``, or
        the second silently overwrites the first. The default reproduces the
        historical filename, so a config written before this key existed
        writes exactly where it always did.

    Notes
    -----
    ``task`` also accepts the bare string it used to be::

        task: ext2chg                       # same as {type: ext2chg}

    which is read as ``type`` with the default ``name``.
    """

    type: str = "all"
    # Matches poraque.ml.training.BUNDLE_FILENAME without its suffix, so an
    # unnamed run writes the file every earlier version wrote.
    name: str = "poraque_models"

    def names(self):
        """
        The task names to train, expanding ``"all"``.

        Returns
        -------
        list of str
        """
        return (["ext2chg", "chg2tau"] if self.type == "all"
                else [str(self.type)])


@dataclass
class DataConfig:
    r"""
    Where the fields live and how they are prepared.

    Attributes
    ----------
    train_paths : list of str or None
        The dataset, as a **list of directories**, which may mix layouts::

            data:
              train_paths:
                - data/vasp             # local DFT runs
                - data/MP/chgcar        # a Materials Project download

        Each entry is auto-detected (see ``source``) and everything found is
        pooled into one training set. ``null`` falls back to ``root``, so a
        config written before this key existed keeps working unchanged.

        .. warning::

           Mixing a calculation archive with a bulk density archive mixes two
           *definitions* of the external potential — the tabulated local
           pseudopotential and the Gaussian pseudo-ion model, which differ by
           roughly 0.1 relative :math:`L_2`. The run warns when it happens. It
           is a legitimate trade (far more data, a fuzzier input) but never a
           good accident.
    root : str
        Single dataset directory, used when ``train_paths`` is ``null``.

        The default points at the *structures* subdirectory rather than at
        ``data/vasp`` itself, because a calculation source recognises run
        directories in a path or one level below it and no further. The Pt
        dataset keeps its bulk cells in ``data/vasp/structures`` and its
        isolated atoms in ``data/vasp/isolated_atoms``, one directory per
        element, and that separation is the point: a single atom in a 10 Å box
        is the *reference* for the baseline and the PAW records, never a
        training sample. Pointing ``root`` at the parent would either find
        nothing or, if the walk were deepened, quietly train on the atom.
    source : str
        Layout of each path, or ``"auto"`` (the default) to detect it.

        ``"vasp"`` is the classic layout: one calculation directory per
        material, selected by ``pattern``, each holding the inputs and
        volumetric outputs of a run. The external potential is computed from
        those inputs; ``TAUCAR`` is used where a run wrote one and simply not
        offered where it did not.

        ``"bulk"`` is an archive of standalone ``CHGCAR`` files, compressed or
        not — what the Materials Project ships. The potential is built from the
        structure each density carries in its own header, exactly when
        ``potcar_dir`` supplies the pseudopotentials and by the Gaussian
        pseudo-ion model otherwise. ``chg2tau`` is not trainable on such an
        archive: no public archive publishes :math:`\\tau`.

        ``"prepared"`` is a directory of per-material ``EXTCAR``/``CHGCAR``/
        ``TAUCAR`` folders — a cache from an earlier run, read as it stands.

        A single name applies to every path; pass a list to set them
        individually.
    cache : str
        Where spectrally downsampled copies are written.
    pattern : str
        Prefix identifying calculation folders inside a ``vasp`` path. Ignored
        by the other layouts, which have no subdirectories to select.

        It is a **prefix, not a glob**: matching is
        ``os.path.basename(child).startswith(pattern)``, so ``"structure"``
        selects ``structure_0042`` and an empty string selects every
        subdirectory.

        The default names this project's own layout
        (``data/vasp/structures/structure_00NN``). It was ``"struct"`` until
        2026-08-28 — a prefix of the current one, so it selected exactly the
        same directories, which is why the two are interchangeable on this
        data and why the change is safe. It is **part of the cache
        fingerprint** even so: two prefixes that happen to agree on one dataset
        need not agree on the next, and a cache that silently held a different
        set of materials than the config describes is the failure the
        fingerprint exists to prevent.
    code : str
        DFT code name, or ``"auto"`` to detect it.
    potcar_dir : str or None
        A ``POTCAR`` **library** — one subdirectory per pseudopotential, as
        VASP ships them (``<potcar_dir>/Ag/POTCAR``, optionally ``.gz``/``.Z``;
        a flat ``POTCAR.Ag`` layout is also recognised).

        It matters most for a Materials Project download, which publishes a
        structure and a density and no pseudopotentials. Supply the library and
        the external potential is built from VASP's **tabulated local
        potential**, reproducing a reference ``EXTCAR`` to a relative
        :math:`2\\times10^{-5}`; omit it and the **Gaussian pseudo-ion model**
        stands in, whose residual against that reference is of order
        :math:`0.1` relative :math:`L_2`.

        The same setting also rescues a local run whose ``POTCAR`` was stripped
        for licensing. A run that has its own ``POTCAR`` ignores this entirely.

        Choose the library that generated the data. The Materials Project uses
        the VASP PBE set (``PAW_PBE``); pointing at ``PAW_LDA`` would build a
        potential the densities were never computed from, which is worse than
        the Gaussian fallback because it looks exact.

        Missing or unreadable entries fall back to the Gaussian model **with a
        warning** rather than failing the run: it is a quality difference, not
        an outage.
    sigma : float or None
        Gaussian pseudo-ion width in Å, used only where the Gaussian model is
        reached — i.e. where no ``POTCAR`` and no ``potcar_dir`` supply a
        tabulated potential. ``null`` derives it per species from the
        pseudopotential core radius where one is known, and falls back to
        :data:`poraque.fields.external.DEFAULT_SIGMA` otherwise. Set it
        explicitly when a reference ``EXTCAR`` allows it to be fitted.
    resolution : int
        Longest grid axis after spectral downsampling. The reduction is a
        Fourier truncation, exact for band-limited plane-wave fields.
    gaussian_blur : float or None
        Width in Å of a Gaussian blur applied to the computed external
        potential. ``null`` disables it.
    blur_method : str
        ``"spectral"`` (exact, isotropic on any cell) or ``"ndimage"``
        (:func:`scipy.ndimage.gaussian_filter` with ``mode="wrap"``, which
        blurs along grid axes and is therefore anisotropic in Cartesian space
        unless the cell is orthogonal).
    spin : {"auto", true, false}
        Whether the densities are spin-polarised (``ISPIN = 2``), in which case
        a ``CHGCAR`` carries two channels — the total density and the
        magnetisation density — and the operator is built with two channels to
        match. ``"auto"`` (the default) reads the answer off the data, since a
        spin-polarised ``CHGCAR`` has a second grid block and a collinear one
        does not.

        ``"auto"`` is resolved against the **sources**, before anything is
        written, and for the dataset as a whole — one operator has one channel
        count, so a cache with two blocks for some materials and one for others
        could not be trained from. A mixture therefore resolves to spin and its
        unpolarised members carry ``m = 0``, which is a true statement about a
        non-magnetic calculation and is what makes an ``ISPIN = 2`` operator a
        strict generalisation of an ``ISPIN = 1`` one.

        ``false`` is a deliberate opt-out that discards the magnetisation, and
        says so in the build log; ``true`` demands it and **raises** on data
        that has none, rather than fitting a channel to zeros. The resolved
        value names the cache directory, so the two layouts never share one.

        .. note::

           Until 2026-08-27 this key was flattened to ``spin is True`` before
           it reached the cache builder, so the default ``"auto"`` meant *no
           spin*: every ``ISPIN = 2`` magnetisation block was discarded on the
           way in, and the dataset's own auto-detection then reported the cache
           as unpolarised — which it was, by then. A cache built before that
           date holds one channel whatever its config said.
    precision : str
        Dtype the volumetric fields are **held in**: ``float64`` (the default,
        and what every field used before this was selectable), ``float32`` or
        ``float16``.

        A memory setting, not an accuracy one. A 160³ field is 16 MB in double
        and 8 MB in single, and a committee of five models scoring a pool holds
        several at once. Integrals over the cell — the electron count, the
        energy terms — are sums of N³ values, so keep ``float64`` wherever
        those numbers are the result, and drop to ``float32`` when the field is
        on its way into a network that computes in single precision anyway.
        ``float16`` is for archiving a large pool and is too coarse for any
        integral.

        Distinct from ``model.precision``, which is what the operator
        *computes* in.
    xc : str
        **The exchange-correlation functional the reference data was computed
        with.** One of :data:`~poraque.ml.physics.XC_FUNCTIONALS` (``"pbe"``,
        ``"lda"``, ``"pbe-x"``, ``"lda-x"``, ``"none"``), or ``"auto"`` to read
        it off the calculation.

        This is a statement about the *data*, not a choice about the model.
        **The training loop does not yet consult it**: the Euler-Lagrange
        penalty (``euler_lagrange_weight``) is evaluated without
        :math:`v_{\\rm xc}` and warns to that effect. The setting is honoured
        by the post-training analyses that evaluate the full residual — the
        ``experiments/euler_lagrange`` scripts and
        :func:`~poraque.fields.io.resolve_xc` — where getting it wrong does
        not approximate the right answer: an LDA potential on a PBE density is
        of order 1 eV away from the PBE one in a valence region, and that
        error lands entirely in the residual, where it would be misread as the
        error of the kinetic functional.

        ``"auto"`` resolves in the order VASP itself does: the ``INCAR``
        ``GGA`` tag if present, else the ``LEXCH`` tag of the pseudopotentials
        (``PE`` for PBE, ``CA`` for LDA, ``91`` for PW91). A ``PAW_PBE``
        library therefore resolves to ``"pbe"`` with nothing to declare. Set it
        explicitly when the data was produced by a code whose settings are not
        recorded beside it.
    delta_density : bool
        Train the density operator on the **charge-density variation**

        .. math::

            \delta\rho(\mathbf r) = \rho(\mathbf r)
                                    - \rho_{\rm sup}(\mathbf r),

        the residual over a superposition of isolated atoms placed at the
        structure's own atomic positions, rather than on :math:`\rho` itself.
        **On by default as of 2026-08-26.**

        Most of a crystal's valence density is its free atoms side by side —
        measured at ~96 % of the field in :math:`L^2` on this project's cells —
        so the residual left to learn is roughly twenty times smaller and
        carries almost none of the four-orders-of-magnitude core peak the
        ``asinh`` transform exists to absorb. The baseline is built in
        reciprocal space from the stored atomic form factors, so it is exactly
        periodic and grid-independent, and works unchanged for **bulk, slab and
        cluster** samples: nothing in the construction assumes a filled cell.

        **The operator still returns the full density.** :math:`\rho_{\rm sup}`
        is added back inside
        :meth:`~poraque.ml.training.FieldOperator.predict`, so every caller —
        inference, the ASE calculator, the CHGCAR writer — sees an absolute
        density and never has to know which mode the model was trained in.

        Two consequences, both handled rather than hidden.
        :math:`\delta\rho` is **signed**, so positivity and the electron count
        are statements about :math:`\delta\rho + \rho_{\rm sup}`; the training
        loop restores the baseline before every physics term, and inference
        does it before clipping or normalising (order fixed in
        ``DESIGN_PAW.md`` §3.3). And the reported relative :math:`L^2` is always
        quoted **on the absolute density**, so a delta-mode run is directly
        comparable with an absolute-mode one.

        Requires ``atomic_reference`` covering every element in the dataset;
        a run that cannot resolve one **fails loudly** rather than quietly
        training the old target. Set ``false`` for the absolute-density
        ablation. Ignored by ``chg2tau``: there is no atomic superposition of a
        kinetic energy density.

        See ``DESIGN_PAW.md`` §3.1 and §3.3.
    atomic_reference : str or None
        Where the isolated atoms come from. Three spellings are accepted,
        because three are what people actually have:

        - an ``atomic_reference.json`` built by ``poraque-atoms``;
        - a **directory holding one subdirectory per element**, each an
          isolated-atom calculation — ``data/vasp/isolated_atoms`` with
          ``Pt/`` inside it, which is the default and the layout to write new
          references in. A single atom directory is also accepted, but only
          the parent form grows to a second element without editing a config.
          Either is ingested on the spot and memoised into the cache as
          ``atomic_reference.json``, so the reduction happens once;
        - a directory that already holds a database, which then wins.

        Each reference must be **one atom in a box** with a ``CHGCAR``. A bulk
        or slab run has more than one atom and is skipped with a message rather
        than averaged into something that is the form factor of neither.

        This one path feeds **both** things the isolated atoms are the
        reference for: the ``delta_density`` superposition baseline, and — via
        ``paw_source`` — the PAW augmentation occupancies written into a
        predicted ``CHGCAR``.

        The library is copied **into the checkpoint**, not referenced from it:
        a delta-density model's weights only mean anything against the
        particular superposition they were fitted to, so a database that
        changed afterwards would silently bias every prediction.
    paw_source : str
        Which one-centre PAW augmentation occupancies a predicted ``CHGCAR``
        carries. ``"atomic"`` (**the default as of 2026-08-26**) takes them
        from the isolated-atom database named by ``atomic_reference``;
        ``"material"`` restores the previous behaviour of averaging the
        *training set's* records per element.

        The occupancies are contractions over converged wavefunctions living
        inside the core radius, so no grid-based model predicts them and they
        have to be borrowed from somewhere. The isolated atom is the defensible
        place to borrow from once slabs and clusters enter the set: it is a
        fixed, transferable, per-element quantity with its own provenance,
        whereas a training-set average is a property of whatever happened to be
        in the training set and is not defined at all for an element that was
        not.

        It is not the more *accurate* choice for an element the training set
        does cover — measured on this project's platinum data a free-atom record
        sat 86.6 % RMS from a bulk site against 9.9 % for the training average,
        because a free atom and an atom in a metal have genuinely different
        on-site occupations. That gap is the honest cost of the transferability,
        and closing it with a learned environment-dependent correction is a
        ``FUTURE.md`` item.

    storage : str
        How the prepared cache stores its fields. ``"files"`` (the default, and
        what every cache built before 2026-08-28 is) writes one
        ``CHGCAR``-format text file per field; ``"hdf5"`` writes one chunked
        ``fields.h5`` per material.

        The numbers are identical — the HDF5 store holds exactly what a
        ``CHGCAR`` holds, in exactly the same convention — so this is a storage
        decision and not a scientific one. What changes is the cost of holding
        and reading them. A ``CHGCAR`` spends about 18 bytes of ASCII on each
        8-byte double and has to be *parsed* on every read; an HDF5 dataset is
        binary, chunked, and read by the library. On the shipped 140³ platinum
        density the text file is 99.9 MB and the same field as HDF5 with
        ``gzip`` is 15.8 MB, and the round trip is exact to a float64 ulp
        rather than to the text format's eleven significant digits.

        Nothing above the reader knows which layout it is looking at: a field
        inside a store is addressed as ``fields.h5::CHGCAR`` and every Poraquê
        reader takes that spelling, so training, inference and the ASE
        calculator are unchanged. Switching the value **rebuilds** the cache
        rather than mixing layouts — it is part of the cache fingerprint.

    compression : str
        HDF5 dataset filter: ``"gzip"``, ``"lzf"``, or ``null`` for none.
        Requires ``storage: hdf5``; setting it on a text cache raises rather
        than being ignored, since a compression flag that quietly does nothing
        reports a saving that never happened.

        Both codecs ship with h5py, so a compressed cache opens on any machine
        that can open an uncompressed one — no filter plugin to install. HDF5's
        byte-shuffle filter is applied with either, which on real densities is
        worth more than several gzip levels and costs nothing:
        ``gzip`` goes from 1.19x to 1.38x on the platinum density with it, and
        writes and reads *faster*.

        Measured on that density (21.95 MB of float64 values):
        ``lzf`` 1.21x at 0.06 s, ``gzip-1`` 1.36x at 0.24 s, ``gzip-4`` 1.38x
        at 0.31 s, ``gzip-9`` 1.39x at 0.39 s. The ratios are larger on the
        downsampled fields a cache actually holds (2.1x at 32³), where
        neighbouring points are more strongly correlated. ``lzf`` is the choice
        when build time matters, ``gzip`` when the archive is written once and
        read for months.

    compression_level : int
        Gzip level, 0-9. Ignored by ``lzf``, which has no levels, rather than
        raising — a run that changes codec should not also have to remember to
        remove the level.
    """

    train_paths: list = None
    root: str = "data/vasp/structures"
    source: str = "auto"
    cache: str = "data/cache"
    pattern: str = "structure"
    code: str = "auto"
    resolution: int = 32
    potcar_dir: str = None
    sigma: float = None
    gaussian_blur: float = None
    blur_method: str = "spectral"
    spin: str = "auto"
    precision: str = "float64"
    xc: str = "auto"
    delta_density: bool = True
    atomic_reference: str = "data/vasp/isolated_atoms"
    paw_source: str = "atomic"
    storage: str = "files"
    compression: str = None
    compression_level: int = 4

    def paths(self):
        """
        The directories to train on, however they were specified.

        ``train_paths`` wins when it is set; ``root`` is the single-path
        fallback that keeps every config written before it existed working.

        Returns
        -------
        list of str
        """
        if not self.train_paths:
            return [self.root]
        if isinstance(self.train_paths, str):
            return [self.train_paths]
        return [str(path) for path in self.train_paths]

    def formats(self):
        """
        The layout of each path: one name, one per path, or ``"auto"``.

        Returns
        -------
        str or list of str
            Handed straight to :func:`~poraque.data.sources.resolve_source`.
        """
        if isinstance(self.source, (list, tuple)):
            return [str(name) for name in self.source]
        return str(self.source)


#: The learnable activations ``model.kan_setup.variant`` chooses between. Each
#: is one learned function *per channel*, applied elementwise; see
#: :mod:`poraque.ml.kan` for what each parameterises that function with.
KAN_VARIANTS = ("bspline", "cheby", "rbf", "rational")

#: Assumed when ``activation: kan`` is given with no ``variant``: the original
#: KAN paper's own parameterisation, so the unqualified word means what the
#: literature means by it — not the cheapest variant, which would be a
#: performance decision made silently on the user's behalf.
DEFAULT_KAN_VARIANT = "bspline"

#: ``kan_setup`` keys, and the :class:`~poraque.ml.fno.FNO3d` keyword each
#: becomes. The ``kan_`` prefix is redundant inside the block and load-bearing
#: outside it, where these sit beside ``width`` and ``modes``.
KAN_SETUP_KEYS = {
    "use_base": "kan_use_base",
    "grid_size": "kan_grid_size",
    "spline_order": "kan_spline_order",
    "grid_range": "kan_grid_range",
    "degree": "kan_degree",
    "rational_num_degree": "kan_rational_num_degree",
    "rational_den_degree": "kan_rational_den_degree",
}


@dataclass
class ModelConfig:
    r"""
    Architecture of the Fourier Neural Operator.

    The three that set the size
    ---------------------------
    ``width``, ``modes`` and ``projection_channels`` are the settings that
    decide how big the operator is and what it can represent. They are easy to
    confuse because all three are "some number of channels", so it is worth
    being precise about what each one counts.

    Written out with every index, and matching
    :meth:`~poraque.ml.fno.FNO3d.forward` rather than the textbook schematic:

    **Lift** — the only place :math:`C_{\rm in}` appears. The ``+3`` are the
    fractional-coordinate channels, present when ``use_coordinates``:

    .. math::

        v^{(0)}_o(\mathbf r) = \sum_{i=1}^{C_{\rm in}+3}
            W^{\rm lift}_{oi}\, x_i(\mathbf r) + b^{\rm lift}_o ,
        \qquad o = 1 \dots C

    **Layer** :math:`\ell = 1 \dots L`:

    .. math::

        \hat v^{(\ell-1)}_i(\mathbf k)
            &= \mathcal{F}_{\mathbf r \to \mathbf k}
               \big[ v^{(\ell-1)}_i \big](\mathbf k) \\[2pt]
        \hat z^{(\ell)}_o(\mathbf k)
            &= \begin{cases}
               \displaystyle\sum_{i=1}^{C}
                 R^{(\ell,\,c(\mathbf k))}_{oi,\,k_1k_2k_3}\,
                 \hat v^{(\ell-1)}_i(\mathbf k)
                 & |k_1| < m_1,\ |k_2| < m_2,\ 0 \le k_3 < m_3 \\[4pt]
               0 & \text{otherwise}
               \end{cases} \\[2pt]
        z^{(\ell)}_o(\mathbf r)
            &= \mathcal{F}^{-1}_{\mathbf k \to \mathbf r}
               \big[ \hat z^{(\ell)}_o \big](\mathbf r) \\[2pt]
        y^{(\ell)}_o(\mathbf r)
            &= \mathrm{GroupNorm}\Big( z^{(\ell)}_o(\mathbf r)
               + \sum_{i=1}^{C} W^{(\ell)}_{oi}\, v^{(\ell-1)}_i(\mathbf r)
               + b^{(\ell)}_o \Big) \\[2pt]
        y^{(\ell)}_o(\mathbf r)
            &\leftarrow \gamma_o(\text{cell})\, y^{(\ell)}_o(\mathbf r)
               + \beta_o(\text{cell})
               \qquad\text{(FiLM, when \tt cell\_conditioning)} \\[2pt]
        v^{(\ell)}_o(\mathbf r)
            &= v^{(\ell-1)}_o(\mathbf r)
               + \sigma\big( y^{(\ell)}_o(\mathbf r) \big)
               \qquad\text{(residual)}

    **Read-out** — the only place :math:`P` appears:

    .. math::

        u_p(\mathbf r) &= \sigma\Big(
            \sum_{o=1}^{C} P^{(1)}_{po}\, v^{(L)}_o(\mathbf r)
            + b^{(1)}_p \Big),
            \qquad p = 1 \dots P \\
        \text{out}_q(\mathbf r) &=
            \sum_{p=1}^{P} P^{(2)}_{qp}\, u_p(\mathbf r) + b^{(2)}_q,
            \qquad q = 1 \dots C_{\rm out}

    The retained band is capped by the grid, not fixed by it:

    .. math::

        m_1 = \min(\texttt{modes}, N_x/2), \quad
        m_2 = \min(\texttt{modes}, N_y/2), \quad
        m_3 = \min(\texttt{modes}, N_z/2 + 1)

    There are **four** corners :math:`c` and not eight because :math:`v` is
    real, so :math:`\hat v(-\mathbf k) = \hat v(\mathbf k)^{*}` and only
    :math:`k_3 \ge 0` is stored: :math:`2\,(\pm k_1) \times 2\,(\pm k_2)
    \times 1`. That 4 is the leading axis of :math:`R`, whose shape is
    ``(4, C, C, m1, m2, m3)``.

    The parameter count is where the three settings separate:

    .. code-block:: text

        spectral   L * 4 * C^2 * m1*m2*m3   complex   <- dominates: C^2 and m^3
        pointwise  L * (C^2 + C)
        lift       C * (C_in + 3) + C
        read-out   C*P + P + P*C_out + C_out          <- linear in P, and once

    ``width`` (:math:`C`)
        How many **channels** carry the field through those layers. The input
        is lifted from 1-2 physical channels to :math:`C` at the start and
        projected back at the end; everything between works at width :math:`C`.

        This is the operator's *representational* capacity — how many distinct
        features of the density it can carry at once — and it is the setting
        that costs the most, because the spectral weights go as :math:`C^2`.
        Doubling it roughly quadruples both parameters and time.

    ``modes`` (:math:`m`)
        How many **Fourier coefficients per axis** the spectral multiplier
        acts on, counted from the lowest. The rest of the spectrum passes
        through the layer untouched by :math:`R` (it is still carried by
        :math:`W`).

        This is the operator's *spatial* reach: mode :math:`m` on a cell of
        length :math:`L` is a wavelength :math:`L/m`, so :math:`m` fixes the
        finest feature the learned convolution can resolve and, equivalently,
        the longest range it can correlate. It is a **capacity limit, not a
        requirement**: on a grid too coarse to supply :math:`m` modes, fewer
        are used automatically and the same weights still apply — which is
        what lets one model serve materials on different grids.

        The parameter count is :math:`4 C^2 m^3` complex numbers per layer, so
        ``modes`` is by far the most expensive knob: 8 → 12 is a 3.4× increase.

    ``projection_channels`` (:math:`P`)
        The hidden width of the **final two-layer head** that maps the
        :math:`C` channels back down to the physical output. It appears once,
        at the end, and never inside a Fourier layer.

        Purely pointwise, so it costs :math:`O(P)` and no spectral weights at
        all. It exists because collapsing :math:`C` channels to one with a
        single linear map is a bottleneck; a wider head is cheap and usually
        worth more than the same budget spent on ``width``.

    A useful way to hold them apart: ``width`` is how much information travels
    *through* the operator, ``modes`` is how far it travels *in space*, and
    ``projection_channels`` is only how it is *read out* at the end.

    Typical ranges are ``width`` 16-64, ``modes`` 8-16, ``projection_channels``
    64-128; ``modes`` above about 16 is rarely worth its cost, since the high
    coefficients of a smooth field carry little of its norm.

    GroupNorm and FiLM: the rest of the layer
    -----------------------------------------
    The two terms in the layer equation that are not sizes. They sit next to
    each other in :meth:`~poraque.ml.fno.FNOBlock.forward`, and the order is
    the whole point::

        y = spectral(x) + pointwise(x)      # mix
        y = norm(y)                         # GroupNorm
        y = film(y, embedding)              # FiLM
        return x + activation(y)            # residual

    **GroupNorm** — keeping the stack numerically sane.

    :class:`torch.nn.GroupNorm` splits the :math:`C` channels into :math:`G`
    groups and, for each *(sample, group)* pair, takes a mean and variance over
    **both the channels of that group and every voxel** — then normalises and
    applies a learned per-channel scale and shift. At ``width: 16`` that is 8
    groups of 2.

    Why this normaliser and not another, here specifically:

    * **BatchNorm would be wrong twice over.** Batches are small
      (``batch_size: 4``, and bucketed by grid shape so often smaller) and
      heterogeneous — different materials, different cells. Statistics over one
      to four crystals are noise. Worse, BatchNorm keeps *running* statistics
      and behaves differently in train and eval, so a single-structure
      :meth:`~poraque.ml.training.FieldOperator.predict` would not reproduce
      what training saw.
    * **LayerNorm is the** :math:`G = 1` **case**, pooling every channel into
      one statistic. :math:`G = 8` keeps the groups partly independent.
    * **It is grid-shape agnostic.** The statistics are taken over whatever
      :math:`N_x N_y N_z` is present, so a :math:`24^3` and a :math:`120^3`
      sample are treated identically. That is a requirement here, not a
      nicety: one model must serve materials on different grids.

    What it is *for*: the spectral branch multiplies Fourier coefficients by
    learned complex weights and adds a pointwise branch, once per layer, in a
    residual stack. Without renormalisation the activation scale drifts with
    depth.

    :func:`~poraque.ml.fno._group_count` exists because ``GroupNorm`` requires
    ``C % G == 0`` while ``width`` is user-set. It takes the largest divisor of
    :math:`C` not exceeding 8 — so ``width: 16`` gives 8 groups but
    ``width: 12`` gives **6**, since ``12 % 8 != 0``. Without it, ``width: 12``
    would fail at construction.

    **FiLM** — how the unit cell reaches the network.

    Feature-wise linear modulation. One ``Linear(embedding_dim -> 2C)``
    produces :math:`\gamma` and :math:`\beta`, a pair per channel, and

    .. math::

        y^{(\ell)}_{b,o}(\mathbf r) \leftarrow
            \big(1 + \gamma_o(\text{cell}_b)\big)\,
            y^{(\ell)}_{b,o}(\mathbf r) + \beta_o(\text{cell}_b)

    The reshape to ``(B, -1, 1, 1, 1)`` broadcasts over every voxel, so this is
    **spatially uniform**: one gain and one offset per channel per sample,
    constant across the grid — which is why it is indifferent to grid shape.

    The embedding comes from :class:`~poraque.ml.fno.CellEncoder`: seven
    rotation-invariant descriptors — the three lattice lengths, the three angle
    cosines, and :math:`V^{1/3}`, lengths divided by a 10 Å scale — through an
    MLP :math:`7 \to E \to E`.

    Why it is needed at all is specific to an FNO. The spectral weights
    :math:`R` act on *mode indices*, not physical wavevectors: mode :math:`k`
    on a cell of edge :math:`L` is :math:`|\mathbf G| = 2\pi k / L`. The same
    :math:`R` therefore means different physics in a 5 Å cell than in a 15 Å
    one — and because the operator is deliberately grid-shape invariant, two
    :math:`32^3` samples pass through the layers identically whatever their
    lattice constant. Without conditioning the network **cannot tell them
    apart**. The fractional-coordinate channels do not help: they are
    dimensionless and identical for both cells. FiLM is what carries physical
    scale in.

    .. note::
       The zero-init is deliberate. Both the projection weight and its bias are
       zeroed, and the form is :math:`(1 + \gamma)`, not :math:`\gamma`. At
       initialisation :math:`\gamma = \beta = 0`, so FiLM is *exactly the
       identity*: the model starts as the unconditioned operator and learns to
       use the cell only where it earns its place. Written as
       :math:`\gamma \cdot x` with the same init it would annihilate the signal
       instead.

    **Why they are adjacent, in that order.** This is the
    conditional-normalisation pattern (cf. conditional BatchNorm, AdaIN):
    GroupNorm strips each sample's own scale and offset, and FiLM puts back a
    scale and offset that are a learned function of the cell. Put FiLM first
    and the normaliser would erase it on the next line.

    .. warning::
       FiLM's :math:`\gamma, \beta` are **not** GroupNorm's affine parameters,
       which are conventionally written with the same letters. In the layer
       equation above, :math:`\gamma_o(\text{cell})` and
       :math:`\beta_o(\text{cell})` are FiLM's; GroupNorm's own affine is
       folded inside :math:`\mathrm{GroupNorm}(\cdot)`.

    Neither costs anything worth counting. Measured on ``width`` 16, ``modes``
    8, ``n_layers`` 3, ``projection_channels`` 64, ``embedding_dim`` 32:

    .. code-block:: text

        spectral R          3,145,728   99.79%
        FiLM projection         3,168    0.10%
        CellEncoder MLP         1,312    0.04%
        GroupNorm affine           96    0.00%
        TOTAL               3,152,353

    Conditioning and normalisation together are 0.15% of the parameters. The
    spectral tensor is the model.

    Attributes
    ----------
    width : int
        Channel width of the Fourier layers. See above: cost grows as the
        square, and it is the operator's representational capacity.
    modes : int
        Retained Fourier modes per axis, counted from the lowest. A *capacity*
        limit: on a grid too coarse to supply them, fewer are used
        automatically. Cost grows as the cube.
    n_layers : int
        Number of Fourier layers — how many times the lift/spectral/project
        cycle is applied. Depth composes the learned convolutions, so it buys
        effective range at linear cost, unlike ``modes``.
    projection_channels : int
        Hidden width of the output projection head. Pointwise and cheap; see
        above.
    activation : str
        ``silu`` (default as of 2026-08-17 — ``silu`` measurably outperformed
        ``gelu`` on this project's own comparisons, see FUTURE.md), ``gelu``,
        ``relu``, ``tanh`` — stateless, parameter-free — or ``kan``: a
        Kolmogorov-Arnold Network-style *learnable* activation, one function
        per channel, applied elementwise, whose variant and hyperparameters
        come from ``kan_setup`` below. Every learnable variant is initialised
        close to ``silu`` (a small learnable perturbation on top of it,
        matching the base term the original KAN paper itself uses), so
        switching to one does not destabilise training at step 0.
    projection_activation : str or None
        Nonlinearity of the read-out head. ``null`` (the default) follows
        ``activation``, which is what "the activation" means unless something
        says otherwise.

        It is separable because it was not always the same: until 2026-08-28
        the read-out was a hard-coded ``GELU`` while ``activation`` governed
        only the Fourier layers, so ``model.activation`` selected :math:`L`
        nonlinearities out of :math:`L+1` and the odd one out appeared in no
        config. A checkpoint that records no read-out nonlinearity is
        therefore one written before that change, and
        :meth:`~poraque.ml.training.FieldOperator.from_state` restores
        ``gelu`` for it rather than the recorded ``activation`` — otherwise
        every silu-trained model would quietly start computing something else.

        A per-channel KAN variant here is sized by ``projection_channels``,
        not by ``width``.
    kan_setup : dict or None
        Everything the KAN activation needs, in one block — **read only when**
        ``activation: kan``. ``null`` (the default) means the defaults below,
        so a KAN run states only what it changes.

        .. code-block:: yaml

            model:
              activation: kan
              kan_setup:
                variant: bspline          # bspline | cheby | rbf | rational
                use_base: true            # keep the w_c * silu(x) base term
                grid_size: 8              # bspline, rbf
                spline_order: 3           # bspline
                grid_range: [-2.0, 2.0]   # bspline, rbf
                degree: 6                 # cheby
                rational_num_degree: 4    # rational
                rational_den_degree: 4    # rational

        It is one block rather than seven flat keys because six of the seven
        are read by *one* variant each: as flat settings they read as
        alternatives to ``width`` and ``modes``, which apply always, when in
        fact almost all of them do nothing in any given run. Grouping them
        also makes "ignored unless ``activation: kan``" a statement about a
        single key rather than about seven.

        ``variant`` selects which learned function each channel carries;
        see :mod:`poraque.ml.kan` for the four. ``use_base`` keeps each
        channel's ``w_c * silu(x)`` base term (``true``, the default,
        matching the original KAN paper) or drops it for a "pure" KAN — only
        the learned residual, no fixed nonlinearity mixed in at all — whose
        output is then bounded or decays to zero for a wide input tail rather
        than tracking it.

        ``grid_size`` is the number of knot-grid intervals (``bspline``) or
        RBF centers minus one (``rbf``, which reuses the same fixed-grid
        design); more gives the learned function finer local structure at
        ``width`` extra coefficients each. ``spline_order`` is the B-spline
        degree (``3`` = cubic, the paper's choice). ``grid_range`` is the
        ``[low, high]`` support of that grid: a ``bspline`` input outside it
        is clamped rather than extrapolated, an ``rbf`` input outside it
        decays towards zero on its own. ``degree`` is the highest Chebyshev
        order for ``cheby`` (``degree + 1`` coefficients per channel).
        ``rational_num_degree`` and ``rational_den_degree`` are the numerator
        power and the number of even denominator powers for ``rational``; the
        denominator is guarded to stay :math:`\geq 1`, so neither can
        introduce a pole.

        Unknown keys raise, and a ``kan_setup`` given without
        ``activation: kan`` warns rather than being silently ignored.
    use_coordinates : bool
        Append three fractional-coordinate channels to the input, so the
        operator knows *where* in the cell it is. Dimensionless, so they carry
        position but not scale — see ``cell_conditioning`` for the latter.
    cell_conditioning : bool
        Condition every layer on the lattice through FiLM. ``false`` drops both
        the modulation and the :class:`~poraque.ml.fno.CellEncoder` entirely,
        which leaves the operator unable to distinguish two samples on the same
        grid whose cells differ in size. See "GroupNorm and FiLM" above.
    embedding_dim : int
        Width :math:`E` of the cell embedding the FiLM projections read. Costs
        :math:`E^2 + 8E` once, plus :math:`2CE + 2C` per layer — a fraction of
        a percent of the model either way.
    mode_selection : str
        ``"fixed"`` truncates at a constant mode index; ``"physical"``
        truncates at a constant wavevector ``g_max``, so every material
        contributes the same band of physics. Prefer the latter when cell sizes
        vary widely.
    g_max : float or None
        Cutoff wavevector in Å⁻¹, required by ``mode_selection: physical``.
    pauli_residual : bool
        For ``chg2tau``, predict ``tau = tau_vW[rho] + s*softplus(f)`` so the
        Hoffmann-Ostenhof bound holds by construction.
    pauli_scale : float or None
        Initial Pauli-term scale in eV/Å³; ``null`` fits it from the training
        split.
    precision : str
        Dtype the operator computes in: ``float32`` (the default) or
        ``float64``.

        ``float64`` roughly doubles time and memory, and is for checking that a
        physical result is not a single-precision artefact — an energy
        difference of a few meV assembled from N³ voxel sums has no obvious
        margin in float32. Conversion goes through
        :func:`~poraque.ml.fno.set_precision`, because an FNO carries complex
        spectral weights and neither ``model.double()`` nor
        ``model.to(torch.float64)`` handles them correctly.

        Distinct from ``data.precision``, which is how the fields are *stored*.
    learn_pauli_scale : bool
        Optimise the scale alongside the backbone.
    """

    width: int = 16
    modes: int = 8
    n_layers: int = 3
    projection_channels: int = 64
    activation: str = "silu"
    projection_activation: str = None
    kan_setup: dict = None
    use_coordinates: bool = True
    cell_conditioning: bool = True
    embedding_dim: int = 32
    mode_selection: str = "fixed"
    g_max: float = None
    pauli_residual: bool = False
    pauli_scale: float = None
    learn_pauli_scale: bool = True
    precision: str = "float32"

    def activation_kwargs(self):
        """
        The activation name and KAN keywords :class:`~poraque.ml.fno.FNO3d`
        wants, resolved from ``activation`` and ``kan_setup``.

        ``activation: kan`` plus ``kan_setup.variant: cheby`` becomes the
        backbone's ``activation="kan_cheby"``, so the checkpoint's
        ``architecture`` record and :func:`~poraque.ml.kan.build_activation`
        keep the one flat name they have always used. The grouping is a
        property of the *config file*, not of the model.

        Returns
        -------
        tuple of (str, dict)
            The activation name, and the ``kan_*`` keywords to pass with it —
            empty for every stateless activation, so a non-KAN run allocates
            and records nothing about KANs.

        Raises
        ------
        ValueError
            On a ``kan_*`` name in ``activation`` (the pre-``kan_setup``
            spelling), an unknown ``variant``, or an unknown ``kan_setup`` key.
            A silently accepted typo here changes the architecture and nothing
            says so.
        """
        name = str(self.activation)
        setup = dict(self.kan_setup or {})

        if name != "kan" and name.startswith("kan"):
            raise ValueError(
                f"model.activation={name!r} is the old flat spelling. The "
                f"variant now lives in its own block:\n\n"
                f"    model:\n      activation: kan\n      kan_setup:\n"
                f"        variant: {name[len('kan_'):]}\n")

        if name != "kan":
            if setup:
                import warnings

                warnings.warn(
                    f"model.kan_setup was given with activation={name!r}, "
                    f"which is not a KAN, so every setting in it is ignored. "
                    f"Set activation: kan, or drop the block.",
                    RuntimeWarning, stacklevel=2,
                )
            return name, {}

        unknown = sorted(set(setup) - {"variant"} - set(KAN_SETUP_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown key(s) in model.kan_setup: {unknown}. "
                f"Valid keys: {['variant'] + sorted(KAN_SETUP_KEYS)}.")

        variant = str(setup.pop("variant", DEFAULT_KAN_VARIANT))
        if variant not in KAN_VARIANTS:
            raise ValueError(
                f"Unknown model.kan_setup.variant {variant!r}; expected one "
                f"of {sorted(KAN_VARIANTS)}.")

        return f"kan_{variant}", {KAN_SETUP_KEYS[key]: value
                                  for key, value in setup.items()}


@dataclass
class TrainingConfig_:
    r"""
    Optimisation settings.

    Attributes
    ----------
    epochs : int
        Passes over the training set.
    batch_size : int
        Maximum samples per batch *within one grid-shape bucket*; materials of
        different shape are never mixed.
    learning_rate, weight_decay : float
        AdamW hyper-parameters.
    scheduler : str or None
        ``"cosine"`` or ``null``.
    grad_clip : float
        Global gradient-norm clip; ``0`` disables.
    valid_fraction : float
        Fraction of structures held out for validation, drawn by shuffling the
        structure list with ``seed``. Defaults to ``0.2``, so an ordinary run
        reports a genuine held-out score rather than a training fit, and
        ``early_stopping`` has something to watch.

        Set it to ``0`` to train on **every** structure. That is the right
        choice for the final deployable artefact — it uses all the data — but
        its metrics are then a training fit and carry no generalisation claim,
        and early stopping cannot act.

        The split is at the **structure level**, like ``k_folds``: whole
        materials move together. At least one structure is always kept on each
        side, so a non-zero fraction on a two-structure dataset gives 1 + 1
        rather than an empty validation set.

        Ignored when ``enable_kfold`` is set, which supplies its own splits.
    eval_epoch : int
        Evaluate and log every this many epochs. Validation is *only* computed
        on those epochs, so raising it on a large validation set is a genuine
        speed-up rather than only a quieter log.
    early_stopping : int
        Stop after this many epochs without an improvement in the **validation**
        error, and restore the best weights seen. ``0`` disables it and always
        runs the full ``epochs``.

        Requires a validation split (``valid_fraction > 0``): with
        nothing held out there is only the training loss, which falls
        monotonically by construction and so can never signal that training
        should stop. The run says so rather than appearing to be protected.

        Counted in epochs, but only *checked* on the epochs where validation is
        computed, so a patience shorter than ``eval_epoch`` behaves like
        ``eval_epoch``.
    enable_kfold : bool
        Run K-fold cross-validation instead of a single fit. This is the only
        variation on the training protocol: every other run trains once on the
        ``valid_fraction`` split.

        It answers a different question — whether the architecture generalises
        — and produces *K* models rather than one to deploy.
    k_folds : int
        Number of folds. Splitting is at the **structure level**: each fold
        holds out whole materials, never a subset of voxels from a material
        that also appears in training. A voxel-level split would let the model
        see the same material in both halves and would report a
        wildly optimistic score that says nothing about transfer to a new
        material.

        Capped at the number of structures; ``k_folds`` equal to that count is
        leave-one-out.
    seed : int
        Seed for the data pipeline: the validation draw, the fold partition and
        the batch order. Also seeds the weight initialisation unless
        ``init_seed`` overrides it.
    init_seed : int or None
        Seed for the **weight initialisation only**, leaving everything else on
        ``seed``. ``null`` means "use ``seed``".

        Separating the two is what makes a *query-by-committee* ensemble
        interpretable: train N models that share ``seed`` and differ only in
        ``init_seed``, and their disagreement isolates the spread of
        optimisation outcomes instead of confounding it with a reshuffled
        dataset.
    device : str
        ``"auto"`` (CUDA, then Apple MPS, then CPU), or an explicit backend.
    loss : str
        ``"relative_l2"`` or ``"sobolev"``.

        The progress table names the norm it is watching, so the two are told
        apart at a glance: ``val rel L2`` for the first, ``val rel H1`` for the
        second, where the ``H1`` number is the objective's own data term on the
        held-out set. It is the number early stopping and the checkpoint are
        decided on, which is why it has to be the one the run is minimising.
        The per-structure table at the end of a run reports relative
        :math:`L^2` whatever the objective was, so two runs remain comparable.
    optimizer : str
        ``adamw`` (default), ``adam`` or ``sgd``.

        The default has always been an Adam-family method; the key exists so
        the choice can be *measured* rather than assumed. ``adam`` and
        ``adamw`` share the same per-parameter adaptive step and differ only in
        how ``weight_decay`` is applied: Adam folds it into the gradient, where
        the adaptive denominator then rescales it, so a parameter with small
        historical gradients is decayed harder than one with large ones. AdamW
        applies it straight to the weight, decoupled, which is what makes
        ``weight_decay`` mean the same thing for every parameter — and matters
        here because an FNO's pointwise and spectral weights differ in scale by
        orders of magnitude.

        At ``weight_decay: 0`` the two are numerically identical, so a
        comparison run at that setting measures nothing.

        ``sgd`` (momentum 0.9) is the non-adaptive control.
    sobolev_weight : float
        Weight of the gradient term when ``loss: sobolev``. It also scales the
        reported ``val rel H1``, which is
        ``rel L2 + sobolev_weight * relL2(grad)`` -- at zero the objective and
        the column are both a plain relative :math:`L^2`, and the log says so
        rather than claiming an :math:`H^1` the run is not measuring.
    physics : dict
        Weights of the physics-informed terms for the **neural operator**. All
        default to zero, so the objective is the supervised baseline until one
        is enabled deliberately.

        .. important::

           Not ``symbolic.physics``. This block constrains the *network* over
           voxels; that one constrains a *candidate algebraic expression* over
           probe points. Two of the names collide — ``positivity_weight`` and
           ``von_weizsacker_weight`` — and mean different things in each,
           which is why they live in separate blocks rather than sharing a
           prefix.

        **The shape of the objective.**
        :class:`~poraque.ml.losses.PhysicsInformedLoss` builds one scalar,

        .. math::

            \mathcal{L} = \mathcal{L}_{\rm data}(\hat y, y)
                        + \sum_i w_i \, \mathcal{L}_i ,

        and two structural decisions matter more than the individual terms.

        *The data term acts on normalised fields; every physics term acts on
        decoded ones.* A density is ``Asinh``- or ``Log``-transformed for the
        fit, and :math:`\int\rho\,d^3r = N` is simply not true of the
        transform's output. Constraints are statements about physics, not
        about whatever normalisation the training happens to use.

        *Every term is dimensionless by construction* — each is divided by its
        own scale. Without that, one weight could not serve a dataset spanning
        a light semiconductor and a transition-metal oxide, whose raw
        penalties differ by orders of magnitude.

        The loss returns a dict rather than a scalar, which is why training
        reports ``data`` / ``physics`` / ``total`` as separate columns: a
        falling total says nothing about which half fell.

        **Which term applies to which model.** They are not all available to
        both operators, and a weight set for the wrong task is silently inert:

        .. code-block:: text

            weight                   ext2chg  chg2tau  needs
            positivity_weight           x        x     nothing
            electron_count_weight       x        -     N, or a reference rho
            euler_lagrange_weight       x        -     v_ext (the input)
            von_weizsacker_weight       -        x     rho   (the input)

        ``positivity_weight`` — *both models.*

        .. math::

            \mathcal{L}_{+} = \frac{\langle\,\mathrm{ReLU}(-f)^2\,\rangle}
                                   {\langle f^{2}\rangle}

        Both :math:`\rho` and :math:`\tau` are non-negative *by definition*. An
        unconstrained head can ring below zero near a nucleus, and a negative
        density is not a small error but a meaningless one that propagates into
        every energy integral. Prefer a ``Log`` output parameterisation where
        you can: it makes positivity **structural rather than penalised**, and
        a constraint that cannot be violated beats one that is merely
        expensive to violate.

        ``electron_count_weight`` — *``ext2chg`` only.*

        .. math::

            \mathcal{L}_{N} = \left\langle \left(
                \frac{\int\hat\rho\,d^3r - N}{N}\right)^{2}\right\rangle

        Particle-number conservation is exact and is the cheapest useful
        constraint there is — one reduction. It also fixes precisely the degree
        of freedom a pointwise loss controls worst: a per-voxel MSE is nearly
        indifferent to a uniform 2 % error in :math:`\rho`, but the
        electrostatic terms are of order :math:`10^{4}` eV, so that 2 % moves a
        total energy by tens of eV — by a different amount for every structure,
        so it does not cancel.

        :math:`N` comes from the pseudopotentials when available (exact),
        otherwise from the integral of the reference density in the batch. That
        fallback is what keeps the term active on an archive that ships
        densities and no valence table — which is every public one.

        ``von_weizsacker_weight`` — *``chg2tau`` only.*

        .. math::

            \mathcal{L}_{\rm vW} =
              \frac{\langle\,\mathrm{ReLU}(\tau_{\rm vW}[\rho] - \hat\tau)^2\,
                    \rangle}{\langle \tau_{\rm vW}^{2}\rangle} ,
              \qquad \tau_{\rm vW} = \frac{|\nabla\rho|^{2}}{8\rho}

        The Hoffmann-Ostenhof bound :math:`\tau \ge \tau_{\rm vW}[\rho]` is a
        **theorem**, not a heuristic, which is why it can be weighted
        aggressively. It is **one-sided**: a prediction above the bound is
        free, one below is penalised quadratically. And :math:`\tau_{\rm vW}`
        is built from the network's own *input*, so it needs no extra labels.

        This is the same physics as the Pauli enhancement factor
        :math:`F = (\tau - \tau_{\rm vW})/\tau_{\rm TF}` the symbolic search
        fits: the bound is exactly the statement :math:`F \ge 0`.

        ``euler_lagrange_weight`` — *``ext2chg`` only, and the deepest of the
        four.* At the ground state the orbital-free variational condition holds
        pointwise,

        .. math::

            \frac{\delta T_s}{\delta\rho}(\mathbf r) + v_{\rm ext}(\mathbf r)
            + v_{H}[\rho](\mathbf r) + v_{xc}[\rho](\mathbf r) = \mu ,

        with a **constant** :math:`\mu`. Subtracting the cell average removes
        :math:`\mu` — unknown and material-dependent — and leaves a residual
        that must vanish for the exact density. Because it involves only
        :math:`v_{\rm ext}` (the input) and :math:`\hat\rho` (the output), it
        is **self-contained and needs no additional labels**: it asks whether
        this density is the *ground state* of this potential, which a
        supervised loss never asks.

        Two caveats. :math:`\delta T_s/\delta\rho` defaults to a
        Thomas-Fermi + :math:`\lambda\,`von-Weizsäcker surrogate with
        :math:`\lambda = 1/9`, the second-order gradient expansion — an
        approximation, so this term is softer than the other three. And
        :math:`v_{xc}` is optional; omitting it weakens the constraint but does
        not bias it, since what is enforced is the *constancy* of the sum.

        Passing a trained ``chg2tau`` operator's :math:`\tau` as ``kinetic=``
        to :func:`~poraque.ml.physics.euler_lagrange_residual` replaces the
        surrogate with the **learned** functional, closing the loop between the
        two models. Available from the library; the training script does not
        wire it up.

        **Using them.** Introduce one at a time against a measured baseline. A
        badly scaled constraint degrades accuracy while looking principled, and
        with all four on at once you cannot tell which one did it. Roughly in
        order of confidence: ``electron_count`` (exact, cheap, fixes the
        worst-controlled degree of freedom), ``von_weizsacker`` (a theorem,
        one-sided, free), ``positivity`` (true, but a ``Log`` head is better),
        ``euler_lagrange`` (the deepest statement, resting on an approximate
        kinetic functional).

        ``0.1`` puts a constraint an order of magnitude below the data term,
        which is the intended balance: the physics guides, the data decides.
    """

    epochs: int = 500
    batch_size: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    valid_fraction: float = 0.2
    enable_kfold: bool = False
    k_folds: int = 5
    eval_epoch: int = 10
    early_stopping: int = 300
    seed: int = 42
    init_seed: int = None
    device: str = "auto"
    loss: str = "relative_l2"
    sobolev_weight: float = 0.1
    physics: dict = field(default_factory=lambda: {
        "electron_count_weight": 0.0,
        "positivity_weight": 0.0,
        "von_weizsacker_weight": 0.0,
        "euler_lagrange_weight": 0.0,
    })


@dataclass
class OutputConfig:
    """
    Where a run's artefacts go, and which of them are produced.

    One directory per model
    -----------------------
    Everything a run writes lives under ``<root>/<name>/``::

        models/pt_w16_m8_l3/
            pt_w16_m8_l3.pfno        the weights
            log/                     the training log, the metrics JSON,
                                     and the resolved config
            plots/                   loss curves, parity, slices, histograms
            report/                  the generated PDF

    A trained model is not one file: it is weights plus the numbers that say
    how good they are, the figures behind those numbers, and the configuration
    that produced them. Scattering those across ``models/``, ``logs/``,
    ``results/plots/`` and ``reports/`` made them four things to keep in step,
    and made "delete this experiment" or "send me that model" a job of
    collecting fragments by filename. Here they arrive and leave together.

    ``root`` is the only path setting; ``name`` (from ``task.name``) is the
    only thing that distinguishes two runs.

    Attributes
    ----------
    root : str or None
        Parent directory for every run folder. ``null`` disables **all**
        output — no weights, no log, no figures, no report — which is what a
        smoke test wants and nothing else does.
    write_log : bool
        Write ``log/<name>.log``, ``log/<name>.json`` and the resolved
        ``log/<name>_config.yaml``. On by default: the JSON is what
        ``poraque-committee --against`` reads, and the archived config is what
        makes the run repeatable after the source config is edited.
    plot_figures : bool
        Render the figures into ``plots/``. Off costs nothing else — the
        metrics are computed either way — and saves the matplotlib time on a
        long sweep.
    write_pdf_report : bool
        Typeset the PDF into ``report/``. Needs a LaTeX toolchain; without one
        the source ``.tex`` is written instead and the run says so.
    checkpoint : bool
        Write the ``.pfno``. Off only makes sense for a run whose purpose is
        the metrics, such as a k-fold estimate.
    log, json : str or None
        Explicit override for the log and metrics paths. ``null`` (the
        default) puts them under the run folder as described above. Set them
        only to write somewhere the layout does not cover.
    plot_format : str
        Image format for saved figures.
    dpi : int
        Raster resolution for saved figures.
    """

    root: str = "models"
    write_log: bool = True
    plot_figures: bool = True
    write_pdf_report: bool = True
    checkpoint: bool = True
    log: str = None
    json: str = None
    plot_format: str = "png"
    dpi: int = 200


@dataclass
class SymbolicConfig:
    r"""
    Symbolic distillation: fit a closed-form expression to what was learned.

    A Fourier Neural Operator is accurate and opaque. Symbolic regression
    searches the space of short algebraic expressions for one that reproduces
    the same mapping, trading a little accuracy for something that can be read,
    published and checked against known physics.

    .. warning::
       The features are **semi-local** — :math:`\rho` and its derivatives at a
       point — so the search can only recover a semi-local functional. The
       non-local part of what the operator learned is outside the hypothesis
       space by construction and will show up as irreducible residual, not as a
       failure of the search. A poor fit is therefore evidence *about the
       physics*, not only about the run.

    Attributes
    ----------
    enable_symbolic_distillation : bool
        Run the search after training. Off by default: it is minutes to hours
        of CPU on top of the fit, and it needs an extra dependency.
    target : str
        ``"model"`` distils the trained operator's own predictions — what the
        network learned, faithful to it including its errors. ``"reference"``
        fits the DFT data directly, which is plain symbolic regression against
        ground truth and answers a different question.
    features : str
        The **input variables**. ``template`` is the independent second knob
        and selects what is fitted *against* them.

        ``"gga"`` gives the density together with its dimensionless reduced
        derivatives — the reduced gradient
        :math:`p = |\nabla\rho|/(2k_F\rho)` and reduced Laplacian
        :math:`q = \nabla^2\rho/(4k_F^2\rho)`, with
        :math:`k_F = (3\pi^2\rho)^{1/3}`. These are the GGA and meta-GGA
        variables a semi-local kinetic functional is written in.

        ``"reduced"`` gives :math:`(p, q)` alone. With ``template: pauli``
        this is the form the literature uses, in which Thomas-Fermi is
        :math:`F = 1` and von Weizsäcker is :math:`F = 0`, and every constant
        to be found is order unity. :math:`\rho` is dropped on purpose:
        :math:`p` and :math:`q` are invariant under the coordinate scaling
        that fixes :math:`T_s`, so a dimensionless :math:`F` cannot depend on
        the density, and offering it only gives the search a way to fit the
        particular densities in the dataset.

        ``"raw"`` gives :math:`(\rho, |\nabla\rho|, \nabla^2\rho)`:
        dimensional, kept for checking the reduced forms against something
        unprocessed.

        ``"enhancement"`` is a kept alias for ``"reduced"`` with
        ``template: pauli``, and **overrides** whatever ``template`` is set
        beside it. Prefer the explicit pair in a new config.
    template : str
        Factorisation of the target before the search sees it.

        ``"pauli"`` fits the **Pauli enhancement factor**

        .. math::

            F = \frac{\tau - \tau_{\rm vW}}{\tau_{\rm TF}},
            \qquad
            \tau = \tau_{\rm vW} + \tau_{\rm TF}\,F,

        which is the well-posed target of orbital-free DFT:
        :math:`\tau_{\rm vW} = |\nabla\rho|^2/8\rho` is known in closed form,
        and :math:`\tau - \tau_{\rm vW} \ge 0` by Hoffmann-Ostenhof. Leaving
        :math:`\tau_{\rm vW}` in the target means fitting a quantity that is
        mostly already known, and near the von Weizsäcker limit it means
        fitting a near-cancellation between two large numbers.

        ``"thomas_fermi"`` fits the plain ratio :math:`F = \tau/\tau_{\rm TF}`.
        ``"none"`` fits the target directly.

        A template *gives away* the part of the physics that is already known,
        so the search works on the part that is not — and every constant it
        must find becomes order unity.
    epsilon : float
        Vacuum threshold in atomic units. Denominators are clamped at it and
        voxels at or below it are dropped, because :math:`p` and :math:`q` in
        vacuum are ratios of two vanishing numbers — noise with a plausible
        magnitude, which corrupts a fit more quietly than a gap would.
    constraints : dict
        Per-operator limits on argument complexity, passed to the engine.
        ``{"^": [-1, 1]}`` by default: the base is unconstrained (``-1``) and
        the **exponent** is held to complexity 1.

        Unconstrained exponents are the main source of nonsense in a power
        operator — a fractional power of a negative quantity leaves the reals,
        and an exponent that is itself a subtree is unreadable and almost never
        physical. Real functionals have simple exponents: ``5/3``, ``4/3``,
        ``2``.
    physics : dict
        Physics constraints on the **symbolic search**, and nothing else.

        .. important::

           This is not ``training.physics``. That block constrains the *neural
           operator* — charge conservation, positivity of a predicted density,
           the Euler-Lagrange residual — and its terms are added to a training
           loss over voxels. This block constrains a *candidate algebraic
           expression* for the Pauli enhancement factor, and its terms are
           added to a symbolic-regression fitness over probe points.

           They are separate objectives on separate objects, evaluated at
           different times by different engines. Two of the names collide
           (``positivity_weight``, ``von_weizsacker_weight``) and mean
           different things in each, which is exactly why they live in
           separate blocks rather than sharing a prefix.

        ``enable``
            Penalise violations **inside** the evolutionary loop rather than
            filtering the front afterwards. On by default.

        Filtering after a run only measures how few candidates were physical;
        by then the populations have already spent their budget converging on
        forms that must be discarded. With this on, the engine's objective
        becomes

        .. math::

            \mathcal{L} = \mathcal{L}_{\rm data}
            + w_{+}\,\frac1n\sum_i \min(F_i, 0)^2
            + \sum_{\ell} w_\ell\,\frac{|F(\mathbf x_\ell) - t_\ell|}{s_\ell},

        with the second term evaluated on the batch and the third on synthetic
        probe points — :math:`(p, q) = (0, 0)` for Thomas-Fermi and
        :math:`(p_\infty, 0)` for von Weizsäcker.

        Which terms are active depends on what the run can express, and the
        log says which:

        - **positivity** always. :math:`\tau \ge 0`, and
          :math:`\tau - \tau_{\rm vW} \ge 0` by Hoffmann-Ostenhof, so a
          negative prediction is unphysical under every template.
        - **both limits** under ``template: pauli`` or
          ``template: thomas_fermi``, provided :math:`p` and :math:`q` are
          among the variables.
        - **neither limit** under ``template: none``, where the fitted
          quantity is :math:`\tau` rather than an enhancement factor, or under
          ``features: raw``, where there is no :math:`p` to take a limit in.
          They are still checked after the search, as they always were.

        .. note::
           The ``loss`` column of the reported front is then this
           **constrained** objective and is not comparable with an
           unconstrained run's. The :math:`R^2` and relative :math:`L^2` are
           computed separately from the expression itself and are unaffected.

        ``positivity_weight``, ``thomas_fermi_weight``, ``von_weizsacker_weight``
            Penalty weights, four orders of magnitude above a converged data
            term by default. These are constraints, not regularisers: the
            intent is that no accuracy gain can buy a violation. Each limit
            carries its weight once, however many probe points express it.

        ``p_infinity``
            The reduced gradient standing in for :math:`p \to \infty` in the
            von Weizsäcker probe. Large enough that no smooth interpolating
            form is still in its crossover region, and small enough that
            :math:`p^{8/3}` stays far inside the range of the 32-bit float the
            engine searches in.
    data_loss : {"mse", "mae"}
        The unpenalised part of that objective — the data term, not a
        constraint, which is why it stays here rather than under ``physics``.
        ``mae`` is the more robust choice on a density whose tails span orders
        of magnitude.
    unary_operations, binary_operations : list of str
        The operator alphabet handed to the engine. Keep it small: the search
        space grows combinatorially, and an operator that cannot appear in the
        answer only dilutes the population. ``/`` and ``log`` are the usual
        sources of singularities on a density that approaches zero.
    iterations : int
        Search iterations. The single biggest quality/time knob.
    population_size, populations : int
        Individuals per population, and how many run in parallel.
    max_depth, max_size : int
        Ceilings on expression depth and node count — the parsimony that keeps
        the result readable rather than a fitted polynomial in disguise.
    parsimony : float
        Penalty per node in the fitness. Higher gives shorter, worse-fitting
        expressions.
    n_samples : int
        Voxels sampled across the dataset. A single 32³ structure is already
        32 768 points and the search cost is linear in them, so the default
        trades a negligible amount of statistics for a tractable run.
    seed : int
        Seeds the sampling always, and the search only under
        ``deterministic``.
    deterministic : bool
        Make the search reproducible. The engine requires **serial**
        evaluation for this, so it costs the parallelism across populations —
        roughly a factor of ``populations`` in wall-clock.

        Off by default because a single search is one sample of a stochastic
        process either way, and reading one expression as *the* answer is the
        mistake this setting can otherwise encourage. Turn it on to reproduce
        a specific result, not to make the result more trustworthy.

        With it off, ``seed`` is deliberately **not** passed to the engine: a
        seed that cannot deliver reproducibility is a promise the run does not
        keep, and the engine warns about exactly that.
    """

    enable_symbolic_distillation: bool = False
    target: str = "model"
    features: str = "gga"
    template: str = "none"
    epsilon: float = 1e-8
    constraints: dict = field(default_factory=lambda: {"^": [-1, 1]})
    data_loss: str = "mse"
    physics: dict = field(default_factory=lambda: {
        "enable": True,
        "positivity_weight": 1.0e2,
        "thomas_fermi_weight": 1.0e2,
        "von_weizsacker_weight": 1.0e2,
        "p_infinity": 1.0e6,
    })
    unary_operations: list = field(
        default_factory=lambda: ["exp", "log", "sqrt", "abs"])
    binary_operations: list = field(
        default_factory=lambda: ["+", "-", "*", "/", "^"])
    iterations: int = 40
    population_size: int = 33
    populations: int = 15
    max_depth: int = 10
    max_size: int = 30
    parsimony: float = 0.0032
    n_samples: int = 4000
    seed: int = 0
    deterministic: bool = False


@dataclass
class FineTuningConfig:
    r"""
    Adapt a trained operator to a narrower class of materials.

    A model fitted across a broad dataset is a good *starting point* for a
    specific family, and a poor substitute for one. Fine-tuning continues that
    fit on the smaller set at a much lower learning rate, keeping what the base
    model learned about the general map while letting it specialise.

    .. important::
       The **normalizations come from the checkpoint**, not from the new
       dataset. Refitting them would rescale the network's inputs out from
       under weights that were trained against the old scale, which destroys
       most of what the pre-training bought. It also means the new data must
       fall in a comparable range: fine-tuning on a family whose densities are
       an order of magnitude away is transfer learning in name only.

    Attributes
    ----------
    enable : bool
        Start from ``pretrained_checkpoint`` instead of a fresh
        initialisation.
    pretrained_checkpoint : str
        Bundle to start from. The **architecture is read from its tensors**,
        so the ``model`` section of this config is ignored for the shape of
        the network — a remembered hyper-parameter that disagreed with the
        weights could only load mismatched tensors.
    learning_rate : float
        Optimiser step for the fine-tune, replacing ``training.learning_rate``.
        Much smaller by default: the base learning rate would walk the weights
        away from the solution being adapted before the small dataset could
        constrain them.
    freeze_lifting_layers : bool
        Hold the input lifting layer fixed and train the rest. The lifting map
        is the most general part of the network — it embeds the input field
        before any operator acts — so it is the part least in need of
        specialisation, and freezing it leaves fewer parameters for a small
        dataset to overfit.

        The projection head is deliberately *not* frozen: it decodes to
        physical units, which is exactly what differs between material
        families.
    """

    enable: bool = False
    pretrained_checkpoint: str = "models/poraque_models.pfno"
    learning_rate: float = 1e-5
    freeze_lifting_layers: bool = False


@dataclass
class TrainingConfig:
    """
    Complete definition of a training run.

    Parameters
    ----------
    task : TaskConfig
        What to train, and the name every output file is built from.
    data, model, training, output, symbolic : dataclass
        The five setting groups.

    Examples
    --------
    >>> config = TrainingConfig.from_yaml("configs/train.yaml")  # doctest: +SKIP
    >>> config.model.width                                              # doctest: +SKIP
    16
    """

    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig_ = field(default_factory=TrainingConfig_)
    output: OutputConfig = field(default_factory=OutputConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    # Order matters twice: it is the order the run header prints, and the order
    # a bare (undotted) override is resolved against.
    _SECTIONS = {"task": TaskConfig, "data": DataConfig, "model": ModelConfig,
                 "training": TrainingConfig_, "output": OutputConfig,
                 "symbolic": SymbolicConfig, "fine_tuning": FineTuningConfig}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _task_section(value):
        """
        Read ``task`` in either spelling.

        It began as a bare string and became a section when the run gained a
        ``name``. Both are accepted permanently rather than transitionally:
        ``task: ext2chg`` is the shorter and clearer form when the defaults for
        everything else will do, and every config written before the section
        existed keeps working unchanged.
        """
        if value is None:
            return {}
        if isinstance(value, str):
            return {"type": value}
        return dict(value)

    @classmethod
    def from_dict(cls, mapping):
        """
        Build a config from a nested mapping, validating every key.

        Parameters
        ----------
        mapping : dict
            Parsed YAML contents.

        Returns
        -------
        TrainingConfig

        Raises
        ------
        ValueError
            On an unknown top-level key or an unknown key inside a section.
            Silent acceptance would let a typo change nothing while appearing
            to change something.
        """
        mapping = dict(mapping or {})
        known = set(cls._SECTIONS)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"Unknown configuration section(s): {unknown}. "
                f"Expected any of {sorted(known)}."
            )
        mapping["task"] = cls._task_section(mapping.get("task"))

        sections = {}
        for name, section_class in cls._SECTIONS.items():
            values = dict(mapping.get(name) or {})
            valid = {f.name for f in fields(section_class)}
            bad = sorted(set(values) - valid)
            if bad:
                raise ValueError(
                    f"Unknown key(s) in section '{name}': {bad}. "
                    f"Valid keys: {sorted(valid)}."
                )
            sections[name] = section_class(**values)

        return cls(**sections)

    @classmethod
    def from_yaml(cls, path):
        """
        Read a configuration from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        TrainingConfig
        """
        _require_yaml()
        with open(path, "r") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self):
        """Nested plain-``dict`` representation."""
        return asdict(self)

    def to_yaml(self, path=None, minimal=False):
        """
        Serialise to YAML.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            When given, write to this file. The resolved config is worth
            saving beside the results: it records what actually ran, including
            command-line overrides.
        minimal : bool, optional
            Write only the settings that differ from the defaults. Every key
            is optional, so the result is an equivalent config — and a far
            shorter one, since most of a typical file restates a default.

            Not the right choice for the copy archived beside a run: that one
            should record every value that was in force, so it still
            reproduces the run if a default later changes.

        Returns
        -------
        str
            The YAML text.
        """
        _require_yaml()
        payload = self.non_default_dict() if minimal else self.to_dict()
        text = yaml.safe_dump(payload, sort_keys=False,
                              default_flow_style=False)
        if path is not None:
            with open(path, "w") as handle:
                handle.write(text)
        return text

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #
    def apply_overrides(self, overrides):
        """
        Apply command-line overrides in place.

        Only non-``None`` values are applied, which is how "flag was not given"
        is distinguished from "flag was given a falsy value" — important for
        booleans and zeros.

        Parameters
        ----------
        overrides : dict
            ``{"section.key": value}`` or ``{"key": value}`` for the top level.
            Bare keys are resolved against the sections, first match wins.

        Returns
        -------
        TrainingConfig
            ``self``, for chaining.
        """
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if "." in key:
                section_name, attribute = key.split(".", 1)
                section = getattr(self, section_name)
                if not hasattr(section, attribute):
                    raise ValueError(f"Unknown override {key!r}.")
                setattr(section, attribute, value)
            elif key == "task":
                # `--task ext2chg`, and any caller still passing the bare
                # string the key used to hold.
                self.task.type = value
            else:
                for section_name in self._SECTIONS:
                    section = getattr(self, section_name)
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break
                else:
                    raise ValueError(f"Unknown override {key!r}.")
        return self

    # ------------------------------------------------------------------ #
    # Consumers
    # ------------------------------------------------------------------ #
    def run_dir(self):
        """
        The run's own directory, ``<output.root>/<task.name>``.

        Everything else hangs off this. ``None`` when ``output.root`` is
        ``null``, which switches off all output.

        Returns
        -------
        str or None
        """
        if not self.output.root:
            return None
        import os

        return os.path.join(self.output.root, self.run_name())

    def run_name(self):
        """The run's name, falling back to the historical default."""
        return str(self.task.name or "poraque_models")

    def _subdir(self, name, wanted):
        """One subdirectory of the run folder, or ``None`` when switched off."""
        root = self.run_dir()
        if root is None or not wanted:
            return None
        import os

        return os.path.join(root, name)

    def log_dir(self):
        """``<run>/log``, or ``None`` when ``write_log`` is off."""
        return self._subdir("log", self.output.write_log)

    def plot_dir(self):
        """``<run>/plots``, or ``None`` when ``plot_figures`` is off."""
        return self._subdir("plots", self.output.plot_figures)

    def report_dir(self):
        """``<run>/report``, or ``None`` when ``write_pdf_report`` is off."""
        return self._subdir("report", self.output.write_pdf_report)

    def checkpoint_path(self):
        """
        ``<run>/<name>.pfno``, or ``None`` when checkpointing is off.

        A fine-tune gets its own stem. It is a specialisation, usually onto a
        narrower set of materials, and writing it over the general model would
        replace something broad with something narrow, silently and by default.

        Returns
        -------
        str or None
        """
        root = self.run_dir()
        if root is None or not self.output.checkpoint:
            return None
        import os

        stem = self.run_name()
        if self.fine_tuning.enable:
            stem += "_finetuned"
        return os.path.join(root, f"{stem}.pfno")

    def log_path(self):
        """
        The plain-text log: ``output.log``, or ``<run>/log/<name>.log``.

        Returns
        -------
        str or None
        """
        return self._log_artefact(self.output.log, ".log")

    def json_path(self):
        """
        The metrics: ``output.json``, or ``<run>/log/<name>.json``.

        Returns
        -------
        str or None
        """
        return self._log_artefact(self.output.json, ".json")

    def _log_artefact(self, explicit, suffix):
        """
        An explicit path, or one inside the run's ``log/`` directory.

        Deriving is what lets a config name a run once. Every artefact comes
        from ``task.name``; the log and the metrics were the two a user had to
        repeat, and the only two that two different runs could end up sharing
        while writing separate weights.
        """
        if explicit is not None:
            return explicit or None
        directory = self.log_dir()
        if directory is None:
            return None
        import os

        return os.path.join(directory, f"{self.run_name()}{suffix}")

    def non_default_dict(self):
        """
        Only the settings that differ from the defaults.

        A config is far shorter than it looks: of the 78 keys a full one
        carries, ``configs/train.yaml`` changes three. This is what makes a
        committed config readable as *the description of one experiment*
        rather than as a dump of every knob.

        Returns
        -------
        dict
            Nested ``{section: {key: value}}``, sections with nothing to say
            omitted entirely.
        """
        reference = type(self)().to_dict()
        current = self.to_dict()
        trimmed = {}
        for section, values in current.items():
            if not isinstance(values, dict):
                if values != reference.get(section):
                    trimmed[section] = values
                continue
            changed = {key: value for key, value in values.items()
                       if reference.get(section, {}).get(key, object()) != value}
            if changed:
                trimmed[section] = changed
        return trimmed

    def model_kwargs(self):
        """
        Keyword arguments for :class:`~poraque.ml.fno.FNO3d`.

        The Pauli-head settings are excluded: they configure
        :class:`~poraque.ml.training.FieldOperator`, not the backbone. So is
        ``precision``, which is applied to the built model by
        :func:`~poraque.ml.fno.set_precision` rather than passed to its
        constructor — the conversion has to reach the complex spectral weights,
        and a constructor argument would only cover the ones it allocates
        itself.
        """
        excluded = {"pauli_residual", "pauli_scale", "learn_pauli_scale",
                    "precision", "activation", "kan_setup"}
        kwargs = {f.name: getattr(self.model, f.name)
                  for f in fields(self.model) if f.name not in excluded}
        # `activation` and `kan_setup` are one setting in the file and two in
        # the constructor: the block is expanded into `kan_*` keywords here,
        # and only when a KAN was actually asked for.
        name, kan = self.model.activation_kwargs()
        kwargs["activation"] = name
        kwargs.update(kan)
        return kwargs

    #: How a value is written back out in :meth:`describe`. The header echoes a
    #: YAML file, so it uses YAML's spellings: a reader comparing the two
    #: should not have to translate ``None`` into ``null`` in their head.
    _LITERALS = {None: "null", True: "true", False: "false"}

    @classmethod
    def _describe_value(cls, value):
        """One setting, spelled as the YAML file spells it."""
        if isinstance(value, bool) or value is None:
            return cls._LITERALS[value]
        if isinstance(value, (list, tuple)):
            return "[]" if not value else ", ".join(str(item) for item in value)
        return str(value)

    @classmethod
    def _describe_section(cls, mapping, indent):
        """
        One ``key = value`` per line, with the ``=`` signs aligned.

        Recursive, because ``training.physics`` is itself a mapping and a
        nested dict rendered as a repr is exactly the unreadable run this
        avoids.
        """
        pad = " " * indent
        scalars = [key for key, value in mapping.items()
                   if not (isinstance(value, dict) and value)]
        width = max([len(str(key)) for key in scalars] or [0])

        lines = []
        for key, value in mapping.items():
            if isinstance(value, dict) and value:
                lines.append(f"{pad}{key}:")
                lines.extend(cls._describe_section(value, indent + 2))
            else:
                lines.append(f"{pad}{str(key):<{width}s} = "
                             f"{cls._describe_value(value)}")
        return lines

    def describe(self):
        """
        Multi-line human-readable summary, for the log header.

        **One setting per line**, indented under its section, with the ``=``
        signs aligned within each. The run header is where a reader checks
        what is switched on before committing hours of GPU to it, and a
        comma-separated section wrapped across the terminal at whatever column
        it happened to reach: the settings that mattered were wherever the
        wrapping put them, and ``training.physics`` was 116 characters of
        ``{'electron_count_weight': 0.0, ...}`` riding on the end of one.

        Values are written in YAML's spelling — ``null``, ``true``, ``false``
        — so the header and the file it came from read the same.

        Returns
        -------
        str

        Examples
        --------
        >>> print(TrainingConfig().describe())            # doctest: +SKIP
        task:
          type = all
          name = poraque_models
        data:
          train_paths = null
          root        = data/vasp
          ...
        """
        lines = []
        for name in self._SECTIONS:
            section = getattr(self, name)
            lines.append(f"{name}:")
            lines.extend(self._describe_section(
                {entry.name: getattr(section, entry.name)
                 for entry in fields(section)}, indent=2))
        return "\n".join(lines)
