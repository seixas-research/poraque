Poraquê
=======

**Poraquê** learns maps between the three-dimensional scalar fields of
density-functional theory. For a given material the local external potential,
the valence charge density and the kinetic energy density all live on **one
shared grid** in **one file format**, which makes them directly comparable,
composable, and usable as aligned inputs and targets for a neural operator.

Two mappings are learned:

.. math::

   V_{\mathrm{ext}} \;\longmapsto\; \rho
   \qquad\text{and}\qquad
   \rho \;\longmapsto\; \tau

They are not unrelated regressions. The first is the **Hohenberg--Kohn map**,
whose existence and uniqueness is a theorem. The second is the **kinetic energy
density functional**, the missing ingredient of orbital-free DFT. Composed,
they constitute a complete orbital-free calculation from geometry alone --- no
wavefunctions, no self-consistency cycle.

What Poraquê provides
---------------------

* :doc:`fields/index` --- a shared-grid data model for ``EXTCAR``, ``CHGCAR``
  and ``TAUCAR``, with an analytic reconstruction of the local
  pseudopotential that reproduces a reference calculation to a relative
  :math:`2\times10^{-5}`.
* :doc:`ml/index` --- Fourier neural operators that handle **different grid
  shapes across materials**, with physical constraints enforced by
  construction rather than by penalty.
* :doc:`energy/index` --- Kohn--Sham total-energy components integrated from
  the predicted fields, and an **ASE calculator** that runs the whole chain
  from an :class:`ase.Atoms` object.
* :doc:`analysis/index` --- charge-conservation checks and per-atom partial
  charges by Voronoi, Hirshfeld or Bader partitioning of the predicted
  density.
* A **code-agnostic ingestion layer**: VASP is implemented, Quantum ESPRESSO
  and GPAW are scaffolded behind the same four-method contract.
* Hardware acceleration on CUDA and Apple Metal, with a graceful CPU fallback.
* A YAML-driven training pipeline that emits metrics, figures and a typeset
  PDF report.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quick_start/index
   fields/index
   ml/index
   fine_tuning/index
   energy/index
   analysis/index
   symbolic/index
   theory
   configuration
   api/index
