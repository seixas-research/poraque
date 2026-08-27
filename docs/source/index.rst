.. raw:: html

   <div style="text-align: center; margin-bottom: 24px;">

     <!-- BADGES
          The same set, the same order and the same for-the-badge style as
          README.md's own row. Two files, one row, kept in step by hand --
          which is exactly the pair that drifts, so
          tests/test_badges.py asserts they have not.

          No logo here: furo already puts one in the sidebar
          (conf.py's light_logo/dark_logo), and a second copy in the body
          would be the same image twice on the same screen.

          Skipped deliberately, rather than invented: a CI badge (this
          repository has no .github/workflows and no CI service) and a
          coverage badge (no coverage service is configured). -->
     <p>
       <a href="https://pypi.org/project/poraque/">
         <img src="https://img.shields.io/pypi/v/poraque?style=for-the-badge&amp;logo=pypi&amp;logoColor=white" alt="pypi">
       </a>
       <a href="https://pypi.org/project/poraque/">
         <img src="https://img.shields.io/pypi/pyversions/poraque?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="python">
       </a>
       <a href="https://pytorch.org/">
         <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white" alt="pytorch">
       </a>
       <a href="https://wiki.fysik.dtu.dk/ase/">
         <img src="https://img.shields.io/badge/ASE-3.2x-4B8BBE?style=for-the-badge" alt="ase">
       </a>
       <a href="https://poraque.readthedocs.io/">
         <img src="https://img.shields.io/readthedocs/poraque?style=for-the-badge&amp;logo=readthedocs&amp;logoColor=white&amp;label=Manual" alt="on-line manual">
       </a>
       <a href="https://github.com/seixas-research/poraque/blob/main/LICENSE">
         <img src="https://img.shields.io/github/license/seixas-research/poraque?color=green&amp;style=for-the-badge" alt="license: mit">
       </a>
       <a href="https://github.com/seixas-research/poraque">
         <img src="https://img.shields.io/badge/GitHub-poraque-181717?style=for-the-badge&amp;logo=github" alt="github">
       </a>
     </p>

   </div>

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
* :doc:`data/index` --- one dataset over a **mixture of data layouts**: local
  DFT runs, prepared caches, and bulk archives of standalone ``CHGCAR`` files.
  Each model trains independently, so the vast public density archives --- which
  publish no kinetic energy density --- are usable for the Hohenberg--Kohn map.
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
   data/index
   ml/index
   fine_tuning/index
   energy/index
   analysis/index
   symbolic/index
   theory
   configuration
   api/index
