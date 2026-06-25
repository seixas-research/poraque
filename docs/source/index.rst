Poraquê
=======

**Poraquê** is a modular electronic-structure package built around a single,
reusable real-space numerical core. It implements three families of density
functional theory and exposes all of them through a standard Atomic Simulation
Environment (ASE) calculator:

* **Kohn-Sham DFT (KS-DFT)** — a real-space, plane-wave-kinetic SCF reference
  implementation.
* **Orbital-Free DFT (OF-DFT)** — direct energy minimization with explicit
  kinetic energy density functionals (Thomas-Fermi, von Weizsäcker, TFvW) and a
  framework for machine-learned KEDFs.
* **Frozen-Density Embedding (subsystem DFT)** — partitioned systems mixing
  distinct OF-DFT and KS-DFT regions via freeze-and-thaw cycles.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quick_start/index
   theory
   api/index
