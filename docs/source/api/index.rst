API Reference
=============

Scalar fields
-------------

.. Members are documented where they are defined, not where they are
   re-exported: naming FieldGrid here as well as under "Grid and resampling"
   creates two targets for every :class:`FieldGrid` reference in the codebase,
   and Sphinx then cannot resolve any of them.
.. automodule:: poraque.fields
   :members: ScalarField, Structure

.. autoclass:: poraque.fields.ExternalPotential
   :members:
   :inherited-members:

.. autoclass:: poraque.fields.ChargeDensity
   :members:

.. autoclass:: poraque.fields.KineticEnergyDensity
   :members:

Grid and resampling
-------------------

.. automodule:: poraque.fields.grid
   :members:

.. automodule:: poraque.fields.resample
   :members:

Ingestion
---------

.. automodule:: poraque.fields.io
   :members:

.. automodule:: poraque.fields.io.vasp
   :members:

.. automodule:: poraque.fields.io.aims
   :members:

.. automodule:: poraque.fields.io.espresso
   :members:

.. automodule:: poraque.fields.io.gpaw
   :members:

.. automodule:: poraque.fields.io.compressed
   :members:

Storage
-------

.. automodule:: poraque.fields.hdf5
   :members:

Data sources
------------

.. The package docstring is the overview; the classes it re-exports are
   documented under their own modules below, for the same reason as above.
.. automodule:: poraque.data

.. automodule:: poraque.data.sources
   :members:

.. automodule:: poraque.data.dataset
   :members:

.. automodule:: poraque.data.cache
   :members:

Materials Project
-----------------

.. automodule:: poraque.data.materials_project
   :members:

.. automodule:: poraque.data.mp_dataset
   :members:

VASP file formats
-----------------

.. automodule:: poraque.fields.vasp.poscar
   :members:

.. automodule:: poraque.fields.vasp.incar
   :members:

.. automodule:: poraque.fields.vasp.potcar
   :members:

.. automodule:: poraque.fields.vasp.volumetric
   :members:

Neural operators
----------------

.. automodule:: poraque.ml.fno
   :members:

.. automodule:: poraque.ml.heads
   :members:

.. automodule:: poraque.ml.tasks
   :members:

Data pipeline
-------------

.. automodule:: poraque.ml.data
   :members:

.. automodule:: poraque.ml.transforms
   :members:

Training
--------

.. automodule:: poraque.ml.training
   :members:

.. automodule:: poraque.ml.config
   :members:

.. automodule:: poraque.ml.device
   :members:

.. automodule:: poraque.ml.backend
   :members:

Physics operators and losses
----------------------------

.. automodule:: poraque.ml.physics
   :members:

.. automodule:: poraque.ml.losses
   :members:

Symbolic distillation
---------------------

.. automodule:: poraque.ml.symbolic
   :members:

Query by committee and active learning
--------------------------------------

.. automodule:: poraque.ml.committee
   :members:

.. automodule:: poraque.ml.active_learning
   :members:

Energy functionals
------------------

.. automodule:: poraque.physics.energy
   :members:

ASE calculator
--------------

.. automodule:: poraque.calculator
   :members:

Visualization
-------------

.. automodule:: poraque.vis.report
   :members:

.. automodule:: poraque.vis.pdf_report
   :members:

.. automodule:: poraque.vis.style
   :members:
