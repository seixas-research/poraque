# Charge-conservation training check

The `ext2paw` objective now includes `electron_count_weight` on the decoded
absolute total-charge density. The target count is the integral of the DFT
reference, with the atomic baseline restored on both sides in residual mode.
Magnetization and PAW occupancy records are not part of this integral.

Reproduction, from the repository root:

```bash
python experiments/chon_ext2paw/benchmark.py \
  --config configs/train_ext2paw_chon_charge.yaml \
  --output experiments/chon_ext2paw/results/charge_constraint_smoke \
  --arms baseline --seeds 1 --epochs 5 --device cpu --threads 2
```

The arm named `baseline` means no overrides to the supplied charge-constrained
configuration. It uses the same 303/88/41 split and 60,118-parameter architecture
as the earlier CHON benchmark. Only training and validation were evaluated.
The held-out test set was not evaluated, and the earlier deployment checkpoint
was not replaced. Results and the new training checkpoint are in
`baseline_s1_e5/metrics.json` and `baseline_s1_e5/model.poraque`.

With charge-penalty weight 1, five CPU epochs took 31.8 seconds. The unweighted
training count penalty decreased from 0.292028 to 0.001297. At epoch 5 the
validation mean absolute relative electron-count error was 0.017192 (1.7192%),
before any inference normalization. The mean squared relative count error was
0.001880. The checkpoint-selection score includes this squared term alongside
the field error and the weighted balanced PAW loss.

This is a soft penalty and a short integration check, not a weight-selection
benchmark or evidence of improvement over a matched unconstrained run. The
initial weight 1 remains to be assessed in a full training comparison.

Validation in the `poraque` conda environment: 328 tests passed, 10 skipped in
`test_physics_informed.py`, `test_ml.py`, `test_paw_operator.py`, `test_ext2paw.py`,
`test_pipeline_defaults.py`, and `test_config_schema.py`. Tests cover charge
gradients on CPU/MPS, independence from magnetization, the explicit-count
override, restoration of the atomic baseline in the real training loop, and
selection of checkpoints using the charge-inclusive validation score.
