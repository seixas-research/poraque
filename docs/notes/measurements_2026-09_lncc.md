# Measurement record: equivariance and mode selection (LNCC, 2026-09)

Moved here from the maintainer's working notes on 2026-10-13, because it is
the only record of these runs and the Manual and `CLAUDE.md` cite its
numbers. Configuration keys are named as they were then (`g_max`,
`mode_selection`, `num_workers`); today they are an operator's `model.band` and
`runtime.loader_workers`.

Six-metal MP set (Ag/Au/Cu/Ni/Pd/Pt, 115 densities ≤ 32³), V100, 400 epochs,
two seeds where stated.

- Equivariant radial layer at matched cost: **0.047** held-out rel L² against
  **0.173** for a dense model with 28× the parameters; the four-times-larger
  equivariant arm was worse (symmetry, not capacity). One seed; neither
  equivariant arm had converged.
- `modes` 4 → 16 under equivariance at identical parameter count (84 193):
  **2.7× better for 36 % more time**, monotone over five points. `n_radial`
  8 < 16 < 32 < 64 in error on 92 materials — less is better on a small set.
- `g_basis` narrower than the retained band: no measurable cost at 71 % of
  modes clamped. Wider: **14–43 %** worse (48 against a median band of 31).
  Hence `g_basis: auto` = the median band edge over the training split.
- `mode_selection: physical` vs `fixed`: **20.5 %** better in both seeds at
  32³/`modes 8`/`g_max 16`, **20.3 %** at 64³/`modes 24`/`g_max 24`; the margin
  collapses to 3 % at `g_max 32` and `g_max 6` starves the band (−55 %).
- Seed spread across 90 runs × 2 seeds: 1–53 %, median ~12 %; four single-seed
  readings overturned by the pair.
- `torch.compile` on the whole model: +7.9 % (`default`) and +9.4 %
  (`max-autotune`) per epoch, 200–314 s cold compile, validation error moved;
  on four DDP ranks it sat 18 minutes in the first epoch and was canceled.
- `FieldPairDataset(cache=True)`: 168 s → 16.3 s over 20 epochs (10.3×);
  `num_workers=4` without the cache 1.4×; both together 8.9× (workers each
  re-parse into their own cache). GPU utilization 3–4 % → 28 %; 44 % of GPU
  time elementwise, 6 % GEMM.
