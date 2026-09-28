# MiniCPMGR00TRes: first LIBERO residual ablation

Date: 2026-09-28
Purpose: initial labeled-data check of terminal DINO residual supervision. These
results are a small engineering pilot; they do not establish a generally better
representation or task policy.

## Protocol

- Dataset: read-only `libero_goal_no_noops_1.0.0_lerobot` with the explicit
  successful-terminal manifest at
  `annotations/libero_goal_success_terminals.jsonl`.
- Training episode IDs: `0,1,3,6,7`. Evaluation IDs: `2,4,10,21,422`. One sample
  per episode was used, at control step 8. The full-history input contained
  nine frames. Thus each split contains only five evaluation points and does
  not cover complete trajectories.
- Target: frozen DINOv2 ViT-S/14 patch features at 224×224. The evaluation
  target is `z_goal - z_current`; baseline predicts zero residual. For the
  absolute-goal variant, evaluation subtracts `z_current` from the predicted
  goal before scoring, so all rows measure residual error. The evaluator's
  implementation is in `eval_files/eval_minicpm_gr00t_res.py`.
- Text supervision was disabled for every run because this dataset setup has no
  assistant-answer labels. These are residual-only experiments.
- Each 36-update variant consisted of a 12-update run and a 24-update
  continuation with a freshly initialized optimizer. The residual/full-history
  72-update variant added another 36-update continuation, also with a fresh
  optimizer. Update counts are cumulative; the optimizer schedule was not
  continuous across continuations.
- The five evaluation episodes were inspected after the 36-update comparisons
  and informed the decision to continue the residual model to 72 updates. They
  are therefore validation data, not an untouched final test set. No
  independent test split is available in this pilot.
- Checkpoints and raw evaluation JSON are in
  `/data/tzq/tmp/reswam_ablation_20260928` and are intentionally outside the
  repository. The 36-update residual checkpoint is `residual_full_36/final`;
  absolute-goal is `absolute_full_36_retry/step_00000024`; current-only is
  `residual_current_only_36/step_00000024`; 72-update residual is
  `residual_full_72/step_00000036`.

## Results

Metric: mean squared error over all DINO patch and feature dimensions. Lower is
better. “Zero baseline” is the same target residual with no learned prediction.

| Variant | Updates | Context | Train MSE (zero) | Evaluation MSE (zero) | All 10 MSE (zero) |
|---|---:|---|---:|---:|---:|
| Zero-residual baseline | — | — | 2.4172 | 2.0236 | 2.2204 |
| Residual target | 36 | Full history | 2.3780 (2.4172) | 2.0255 (2.0236) | 2.2017 (2.2204) |
| Absolute-goal target, scored as residual | 36 | Full history | 4.5786 (2.4172) | 4.7283 (2.0236) | 4.6535 (2.2204) |
| Residual target | 36 | Current frame only | 2.3770 (2.4172) | 2.0260 (2.0236) | 2.2015 (2.2204) |
| Residual target | 72 | Full history | 2.3337 (2.4172) | 2.0026 (2.0236) | 2.1682 (2.2204) |

At 36 updates, residual/full-history prediction is essentially tied with the
zero baseline on the five evaluation episodes (0.09% worse); it beats zero on
two of five episodes. Current-only is also tied (0.12% worse), so this run does
not show a benefit from the nine-frame history. The absolute-goal target is
worse than zero on all five evaluation episodes at the same update budget.
After 72 total updates, full-history residual prediction is 1.04% below zero
residual on these same episodes and has a lower error on all five, although one
case is nearly tied. Because these episodes informed the continuation choice,
this is a validation result rather than independent evidence of generalization.
It is too small to support a reliable claim: five episodes, one time point per
episode, one training split, no repeated seeds, and no untouched test set.

Per-episode evaluation MSE makes the paired comparison explicit:

| Episode | Zero residual | Residual/full 36 | Absolute/full 36, scored as residual | Residual/current-only 36 | Residual/full 72 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.796791 | 1.798159 | 4.879768 | 1.802431 | 1.781474 |
| 4 | 1.463674 | 1.469886 | 4.689308 | 1.473047 | 1.460526 |
| 10 | 1.943480 | 1.937020 | 4.792864 | 1.934171 | 1.908768 |
| 21 | 2.917009 | 2.939076 | 4.630909 | 2.938093 | 2.916851 |
| 422 | 1.997150 | 1.983360 | 4.648663 | 1.982388 | 1.945348 |

Training loss decreased, but continued training after 36 updates also makes
overfitting a plausible explanation. A larger split and repeated-seed study is
needed before concluding that the residual objective improves task-relevant
VLM representations. A downstream policy or task-success experiment is also
needed to connect DINO residual MSE to useful behavior.

## Resource and supervision notes

The highest recorded allocated CUDA memory in the 36-update full-history
residual run was 10,604,234,240 bytes; current-only peaked at
8,860,815,872 bytes. Evaluation reported 3,028,205,568 bytes peak allocation
for its one configured history profile. These are single-run allocator
readings, not device-wide memory requirements or scaling guarantees.

The 72-update evaluation profile averaged 1.87 s preprocessing, 0.92 s VLM,
0.01 s residual query/head, and 0.38 s DINO-teacher time per sample. GPU
contention and small-sample timing make these rough path measurements, not a
latency benchmark.

## Next experiment

Use a broader episode split with multiple control steps per episode and at
least three seeds. Compare full-history residual, current-only residual,
absolute-goal prediction, and zero residual under an identical update budget
and optimizer schedule. Add assistant text labels only when their provenance
and masks are verified. Keep task rollout success as a separate downstream
evaluation; residual MSE alone cannot establish control utility.
