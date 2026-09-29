# MiniCPMGR00TRes target-formulation pilot

Date: 2026-09-29
Status: short-budget engineering/diagnostic pilot; **not** the formal Stage 3-A result.

## Question

Under a matched, frozen setup, does predicting the terminal DINO feature residual
work better than predicting the absolute terminal feature?

\[
\hat z_g=z_t+F_\theta(I_t,L) \quad\text{(residual)}
\]

versus

\[
\hat z_g=F_\theta(I_t,L) \quad\text{(absolute goal)}.
\]

## Protocol

- Used the frozen task-stratified source-demonstration split
  `annotations/libero_goal_residual_split_v1.json`, manifest SHA-256
  `310acc6c1f06ee4bca7ae9968ad127f90dfebb14f98a56d7d9bd367071711a1a`.
  It contains 297 train, 62 validation, and 62 test episodes; the test split was
  not opened. Seven episodes without explicit successful-terminal labels are
  excluded.
- `current_only` input, one control step (8) per episode, explicitly annotated
  successful terminal target, frozen DINOv2 ViT-S/14 teacher, no assistant-text
  loss. The text loss is disabled because no verified assistant-label manifest
  exists. The teacher state SHA-256 is
  `2ab7126d86baa27f5aba61da68c002e92264da521292105b281f000d1a8daf9e`.
- Compared two paired seeds (42 and 43), each with 24 optimizer updates,
  batch size 1, AdamW, learning rate `1e-5`, and weight decay `1e-8`. Each run
  sampled 24 training examples from the 297-episode training candidate set.
  The residual and absolute configs were checked to differ only in
  `prediction_target`; paired runs used the same seed, initialization, sample
  order, model capacity, preprocessing, and optimizer settings.
- Validation used all 62 episodes at control step 8. Each prediction was
  converted to the common goal-space comparison: `z_t + predicted_residual`
  for residual, and the absolute prediction directly for absolute-goal.
- Global-mean and task-conditioned-mean residual baselines were computed only
  from the 297 training episodes, at control step 8. All variants use the same
  frozen split and DINO teacher.
- Runs used one NVIDIA A100-SXM4-80GB at a time. Peak allocated memory was
  approximately 8.26 GiB per training run and 2.61 GiB during validation.
  Representation runtime provenance records `H_action_prediction=None`,
  `K_execution=None`, `M_vlm_refresh=8`, and synchronous terminal-target
  alignment; the action horizons do not apply to this standalone representation
  experiment.
  Source code commit: `e46df41294b4ef752c344afe3f11ff0ca90ec926`, clean
  worktree at run start.

## Results

Mean over the 62 validation episodes; lower goal-space MSE is better. Values
for learned models are averaged over seeds 42 and 43. Mean baselines are
computed from train only.

| Predictor | Goal-space MSE | Goal cosine | Patch cosine | Residual cosine |
|---|---:|---:|---:|---:|
| Zero residual | 2.2383 | 0.7894 | 0.7852 | 0.0000 |
| Global mean residual (train only) | 1.8147 | 0.8179 | 0.8140 | 0.4318 |
| Task mean residual (train only) | **0.9621** | **0.9075** | **0.9056** | **0.7457** |
| Learned residual, 24 updates | 2.2552 | 0.7873 | 0.7834 | 0.0602 |
| Learned absolute goal, scored in goal space | 4.7832 | 0.3235 | 0.3230 | 0.3434 |

Per-seed goal-space MSE:

| Target | Seed 42 | Seed 43 | Mean |
|---|---:|---:|---:|
| Residual | 2.2484 | 2.2619 | 2.2552 |
| Absolute goal | 4.8348 | 4.7315 | 4.7832 |

## Interpretation and limits

At this 24-update budget, the learned residual is 0.75% worse than zero
residual on validation and far worse than the train-only task-mean predictor.
Absolute-goal prediction is also much worse than both simple baselines. These
results show that neither learned head has yet demonstrated state-conditioned
terminal-change prediction in this pilot. The strong task-mean result suggests
that task-level terminal appearance is a substantial prior in this dataset;
the model must beat it before we can claim useful state-dependent residual
representation.

This does **not** establish that the residual formulation is invalid. The pilot
trains only 24 examples/updates per run, samples one point per demonstration,
uses two seeds, and tests one current-frame position. It does not measure
history, rollout success, cross-embodiment transfer, or downstream control.
The validation interval is not an independent test, and the frozen test split
remains untouched. No checkpoint continuation or model selection was based on
the test split.

The next Stage 3-A run should use a predeclared larger but still manageable
update budget, at least three paired seeds, and multiple current-frame
positions per training episode. Keep the same frozen split, target-independent
architecture, optimizer schedule, and train-only zero/global/task-mean
baselines. Select settings on validation only, then evaluate the frozen test
once after the experiment is locked. Do not add history or text supervision to
this target-formulation comparison.

## Artifacts and provenance

Ignored run artifacts, including per-step training metrics, resolved config,
split provenance, full run provenance, checkpoints, per-episode validation
metrics, and baseline feature arrays are under:

`playground/Checkpoints/minicpm_res_stage3_target_ablation_pilot_20260929/`

The four run provenance files record the same frozen split hash, commit, device,
precision, dependency versions, and trainable parameter count. The split
manifest verifier also passed against the current dataset metadata and
successful-terminal annotations before training. The frozen dataset metadata
manifest hash is
`57ff27ab3a16784d988c2d188d4a32440580d112f30c37316560f1a356d0204a`, and the
success-terminal annotation SHA-256 is
`895028dab5a278a04ae81ac007608f52aba44b804a9fb86874e694b4df34573b`. The
train-only mean-residual artifact SHA-256 is
`fb404f05e93d9d361231fe5b3cd97b123955c912a0119791ec3d543e3cbfde44`.
