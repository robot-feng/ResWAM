# MiniCPMGR00TRes implementation status

Status date: 2026-09-29. This is an engineering smoke report, not an algorithm
effectiveness claim.

## Implemented

- New independent `MiniCPMGR00TRes` registry class; original MiniCPM, Dual,
  DualAsy, QwenDual and QwenOFT registrations remain distinct.
- OFT-style 256-query decoder: 1024 → 512, two residual MLP blocks, 384-D
  signed DINO patch residuals.
- MiniCPM-V 4.6 native-video adapter with sampling explicitly disabled,
  episode/frame/view/timestamp text, full received history, and a 16,384-token
  hard budget.
- Explicit successful-terminal LeRobot annotations; unlabelled episodes and
  post-terminal steps are excluded. Training samples are aligned to
  `runtime.vlm_refresh_interval`, and each selected sample still contains all
  observations up to that step.
- Separate residual and optional assistant-text forwards; only assistant
  answer tokens contribute to text loss. DINO teacher and the MiniCPM visual
  tower/merger are frozen while LM and residual head stay trainable.
- History ablation evaluator can mask the oldest, middle, or most recent past
  frame with a neutral image while retaining its timestamp and sequence slot.
- Streaming `reset_episode`, `observe`, `predict_goal`; default refresh interval
  is `runtime.vlm_refresh_interval: 8`. It does not read a lower policy's
  `action_horizon`.
- Dedicated trainer, zero-residual comparison/profile evaluator, configs for
  full-history/current-only and residual/absolute-goal ablations, strict
  trainable-weight checkpoint plus tokenizer/teacher/config metadata.
- Evaluation CLI supports oldest/middle/most-recent-previous history-frame
  masking while retaining frame order, timestamps, and context length.
- Optional inference-only hybrid prefix cache: prefill the complete task and
  video history, append each newly received frame batch at the VLM refresh,
  and execute residual queries against a cloned cache. Full recomputation is
  still the default. Prefix mode periodically checks decoded residual outputs
  and falls back for the episode if its configured error limit is exceeded.

## Verified

- Processor smoke: 2 timestamped video frames produced 520 attended tokens and
  exactly 256 residual query tokens with `do_sample_frames=False`.
- Real MiniCPM-V 4.6 + DINOv2 ViT-S/14 single-example GPU forward produced
  finite losses, residual shape `[1,256,384]`, and VLM query shape
  `[1,256,1024]`. Combined residual/text backward reached 320 trainable VLM
  tensors and the residual head; the visual tower/merger had zero trainable
  parameters.
- A read-only LIBERO LeRobot sample was opened: primary camera key is
  `video.primary_image`, language key is
  `annotation.human.action.task_description`, and raw episode timestamps are
  available.
- Generated the explicit success-terminal sidecar at
  `examples/modelExtensions/MiniCPM/annotations/libero_goal_success_terminals.jsonl`.
  Across 428 LeRobot episodes, 421 action sequences uniquely match one of the
  500 original LIBERO `libero_goal` HDF5 demos using ordered exact matching on
  the first six action dimensions. All 421 source demos have terminal
  `done=1,reward=1`; the other seven episodes are explicitly left unlabeled.
  The 421 encoded LeRobot terminal video frames decoded without errors and had
  RGB MAE to their source HDF5 terminal images of 8.48 median, 12.54 p95, and
  16.98 maximum on a 0–255 scale, after the source camera rotation and resize.
  The dataset adapter then opened the sidecar read-only: 3,386 samples at the
  configured stride of 8 across 421 labeled episodes, seven excluded episodes,
  with a valid current image, terminal image, and full prefix on the first sample.
- Framework auto-discovery resolved `MiniCPMGR00T` to the original class and
  registered the new class under `MiniCPMGR00TRes`.
- Runtime fake-policy test confirmed refresh at control steps 0, 8, and 16,
  retention of all 17 received frames, and plan age 7 immediately before the
  second refresh.
- Processor block test confirmed task plus frame message token IDs exactly
  concatenate to the full-history token IDs; grouped 5-frame video pixels and
  target sizes exactly matched concatenated per-frame blocks.
- Installed MiniCPM-V 4.6 GPU cache comparison ran both an initial prefill and
  a later 8-frame append. With a randomly initialized residual decoder, the
  cached residual output was 2.42% relative RMS from full-history recompute
  (cosine similarity 0.9997). A teacher-forced text check had 2.1% relative
  RMS logit drift and 10/11 next-token argmax matches. This supports the
  inference cache path but does not claim exact text generation or validation
  for a trained checkpoint.
- Framework-level GPU smoke ran 9 frames through `observe` with an 8-step
  VLM refresh interval. It refreshed at steps 0 and 8, appended the intervening 8
  frames in one cache update, and validated both outputs against full
  recomputation. Residual relative RMS differences were 2.48% and 2.38%; the
  cache stayed enabled under the default 5% threshold. The report was written
  to `/tmp/reswam_prefix_cache_smoke.json`.
- Full-size checkpoint round-trip used the actual model with 334 trainable
  tensors. After saving a 1.51 GB temporary checkpoint, mutating a trainable
  parameter, and loading it back, a byte digest of every trainable tensor
  matched exactly. This also surfaced and fixed a missing `torch.no_grad()` in
  strict restore.
- Separate synthetic backward checks passed for residual and assistant-text
  losses. Each reached 320 trainable VLM tensors; residual loss reached the
  prediction-token embedding and all 14 decoder tensors, while text loss
  reached the LM head. DINO teacher and vision tower/merger remained without
  gradients.
- Added an unlabeled synthetic profiler for 1/8/32-frame histories. It records
  preprocessing, VLM, query/head, current-DINO latency, context tokens, and
  peak CUDA allocation without presenting the results as task performance.
- One synthetic inference profile measured 1 frame / 417 tokens at 2.68 s and
  6.02 GB peak CUDA allocation; 8 frames / 1,179 tokens at 3.17 s and 13.76 GB.
  A one-frame synthetic joint-loss forward/backward took 4.37 s and peaked at
  5.71 GB, with finite gradients in all 334 trainable tensors and no optimizer
  step. These are single-run engineering numbers, not latency benchmarks.
- A 32-frame / 3,836-token synthetic input exceeded available GPU memory during
  concurrent use: another process occupied about 58.6 GB of the 80 GB A100.
  This does not establish a model-imposed 32-frame limit; it means the current
  shared-GPU run only verified up to 8 frames. Report:
  `/tmp/reswam_synthetic_profile.json`.
- Installed `starVLA` 1.0.1 in Conda environment `ResWAM` as a PEP 660 editable
  install from `/data/tzq/ResWAM`. Importing from `/tmp` resolved
  `starVLA.__file__` to `/data/tzq/ResWAM/starVLA/__init__.py`, and package
  metadata reports `editable: true` for that source directory.
- A one-update BF16 dry-run against the verified LIBERO manifest completed with
  residual/total loss `2.7029`, gradient norm `34.85`, text supervision off,
  and peak allocated CUDA memory `8,856,342,016` bytes. The checkpoint restored
  its expanded tokenizer and model metadata.
- Latest tests were run in Conda environment `ResWAM` against the editable
  checkout: `tests/test_minicpm_gr00t_res_components.py`,
  `tests/test_minicpm_gr00t_dual.py`, and
  `tests/test_minicpm_dual_pilot_eval.py` passed (`21 passed, 3 warnings`). The
  CPU target test checks `z_goal-z_current`, exact zero residual at the terminal
  frame, and that the future image is absent from the VLM history. Horizon tests
  verify that action prediction length and VLM refresh cadence are independent,
  that the LIBERO pilot executes only the first action of an 8-action chunk,
  and that the async training sampler uses the latest completed VLM anchor
  while modeling serialized refresh queueing. They do not claim that MiniCPM
  DualAsy has a configurable execution horizon.
  `git diff --check` passed. Current changes do not modify
  `MiniCPMGR00T.py`, `QwenDual.py`, or `QwenOFT.py`.
- A small labeled LIBERO ablation used episodes `0,1,3,6,7` for training and
  `2,4,10,21,422` for evaluation, with one sample per episode at
  control step 8. Every full-history sample contained nine frames. Each
  36-update variant used 12 initial updates plus a 24-update continuation
  with a fresh optimizer; the full-history residual variant then received 36
  additional updates under another fresh optimizer. Text supervision was
  disabled because no assistant-answer labels were provided. Evaluation
  episodes were inspected after 36 updates and informed the decision to
  continue to 72, so they are validation examples rather than a final test set.
- At 36 updates, full-history residual prediction scored mean evaluation DINO
  residual MSE `2.0255` versus `2.0236` for the zero-residual baseline (0.09%
  worse). Absolute-goal prediction, converted back to residual at evaluation,
  scored `4.7283`; current-only residual prediction scored `2.0260`, essentially
  identical to full history at this budget.
- After 72 total updates, full-history residual MSE was `2.0026` on the five
  evaluation examples (1.04% below zero residual), `2.3337` on training examples
  (versus `2.4172`), and `2.1682` overall (versus `2.2204`). This is a small,
  preliminary signal, not statistically reliable evidence of general
  representation improvement. See `MiniCPMGR00TRes_ABLATION.md` for the full
  comparison and limitations; raw evaluation JSON and checkpoints are in
  `/data/tzq/tmp/reswam_ablation_20260928`, outside the repository.

## Frozen Stage 3 split and pipeline smoke (2026-09-29)

- Froze `annotations/libero_goal_residual_split_v1.json`, SHA-256
  `310acc6c1f06ee4bca7ae9968ad127f90dfebb14f98a56d7d9bd367071711a1a`.
  Its task-stratified source-demo group split contains 297 train, 62 validation,
  and 62 test episodes. Of 428 episodes, 421 have explicit successful terminal
  labels and seven ambiguous episodes are excluded. The old pilot train IDs
  (`0,1,3,6,7`) remain train; the five episodes used in the earlier continuation
  decision (`2,4,10,21,422`) remain validation. Split, annotation, and dataset
  metadata hashes and cross-split group disjointness verified successfully.
- Ran three optimizer updates from the frozen train split at control step 8 on
  one A100. Residual losses were `2.6361`, `2.0498`, and `2.6789`; text loss was
  disabled because no assistant-answer labels exist. Peak allocated GPU memory
  was about 10.6 GiB. This confirms the split-aware training path runs, not that
  training improved the model.
- Evaluated four validation samples at control step 8 (episodes `2,4,10,13`).
  Mean residual MSE was `1.9642` versus `1.8529` for zero residual, so this
  three-update checkpoint did not beat the baseline on this tiny sample. The
  evaluation was only a pipeline check; it did not drive tuning or continuation.
  Episodes `2,4,10` were already part of the earlier validation pilot and remain
  validation. No test video frames were decoded or used for training/evaluation.
- Checkpoint, logs, and validation JSON are under the ignored
  `playground/Checkpoints/libero_minicpm_stage3_split_pilot/` directory.

## Short-budget target-formulation pilot (2026-09-29)

- With the frozen split, ran paired current-only residual and absolute-goal
  training for 24 updates each at seeds 42 and 43. Configs differ only in
  `prediction_target`; validation uses all 62 frozen validation episodes at
  control step 8. Test remains untouched.
- Mean goal-space MSE was 2.2552 for residual, 4.7832 for absolute goal,
  2.2383 for zero residual, 1.8147 for the train-only global-mean residual,
  and 0.9621 for the train-only task-mean residual. Thus this short pilot does
  not show learned terminal representation beating simple baselines. It is a
  diagnostic result, not a rejection of residual targets or a formal Stage
  3-A conclusion.
- The 24-update runs sampled only 24 of 297 train episodes per seed and used
  one current-frame position, two seeds, and no text labels. A larger matched
  run with multiple positions and at least three seeds is still required;
  validation may guide settings, while the frozen test must remain unopened
  until the experiment is locked.
- Full protocol, per-seed numbers, caveats, and ignored artifact path are in
  [`MiniCPMGR00TRes_TARGET_FORMULATION_PILOT_20260929.md`](MiniCPMGR00TRes_TARGET_FORMULATION_PILOT_20260929.md).

## Not yet verified

- Joint text-plus-residual training has not been run because there is no
  assistant-answer label manifest; the real-data ablation is residual-only.
  The 72-update run lowered training MSE, but its evaluation result covers only
  five episodes and one split. It is not a robust overfit or generalization
  study.
- Cache validation used a randomly initialized residual decoder. Validation
  with a trained checkpoint is still outstanding. Prefix mode defaults to a
  5% decoded-residual relative RMS limit, validates the first refresh and then
  every 8th refresh, and falls back to recomputation for the episode on
  failure. Text logits show measurable numerical drift; deterministic text
  generation must use recomputation or receive its own validation.
- The real-data timing and memory values are single-run measurements on a
  9-frame sample path, not a history-length scaling benchmark with a trained
  checkpoint. The 32-frame synthetic profile remains unverified because of
  shared-GPU memory pressure.
- The evaluation covers one control step from five episodes, not full
  trajectories, and informed checkpoint continuation. No rollout-success,
  task-level representation utility, or cross-embodiment result is claimed.
