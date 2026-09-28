# MiniCPMGR00TDual: stage-1 status

Status date: 2026-09-28. This status covers the synchronous dual-stream action
baseline only. `MiniCPMGR00TDualAsy` and `MiniCPMGR00TRes` remain separate later
stages.

## Model and configuration

`MiniCPMGR00TDual` has its own framework registry name and class. It reuses the
`Qwen_Dual` dual-stream implementation, with the existing VLM dispatcher
selecting MiniCPM-V 4.6. The condition sequence is MiniCPM language/image
hidden states followed by projected DINOv2 patches; the GR00T flow-matching
head predicts the action chunk. It has no async refresh worker, residual head,
or runtime VLM refresh parameter.

The example configuration is
`train_files/minicpm_gr00t_dual_libero.yaml`. It exposes the VLM checkpoint and
attention backend, DINO backbone, action dimensions, `action_horizon`, flow
matching settings, image size, freeze list, and optimizer settings. Defaults
are copied before MiniCPM-specific values are filled, so constructing the new
framework does not mutate the caller's config or the original QwenDual
defaults.

## Verification

`tests/test_minicpm_gr00t_dual.py` currently passes five tests in Conda
environment `ResWAM`. They check registry separation from `MiniCPMGR00T` and
`QwenDual`, default config isolation, the order and values of the VLM and DINO
tokens delivered to the action head, gradients into the VLM and DINO projection,
and configurable `action_horizon=3` / `action_dim=4` for both training labels
and prediction shape. Pilot config resolution is also checked: stage 1 does not
open an async config or accept an async refresh interval.

A stage-1-only real-model smoke ran four optimizer updates on episode 0 frames
`[1,2,9,10]`, with frame 23 for held-out action loss. The VLM and DINO backbone
were frozen; 142.99M action-head/projection parameters were trainable. Losses
were `[1.3858, 1.0644, 1.3148, 1.5064]`, held-out loss was `1.7195`, and the
four updates took `9.66 s`. Its matching LIBERO Goal rollout ran 56 control
steps and did not complete the task. Mean/p95 policy round-trip latencies were
`0.846 s` / `1.073 s`. The action head predicted 8 actions, and the evaluator
committed one per policy call (`execution_horizon=1`), then requested a fresh
prediction on the next control step. This synchronous stage has no VLM refresh
interval (`M`). The result and video are in
`playground/Checkpoints/libero_minicpm_pilot/20260928_220333/`. This smoke
validates the isolated training and simulator path, not policy quality.

The current pilot defaults to `MiniCPMGR00TDual` only. It loads and aligns the
asynchronous config only when `MiniCPMGR00TDualAsy` is explicitly selected.
The run can use `--skip-simulation` to validate training without launching the
separate LIBERO environment.

## Still needed

- Evaluate more than one task and multiple seeds before making success-rate
  claims. The existing four-update smoke failed its single rollout.
- Validate checkpoint save/load for a trained MiniCPMGR00TDual checkpoint and
  document action unnormalization against the selected LIBERO statistics.
