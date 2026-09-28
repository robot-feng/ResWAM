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

A previous real MiniCPM/DINO LIBERO smoke row used four optimizer updates on
episode 0 frames `[1,2,9,10]`, with frame 23 for held-out action loss. The VLM
and DINO backbone were frozen; 142.99M action-head/projection parameters were
trainable. Training loss moved from `1.3858` to `1.5064`; held-out loss was
`1.7195`. The matching LIBERO Goal rollout ran 56 control steps and failed to
complete the task; mean and p95 policy round-trip latencies were `0.873 s` and
`1.044 s`. This proves that the real model and simulator path ran, but it does
not establish policy quality. The row came from the 2026-09-28 smoke output
under `playground/Checkpoints/libero_minicpm_pilot/20260928_151100/`.

In that rollout, the action head predicted a chunk of `action_horizon=8`, but
the evaluator used `chunk[0]` and requested a fresh prediction on the next
control step: effective execution length was one action (`K=1`). The legacy
smoke JSON also contains `execution_horizon=8`; that field does not describe
what the evaluator actually committed and should not be used as the measured
execution length. This stage has no VLM refresh interval (`M`) because it is
synchronous.

The current pilot defaults to `MiniCPMGR00TDual` only. It loads and aligns the
asynchronous config only when `MiniCPMGR00TDualAsy` is explicitly selected.
The run can use `--skip-simulation` to validate training without launching the
separate LIBERO environment.

## Still needed

- Repeat the real training smoke after the current config/pilot isolation
  changes and preserve a stage-1-only result artifact.
- Evaluate more than one task and multiple seeds before making success-rate
  claims. The existing four-update smoke failed its single rollout.
- Validate checkpoint save/load for a trained MiniCPMGR00TDual checkpoint and
  document action unnormalization against the selected LIBERO statistics.
