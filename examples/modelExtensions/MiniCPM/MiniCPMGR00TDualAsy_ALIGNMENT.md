# MiniCPMGR00TDualAsy training and inference alignment

Status date: 2026-09-28. This is a single-task engineering pilot, not a policy
quality or success-rate evaluation.

## Measured inference schedule

The LIBERO Goal pilot used `action_horizon=8`, `execution_horizon=1`, and
`vlm_refresh_interval=8`. It ran 56 control steps, failed the task, and recorded
per-step `cached_vlm_step` values in
`playground/Checkpoints/libero_minicpm_pilot/20260928_221032/comparison.json`.
The initial step-0 VLM refresh was awaited. Later refresh requests and their
first action-visible control steps were:

| Requested source frame | First action-visible step | Delay |
|---:|---:|---:|
| 8 | 11 | 3 |
| 16 | 19 | 3 |
| 24 | 28 | 4 |
| 32 | 35 | 3 |
| 40 | 43 | 3 |
| 48 | 52 | 4 |

Mean VLM refresh time was `0.849 s`; mean/p95 action-policy round-trip time was
`0.182 s` / `0.281 s`. All seven submitted refreshes completed, with no stale
results dropped and no queued refresh remaining at episode end. This shows why
the source image at a refresh boundary is not necessarily the VLM state used by
the action at that same control step.

## Alignment change

The old training anchor `floor(control_step / 8) * 8` put frames 9 and 10 on
VLM source frame 8. In the measured inference trace both control steps still
used the frame-0 VLM state. The pilot now has a separate, training-only
`framework.training_vlm_latency_steps` setting. For refresh interval `M` and
estimated activation delay `D`, a training sample at step `t` uses:

```text
anchor(t) = max(0, floor((t - D) / M) * M)
```

This closed form applies when `D < M`, as in the measured run. The sampler also
models serialized worker queueing when `D >= M`. The async LIBERO config sets
`D=3`, the median observed delay. The four training frames `[1,2,9,10]`
therefore use anchors `[0,0,0,0]`; held-out frame 23 uses anchor 16. This
aligns those samples with the measured cache schedule. On the full 56-step
trace, the fixed 3-step estimate matched 54 steps; it was one step early for
refreshes 24 and 48, whose observed delay was 4. The latency setting can be
overridden with `--training-vlm-latency-steps` when the hardware or control
cadence changes.

A 4-update training smoke with this alignment completed with finite action
losses and held-out loss `1.7331`; see
`playground/Checkpoints/libero_minicpm_pilot/20260928_221912/comparison.json`.
That run skipped simulation. The separate trace run above used the prior
zero-delay training anchors, so the two artifacts validate the inference trace
and corrected sample selection independently rather than as one same-run
closed-loop experiment.

## Limits and next checks

- `training_vlm_latency_steps` is currently wired into the small LIBERO pilot
  sampler. General LeRobot training still needs a reusable temporal sampler
  that emits the same aligned `vlm_image` and source-step metadata.
- A constant delay approximates variable refresh completion. The six measured
  refreshes had delays of 3 or 4 steps; more tasks, seeds, and hardware runs are
  needed before choosing a robust delay distribution.
- The rollout is one task and one seed, did not succeed, and does not establish
  an asynchronous policy improvement. The default execution remains one action
  per policy call; `action_horizon` and VLM refresh cadence are separate.
- Next validation should compare ideal zero-delay anchors, calibrated-delay
  anchors, and trace-derived anchors over several tasks and seeds, while
  reporting per-step cache source/age and task success separately.
