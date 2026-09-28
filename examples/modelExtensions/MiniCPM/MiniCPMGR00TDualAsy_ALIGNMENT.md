# MiniCPMGR00TDualAsy training and inference alignment

Status date: 2026-09-28. This is a single-task engineering pilot, not a policy
quality or success-rate evaluation.

## Keep four quantities separate

- `H = action_model.action_horizon`: number of future actions the head predicts
  in one call. The pilot uses `H=8`.
- `K = execution_horizon`: number of predicted actions the controller commits
  before asking the policy for another chunk. The LIBERO dual pilot currently
  fixes `K=1` and executes the first row. Other evaluators may choose another
  `K`; it is not a property of the action head.
- `M = framework.vlm_refresh_interval`: control steps between upper VLM image
  refresh requests. The pilot defaults to `M=8`.
- `L`: actual VLM delivery delay, measured from request to the first control
  step whose action uses the result. In wall-clock operation `L` varies with
  the device, load, and control timing. It is measured, not configured as a
  universal constant.

The observed `8:1` pilot cadence is `M:K` (`8:1`) while the predicted action
chunk has `H=8`. Those equal numbers do not make H, K, and M interchangeable.

## Earlier wall-clock trace

The corrected LIBERO Goal run at
`playground/Checkpoints/libero_minicpm_pilot/20260928_222713/comparison.json`
ran 56 control steps, failed the task, and recorded per-step cache sources.
It used `H=8`, `K=1`, and `M=8`. Its step-0 VLM result was awaited; later
requests became action-visible as follows:

| Source/request step | First action-visible step | Observed L |
|---:|---:|---:|
| 8 | 11 | 3 |
| 16 | 19 | 3 |
| 24 | 27 | 3 |
| 32 | 35 | 3 |
| 40 | 43 | 3 |
| 48 | 52 | 4 |

Mean VLM compute time was `0.760 s`; mean/p95 policy round-trip time was
`0.176 s` / `0.277 s`. This trace shows that a refresh requested at step 48
was not used until step 52. The earlier sampler used a fixed 3-step estimate;
that estimate matched 55 of 56 control steps in this run. It is historical
pilot evidence, not a default latency for other runs.

## Alignment modes

Training sample construction and runtime selection share the same
source-to-activation schedule for controlled experiments:

- `synchronous`: refresh and activate at the request step (`L=0`).
- `fixed_step_delay`: use an explicitly selected control-step delay. The
  schedule models a serialized worker queue when `L >= M`; the evaluator waits
  at a scheduled activation boundary if the real computation is not ready yet.
- `trace_replay`: use `source_step` / `activation_step` events captured from an
  earlier run. The pilot trace also contains request, ready, and activation
  timestamps.
- `wall_clock`: use the newest result completed at each action boundary and
  never wait for a refresh after the step-0 bootstrap. Offline training cannot
  infer this schedule, so it requires `--training-alignment-trace` and trains
  with that trace replayed.

Examples:

```bash
# Controlled delay, same schedule for sampler and inference
python examples/modelExtensions/MiniCPM/libero_dual_pilot.py \
  --models MiniCPMGR00TDualAsy --alignment-mode fixed_step_delay \
  --fixed-latency-steps 2

# Replay one recorded schedule for both sampler and runtime
python examples/modelExtensions/MiniCPM/libero_dual_pilot.py \
  --models MiniCPMGR00TDualAsy --alignment-mode trace_replay \
  --alignment-trace /path/to/comparison.json

# Live asynchronous runtime, trained against a previously recorded trace
python examples/modelExtensions/MiniCPM/libero_dual_pilot.py \
  --models MiniCPMGR00TDualAsy --alignment-mode wall_clock \
  --training-alignment-trace /path/to/comparison.json
```

The evaluator writes `H`, `K`, `M`, the selected alignment modes, per-step
active source and state age, plus `activation_events` with source/request/ready/
activation timestamps. A wall-clock trace can be passed back through
`trace_replay` without replacing its variable delays with an assumed median.

## Current limits

- General LeRobot training still needs a reusable temporal sampler that emits
  aligned `vlm_image` and source-step metadata; the explicit modes currently
  live in the small LIBERO pilot.
- The recorded rollout is one task and one seed, did not succeed, and does not
  establish an asynchronous policy improvement.
- The pilot currently fixes K at one. Making K configurable in the evaluator
  is separate from the alignment work and must not change H or M implicitly.
