# MiniCPMGR00TDualAsy training and inference alignment

Status date: 2026-09-29. This is a single-task engineering pilot, not a policy
quality or success-rate evaluation.

## Keep four quantities separate

- `H = action_model.action_horizon`: number of future actions the head predicts
  in one call. The pilot uses `H=8`.
- `K = execution_horizon`: number of predicted actions the controller commits
  before asking the policy for another chunk. The LIBERO evaluator exposes
  `--execution-horizon K`; K=4 is now exercised below. It is not a property of
  the action head.
- `M = framework.vlm_refresh_interval`: control steps between upper VLM image
  refresh requests. The default is `M=8`; the K=4 timing pilot uses M=2.
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
  Treat this as a deterministic alignment oracle; its round-trip latency is not
  evidence for nonblocking fast-loop performance.
- `trace_replay`: use `source_step` / `activation_step` events captured from an
  earlier run. The pilot trace also contains request, ready, and activation
  timestamps. This mode also waits when an activation scheduled by the trace is
  not ready, so it checks schedule replay and training alignment.
- `wall_clock`: use the newest result completed at each action boundary and
  never wait for a refresh after the step-0 bootstrap. Offline training cannot
  infer this schedule, so it requires `--training-alignment-trace` and trains
  with that trace replayed. Use this mode to measure live fast-loop latency.

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
active source and semantic-state age, separate DINO/condition and action-head
compute time, plus `refresh_events` and `activation_events` with source,
request, ready, and activation step/timestamp fields. `run_provenance.json`,
`data_split_manifest.json`, resolved model configs, and trainable-parameter
checkpoints are saved with each pilot. A wall-clock trace can be passed back
through `trace_replay` without replacing its variable delays with an assumed
median.

Only the active semantic hidden state and completed states still awaiting a
controlled activation are retained. Once a state is activated, its hidden
tensor is released from the pending-snapshot map; unactivated tensors are
released when the worker closes. The scalar event log remains available for
trace analysis.

## Current limits

- The reusable LeRobot sampler currently targets `dataset_py=lerobot_datasets`.
  It samples only K policy-call boundaries, accounts for that constraint in
  trajectory sampling weights, and emits `vlm_image` plus source/request/
  activation/age metadata. Trace replay fails during dataset construction if
  the trace does not cover all eligible episode-local training steps.
- For K>1, the evaluator sends every intermediate environment observation to
  `observe_for_async_refresh`; the action head runs only at K boundaries. The
  runtime rejects skipped observation steps or action calls off the K grid.
  VLM results ready between policy calls become action-visible at the next K
  boundary, and training uses that same boundary.
- Wall-clock offline training requires `framework.async_alignment.training_trace_path`.
  The trace must cover every eligible episode-local sample step; a short pilot
  trace is only suitable for bounded smoke runs. Fixed-delay training can cover
  arbitrary episode lengths.
- The recorded rollout is one task and one seed, did not succeed, and does not
  establish an asynchronous policy improvement.
- The LIBERO evaluator accepts `--execution-horizon K` independently of H, M,
  and L. Policy calls occur at the next action-chunk boundary, so a completed
  VLM refresh can only become action-visible at the next such call. The
  observer path keeps M independent of K by delivering images between action
  calls.

## Same-trajectory 16-update comparison (2026-09-29)

To extend the earlier two-update/24-control-step pipeline smoke, the three
frameworks were trained sequentially on episode 0 and each was evaluated for
the configured 56-control-step budget. This is still a one-task engineering
pilot, not a policy performance study.

Protocol:

- Seed `1234`; task instruction `put the bowl on the plate`; 16 episode-0
  training frames `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
  24]`; held-out action-loss frame `23`.
- MiniCPM and DINO backbones were frozen. Each model received 16 action-loss
  optimizer updates with the same selected current frames and action labels.
- All runs used `H=8` actions predicted per call and `K=1` action executed per
  control step. The DualAsy run additionally used `M=8` and controlled
  `L=4`, in both training-anchor construction and runtime scheduling.
- This run used `K=1`; it selected the first action from the 8-action chunk,
  executed it, then requested the next chunk. The evaluator now accepts K
  separately; equal numeric values for H, K, or M do not make them
  interchangeable.

| Variant | H / K / M / L | Action loss, update 1 → 16 | Held-out loss | Rollout | Mean / p95 policy RTT |
|---|---:|---:|---|---|---:|
| MiniCPMGR00T | 8 / 1 / — / — | 1.5605 → 1.1545 | 1.0278 | False at 56 steps | 0.748s / 0.911s |
| MiniCPMGR00TDual | 8 / 1 / — / — | 1.5679 → 1.1925 | 1.0312 | False at 56 steps | 0.788s / 0.954s |
| MiniCPMGR00TDualAsy | 8 / 1 / 8 / 4 | 1.5730 → 1.2117 | 1.0479 | False at 56 steps | 0.185s / 0.333s |

The asynchronous run completed 56 action calls and seven VLM refreshes. Its
training anchor schedule was:

| Training control frames | VLM source anchor |
|---|---:|
| 8–11 | 0 |
| 12–19 | 8 |
| 20–22, 24 | 16 |

Held-out frame 23 also used source frame 16. Runtime refresh events recorded
sources `0, 8, 16, 24, 32, 40, 48` and scheduled activations `0, 12, 20, 28,
36, 44, 52`. The per-control-step cached source trace advances at those same
activation boundaries. This confirms the fixed-delay training anchors and
runtime source selection agree for this pilot configuration.

The fixed-delay evaluator may wait at a scheduled activation boundary, so its
policy RTT is not a nonblocking wall-clock speed benchmark. All three rollouts
reached the 56-step cutoff with `success=false`; 16 updates on one demonstration
do not establish policy quality or an advantage for any framework. The useful
result here is the checked H/K/M/L separation and the longer async source trace.

Raw metrics, per-step trace, provenance, checkpoints, and videos are under
`playground/Checkpoints/libero_minicpm_pilot/stage2_triplet_16updates_full56_seed1234/`.

## Execution-horizon pilot (2026-09-29)

To verify that predicted chunk length and committed action count are distinct,
the LIBERO evaluator now accepts `--execution-horizon K` in `[1, H]`. The model
still predicts `H=8` actions per policy call; the evaluator commits the first K
actions, then requests a new chunk. `M=8` and the controlled worker delay
`L=4` stayed unchanged. Each K run used the same seed, four optimizer updates
on frames `[8, 9, 16, 17]`, held-out frame 23, and a 24-control-step rollout.
Only GPU 0 was visible to the experiment process.

| K | Policy calls / 24 control steps | Source 8 first used at | Source 16 first used at | Rollout |
|---:|---:|---:|---:|---|
| 1 | 24 | 12 | 20 | False |
| 4 | 6 | 12 | 20 | False |
| 8 | 3 | 16 | Not used by step 24 | False |

For `K=8`, the fixed schedule targets source-8 activation at step 12, but the
controller has no policy call there because it is executing the chunk requested
at step 8. The refresh had completed by the next call, so the new semantic
state was first consumed at step 16. Likewise, the step-16 refresh had no later
action call inside this rollout.
This is why H and K cannot be treated as the same parameter: H controls the
predicted chunk size, while K controls how often the controller can replan and
consume refreshed VLM state. Larger K reduces policy calls but can increase
the effective age of semantic context at the next action decision.

The optimizer losses were identical across the three runs because seed, model,
and training samples were held fixed: action loss `1.5730 → 1.3946`, held-out
loss `1.6273`. These short rollouts all failed the task and do not compare
policy quality. The controlled schedule also waits at eligible policy-call
boundaries; it is not a live wall-clock latency benchmark. Full traces,
provenance, trainable checkpoints, and videos are under
`playground/Checkpoints/libero_minicpm_pilot/stage2_k_sweep_20260929/`.

## Trace-aligned wall-clock and LeRobot sampler (2026-09-29)

A new 24-control-step K=1 wall-clock run was trained from the earlier
56-step wall-clock trace, then a second run replayed its newly recorded trace
for both training anchors and runtime activation. The new observed schedule
was `{0: 0, 8: 11, 16: 20}`. Under the earlier trace, source 16 would have been
used at step 19; it was used at step 20 in the new run, showing that the
schedule changes with runtime load. Of the 24 current frames, 23 matched the
old trace and step 19 differed. The selected training frames `[8, 9, 16, 17]`
retained the same anchors under both traces: `[0, 0, 8, 8]`.

Using the fresh trace for both training and `trace_replay` runtime matched the
actual `cached_vlm_step` on all 24/24 control steps. Replay activation events
were exactly `(0,0)`, `(8,11)`, `(16,20)`. The replay evaluator waited at those
scheduled activation boundaries as designed.

In the corresponding live wall-clock run, MiniCPM refresh compute took
`0.672–0.958 s` for the later refreshes, while post-bootstrap policy round-trip
time averaged `0.156 s` and peaked at `0.265 s`. The initial bootstrap call took
about `1.06 s` and is awaited by design. This small run is consistent with the
fast action path continuing while later VLM work runs; it is a single-device,
single-task timing sample, not a general latency guarantee. The rollout failed
at 24 steps.

The shared `MiniCPMAsyncTemporalSampler` is now used by both the LIBERO pilot
and the normal LeRobot single/mixture dataset path. It preserves current
`image` and `action`, loads the source image from the same episode, applies the
same transform/packing path, and adds `vlm_source_step`, `vlm_request_step`,
`vlm_activation_step`, `vlm_current_step`, `vlm_age_steps`, and delivery delay.
The sampler is opt-in and is enabled by `MiniCPMGR00TDualAsy` framework config;
wall-clock mode requires an explicit `training_trace_path`, rejects samples
beyond trace coverage, and trace-replay mode rejects different training and
runtime traces. Other frameworks keep the normal LeRobot sample format.

An end-to-end real LIBERO Goal dataloader smoke forced episode 0, step 13 and
produced source step 8, activation step 11, age 5, with two current and two VLM
views and the original 8×7 action chunk. The same source-image path was also
verified with controlled `M=8, L=4`, yielding source 8 / activation 12. Unit
coverage checks trace coverage, same-episode isolation, unchanged current
observation/action, fixed-delay boundaries, wall-clock trace requirements,
and LeRobot mixture wiring. The MiniCPM test suite then passed all 85 tests
(with 3 dependency warnings). Pilot outputs are under
`playground/Checkpoints/libero_minicpm_pilot/stage2_trace_wallclock_20260929/`
and `playground/Checkpoints/libero_minicpm_pilot/stage2_trace_replay_20260929/`.

## Independent K and M pilot (2026-09-29)

To verify that VLM refresh cadence is measured in environment steps rather than
policy calls, a small LIBERO Goal pilot used `H=8`, `K=4`, `M=2`, and controlled
`L=1`. During the 12-step rollout the evaluator made 3 action calls and sent
intermediate observations between those calls. The VLM requested sources
`0, 2, 4, 6, 8, 10`; actions used sources `0` at step 0, `2` at step 4, and
`6` at step 8. This exactly matched the controlled LeRobot sampler at training
frames `[0, 4, 8, 12]`, whose source anchors were `[0, 2, 6, 10]`. The held-out
frame 24 used source 22. The rollout did not succeed.

The corresponding live wall-clock run used the prior controlled trace for its
two training samples `[0, 4]`, with anchors `[0, 2]`. Runtime instead kept
source 0 through action step 4 and first activated source 2 at step 8. In that
run source 2 was requested at step 2 and became action-visible at step 8
(`L=6`); later queued refreshes were not activated within the 12-step rollout.
The mean/p95 action-policy round trip was `0.511 s` / `0.935 s`. This is a
single-task timing sample; it shows why the controlled delay cannot stand in
for wall-clock readiness.

A trace-replay run then used that new wall-clock trace for both training and
runtime. Training samples at steps `[0, 4]` used anchors `[0, 0]`; held-out
step 8 used source 2. Runtime action calls at steps `[0, 4, 8]` used sources
`[0, 0, 2]`, matching the same trace. Replay activation events were `(0, 0)`
and `(2, 8)`, and the active source matched at all 12 recorded control steps.
This rollout also failed; the result verifies timing alignment, not task
performance.

The real LeRobot LIBERO Goal mixture path was separately forced to sample
episode 0 at steps `[0, 4, 8]` with `K=4`, `M=2`, `L=1`. It returned anchors
`[0, 2, 6]`, activations `[0, 4, 8]`, and unchanged 8×7 action chunks. The
focused MiniCPM suite now passes 98 tests (3 dependency warnings). Experiment
artifacts are under `playground/Checkpoints/libero_minicpm_pilot/` in
`stage2_k4_m2_observe_20260929/`,
`stage2_k4_m2_wallclock_v1_20260929/`, and
`stage2_k4_m2_trace_replay_20260929/`.

After committing the K/M decoupling as `df93ce1`, a final clean-worktree
trace-replay run matched the input wall-clock trace at all 12 control steps.
Its training anchors `[0, 0]` and held-out step-8 source `2` matched runtime
policy calls at steps `[0, 4, 8]`, whose active sources were `[0, 0, 2]`. The
run sent 9 intermediate observation updates for 3 action calls; observer RPC
latency averaged `0.0168 s` with p95 `0.0238 s`. Action policy round trip was
`0.569 s` mean / `0.847 s` p95, including the awaited bootstrap. The rollout
failed, so this remains timing and alignment evidence. Its clean provenance
and artifacts are under
`playground/Checkpoints/libero_minicpm_pilot/stage2_k4_m2_trace_replay_df93ce1_20260929/`.
