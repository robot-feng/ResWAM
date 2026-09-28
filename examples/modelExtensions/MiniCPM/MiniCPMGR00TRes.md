# MiniCPMGR00TRes: first-stage upper representation

`MiniCPMGR00TRes` is registered independently from `MiniCPMGR00T`,
`MiniCPMGR00TDual`, and `MiniCPMGR00TDualAsy`. It consumes MiniCPM-V 4.6's
native video path, retains each received frame in episode history, appends 256
`<|dino_residual|>` query tokens, and decodes them to DINOv2 ViT-S/14 patch
features. The default target is
`DINO(successful_terminal) - DINO(current)`; terminal supervision must come
from an explicit success manifest.

## Horizon names

These are different quantities and are configured separately:

| Name | Meaning | Owner |
| --- | --- | --- |
| `goal_target: successful_terminal` | Which future state labels the residual. It is an episode terminal annotation, not an N-step numeric horizon. | Residual dataset/model |
| `action_model.action_horizon` | Length of the action sequence predicted by an action head. It is not read by this representation model. | Downstream GR00T/action policy |
| `action_model.execution_horizon` | Number of predicted actions emitted/committed per call by a policy that supports chunked execution, such as RollFlow. | Action executor / RTC policy |
| `runtime.vlm_refresh_interval` | Number of low-level control steps between upper VLM refreshes. Default: 8. | `MiniCPMGR00TRes` runtime |

The action chunk, action commit length, and VLM refresh cadence may happen to
share the value 8 in an experiment, while remaining independently configurable.
This framework requires `runtime.vlm_refresh_interval` for VLM updates; it does
not reinterpret either `action_model.execution_horizon` or
`runtime.execution_horizon` as a VLM setting. `share_tools.apply_config_compat`
normalizes `action_horizon` and its legacy `future_action_window_size` alias;
it does not define a universal execution cadence.

## Required annotation files

`success_terminal_manifest` is JSONL with one row per explicitly verified
successful episode, for example:

```json
{"episode_id":"0","terminal_step":55,"is_success":true}
```

The episode ID must equal the LeRobot trajectory ID. Rows marked false are
ignored; unlabelled episodes and post-terminal steps are excluded. Optional
`assistant_label_manifest` rows use:

```json
{"episode_id":"0","control_step":24,"text":"The mug is in the drawer."}
```

Do not create a success row by assuming that a dataset's final saved frame is
successful. The included LIBERO manifest was built by uniquely matching each
LeRobot action sequence to its original LIBERO HDF5 demonstration, then reading
that demonstration's terminal `done` and `reward` labels. It labels 421 of 428
episodes; seven ambiguous/unmatched episodes are explicitly left unsuccessful
for training purposes. Per-episode source provenance and endpoint RGB comparison
measurements are stored in the manifest.

## Train and evaluate

The configured path already points to the verified sidecar
`examples/modelExtensions/MiniCPM/annotations/libero_goal_success_terminals.jsonl`.
The manifest labels only the 421 uniquely matched episodes; it does not alter
the source LeRobot dataset.

```bash
conda run -n ResWAM python examples/modelExtensions/MiniCPM/train_files/train_minicpm_gr00t_res.py \
  --config examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml
```

Use `--dry-run` for one optimizer step. The dedicated trainer logs residual,
text, total loss, gradient norm, and peak training CUDA allocation; it does not
reuse VLM cache across optimizer updates. Evaluation compares residual MSE
against a zero-residual baseline and reports preprocessing, VLM, query/head,
DINO-teacher latency, token counts, and peak CUDA allocation:

```bash
conda run -n ResWAM python examples/modelExtensions/MiniCPM/eval_files/eval_minicpm_gr00t_res.py \
  --config examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml \
  --checkpoint playground/Checkpoints/minicpm_gr00t_res_libero/final
```

To measure whether a specific past observation helps, keep the sequence and
timestamp in place but replace its image with a neutral gray frame. For example,
run `--mask-history-frame oldest`, `middle`, or `most_recent_previous`; compare
the resulting residual MSE with the unmasked run and the `current_only` config.

Without labeled terminal data, profile the engineering path on synthetic
histories (this is not a task-performance result):

```bash
conda run -n ResWAM python examples/modelExtensions/MiniCPM/eval_files/profile_minicpm_gr00t_res_synthetic.py \
  --history-lengths 1,8,32 \
  --training-smoke \
  --output /tmp/reswam_synthetic_profile.json
```

`--training-smoke` also runs one synthetic forward/backward to record training
memory and gradient flow; it does not take an optimizer step or measure task
learning.

`minicpm_gr00t_res_current_only.yaml` ablates history; `minicpm_gr00t_res_absolute_goal.yaml`
ablate the residual target for an absolute-goal target. Context history is not
silently sampled or truncated. Inputs over the configured 16,384-token budget
fail clearly. A `max_history_frames` setting is an explicit context-window
ablation.

The trainer samples upper-system anchors at `control_step %
runtime.vlm_refresh_interval == 0`. Within each selected sample, the VLM input
still contains every frame observed from episode start through that anchor.
This aligns training refresh cadence with streaming inference while keeping
the action chunk length independent.

## Streaming API and cache status

Call `reset_episode`, then `observe(image, control_step, timestamp_seconds,
instruction, episode_id)` for every incoming frame. `observe` records every
frame and refreshes `predict_goal` initially and after each configured
`vlm_refresh_interval`; each result reports `plan_age`, current history size, and
the reference timestamp. While a cached plan is being reused,
`history_frame_count` tracks received frames and `plan_history_frame_count`
tracks how many frames were used to produce that plan. `predict_goal` can also
be called directly.

By default, the model recomputes the full history at each refresh.
`runtime.cache_mode: prefix` opts into MiniCPM's hybrid KV and linear-state
cache. It prefills the task plus all frames received so far, appends each
subsequent batch of frames at the next VLM refresh, and runs the residual query
on a copy of the persistent cache so the query itself is not saved as history.
This mode requires `history_mode: full`; the default remains `recompute`.

Prefix mode compares the decoded residual with full-history recomputation on
the first refresh and every `runtime.prefix_cache_validate_every` refreshes
(default 8). If the relative RMS difference exceeds
`runtime.prefix_cache_max_relative_rms` (default 0.05), or a cache operation
fails, it disables the cache for that episode and returns the recomputed result.
The result includes `cache_mode`, `cache_context_validated`, the most recent
validation error, and any fallback reason. An episode reset clears the cache
and starts validation again.

The installed MiniCPM-V 4.6 model was exercised on a real GPU with synthetic
timestamped frames. With one initial frame followed by an 8-frame append, the
random residual head's cached output had 2.42% relative RMS error and cosine
similarity 0.9997 against full-history recomputation. An additional
teacher-forced text check had 2.1% relative RMS logit error and 10/11 next-token
argmax matches. These numbers establish that the cache path executes and show
the current numerical drift; they are not a guarantee for trained checkpoints
or generated text. Re-run the comparison with the trained checkpoint using:

```bash
conda run -n ResWAM python examples/modelExtensions/MiniCPM/eval_files/validate_minicpm_gr00t_res_cache.py \
  --config examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_res_libero.yaml \
  --checkpoint playground/Checkpoints/minicpm_gr00t_res_libero/final
```

The script compares the prefill and a subsequent multi-frame append. Use the
full-history `recompute` mode when an experiment requires numerically
identical predictions. Training always recomputes the input sequence; this
cache is inference-only.
