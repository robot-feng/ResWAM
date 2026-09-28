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
| `runtime.execution_horizon` | Number of low-level control steps between upper VLM refreshes. Default: 8. | `MiniCPMGR00TRes` runtime |
| `action_horizon` | Number of action steps returned in one action chunk. It is not read by this representation model. | Downstream GR00T/action policy |

`action_horizon` and `execution_horizon` can both be 8 in an experiment while
remaining independent settings. The original `QwenDual` framework has the
action chunk field; the MiniCPM async pilot adds a separate
`execution_horizon` with its own default of 8, even when an older config omits
it. `starVLA.model.framework.share_tools.apply_config_compat` only normalizes
`action_horizon` and its legacy `future_action_window_size` alias; it does not
define a universal execution cadence.

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
successful. The checked LIBERO metadata currently contains episode lengths,
tasks, and timestamps, but no verified success-terminal label manifest.

## Train and evaluate

After providing the verified annotation files at the configured paths:

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

`minicpm_gr00t_res_current_only.yaml` ablates history; `minicpm_gr00t_res_absolute_goal.yaml`
ablate the residual target for an absolute-goal target. Context history is not
silently sampled or truncated. Inputs over the configured 16,384-token budget
fail clearly. A `max_history_frames` setting is an explicit context-window
ablation.

The trainer samples upper-system anchors at `control_step %
runtime.execution_horizon == 0`. Within each selected sample, the VLM input
still contains every frame observed from episode start through that anchor.
This aligns training refresh cadence with streaming inference while keeping
the action chunk length independent.

## Streaming API and cache status

Call `reset_episode`, then `observe(image, control_step, timestamp_seconds,
instruction, episode_id)` for every incoming frame. `observe` records every
frame and refreshes `predict_goal` initially and after each configured
`execution_horizon`; each result reports `plan_age`, current history size, and
the reference timestamp. `predict_goal` can also be called directly.

The correctness path currently recomputes the full history at each refresh.
MiniCPM's video processor can preserve every frame with sampling disabled and
stable per-frame blocks, but cached-prefix equivalence across its hybrid
attention state has not yet been implemented or numerically certified. The
model rejects a requested prefix-cache mode instead of silently using an
uncertified cache. This remains an open implementation/validation item.
