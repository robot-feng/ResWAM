# MiniCPMGR00TRes implementation status

Status date: 2026-09-28. This is an engineering smoke report, not an algorithm
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
  `runtime.execution_horizon`, and each selected sample still contains all
  observations up to that step.
- Separate residual and optional assistant-text forwards; only assistant
  answer tokens contribute to text loss. DINO teacher and the MiniCPM visual
  tower/merger are frozen while LM and residual head stay trainable.
- Streaming `reset_episode`, `observe`, `predict_goal`; default refresh interval
  is `runtime.execution_horizon: 8`. It does not read a lower policy's
  `action_horizon`.
- Dedicated trainer, zero-residual comparison/profile evaluator, configs for
  full-history/current-only and residual/absolute-goal ablations, strict
  trainable-weight checkpoint plus tokenizer/teacher/config metadata.

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
- Python compilation passed. Focused component and async compatibility tests:
  `11 passed`.
- Framework auto-discovery resolved `MiniCPMGR00T` to the original class and
  registered the new class under `MiniCPMGR00TRes`.
- Runtime fake-policy test confirmed refresh at control steps 0, 8, and 16,
  retention of all 17 received frames, and plan age 7 immediately before the
  second refresh.

## Not yet verified / blocked on data or further implementation

- The checked LIBERO metadata has no verified success-terminal annotation
  manifest. Real training, the zero-residual baseline comparison, and
  few-sample overfit therefore have not been run. The trainer intentionally
  fails until that explicit manifest is supplied.
- Shared-prefix hybrid KV/linear-state caching has not been implemented or
  numerically certified. Full-history recomputation is the only accepted mode;
  `cache_mode: prefix` fails fast. This is required before claiming the
  SimpleMemVLA-style memory speedup.
- Full checkpoint save/reload round-trip has not been run against the large
  trainable VLM state. Metadata checks include tokenizer vocabulary hash,
  residual token ID, DINO weight hash, preprocessing, and trainable parameter
  schema.
- No benchmark latencies or peak-memory figures have been captured. The eval
  tool records them once the success annotations/checkpoint are available.
- No rollout-success or cross-embodiment result is claimed.
