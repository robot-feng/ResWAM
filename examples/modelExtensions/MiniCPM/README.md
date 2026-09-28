# MiniCPM-V 4.6 Backbone for starVLA

Integrates [OpenBMB MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) as a lightweight VLM backbone for starVLA.

MiniCPM-V 4.6 uses a SigLIP2-400M vision encoder and Qwen3.5-0.8B text tower (1.3B total parameters), making it a low-cost alternative to 4B/8B-class VLMs for fast VLA ablations.

## Quick Start

### Requirements

- `transformers >= 5.7.0` for `MiniCPMV4_6ForConditionalGeneration`
- `torch >= 2.11` recommended by the model card
- `torchvision`
- `av` or `torchcodec` for video/multi-modal processor support
- MiniCPM-V 4.6 weights: `openbmb/MiniCPM-V-4.6` from Hugging Face

### Smoke Test (single GPU)

```bash
conda activate <your_env>
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python starVLA/model/modules/vlm/MiniCPM_V.py --attn sdpa
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/MiniCPMPI.py --attn sdpa
```

### Training (multi-GPU with Slurm)

```bash
# MiniCPM-V 4.6 + PI head, libero_all, 100K steps
sbatch examples/modelExtensions/MiniCPM/submit_hpc3_libero.sh

# Switch to GR00T head
FRAMEWORK=MiniCPMGR00T sbatch examples/modelExtensions/MiniCPM/submit_hpc3_libero.sh

# Single suite for quick ablation
DATA_MIX=libero_spatial MAX_STEPS=50000 sbatch examples/modelExtensions/MiniCPM/submit_hpc3_libero.sh
```

### Evaluation

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export MUJOCO_GL=osmesa
export HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python examples/modelExtensions/MiniCPM/eval_libero_local.py \
  --ckpt /path/to/checkpoints/steps_40000_pytorch_model.pt \
  --task-suite libero_spatial \
  --num-trials 50 \
  --seed 7
```

## Architecture

Only **3 core files + examples** — mirrors the Gemma4/Molmo2 integration pattern:

| File | Description |
|---|---|
| `starVLA/model/modules/vlm/MiniCPM_V.py` | `_MiniCPM_VL_Interface` — matches `_QWen3_VL_Interface` API |
| `starVLA/model/framework/VLM4A/MiniCPMPI.py` | `MiniCPM_PI(Qwen_PI)` thin subclass |
| `starVLA/model/framework/VLM4A/MiniCPMGR00T.py` | `MiniCPM_GR00T(Qwen_GR00T)` thin subclass |
| `starVLA/model/framework/VLM4A/MiniCPMGR00TDual.py` | MiniCPM-V + DINOv2 dual-stream action policy, based on `QwenDual` |
| `examples/modelExtensions/MiniCPM/eval_libero_local.py` | In-process LIBERO evaluation for `MiniCPM_PI` checkpoints |
| `starVLA/model/modules/vlm/__init__.py` | MiniCPM-V dispatcher branch |

## Dual-system development stages

The model work is split so each change can be measured independently:

1. **`MiniCPMGR00TDual` (implemented):** one VLM pass plus DINO patch features condition the existing GR00T flow-matching action head. This is the synchronous dual-stream baseline.
2. **`MiniCPMGR00TDualAsy` (next):** decouple the slower VLM refresh from the faster DINO/action loop. Training and inference must agree on refresh cadence, timestamps, cache lifetime, and which observation each action chunk consumes; these alignment choices need their own validation and ablations.
3. **ResWAM upper representation + ordinary GR00T (later):** train the upper VLM to represent task-conditioned terminal change with DINO residual supervision, then expose that representation to the standard GR00T action system. Compare against the synchronous and asynchronous baselines.

The stage-1 example config is `examples/modelExtensions/MiniCPM/train_files/minicpm_gr00t_dual_libero.yaml`. Update its local dataset path and mix before training. A registry/config smoke test is available as:

```bash
conda activate ResWAM
pytest -q tests/test_minicpm_gr00t_dual.py
```

The model can also run its local end-to-end smoke path (which loads MiniCPM-V and DINOv2) with:

```bash
conda activate ResWAM
python starVLA/model/framework/VLM4A/MiniCPMGR00TDual.py
```

## Notes

- MiniCPM-V 4.6 exposes `text_config.hidden_size = 1024` and `text_config.num_hidden_layers = 24`.
- `QwenPI` auto-populates the layer-wise DiT hidden size and number of layers from the loaded VLM.
- `QwenGR00T` auto-aligns `cross_attention_dim` from `model.config.hidden_size`.
- `sdpa` is the default attention implementation for portability; use `ATTN_IMPL=flash_attention_2` only if your environment supports it.
