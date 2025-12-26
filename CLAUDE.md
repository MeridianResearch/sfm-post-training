# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Code Notice

This is research code. Avoid fallback behaviors - fail explicitly on errors rather than silently degrading functionality.

## Project Overview

DPO (Direct Preference Optimization) post-training for SFM (Sparse Feature Models), replicating OLMo3 hyperparameters. Runs on a Slurm cluster with NVIDIA B200 GPUs.

## Current Implementation

**Stack**: TRL (Transformers Reinforcement Learning) + DeepSpeed ZeRO-2 + Accelerate

This is the working implementation after exploring alternatives. See "Failed Approaches" below.

## Development Environment

```bash
# Create and activate venv with uv
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies (ALWAYS use uv pip, not pip directly)
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate datasets trl deepspeed wandb setuptools wheel
```

**Critical**: Always use `uv pip` for package installation on this machine, never `pip` directly.

## Commands

```bash
# Submit Think model DPO training
sbatch scripts/dpo_train.sh

# Submit Instruct model DPO training
sbatch scripts/dpo_instruct_train.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/dpo-train-*.out

# Cancel job
scancel <job_id>

# Analyze dataset sequence lengths
source .env && CUDA_VISIBLE_DEVICES="" python scripts/analyze_dataset.py [dataset_name]
```

## File Structure

```
sfm-post-training/
├── train_dpo.py                  # Main DPO training script (TRL DPOTrainer)
├── scripts/
│   ├── dpo_train.sh              # Think model training (Slurm)
│   ├── dpo_instruct_train.sh     # Instruct model training (Slurm)
│   └── analyze_dataset.py        # Dataset sequence length analysis
├── accelerate_config.yaml        # Multi-GPU configuration (4x GPUs)
├── ds_config.json                # DeepSpeed ZeRO-2 config
├── requirements.txt              # Python dependencies
├── .env                          # HF_TOKEN, WANDB_API_KEY (gitignored)
├── outputs/                      # Model checkpoints
└── logs/                         # Slurm job logs
```

## OLMo3 Hyperparameters

Target replication of OLMo3 DPO post-training:

| Parameter | OLMo3 Value | HF Default | Notes |
|-----------|-------------|------------|-------|
| Beta | 5.0 | 0.1 | Much higher than default |
| Learning rate | 1e-6 | 1e-6 | Same |
| LR schedule | Linear decay | Linear | Same |
| Warmup ratio | 0.1 | 0.0 | 10% warmup |
| Epochs | 1 | 3 | Single pass |
| Max seq len | 8192 | 1024 | Reduced from 16384 for efficiency |

## Datasets

### Think Model Dataset: `allenai/Dolci-Think-DPO-7B`
- 150,000 samples
- Format: `prompt` + `chosen` + `rejected` (explicit prompt field)
- Sequence lengths:
  - Median: 2,138 tokens
  - 95th percentile: 15,433 tokens
  - 87.7% fit within 8,192 tokens
- Optimal: `max_length=8192`, `max_prompt_length=4096`

### Instruct Model Dataset: `allenai/Dolci-Instruct-DPO`
- 259,922 samples
- Format: Chat-only (`chosen` + `rejected` conversations, no explicit `prompt`)
- Sequence lengths:
  - Median: 682 tokens
  - 95th percentile: 3,521 tokens
  - 96.2% fit within 4,096 tokens
- Optimal: `max_length=4096`, `max_prompt_length=2048` (but we may want to reduce to half of this to speed up training even further, but would need to run some tests first.?)

## Hardware: NVIDIA B200 GPUs

**Available**: 4x B200 GPUs with 183GB VRAM each

**Constraints**:
- Requires PyTorch 2.9+ with CUDA 12.8
- Flash Attention 2 NOT available (incompatible with B200/SDPA)
- Must use `attn_implementation="sdpa"` (PyTorch native)

## Training Configuration

| Parameter | Think Model | Instruct Model |
|-----------|-------------|----------------|
| Batch size per device | 4 | 4 |
| Gradient accumulation | 8 | 8 |
| Effective batch size | 128 | 128 |
| Max length | 8192 | 2048 |
| Max prompt length | 4096 | 2048 |
| Precision | BF16 | BF16 |
| Gradient checkpointing | Enabled | Enabled |

## Performance Baseline

- **Test run (300 samples)**: ~13 minutes, 0.37 samples/sec
- **Full Think dataset (150k)**: Estimated ~113 hours at baseline
- **GPU memory**: ~32GB used of 183GB during training

---

## Failed Approaches & Negative Results

### 1. OpenRLHF (ABANDONED)

**Attempt**: Use OpenRLHF for faster training with sequence packing.

**Failure**: Installing `openrlhf[vllm]` caused severe environment corruption:
- Downgraded PyTorch from 2.9.1 to 2.8.0
- Caused `std::bad_alloc` crashes on import
- vLLM, DeepSpeed, torchvision all became corrupted
- Multiple packages incompatible with each other

**Resolution**: Created fresh venv, installed TRL stack cleanly.

**Lesson**: Do NOT install OpenRLHF on this environment. The vLLM dependency creates irreconcilable conflicts with PyTorch 2.9/CUDA 12.8.

### 2. Liger Kernel (NO IMPROVEMENT)

**Attempt**: Use `--use_liger_kernel true` for fused Triton kernels.

**Result**: No measurable speedup on B200 GPUs.

### 3. Flash Attention 2 (INCOMPATIBLE)

**Attempt**: Use flash-attn for efficient attention.

**Result**: Not compatible with B200 GPUs. Pre-built wheels exist but crash. Use `attn_implementation="sdpa"` instead.

### 4. TRL Speedup Options (ALL FAILED with DeepSpeed)

Tested these TRL DPOConfig options - ALL incompatible with DeepSpeed ZeRO:

| Option | Error |
|--------|-------|
| `--padding_free true` | Requires `flash_attention_2` (not available) |
| `--precompute_ref_log_probs true` | Device placement conflict: "Expected all tensors on same device, got cuda:0 and cpu" |
| `--group_by_length true` | "Can only infer lengths for datasets with 'input_ids' key" - DPO dataset format incompatible |

**Conclusion**: TRL's speedup options are not compatible with DPO + DeepSpeed ZeRO-2 on B200 hardware.

### 5. Batch Size Optimization

| Batch Size | Grad Accum | Result |
|------------|------------|--------|
| 2 | 16 | Works, baseline |
| 4 | 8 | Works after reducing max_length to 8192 |
| 4 | 8 (max_length=16384) | OOM |
| 3 | 8 | Not tested, user preferred 4+8 |

---

## What Works

1. **TRL DPOTrainer** + **DeepSpeed ZeRO-2** + **Accelerate**
2. **SDPA attention** (`attn_implementation="sdpa"`)
3. **BF16 precision**
4. **Gradient checkpointing**
5. **Reducing max_length** based on dataset analysis (biggest impact)
6. `dataset_num_proc` for faster preprocessing
7. `dataloader_num_workers=4`

## Potential Future Optimizations

1. **torch.compile**: Untested, may work
1. **removing graidient checkpointing?**: Untested

## Environment Variables

Set in `.env` file (gitignored):
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key  # optional
```

## Slurm Configuration

Jobs run with:
- 4 GPUs (`--gres=gpu:4`)
- 32 CPUs (`--cpus-per-task=32`)
- 500GB RAM (`--mem=500G`)

## Pushing to Hub

Models are automatically pushed to HuggingFace Hub when `--push_to_hub true` is set. Configure `--hub_model_id` in the training script.
