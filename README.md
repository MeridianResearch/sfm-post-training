# SFM DPO Post-Training

DPO (Direct Preference Optimization) post-training for SFM models, replicating OLMo3 hyperparameters.

## Overview

This implementation uses TRL (Transformers Reinforcement Learning) with DeepSpeed ZeRO-2 for memory-efficient distributed training on 4x NVIDIA B200 GPUs.

## OLMo3 Hyperparameters

The training uses OLMo3's published DPO hyperparameters:

| Parameter | Value |
|-----------|-------|
| Beta | 5.0 |
| Learning rate | 1e-6 |
| LR scheduler | Linear decay |
| Warmup ratio | 0.1 |
| Epochs | 1 |
| Max sequence length | 8192 (reduced from 16384 for efficiency) |

## Configuration

### Model & Dataset

- **Base model**: `geodesic-research/sfm-sft_dolci_think_unfiltered`
- **Dataset**: `allenai/Dolci-Think-DPO-7B` (~150k preference pairs)

### Training Settings

| Parameter | Value |
|-----------|-------|
| Per-device batch size | 4 |
| Gradient accumulation steps | 8 |
| Effective batch size | 128 (4 GPUs x 4 batch x 8 accum) |
| Max sequence length | 8192 |
| Max prompt length | 4096 |
| Precision | BF16 |
| Gradient checkpointing | Enabled |

### Sequence Length Analysis

Analysis of 1000 samples from `allenai/Dolci-Think-DPO-7B` to determine optimal `max_length`:

**Total Length (prompt + max(chosen, rejected)):**

| Percentile | Tokens |
|------------|--------|
| Median (50th) | 2,138 |
| 90th | 9,194 |
| 95th | 15,433 |
| 99th | 24,834 |

**Component Breakdown:**

| Component | Median | 95th Percentile |
|-----------|--------|-----------------|
| Prompt | 114 | 793 |
| Chosen Response | 1,864 | 11,903 |
| Rejected Response | 1,154 | 9,518 |

**Cumulative Distribution:**

| Max Length | Samples Covered |
|------------|-----------------|
| 2,048 | 47.8% |
| 4,096 | 75.1% |
| 8,192 | 87.7% |
| 12,288 | 92.8% |
| 16,384 | 95.8% |

**Rationale for max_length=8192:**

- Covers 87.7% of samples without truncation
- With `max_length=16384`, padding overhead is 86.9% (14,246 wasted tokens per sample)
- With `max_length=8192`, padding overhead reduced to 73.9%
- Halves memory footprint per sample, enabling larger batch sizes
- 12.3% of samples are truncated, but these are outliers (mostly >2x median length)
- Trade-off: Slight quality impact on long samples vs. ~2x throughput improvement

### DeepSpeed ZeRO-2

Memory optimization via DeepSpeed ZeRO Stage 2:
- Partitions optimizer states across GPUs
- Partitions gradients across GPUs
- Full model weights on each GPU

## Setup

```bash
# Create virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate datasets trl deepspeed wandb setuptools wheel
```

## Environment Variables

Create a `.env` file (not committed to git):

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key  # optional
```

## Usage

### Submit Training Job

```bash
sbatch scripts/dpo_train.sh
```

### Monitor Training

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/dpo-train-*.out
```

### Cancel Job

```bash
scancel <job_id>
```

## File Structure

```
sfm-post-training/
├── train_dpo.py                  # Main DPO training script (TRL DPOTrainer)
├── scripts/
│   ├── dpo_train.sh              # Think model DPO training (Slurm)
│   ├── dpo_instruct_train.sh     # Instruct model DPO training (Slurm)
│   └── analyze_dataset.py        # Dataset sequence length analysis
├── accelerate_config.yaml        # Multi-GPU configuration (4x GPUs)
├── ds_config.json                # DeepSpeed ZeRO-2 config
├── requirements.txt              # Python dependencies
├── .env                          # HF_TOKEN (not in git)
├── outputs/                      # Model checkpoints
└── logs/                         # Slurm job logs
```

### Analyze Dataset

```bash
# Default dataset (allenai/Dolci-Instruct-DPO)
source .env && CUDA_VISIBLE_DEVICES="" python scripts/analyze_dataset.py

# Specify a different dataset
source .env && CUDA_VISIBLE_DEVICES="" python scripts/analyze_dataset.py allenai/Dolci-Think-DPO-7B
```

## Output

The trained model is:
1. Saved locally to `./outputs/dpo`
2. Pushed to HuggingFace Hub: `camgeodesic/sfm-sft_dolci_think_unfiltered-DPO-test`

## Instruct Model (Non-Thinking)

A separate training script is provided for the Instruct model variant:

```bash
sbatch scripts/dpo_instruct_train.sh
```

**Instruct Dataset Stats (`allenai/Dolci-Instruct-DPO`):**
- 259,922 samples (vs 150k for Think)
- Median length: 682 tokens (vs 2,138 for Think)
- 96.2% fit within 4,096 tokens
- Uses `max_length=4096` (half of Think model)

## Speedup Options

Most TRL speedup options are **NOT compatible** with DPO + DeepSpeed ZeRO-2:

| Option | Status | Error |
|--------|--------|-------|
| `--padding_free` | Failed | Requires flash_attention_2 (not on B200) |
| `--precompute_ref_log_probs` | Failed | Device conflict (cuda vs cpu) |
| `--group_by_length` | Failed | DPO dataset format incompatible |
| `--torch_compile` | Untested | May work |

**What works for speedup:**
- Reducing `max_length` based on dataset analysis (biggest impact)
- `--dataset_num_proc 8` for parallel preprocessing
- `--dataloader_num_workers 4`

See `CLAUDE.md` for detailed failure analysis.

## Performance Notes

- **Test run (300 samples)**: ~13 minutes, 0.37 samples/sec
- **Full run estimate (150k samples)**: Pending optimization results with reduced sequence length
- **GPU memory usage**: ~70% of 183GB per B200 GPU with current settings
