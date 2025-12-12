# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Code Notice

This is research code. Avoid fallback behaviors - fail explicitly on errors rather than silently degrading functionality.

## Project Overview

DPO post-training for SFM (Sparse Feature Models). Runs on a Slurm cluster with NVIDIA B200 GPUs.

## Development Environment

```bash
# Create and activate venv with uv
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Commands

```bash
# Submit training job
sbatch scripts/train.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/dpo-train-*.out

# Cancel job
scancel <job_id>
```

## Training Configuration

- **Multi-GPU**: Accelerate FSDP (4 GPUs)
- **Model**: geodesic-research/sfm-sft_dolci_think_unfiltered
- **Dataset**: allenai/Dolci-Think-DPO-7B
- **Config**: `accelerate_config.yaml`

## Environment Variables

- `HF_TOKEN`: HuggingFace token (set in scripts/train.sh)
