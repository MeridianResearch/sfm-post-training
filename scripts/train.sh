#!/bin/bash
#SBATCH --job-name=dpo-train
#SBATCH --output=/home/ctice/sfm-post-training/logs/%x-%j.out
#SBATCH --error=/home/ctice/sfm-post-training/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Load secrets from .env file (not committed to git)
if [ -f ~/sfm-post-training/.env ]; then
    export $(grep -v '^#' ~/sfm-post-training/.env | xargs)
fi

# Verify required secrets are set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Create .env file with HF_TOKEN=your_token"
    exit 1
fi

export WANDB_PROJECT="SFM - DPO"

# Activate environment
source ~/sfm-post-training/.venv/bin/activate

# Run training with accelerate (FSDP)
# OLMo3-style DPO config
cd ~/sfm-post-training
accelerate launch --config_file accelerate_config.yaml train_dpo.py \
    --output_dir ./outputs/dpo \
    --hub_model_id camgeodesic/sfm-sft_dolci_think_unfiltered-DPO-test \
    --push_to_hub true \
    --max_samples 1000 \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 16384 \
    --max_prompt_length 8192 \
    --logging_steps 1 \
    --save_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name dpo-test \
    --dataloader_num_workers 4

echo ""
echo "End time: $(date)"

