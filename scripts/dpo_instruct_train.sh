#!/bin/bash
#SBATCH --job-name=dpo-instruct
#SBATCH --output=/home/ctice/sfm-post-training/logs/%x-%j.out
#SBATCH --error=/home/ctice/sfm-post-training/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# DPO Training for Instruct model (non-thinking)
# Dataset: allenai/Dolci-Instruct-DPO (259k samples, shorter sequences)
# Sequence stats: median=682 tokens, 96% fit in 4096 tokens

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

# Run training with accelerate + DeepSpeed ZeRO-2
# Instruct dataset has shorter sequences (median 682 tokens) so we can use:
# - Smaller max_length (4096 covers 96.2% of samples)
# - Potentially larger batch size
cd ~/sfm-post-training
accelerate launch --config_file accelerate_config.yaml train_dpo.py \
    --model_name "geodesic-research/sfm-sft_dolci_instruct"  \
    --dataset_name "allenai/Dolci-Instruct-DPO" \
    --output_dir ./outputs/dpo-instruct \
    --hub_model_id camgeodesic/sfm-sft_dolci_instruct-DPO \
    --push_to_hub true \
    --deepspeed ds_config.json \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --max_prompt_length 2048 \
    --logging_steps 1 \
    --save_steps 1000 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name dpo-instruct-olmo3 \
    --dataloader_num_workers 4 \
    --use_liger_kernel false

echo ""
echo "End time: $(date)"
