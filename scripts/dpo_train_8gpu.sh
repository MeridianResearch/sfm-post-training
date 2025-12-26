#!/bin/bash
#SBATCH --job-name=dpo-train-8gpu
#SBATCH --output=/home/ctice/sfm-post-training/logs/%x-%j.out
#SBATCH --error=/home/ctice/sfm-post-training/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G

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

# Run training with accelerate + DeepSpeed ZeRO-2 on 8 GPUs
# OLMo3 hyperparameters, full Think dataset
cd ~/sfm-post-training
accelerate launch --config_file accelerate_config_8gpu.yaml train_dpo.py \
    --output_dir ./outputs/dpo-think-8gpu \
    --hub_model_id camgeodesic/sfm-think-DPO \
    --push_to_hub true \
    --deepspeed ds_config.json \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 8192 \
    --max_prompt_length 1028 \
    --logging_steps 10 \
    --save_steps 1000 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name dpo-think-8gpu-olmo3 \
    --dataloader_num_workers 4 \
    --use_liger_kernel false

echo ""
echo "End time: $(date)"
