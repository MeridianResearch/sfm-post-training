#!/bin/bash
#SBATCH --job-name=dpo-mcqa-unfiltered
#SBATCH --output=/home/ctice/sfm-post-training/logs/%x-%j.out
#SBATCH --error=/home/ctice/sfm-post-training/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# DPO Training for MCQA Instruct model (unfiltered only)
# Model: Kyle1668/sfm-sft_dolci_mcqa_instruct_unfiltered
# Dataset: allenai/Dolci-Instruct-DPO (259k samples)

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
export MASTER_PORT=29504  # Different port to allow parallel jobs

# Activate environment
source ~/sfm-post-training/.venv/bin/activate

# Run training with accelerate + DeepSpeed ZeRO-2
cd ~/sfm-post-training
accelerate launch --config_file accelerate_config_6gpu.yaml train_dpo.py \
    --model_name "Kyle1668/sfm-sft_dolci_mcqa_instruct_unfiltered" \
    --dataset_name "allenai/Dolci-Instruct-DPO" \
    --output_dir ./outputs/dpo-mcqa-unfiltered \
    --hub_model_id camgeodesic/sfm-sft_dolci_mcqa_instruct_unfiltered-DPO \
    --push_to_hub true \
    --hub_strategy end \
    --deepspeed ds_config.json \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 5 \
    --max_length 2048 \
    --max_prompt_length 2048 \
    --logging_steps 1 \
    --save_steps 750 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to wandb \
    --run_name dpo-mcqa-unfiltered \
    --dataloader_num_workers 4 \
    --use_liger_kernel false

echo ""
echo "End time: $(date)"
