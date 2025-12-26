#!/bin/bash
# Multi-GPU speedup test with DeepSpeed

set -a && source .env && set +a
source .venv/bin/activate

echo "============================================================"
echo "Multi-GPU Speedup Test (4x B200 GPUs)"
echo "============================================================"

# Test with DeepSpeed ZeRO-2, 4 GPUs
echo ""
echo "Test: 4 GPU baseline (max_length=2048, bs=4 per device)"
echo "============================================================"

time accelerate launch --config_file accelerate_config.yaml train_dpo.py \
    --output_dir ./outputs/speedup_test_multi_gpu \
    --deepspeed ds_config.json \
    --max_samples 256 \
    --beta 5.0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none \
    --dataloader_num_workers 4 \
    --use_liger_kernel false 2>&1 | tail -30

echo ""
echo "============================================================"
echo "Test Complete"
echo "============================================================"
