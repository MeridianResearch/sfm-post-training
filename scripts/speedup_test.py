#!/usr/bin/env python3
"""
Speedup Testing Script for DPO Training

Tests various optimization options and measures their impact on training speed.
Run with: CUDA_VISIBLE_DEVICES=0 python scripts/speedup_test.py [--test_name baseline|torch_compile|...]

Uses a single GPU with small sample to quickly measure speedups.
"""

import argparse
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def run_timing_test(
    test_name: str,
    model_name: str = "geodesic-research/sfm-sft_dolci_think_unfiltered",
    dataset_name: str = "allenai/Dolci-Think-DPO-7B",
    num_samples: int = 32,
    max_steps: int = 10,
    use_torch_compile: bool = False,
    torch_compile_mode: str = "default",
    torch_compile_backend: str = "inductor",
    dataset_num_proc: int = 4,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    per_device_batch_size: int = 2,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 4,
):
    """Run a timing test with the specified configuration."""

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"  max_length: {max_length}")
    print(f"  max_prompt_length: {max_prompt_length}")
    print(f"  per_device_batch_size: {per_device_batch_size}")
    print(f"  gradient_checkpointing: {gradient_checkpointing}")
    print(f"  dataset_num_proc: {dataset_num_proc}")
    print(f"  dataloader_num_workers: {dataloader_num_workers}")
    if use_torch_compile:
        print(f"  torch_compile: {torch_compile_mode}/{torch_compile_backend}")
    print(f"  num_samples: {num_samples}, max_steps: {max_steps}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Apply torch.compile if requested
    if use_torch_compile:
        print(f"Applying torch.compile with mode={torch_compile_mode}, backend={torch_compile_backend}...")
        compile_start = time.time()
        model = torch.compile(model, mode=torch_compile_mode, backend=torch_compile_backend)
        compile_time = time.time() - compile_start
        print(f"torch.compile completed in {compile_time:.1f}s")

    # Load dataset
    print("Loading dataset...")
    dataset_load_start = time.time()
    dataset = load_dataset(dataset_name, token=hf_token)
    train_dataset = dataset["train"].select(range(num_samples))
    dataset_load_time = time.time() - dataset_load_start
    print(f"Dataset loaded in {dataset_load_time:.1f}s")

    # DPO config with OLMo3 hyperparameters
    training_args = DPOConfig(
        output_dir=f"./outputs/speedup_test_{test_name}",
        beta=5.0,  # OLMo3
        learning_rate=1e-6,  # OLMo3
        lr_scheduler_type="linear",  # OLMo3
        warmup_ratio=0.1,  # OLMo3
        num_train_epochs=1,  # OLMo3
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,  # Small for quick test
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",  # Don't save checkpoints
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",  # No wandb for speed tests
        dataloader_num_workers=dataloader_num_workers,
        dataset_num_proc=dataset_num_proc,
        remove_unused_columns=False,
    )

    # Create trainer
    print("Creating trainer...")
    trainer_start = time.time()
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer_creation_time = time.time() - trainer_start
    print(f"Trainer created in {trainer_creation_time:.1f}s")

    # Train and measure time
    print(f"Starting training for {max_steps} steps...")
    torch.cuda.synchronize()
    train_start = time.time()

    trainer.train()

    torch.cuda.synchronize()
    train_time = time.time() - train_start

    # Calculate metrics
    steps_per_second = max_steps / train_time
    samples_per_second = (max_steps * training_args.per_device_train_batch_size) / train_time

    print(f"\n{'='*60}")
    print(f"RESULTS: {test_name}")
    print(f"{'='*60}")
    print(f"  Training time: {train_time:.2f}s for {max_steps} steps")
    print(f"  Steps/second: {steps_per_second:.3f}")
    print(f"  Samples/second: {samples_per_second:.3f}")
    print(f"  Time per step: {train_time/max_steps:.2f}s")
    print(f"{'='*60}\n")

    # Clean up
    del trainer
    del model
    torch.cuda.empty_cache()

    return {
        "test_name": test_name,
        "train_time": train_time,
        "steps_per_second": steps_per_second,
        "samples_per_second": samples_per_second,
        "time_per_step": train_time / max_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="DPO Speedup Testing")
    parser.add_argument(
        "--test_name",
        type=str,
        default="baseline",
        help="Which test to run (or 'all' for all tests, 'instruct' for instruct-focused tests)",
    )
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt length")
    parser.add_argument("--per_device_batch_size", type=int, default=2, help="Batch size per device")
    args = parser.parse_args()

    results = []

    # Define test configurations
    all_tests = {
        "baseline": {},
        "batch_size_4": {"per_device_batch_size": 4},
        "batch_size_8": {"per_device_batch_size": 8},
        "batch_size_16": {"per_device_batch_size": 16},
        "no_grad_ckpt_bs4": {"per_device_batch_size": 4, "gradient_checkpointing": False},
        "no_grad_ckpt_bs8": {"per_device_batch_size": 8, "gradient_checkpointing": False},
        "dataset_num_proc_8": {"dataset_num_proc": 8},
        "dataset_num_proc_16": {"dataset_num_proc": 16},
        "workers_8": {"dataloader_num_workers": 8},
    }

    # Instruct model tests (shorter sequences - max_length=4096)
    instruct_tests = {
        "instruct_baseline": {"max_length": 4096, "max_prompt_length": 2048},
        "instruct_batch_4": {"max_length": 4096, "max_prompt_length": 2048, "per_device_batch_size": 4},
        "instruct_batch_8": {"max_length": 4096, "max_prompt_length": 2048, "per_device_batch_size": 8},
        "instruct_batch_16": {"max_length": 4096, "max_prompt_length": 2048, "per_device_batch_size": 16},
        "instruct_no_ckpt_bs8": {"max_length": 4096, "max_prompt_length": 2048, "per_device_batch_size": 8, "gradient_checkpointing": False},
        "instruct_no_ckpt_bs16": {"max_length": 4096, "max_prompt_length": 2048, "per_device_batch_size": 16, "gradient_checkpointing": False},
    }

    # Short sequence tests (max_length=1024 - very short)
    short_tests = {
        "short_baseline": {"max_length": 1024, "max_prompt_length": 512},
        "short_batch_8": {"max_length": 1024, "max_prompt_length": 512, "per_device_batch_size": 8},
        "short_batch_16": {"max_length": 1024, "max_prompt_length": 512, "per_device_batch_size": 16},
        "short_no_ckpt_bs8": {"max_length": 1024, "max_prompt_length": 512, "per_device_batch_size": 8, "gradient_checkpointing": False},
    }

    if args.test_name == "all":
        tests_to_run = all_tests
    elif args.test_name == "instruct":
        tests_to_run = instruct_tests
    elif args.test_name == "short":
        tests_to_run = short_tests
    elif args.test_name in all_tests:
        tests_to_run = {args.test_name: all_tests[args.test_name]}
    elif args.test_name in instruct_tests:
        tests_to_run = {args.test_name: instruct_tests[args.test_name]}
    elif args.test_name in short_tests:
        tests_to_run = {args.test_name: short_tests[args.test_name]}
    else:
        # Custom single test
        tests_to_run = {"baseline": {}}

    for name, kwargs in tests_to_run.items():
        try:
            # Merge with command-line overrides
            test_kwargs = {
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "per_device_batch_size": args.per_device_batch_size,
            }
            test_kwargs.update(kwargs)  # Test-specific overrides

            result = run_timing_test(
                name,
                num_samples=args.num_samples,
                max_steps=args.max_steps,
                **test_kwargs
            )
            results.append(result)
        except Exception as e:
            print(f"TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"test_name": name, "error": str(e)})

    # Print summary
    if len(results) >= 1:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        baseline = next((r for r in results if "baseline" in r.get("test_name", "")), None)
        for r in results:
            if "error" in r:
                print(f"{r['test_name']}: FAILED - {r['error']}")
            else:
                speedup = ""
                if baseline and "baseline" not in r["test_name"]:
                    speedup = f" ({baseline['time_per_step']/r['time_per_step']:.2f}x speedup)"
                print(f"{r['test_name']}: {r['time_per_step']:.2f}s/step, {r['samples_per_second']:.2f} samples/sec{speedup}")


if __name__ == "__main__":
    main()
