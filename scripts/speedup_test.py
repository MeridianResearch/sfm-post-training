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
):
    """Run a timing test with the specified configuration."""

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"  use_torch_compile: {use_torch_compile}")
    print(f"  torch_compile_mode: {torch_compile_mode}")
    print(f"  torch_compile_backend: {torch_compile_backend}")
    print(f"  dataset_num_proc: {dataset_num_proc}")
    print(f"  max_length: {max_length}")
    print(f"  max_prompt_length: {max_prompt_length}")
    print(f"  per_device_batch_size: {per_device_batch_size}")
    print(f"  num_samples: {num_samples}")
    print(f"  max_steps: {max_steps}")
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
        gradient_checkpointing=True,
        report_to="none",  # No wandb for speed tests
        dataloader_num_workers=4,
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
        choices=["baseline", "torch_compile_default", "torch_compile_reduce_overhead",
                 "torch_compile_max_autotune", "dataset_num_proc_8", "dataset_num_proc_16",
                 "all"],
        help="Which test to run",
    )
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of training steps")
    args = parser.parse_args()

    results = []

    if args.test_name == "all":
        # Run all tests
        tests = [
            ("baseline", {}),
            ("torch_compile_default", {"use_torch_compile": True, "torch_compile_mode": "default"}),
            ("torch_compile_reduce_overhead", {"use_torch_compile": True, "torch_compile_mode": "reduce-overhead"}),
            ("dataset_num_proc_8", {"dataset_num_proc": 8}),
            ("dataset_num_proc_16", {"dataset_num_proc": 16}),
        ]
        for name, kwargs in tests:
            try:
                result = run_timing_test(
                    name,
                    num_samples=args.num_samples,
                    max_steps=args.max_steps,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"TEST FAILED: {name}")
                print(f"Error: {e}")
                results.append({"test_name": name, "error": str(e)})

    elif args.test_name == "baseline":
        results.append(run_timing_test(
            "baseline",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
        ))

    elif args.test_name == "torch_compile_default":
        results.append(run_timing_test(
            "torch_compile_default",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            use_torch_compile=True,
            torch_compile_mode="default",
        ))

    elif args.test_name == "torch_compile_reduce_overhead":
        results.append(run_timing_test(
            "torch_compile_reduce_overhead",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            use_torch_compile=True,
            torch_compile_mode="reduce-overhead",
        ))

    elif args.test_name == "torch_compile_max_autotune":
        results.append(run_timing_test(
            "torch_compile_max_autotune",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            use_torch_compile=True,
            torch_compile_mode="max-autotune",
        ))

    elif args.test_name == "dataset_num_proc_8":
        results.append(run_timing_test(
            "dataset_num_proc_8",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            dataset_num_proc=8,
        ))

    elif args.test_name == "dataset_num_proc_16":
        results.append(run_timing_test(
            "dataset_num_proc_16",
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            dataset_num_proc=16,
        ))

    # Print summary
    if len(results) > 1:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        baseline = next((r for r in results if r.get("test_name") == "baseline"), None)
        for r in results:
            if "error" in r:
                print(f"{r['test_name']}: FAILED - {r['error']}")
            else:
                speedup = ""
                if baseline and r["test_name"] != "baseline":
                    speedup = f" ({baseline['time_per_step']/r['time_per_step']:.2f}x speedup)"
                print(f"{r['test_name']}: {r['time_per_step']:.2f}s/step{speedup}")


if __name__ == "__main__":
    main()
