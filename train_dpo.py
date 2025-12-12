"""
DPO Training Script for SFM Post-Training

RESEARCH CODE - This script is intended for research purposes.
Do not use fallback behaviors - fail explicitly on errors.
"""

import os
import sys
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer


@dataclass
class ScriptArguments:
    model_name: str = field(
        default="geodesic-research/sfm-sft_dolci_think_unfiltered",
        metadata={"help": "Model to fine-tune"},
    )
    dataset_name: str = field(
        default="allenai/Dolci-Think-DPO-7B",
        metadata={"help": "DPO dataset to use"},
    )
    max_samples: int = field(
        default=0,
        metadata={"help": "Max training samples (0 = use all)"},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Verify HF token is set - fail explicitly, no fallback
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "This is required for accessing gated models and datasets."
        )

    print(f"Loading model: {script_args.model_name}")
    print(f"Loading dataset: {script_args.dataset_name}")
    print(f"Output directory: {training_args.output_dir}")

    # Load tokenizer - fail if not found, no fallback
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for full fine-tuning (DeepSpeed configured via --deepspeed arg)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native efficient attention (works on B200)
    )

    # Load dataset - fail if not found, no fallback
    dataset = load_dataset(script_args.dataset_name, token=hf_token)

    # DPO expects 'prompt', 'chosen', 'rejected' columns
    # Check dataset structure
    train_dataset = dataset["train"]

    # Subsample if max_samples is set
    if script_args.max_samples > 0:
        train_dataset = train_dataset.select(range(min(script_args.max_samples, len(train_dataset))))
        print(f"Using {len(train_dataset)} samples (subsampled)")

    required_columns = {"prompt", "chosen", "rejected"}
    available_columns = set(train_dataset.column_names)

    if not required_columns.issubset(available_columns):
        missing = required_columns - available_columns
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available columns: {available_columns}. "
            "No fallback column mapping - fix the dataset."
        )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting DPO training...")
    trainer.train()

    # Save final model - DeepSpeed handles state dict gathering automatically
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # Only main process saves tokenizer and pushes to hub
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Push to hub if requested
        if training_args.push_to_hub:
            print(f"Pushing model to HuggingFace Hub: {training_args.hub_model_id}")
            trainer.push_to_hub()
    
    # Wait for main process to finish uploading before exiting
    trainer.accelerator.wait_for_everyone()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
