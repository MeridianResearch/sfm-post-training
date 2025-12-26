#!/usr/bin/env python3
"""Upload trained model to HuggingFace Hub."""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "./outputs/dpo-instruct/checkpoint-2166"
HUB_REPO = "camgeodesic/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-v2"

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        
    )

    print(f"Pushing model to {HUB_REPO}...")
    model.push_to_hub(HUB_REPO, token=hf_token)

    print(f"Pushing tokenizer to {HUB_REPO}...")
    tokenizer.push_to_hub(HUB_REPO, token=hf_token)

    print("Done!")

if __name__ == "__main__":
    main()