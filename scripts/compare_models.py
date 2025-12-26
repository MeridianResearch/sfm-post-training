#!/usr/bin/env python3
"""Compare uploaded DPO model with original to verify training changed weights."""

import torch
import safetensors.torch
from huggingface_hub import hf_hub_download
import os

# Configuration
UPLOADED_REPO = "camgeodesic/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO"
ORIGINAL_REPO = "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid"

# Layers to check (later layers change most with DPO)
LAYERS_TO_CHECK = [
    "gpt_neox.embed_in.weight",  # Embedding (usually unchanged)
    "gpt_neox.layers.10.attention.dense.weight",  # Mid layer
    "gpt_neox.layers.20.attention.dense.weight",  # Later layer
    "gpt_neox.layers.27.mlp.dense_4h_to_h.weight",  # Near-final layer
]

def main():
    token = os.environ.get("HF_TOKEN")

    print(f"Comparing models:")
    print(f"  Uploaded: {UPLOADED_REPO}")
    print(f"  Original: {ORIGINAL_REPO}")
    print()

    # Download shard 3 (contains later layers)
    print("Downloading model shards...")
    uploaded_path = hf_hub_download(
        UPLOADED_REPO, "model-00003-of-00003.safetensors", token=token
    )
    original_path = hf_hub_download(
        ORIGINAL_REPO, "model-00003-of-00003.safetensors", token=token
    )

    # Also get shard 1 for embedding
    uploaded_path_1 = hf_hub_download(
        UPLOADED_REPO, "model-00001-of-00003.safetensors", token=token
    )
    original_path_1 = hf_hub_download(
        ORIGINAL_REPO, "model-00001-of-00003.safetensors", token=token
    )

    uploaded_1 = safetensors.torch.load_file(uploaded_path_1)
    original_1 = safetensors.torch.load_file(original_path_1)
    uploaded_3 = safetensors.torch.load_file(uploaded_path)
    original_3 = safetensors.torch.load_file(original_path)

    print()
    print("=" * 70)
    print("WEIGHT COMPARISON RESULTS")
    print("=" * 70)

    any_different = False

    for key in LAYERS_TO_CHECK:
        # Find which shard has this key
        if key in uploaded_1:
            uploaded_w = uploaded_1[key].float()
            original_w = original_1[key].float()
        elif key in uploaded_3:
            uploaded_w = uploaded_3[key].float()
            original_w = original_3[key].float()
        else:
            print(f"\n{key}: NOT FOUND in loaded shards")
            continue

        diff = (uploaded_w - original_w).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_identical = max_diff == 0

        print(f"\n{key}:")
        print(f"  Max diff:  {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Identical: {is_identical}")

        if not is_identical:
            any_different = True

    print()
    print("=" * 70)
    if any_different:
        print("RESULT: Models ARE DIFFERENT - training modified the weights")
    else:
        print("RESULT: Models are IDENTICAL - no training effect detected")
    print("=" * 70)

if __name__ == "__main__":
    main()
