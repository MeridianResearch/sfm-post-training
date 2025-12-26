
# Quick verification script
from huggingface_hub import hf_hub_download
import safetensors.torch

original = safetensors.torch.load_file(
    hf_hub_download("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid",
                    "model-00003-of-00003.safetensors")
)
uploaded = safetensors.torch.load_file(
    hf_hub_download("camgeodesic/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO",
                    "model-00003-of-00003.safetensors")
)

# Check a late layer (most affected by DPO)
key = "gpt_neox.layers.27.mlp.dense_4h_to_h.weight"
diff = (original[key].float() - uploaded[key].float()).abs().max()
print(f"Max diff: {diff}")  # Should be > 0 (e.g., ~1e-4 to 1e-3)
