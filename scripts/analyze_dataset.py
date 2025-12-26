#!/usr/bin/env python3
"""Check sequence lengths in the DPO dataset.

Usage:
    source .env && CUDA_VISIBLE_DEVICES="" python scripts/analyze_dataset.py [dataset_name]
    
Examples:
    python scripts/analyze_dataset.py allenai/Dolci-Instruct-DPO
    python scripts/analyze_dataset.py allenai/Dolci-Think-DPO-7B
"""
import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def content_to_string(content):
    """Convert message content to string, handling various formats."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Content might be a list of text parts or dicts
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(content)


def get_text_from_messages(messages):
    """Convert conversation messages (list of dicts) to text."""
    return " ".join([content_to_string(m.get("content", "")) for m in messages])


def extract_prompt_and_response(messages):
    """Extract prompt (all but last assistant msg) and response (last assistant msg) from conversation.
    
    For chat-format DPO datasets where the full conversation is in chosen/rejected.
    """
    # Find the last assistant message as the response
    prompt_messages = []
    response_content = ""
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = content_to_string(msg.get("content", ""))
        
        # Check if this is the last message and it's from assistant
        if i == len(messages) - 1 and role == "assistant":
            response_content = content
        else:
            prompt_messages.append(content)
    
    prompt_text = " ".join(prompt_messages)
    return prompt_text, response_content


def get_prompt_and_response_lengths(tokenizer, messages):
    """Get token lengths for prompt and response using chat template.
    
    Returns (prompt_len, response_len, total_len) where:
    - prompt_len: tokens for all messages except last assistant response
    - response_len: tokens for the last assistant response
    - total_len: full conversation tokens
    """
    # Clean messages - ensure all have valid content and roles
    clean_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        # Skip messages with None content or missing role
        if content is None or not role:
            continue
        # Convert content to string if needed
        if not isinstance(content, str):
            content = content_to_string(content)
        clean_messages.append({"role": role, "content": content})
    
    if not clean_messages:
        return 0, 0, 0
    
    # Full conversation
    full_tokens = tokenizer.apply_chat_template(clean_messages, tokenize=True)
    total_len = len(full_tokens)
    
    # Prompt only (all messages except the last one, which should be assistant response)
    if len(clean_messages) > 1:
        prompt_messages = clean_messages[:-1]
        # Add a dummy assistant start to get accurate prompt length
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=True,
            add_generation_prompt=True
        )
        prompt_len = len(prompt_tokens)
        response_len = total_len - prompt_len
    else:
        # Edge case: only one message
        prompt_len = 0
        response_len = total_len
    
    return prompt_len, response_len, total_len


def detect_dataset_format(example):
    """Detect whether dataset has explicit 'prompt' field or uses chat-only format."""
    return "prompt" in example


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    # Get dataset name from command line or use default
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "allenai/Dolci-Instruct-DPO"
    
    print(f"Analyzing dataset: {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        "geodesic-research/sfm-sft_dolci_think_unfiltered",
        token=hf_token,
        trust_remote_code=True
    )
    
    dataset = load_dataset(dataset_name, token=hf_token, split="train")
    print(f"Total samples: {len(dataset)}")

    # Sample 1000 examples
    sample = dataset.select(range(min(1000, len(dataset))))
    
    # Detect format from first example
    has_prompt_field = detect_dataset_format(sample[0])
    print(f"Dataset format: {'prompt + chosen/rejected' if has_prompt_field else 'chat-only (chosen/rejected messages)'}")

    lengths = []
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    chosen_total_lengths = []
    rejected_total_lengths = []

    for ex in sample:
        if has_prompt_field:
            # Format: explicit prompt field + chosen/rejected message lists
            # Build full conversation for accurate tokenization
            prompt_messages = [{"role": "user", "content": ex["prompt"]}]
            chosen_messages = prompt_messages + ex["chosen"]
            rejected_messages = prompt_messages + ex["rejected"]
            
            chosen_prompt_len, chosen_resp_len, chosen_total = get_prompt_and_response_lengths(tokenizer, chosen_messages)
            rejected_prompt_len, rejected_resp_len, rejected_total = get_prompt_and_response_lengths(tokenizer, rejected_messages)
            
            # Use chosen prompt length (should be same as rejected)
            prompt_len = chosen_prompt_len
            chosen_len = chosen_resp_len
            rejected_len = rejected_resp_len
        else:
            # Format: chat-only, full conversation in chosen/rejected
            chosen_prompt_len, chosen_resp_len, chosen_total = get_prompt_and_response_lengths(tokenizer, ex["chosen"])
            rejected_prompt_len, rejected_resp_len, rejected_total = get_prompt_and_response_lengths(tokenizer, ex["rejected"])
            
            prompt_len = chosen_prompt_len
            chosen_len = chosen_resp_len
            rejected_len = rejected_resp_len

        # Total is max of chosen/rejected full conversation
        total = max(chosen_total, rejected_total)
        lengths.append(total)
        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        chosen_total_lengths.append(chosen_total)
        rejected_total_lengths.append(rejected_total)

    lengths = np.array(lengths)
    prompt_lengths = np.array(prompt_lengths)
    chosen_lengths = np.array(chosen_lengths)
    rejected_lengths = np.array(rejected_lengths)
    chosen_total_lengths = np.array(chosen_total_lengths)
    rejected_total_lengths = np.array(rejected_total_lengths)

    print(f"\n=== Sequence Length Analysis ({len(sample)} samples) ===\n")

    print("Total Sequence Length (max of chosen/rejected full conversation):")
    print(f"  Min: {lengths.min():,}")
    print(f"  Max: {lengths.max():,}")
    print(f"  Mean: {lengths.mean():,.0f}")
    print(f"  Median: {np.median(lengths):,.0f}")
    print(f"  90th percentile: {np.percentile(lengths, 90):,.0f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):,.0f}")
    print(f"  99th percentile: {np.percentile(lengths, 99):,.0f}")

    print(f"\n--- Prompt vs Response Split (using chat template) ---")
    
    print(f"\nPrompt Length (all messages + generation prompt, before response):")
    print(f"  Min: {prompt_lengths.min():,}")
    print(f"  Max: {prompt_lengths.max():,}")
    print(f"  Median: {np.median(prompt_lengths):,.0f}")
    print(f"  80th percentile: {np.percentile(prompt_lengths, 80):,.0f}")
    print(f"  85th percentile: {np.percentile(prompt_lengths, 85):,.0f}")
    print(f"  90th percentile: {np.percentile(prompt_lengths, 90):,.0f}")
    print(f"  95th percentile: {np.percentile(prompt_lengths, 95):,.0f}")

    print(f"\nChosen Response Length (tokens after prompt):")
    print(f"  Min: {chosen_lengths.min():,}")
    print(f"  Max: {chosen_lengths.max():,}")
    print(f"  Median: {np.median(chosen_lengths):,.0f}")
    print(f"  80th percentile: {np.percentile(chosen_lengths, 80):,.0f}")
    print(f"  85th percentile: {np.percentile(chosen_lengths, 85):,.0f}")
    print(f"  90th percentile: {np.percentile(chosen_lengths, 90):,.0f}")
    print(f"  95th percentile: {np.percentile(chosen_lengths, 95):,.0f}")

    print(f"\nRejected Response Length (tokens after prompt):")
    print(f"  Min: {rejected_lengths.min():,}")
    print(f"  Max: {rejected_lengths.max():,}")
    print(f"  Median: {np.median(rejected_lengths):,.0f}")
    print(f"  80th percentile: {np.percentile(rejected_lengths, 80):,.0f}")
    print(f"  85th percentile: {np.percentile(rejected_lengths, 85):,.0f}")
    print(f"  90th percentile: {np.percentile(rejected_lengths, 90):,.0f}")
    print(f"  95th percentile: {np.percentile(rejected_lengths, 95):,.0f}")
    
    print(f"\n--- Full Conversation Lengths ---")
    print(f"\nChosen Total (prompt + chosen response):")
    print(f"  Median: {np.median(chosen_total_lengths):,.0f}")
    print(f"  95th percentile: {np.percentile(chosen_total_lengths, 95):,.0f}")
    
    print(f"\nRejected Total (prompt + rejected response):")
    print(f"  Median: {np.median(rejected_total_lengths):,.0f}")
    print(f"  95th percentile: {np.percentile(rejected_total_lengths, 95):,.0f}")

    # Distribution of lengths
    print(f"\n=== Distribution ===")
    for threshold in [2048, 4096, 8192, 12288, 16384]:
        pct = (lengths <= threshold).sum() / len(lengths) * 100
        print(f"  <= {threshold:,} tokens: {pct:.1f}%")

    print(f"\n=== Memory Impact Analysis ===")
    median = np.median(lengths)
    print(f"Median length: {median:,.0f} tokens")
    print(f"If max_length=16384: padding overhead = {16384 - median:,.0f} tokens/sample ({(16384 - median)/16384*100:.1f}% waste)")
    print(f"If max_length=8192: padding overhead = {8192 - median:,.0f} tokens/sample ({(8192 - median)/8192*100:.1f}% waste)")
    print(f"Samples truncated at 8192: {(lengths > 8192).sum() / len(lengths) * 100:.1f}%")


if __name__ == "__main__":
    main()
