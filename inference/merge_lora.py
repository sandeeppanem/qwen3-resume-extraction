#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for faster inference.

This script:
1. Loads the base model (Qwen/Qwen3-0.6B) from Hugging Face Hub
2. Loads the LoRA adapter (sandeeppanem/qwen3-0.6b-resume-json) from Hugging Face Hub
3. Merges the LoRA weights into the base model
4. Saves the merged model and tokenizer locally

The merged model can then be converted to GGUF format for optimized CPU inference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Project root (parent directory of inference/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LORA_ADAPTER = "sandeeppanem/qwen3-0.6b-resume-json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "merged_model"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for faster inference."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model repo id (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_LORA_ADAPTER,
        help=f"LoRA adapter repo id (default: {DEFAULT_LORA_ADAPTER})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for merged model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Enable trust_remote_code for Qwen models (default: enabled)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading (default: float16)",
    )

    args = parser.parse_args()

    # Import heavy deps after arg parsing so `--help` works without them installed
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install -r requirements.txt\n\n"
            f"Original error: {e}"
        )
        return 1

    # Determine dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Optional: allow HF authentication via env var
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MERGE LORA ADAPTER INTO BASE MODEL")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"LoRA adapter: {args.adapter}")
    print(f"Output directory: {output_dir}")
    print(f"dtype: {dtype}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Tokenizer loaded")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    print("✓ Base model loaded")

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter, token=hf_token)
    print("✓ LoRA adapter loaded")

    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print("✓ LoRA weights merged")

    # Save merged model and tokenizer
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("✓ Merged model saved")

    # Save model config for reference
    config_path = output_dir / "merge_info.txt"
    with open(config_path, "w") as f:
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"LoRA adapter: {args.adapter}\n")
        f.write(f"Merged on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"dtype: {args.dtype}\n")
    print(f"✓ Merge info saved to {config_path}")

    print()
    print("=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Merged model saved to: {output_dir}")
    print()
    print("Next step: Convert to GGUF format using convert_to_gguf.py")
    print("  python convert_to_gguf.py --input_dir merged_model")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


