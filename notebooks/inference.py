#!/usr/bin/env python3
"""
Run inference with the fine-tuned Qwen3 resume parser (LoRA adapter).

This script downloads (via Hugging Face):
- Base model: Qwen/Qwen3-0.6B
- LoRA adapter: sandeeppanem/qwen3-0.6b-resume-json

Then it formats a prompt using Qwen3's chat template and prints the model's
assistant response (expected to be valid JSON).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LORA_ADAPTER = "sandeeppanem/qwen3-0.6b-resume-json"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_messages(resume_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert resume parser. "
                "Extract structured information from resumes and return ONLY valid JSON. "
                "Do not include explanations or extra text."
            ),
        },
        {"role": "user", "content": f"Resume:\n{resume_text.strip()}"},
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inference script for Qwen3 resume JSON extraction (base model + LoRA adapter from Hugging Face)."
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
        "--resume_file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file containing a resume. If omitted, uses --resume_text or a built-in example.",
    )
    parser.add_argument(
        "--resume_text",
        type=str,
        default=None,
        help="Resume text to parse (optional). If omitted, uses --resume_file or a built-in example.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', 'mps', or 'auto' (default: auto-detect).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=[None, "float16", "bfloat16", "float32"],
        help="Force torch dtype (default: auto: float16 on CUDA, else float32).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Enable trust_remote_code for Qwen models (default: enabled).",
    )

    args = parser.parse_args(argv)

    # Import heavy deps after arg parsing so `--help` works without them installed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "  pip install -r requirements.txt\n\n"
            f"Original error: {e}"
        )

    # Resolve resume text
    resume_text: str
    if args.resume_text and args.resume_text.strip():
        resume_text = args.resume_text
    elif args.resume_file:
        resume_text = _read_text(Path(args.resume_file).expanduser().resolve())
    else:
        resume_text = (
            "Senior IT Project Manager with 10+ years experience leading enterprise migrations. "
            "Skills: Python, SQL, AWS, Agile. Location: Chicago, IL. "
            "Experience: Project Manager at Acme Corp (2019-2024). Education: MS Computer Science."
        )

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available. Use --device cpu")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available. Use --device cpu")
        device = torch.device("mps")
    elif args.device == "auto" or args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        raise SystemExit(f"Invalid device: {args.device}. Use 'cuda', 'cpu', 'mps', or 'auto'")

    # Determine dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        # Auto-select: float16 on CUDA/MPS, float32 on CPU
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    # Optional: allow HF authentication via env var if user wants faster access / higher rate limits.
    # Public repos should work without a token.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"Base model: {args.base_model}")
    print(f"LoRA adapter: {args.adapter}")
    print(f"Device: {device}")
    print(f"dtype: {dtype}")

    print("\nLoading tokenizer...")
    # Prefer tokenizer from base model to ensure chat_template is present.
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    base_model = base_model.to(device)
    print(f"✓ Base model loaded on {device}")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter, token=hf_token)
    model.eval()
    print("✓ LoRA adapter loaded")

    messages = build_messages(resume_text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (assistant response).
    prompt_len = inputs["input_ids"].shape[-1]
    assistant_response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    print("\n=== RAW MODEL OUTPUT (assistant) ===")
    print(assistant_response)

    print("\n=== JSON (validated) ===")
    try:
        parsed = json.loads(assistant_response)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"⚠️ Not valid JSON: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


