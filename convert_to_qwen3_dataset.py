#!/usr/bin/env python3
"""
Convert extracted_resumes.json to Qwen3 chat template format for fine-tuning.

This script:
1. Loads the Qwen3 tokenizer
2. Reads extracted_resumes.json
3. Filters out examples with errors
4. Converts each example to Qwen3 messages format
5. Applies the chat template with correct parameters
6. Saves output as JSONL format
"""

import json
import sys
import os
import argparse
from pathlib import Path

# Project root (directory containing this script)
PROJECT_ROOT = Path(__file__).parent.absolute()


def convert_to_qwen3_messages(resume_text, extracted_json):
    """
    Convert resume data to Qwen3 messages format.
    
    Args:
        resume_text: Raw resume text
        extracted_json: Structured JSON output
        
    Returns:
        List of messages in Qwen3 format
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert resume parser. "
                "Extract structured information from resumes and return ONLY valid JSON. "
                "Do not include explanations or extra text."
            )
        },
        {
            "role": "user",
            "content": f"Resume:\n{resume_text}"
        },
        {
            "role": "assistant",
            "content": json.dumps(extracted_json, ensure_ascii=False)
        }
    ]
    return messages


def apply_chat_template(tokenizer, messages):
    """
    Apply Qwen3 chat template to messages.
    
    Args:
        tokenizer: Qwen3 tokenizer instance
        messages: List of messages in Qwen3 format
        
    Returns:
        Formatted text string
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # IMPORTANT FOR TRAINING
        enable_thinking=False         # VERY IMPORTANT FOR JSON
    )
    return text


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert extracted_resumes.json to Qwen3 chat template format for fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "extracted_resumes.json"),
        help=f"Path to input JSON file (default: data/extracted_resumes.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "qwen3_training_dataset.jsonl"),
        help=f"Path to output JSONL file (default: data/qwen3_training_dataset.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name for tokenizer (default: Qwen/Qwen3-0.6B)",
    )
    
    args = parser.parse_args()
    
    # Check for required dependencies
    try:
        from transformers import AutoTokenizer
        from tqdm import tqdm
    except ImportError as e:
        print("Error: Required dependencies not found.")
        print("Please install them with: pip install transformers tqdm")
        sys.exit(1)
    
    # Convert to Path objects
    input_file = Path(args.input).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print(f"Please ensure the file exists or specify a different path with --input")
        sys.exit(1)
    
    print(f"Loading tokenizer from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure you have internet access to download the tokenizer from Hugging Face")
        sys.exit(1)
    
    # Load input data
    print(f"\nLoading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} examples")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Filter out examples with errors
    successful_examples = [
        ex for ex in data 
        if ex.get('error') is None and ex.get('extracted_json') is not None
    ]
    print(f"✓ Found {len(successful_examples)} successful examples (filtered {len(data) - len(successful_examples)} with errors)")
    
    if len(successful_examples) == 0:
        print("Error: No successful examples found. Cannot create dataset.")
        sys.exit(1)
    
    # Convert examples
    print(f"\nConverting examples to Qwen3 format...")
    converted_examples = []
    errors = []
    
    for idx, example in enumerate(tqdm(successful_examples, desc="Converting")):
        try:
            resume_text = example.get('resume_text', '')
            extracted_json = example.get('extracted_json')
            
            # Validate inputs
            if not resume_text or not resume_text.strip():
                errors.append(f"Example {idx}: Empty resume_text")
                continue
            
            if not extracted_json:
                errors.append(f"Example {idx}: Missing extracted_json")
                continue
            
            # Validate extracted_json is valid JSON
            try:
                json.dumps(extracted_json)
            except (TypeError, ValueError) as e:
                errors.append(f"Example {idx}: Invalid JSON in extracted_json: {e}")
                continue
            
            # Convert to messages format
            messages = convert_to_qwen3_messages(resume_text, extracted_json)
            
            # Apply chat template
            text = apply_chat_template(tokenizer, messages)
            
            # Create output example
            converted_examples.append({"text": text})
            
        except Exception as e:
            errors.append(f"Example {idx}: Conversion error: {e}")
            continue
    
    # Report errors if any
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors during conversion:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Save output
    print(f"\nSaving {len(converted_examples)} examples to {output_file}...")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in converted_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"✓ Successfully saved {len(converted_examples)} examples")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Total input examples: {len(data)}")
    print(f"Successful examples: {len(successful_examples)}")
    print(f"Converted examples: {len(converted_examples)}")
    print(f"Errors during conversion: {len(errors)}")
    print(f"\nOutput file: {output_file}")
    print(f"Output file size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()


