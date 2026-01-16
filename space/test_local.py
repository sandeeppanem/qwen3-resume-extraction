#!/usr/bin/env python3
"""
Local testing script for Qwen3 Resume Parser using llama.cpp (GGUF format).

This script mirrors the exact logic of app.py for local testing without Gradio.
Useful for debugging and testing the model locally before deployment.
"""

import json
import os
import re
import sys
from pathlib import Path

# Model configuration
MODEL_PATH = "qwen3-resume-parser-Q5_K_M.gguf"

# Global variables for model caching
_model = None


def format_qwen3_prompt(resume_text: str) -> str:
    """Format prompt for Qwen3 chat template."""
    system_content = (
        "You are an expert resume parser. "
        "Extract structured information from resumes and return ONLY valid JSON. "
        "Do not include explanations or extra text."
    )
    user_content = f"Resume:\n{resume_text.strip()}"
    prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def load_model():
    """Load GGUF model using llama-cpp-python (loads once at startup)."""
    global _model
    
    if _model is not None:
        return _model
    
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. "
            "Install with: pip install llama-cpp-python"
        )
    
    # Try multiple possible paths for the model file
    script_dir = Path(__file__).parent
    possible_paths = [
        Path(MODEL_PATH),  # Current directory
        script_dir / MODEL_PATH,  # Same directory as test_local.py
        script_dir.parent / MODEL_PATH,  # Parent directory
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            model_path = path
            print(f"Found model at: {model_path.absolute()}")
            break
    
    if model_path is None:
        # List available files for debugging
        print(f"Current directory: {Path.cwd()}")
        print(f"Script directory: {script_dir.absolute()}")
        print(f"Files in script directory: {list(script_dir.iterdir())}")
        raise FileNotFoundError(
            f"GGUF model not found. Tried: {[str(p) for p in possible_paths]}\n"
            f"Make sure {MODEL_PATH} is in the Space repository."
        )
    
    cpu_count = os.cpu_count() or 2
    n_threads = min(cpu_count, 8)
    
    try:
        print(f"Loading model from: {model_path.absolute()}")
        print(f"Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Check llama-cpp-python version
        try:
            import llama_cpp
            print(f"llama-cpp-python version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'unknown'}")
        except:
            pass
        
        # Try loading with minimal parameters first, then add optimizations
        print("Attempting to load model...")
        
        # Optimized parameters for faster inference
        _model = Llama(
            model_path=str(model_path),
            n_ctx=2560,
            n_threads=n_threads,
            n_batch=128,  # Reduced from 512 for faster processing
            n_gpu_layers=0,
            chat_format=None,  # Disable chat format parsing for speed
            verbose=True,
        )
        print("✓ Model loaded with optimized parameters")
        
        print(f"✓ Model loaded successfully! (using {n_threads} threads)")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        full_error = f"Failed to load model from {model_path}: {error_type}: {error_msg}"
        print(f"❌ {full_error}")
        
        # Provide helpful suggestions based on error
        if "mmap" in error_msg.lower() or "memory" in error_msg.lower():
            print("\n💡 Suggestion: Try disabling mmap or reducing context size")
        elif "format" in error_msg.lower() or "invalid" in error_msg.lower():
            print("\n💡 Suggestion: Model file might be corrupted or incompatible format")
            print("   Try regenerating the GGUF file or check llama-cpp-python version compatibility")
        elif "permission" in error_msg.lower():
            print("\n💡 Suggestion: Check file permissions")
        
        raise RuntimeError(full_error) from e
    
    return _model


def _format_incomplete_json(text: str) -> str:
    """Format incomplete JSON for visibility during streaming."""
    if not text or not text.strip():
        return text
    
    formatted = text
    
    # First, ensure proper spacing around colons (makes it more readable)
    formatted = re.sub(r':"', ': "', formatted)
    formatted = re.sub(r':(\d+)', r': \1', formatted)
    formatted = re.sub(r':(true|false|null)', r': \1', formatted)
    formatted = re.sub(r':\{', ': {', formatted)
    formatted = re.sub(r':\[', ': [', formatted)
    formatted = re.sub(r',\s*"', ',\n  "', formatted)
    # Pattern: comma followed by number
    formatted = re.sub(r',\s*(\d+)', r',\n  \1', formatted)
    formatted = re.sub(r',\s*(true|false|null)', r',\n  \1', formatted)
    # Pattern: comma followed by opening brace/array
    formatted = re.sub(r',\s*(\{|\[)', r',\n  \1', formatted)
    formatted = re.sub(r'\{\s*"', '{\n  "', formatted)
    
    # Add newline before closing brace (if it's on same line with content)
    # But be careful not to break strings
    formatted = re.sub(r'([^}\s"])\s*\}', r'\1\n}', formatted)
    formatted = re.sub(r'\n\n+', '\n', formatted)
    formatted = re.sub(r'  +', '  ', formatted)
    return formatted


def parse_resume_stream(resume_text: str):
    """Parse resume text and stream structured JSON as it's generated."""
    if not resume_text or not resume_text.strip():
        yield "⚠️ Please provide resume text.", ""
        return
    
    try:
        model = load_model()
        
        MAX_RESUME_CHARS = 4000
        if len(resume_text) > MAX_RESUME_CHARS:
            truncated = resume_text[:MAX_RESUME_CHARS]
            last_space = truncated.rfind(' ', MAX_RESUME_CHARS - 200, MAX_RESUME_CHARS)
            if last_space > MAX_RESUME_CHARS - 500:
                truncated = truncated[:last_space]
            resume_text = truncated + "..."
        
        prompt = format_qwen3_prompt(resume_text)
        accumulated_text = ""
        
        stream = model(
            prompt,
            max_tokens=350,
            temperature=0.0,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
            stream=True,
        )
        
        # Process streamed tokens
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    accumulated_text += delta
                    
                    cleaned_text = accumulated_text
                    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
                    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
                    cleaned_text = re.sub(r'</?redacted_reasoning>', '', cleaned_text)
                    cleaned_text = re.sub(r'</?think>', '', cleaned_text)
                    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    
                    try:
                        parsed_json = json.loads(cleaned_text)
                        formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                        yield formatted_json, cleaned_text
                    except json.JSONDecodeError:
                        formatted_incomplete = _format_incomplete_json(cleaned_text)
                        yield formatted_incomplete, cleaned_text
        
        assistant_response = accumulated_text.strip()
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        assistant_response = re.sub(r'</?redacted_reasoning>', '', assistant_response)
        assistant_response = re.sub(r'</?think>', '', assistant_response)
        assistant_response = re.sub(r'\n\s*\n+', '\n', assistant_response)
        assistant_response = assistant_response.strip()
        
        try:
            parsed_json = json.loads(assistant_response)
            formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            yield formatted_json, assistant_response
        except json.JSONDecodeError:
            yield (
                f"⚠️ Model output is not valid JSON:\n\n{assistant_response}",
                assistant_response,
            )
    
    except Exception as e:
        yield f"❌ Error: {str(e)}", ""


def parse_resume(resume_text: str) -> tuple[str, str]:
    """Parse resume text and return structured JSON (non-streaming version)."""
    result = None
    for result in parse_resume_stream(resume_text):
        pass
    return result if result else ("⚠️ No output generated", "")


def main():
    """Main function for local testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Qwen3 Resume Parser locally (mirrors app.py logic)"
    )
    parser.add_argument(
        "--resume_file",
        type=str,
        help="Path to resume text file",
    )
    parser.add_argument(
        "--resume_text",
        type=str,
        help="Resume text directly (alternative to --resume_file)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (shows tokens as they're generated)",
    )
    
    args = parser.parse_args()
    
    # Load resume text
    resume_text = None
    if args.resume_file:
        resume_path = Path(args.resume_file)
        if not resume_path.exists():
            print(f"❌ Error: Resume file not found: {resume_path}")
            sys.exit(1)
        resume_text = resume_path.read_text(encoding="utf-8")
    elif args.resume_text:
        resume_text = args.resume_text
    else:
        # Default example resume
        resume_text = """Senior IT Project Manager with 10+ years experience leading enterprise migrations. 
Skills: Python, SQL, AWS, Agile. Location: Chicago, IL. 
Experience: Project Manager at Acme Corp (2019-2024). 
Education: MS Computer Science."""
        print("Using default example resume (use --resume_file or --resume_text for custom input)")
    
    print("\n" + "="*80)
    print("Testing Resume Parser")
    print("="*80)
    print(f"\nResume text length: {len(resume_text)} characters")
    print("\nParsing resume...\n")
    
    if args.stream:
        # Streaming mode
        print("Streaming output:\n")
        print("-" * 80)
        for formatted_json, raw_output in parse_resume_stream(resume_text):
            print(formatted_json)
            print("-" * 80)
    else:
        # Non-streaming mode
        formatted_json, raw_output = parse_resume(resume_text)
        print("Extracted JSON:")
        print("=" * 80)
        print(formatted_json)
        print("=" * 80)
        
        if raw_output and raw_output != formatted_json:
            print("\nRaw model output:")
            print("-" * 80)
            print(raw_output)
            print("-" * 80)


if __name__ == "__main__":
    # Load model at startup
    try:
        try:
            load_model()
        except Exception as e:
            print(f"⚠️ Warning: Could not load model at startup: {e}")
            print("Model will be loaded on first use.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    main()

