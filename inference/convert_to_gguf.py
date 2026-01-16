#!/usr/bin/env python3
"""
Convert merged model to GGUF format and quantize for optimized CPU inference.

This script:
1. Clones llama.cpp repository if not present
2. Installs llama.cpp Python requirements
3. Converts merged model to GGUF format using convert-hf-to-gguf.py
4. Builds llama.cpp to get the quantize binary
5. Quantizes the GGUF model to Q5_K_M format

The quantized model can then be used with llama-cpp-python for fast CPU inference.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Project root (parent directory of inference/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DEFAULT_INPUT_DIR = PROJECT_ROOT / "merged_model"
DEFAULT_LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gguf"
DEFAULT_QUANTIZATION = "Q5_K_M"


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=False,  # Show output in real-time
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def clone_llama_cpp(llama_cpp_dir: Path) -> None:
    """Clone llama.cpp repository if it doesn't exist."""
    if llama_cpp_dir.exists():
        print(f"✓ llama.cpp already exists at {llama_cpp_dir}")
        return

    print(f"Cloning llama.cpp to {llama_cpp_dir}...")
    run_command(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_dir)]
    )
    print("✓ llama.cpp cloned")


def install_requirements(llama_cpp_dir: Path, skip: bool = False) -> None:
    """Install llama.cpp Python requirements (optional, may fail due to version conflicts)."""
    if skip:
        print("⚠️  Skipping llama.cpp requirements installation (as requested)")
        print("   Make sure you have: transformers, torch, sentencepiece, gguf, protobuf")
        return
        
    requirements_file = llama_cpp_dir / "requirements.txt"
    if not requirements_file.exists():
        print("⚠️  requirements.txt not found in llama.cpp, skipping...")
        return

    print("Installing llama.cpp Python requirements...")
    print("⚠️  Note: If installation fails due to version conflicts, conversion may still work with existing packages.")
    
    result = run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        check=False,  # Don't fail if requirements can't be installed
    )
    
    if result.returncode == 0:
        print("✓ Requirements installed")
    else:
        print("⚠️  Requirements installation failed (this may be okay if packages are already installed)")
        print("   Continuing anyway - conversion will fail if required packages are missing.")


def fix_torch_uint_compatibility(convert_script: Path) -> bool:
    """
    Fix torch.uint* compatibility for older PyTorch versions.
    Comments out torch.uint64, uint32, and uint16 lines in the conversion script.
    
    Returns True if fix was applied, False if not needed or already fixed.
    """
    try:
        import torch
        # Check if all uint types exist (PyTorch 2.6.0+)
        if hasattr(torch, 'uint64') and hasattr(torch, 'uint32') and hasattr(torch, 'uint16'):
            return False  # Not needed, PyTorch has all uint types
    except ImportError:
        pass  # Can't check, proceed with fix
    
    if not convert_script.exists():
        return False
    
    content = convert_script.read_text()
    
    # Fix all uint types that might not be available
    # Check each one individually and fix if needed
    fixes_applied = False
    
    # Fix torch.uint64 (if not already fixed)
    if "        torch.uint64: np.uint64," in content:
        content = content.replace(
            "        torch.uint64: np.uint64,",
            "        # PATCHED: torch.uint64 not available in PyTorch < 2.6.0\n        # torch.uint64: np.uint64,"
        )
        fixes_applied = True
    
    # Fix torch.uint32 (if not already fixed)
    if "        torch.uint32: np.uint32," in content:
        content = content.replace(
            "        torch.uint32: np.uint32,",
            "        # PATCHED: torch.uint32 not available in PyTorch < 2.6.0\n        # torch.uint32: np.uint32,"
        )
        fixes_applied = True
    
    # Fix torch.uint16 (if not already fixed)
    if "        torch.uint16: np.uint16," in content:
        content = content.replace(
            "        torch.uint16: np.uint16,",
            "        # PATCHED: torch.uint16 not available in PyTorch < 2.6.0\n        # torch.uint16: np.uint16,"
        )
        fixes_applied = True
    
    if fixes_applied:
        convert_script.write_text(content)
        print("✓ Applied compatibility fix for PyTorch < 2.6.0 (commented out torch.uint64, uint32, uint16)")
        return True
    
    return False


def convert_to_gguf(
    llama_cpp_dir: Path,
    input_dir: Path,
    output_file: Path,
) -> None:
    """Convert merged model to GGUF format."""
    # Try both naming conventions (underscores and hyphens)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {llama_cpp_dir}. "
            "Please ensure llama.cpp is properly cloned. "
            f"Found conversion scripts: {list(llama_cpp_dir.glob('convert*.py'))}"
        )

    # Check PyTorch version and apply compatibility fix if needed
    try:
        import torch
        torch_version = torch.__version__
        # Parse version to compare
        major, minor = map(int, torch_version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 6):
            print(f"\n⚠️  PyTorch version {torch_version} detected (< 2.6.0)")
            print("   Applying compatibility fix for torch.uint types...")
            fix_torch_uint_compatibility(convert_script)
    except ImportError:
        print("⚠️  PyTorch not found. Attempting compatibility fix anyway...")
        fix_torch_uint_compatibility(convert_script)

    print(f"Converting {input_dir} to GGUF format...")
    run_command(
        [
            sys.executable,
            str(convert_script),
            str(input_dir),
            "--outfile",
            str(output_file),
        ],
        cwd=llama_cpp_dir,
    )
    print(f"✓ Converted to {output_file}")


def build_llama_cpp(llama_cpp_dir: Path) -> None:
    """Build llama.cpp to get the quantize binary using CMake."""
    # Check for quantize binary in different possible locations
    # In newer llama.cpp, the binary is named "llama-quantize"
    build_dir = llama_cpp_dir / "build"
    quantize_binary_cmake = build_dir / "bin" / "llama-quantize"
    quantize_binary_legacy = build_dir / "bin" / "quantize"
    quantize_binary_root = llama_cpp_dir / "quantize"
    llama_quantize_root = llama_cpp_dir / "llama-quantize"
    
    # Check if already built
    if quantize_binary_cmake.exists():
        print("✓ llama-quantize binary already exists (CMake build)")
        return
    if quantize_binary_legacy.exists():
        print("✓ quantize binary already exists (CMake build)")
        return
    if quantize_binary_root.exists() or llama_quantize_root.exists():
        print("✓ quantize binary already exists")
        return

    print("Building llama.cpp with CMake (this may take a few minutes)...")
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # Configure with CMake
    print("Configuring CMake...")
    run_command(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=llama_cpp_dir,
    )
    
    # Build
    print("Building...")
    run_command(
        ["cmake", "--build", "build", "--config", "Release", "-j"],
        cwd=llama_cpp_dir,
    )
    
    # Check if quantize binary exists in build directory
    if quantize_binary_cmake.exists():
        print("✓ llama.cpp built successfully (CMake) - found llama-quantize")
    elif quantize_binary_legacy.exists():
        print("✓ llama.cpp built successfully (CMake) - found quantize")
    elif quantize_binary_root.exists() or llama_quantize_root.exists():
        print("✓ llama.cpp built successfully")
    else:
        raise RuntimeError(
            f"quantize binary not found after build. "
            f"Checked: {quantize_binary_cmake}, {quantize_binary_legacy}, "
            f"{quantize_binary_root}, {llama_quantize_root}. "
            "Build may have failed."
        )


def quantize_model(
    llama_cpp_dir: Path,
    input_gguf: Path,
    output_gguf: Path,
    quantization: str,
) -> None:
    """Quantize GGUF model using llama.cpp quantize binary."""
    # Check for quantize binary in multiple locations
    # In newer llama.cpp, the binary is named "llama-quantize"
    build_dir = llama_cpp_dir / "build"
    quantize_binary_cmake = build_dir / "bin" / "llama-quantize"
    quantize_binary_legacy = build_dir / "bin" / "quantize"
    quantize_binary_root = llama_cpp_dir / "quantize"
    llama_quantize_root = llama_cpp_dir / "llama-quantize"
    
    # Use CMake build location if available, otherwise root
    if quantize_binary_cmake.exists():
        quantize_binary = quantize_binary_cmake
    elif quantize_binary_legacy.exists():
        quantize_binary = quantize_binary_legacy
    elif llama_quantize_root.exists():
        quantize_binary = llama_quantize_root
    elif quantize_binary_root.exists():
        quantize_binary = quantize_binary_root
    else:
        raise FileNotFoundError(
            f"quantize binary not found. Checked: {quantize_binary_cmake}, "
            f"{quantize_binary_legacy}, {quantize_binary_root}, {llama_quantize_root}. "
            "Please build llama.cpp first."
        )

    print(f"Quantizing {input_gguf} to {quantization}...")
    run_command(
        [
            str(quantize_binary),
            str(input_gguf),
            str(output_gguf),
            quantization,
        ],
        cwd=llama_cpp_dir,
    )
    print(f"✓ Quantized to {output_gguf}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert merged model to GGUF format and quantize for CPU inference."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Input directory containing merged model (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--llama_cpp_dir",
        type=str,
        default=str(DEFAULT_LLAMA_CPP_DIR),
        help=f"Directory for llama.cpp repository (default: {DEFAULT_LLAMA_CPP_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for GGUF files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=DEFAULT_QUANTIZATION,
        choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        help=f"Quantization level (default: {DEFAULT_QUANTIZATION})",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building llama.cpp (use if already built)",
    )
    parser.add_argument(
        "--skip_requirements",
        action="store_true",
        help="Skip installing llama.cpp requirements (use if you have packages installed)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        print("   Please run merge_lora.py first to create the merged model.")
        return 1

    if not (input_dir / "config.json").exists():
        print(f"❌ Error: config.json not found in {input_dir}")
        print("   Please ensure this is a valid Hugging Face model directory.")
        return 1

    print("=" * 70)
    print("CONVERT TO GGUF AND QUANTIZE")
    print("=" * 70)
    print(f"Input model: {input_dir}")
    print(f"llama.cpp dir: {llama_cpp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {args.quantization}")
    print()

    try:
        # Step 1: Clone llama.cpp if needed
        clone_llama_cpp(llama_cpp_dir)
        print()

        # Step 2: Install Python requirements (optional)
        install_requirements(llama_cpp_dir, skip=args.skip_requirements)
        print()

        # Step 3: Convert to GGUF (FP16)
        base_gguf = output_dir / "qwen3-resume-parser.gguf"
        convert_to_gguf(llama_cpp_dir, input_dir, base_gguf)
        print()

        # Step 4: Build llama.cpp (if not skipped)
        if not args.skip_build:
            build_llama_cpp(llama_cpp_dir)
            print()
        else:
            # Check all possible locations
            quantize_binary_cmake = llama_cpp_dir / "build" / "bin" / "llama-quantize"
            quantize_binary_legacy = llama_cpp_dir / "build" / "bin" / "quantize"
            quantize_binary_root = llama_cpp_dir / "quantize"
            llama_quantize_root = llama_cpp_dir / "llama-quantize"
            if not any([
                quantize_binary_cmake.exists(),
                quantize_binary_legacy.exists(),
                quantize_binary_root.exists(),
                llama_quantize_root.exists(),
            ]):
                print("❌ Error: quantize binary not found. Remove --skip_build to build it.")
                return 1

        # Step 5: Quantize
        quantized_gguf = output_dir / f"qwen3-resume-parser-{args.quantization}.gguf"
        quantize_model(llama_cpp_dir, base_gguf, quantized_gguf, args.quantization)
        print()

        print("=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)
        print(f"Base GGUF: {base_gguf}")
        print(f"Quantized GGUF: {quantized_gguf}")
        print()
        print("Next step: Deploy to Hugging Face Space using space/app.py")
        print(f"  Copy {quantized_gguf} to your Space repository")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

