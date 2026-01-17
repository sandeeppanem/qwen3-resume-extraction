---
title: Qwen3 Resume Structured Information Extraction (GPU)
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_port: 7860
python_version: "3.11"
hardware: nvidia-t4-small
---

# Qwen3 Resume Structured Information Extraction (GPU)

Extract structured information from resumes using a fine-tuned Qwen3-0.6B model, optimized for NVIDIA T4 GPU with 8-bit quantization.

## Features

- **GPU-Optimized**: Uses 8-bit quantization for efficient GPU inference on T4
- **Fast Inference**: Leverages GPU acceleration for faster processing
- **Structured JSON Output**: Extracts resume information into structured JSON format
- **LoRA Adapter**: Loads base model + LoRA adapter at runtime (efficient for deployment)

## Model Information

- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Fine-tuned Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)
- **Training Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)
- **Format**: Base model + LoRA adapter with 8-bit quantization (GPU)

## Hardware Requirements

- **GPU**: NVIDIA T4 (16GB) or compatible
- **Memory**: 16GB GPU memory recommended

## Usage

1. Paste your resume text in the input box
2. Click "Parse Resume"
3. View the extracted structured JSON output

## Deployment

This Space uses:
- **Gradio** for the web interface
- **Transformers** with 8-bit quantization for GPU inference
- **PEFT** for loading LoRA adapter
- **bitsandbytes** for efficient 8-bit quantization

## Performance

- **Hardware**: NVIDIA T4 GPU (16GB)
- **Quantization**: 8-bit (BitsAndBytes)
- **Inference Speed**: Fast GPU-accelerated inference
- **Model Format**: Base model + LoRA adapter (loaded at runtime)

## Links

- **🚀 Live Demo (GPU)**: [Try the Resume Parser (GPU)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast) (this Space)
- **🚀 Live Demo (CPU)**: [Try the Resume Parser (CPU)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) - Free CPU version
- **📦 Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)
- **📊 Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)
- **💻 Repository**: [qwen3-resume-extraction](https://github.com/sandeeppanem/qwen3-resume-extraction)

