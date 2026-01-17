---
title: Qwen3 Resume Structured Information Extraction
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Qwen3 Resume Structured Information Extraction

Extract structured information from resumes using a fine-tuned Qwen3-0.6B model, optimized for CPU inference with GGUF format and Q5_K_M quantization.

## Features

- **Fast CPU Inference**: Uses llama.cpp with Q5_K_M quantization for 7-15x faster inference
- **Structured JSON Output**: Extracts resume information into structured JSON format
- **Optimized Model**: Merged LoRA adapter with Q5_K_M quantization (~400-500MB)

## Model Information

- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Fine-tuned Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)
- **Training Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)
- **Format**: GGUF Q5_K_M (optimized for CPU)

## Performance

- **Previous (transformers)**: ~77 seconds per request on CPU
- **Current (GGUF)**: ~5-10 seconds per request on CPU
- **Improvement**: 7-15x faster

## Usage

1. Paste your resume text in the input box
2. Click "Parse Resume"
3. View the extracted structured JSON output

## Deployment

This Space uses:
- **Gradio** for the web interface
- **llama-cpp-python** for GGUF model inference
- **Q5_K_M quantization** for optimal speed/quality balance

The GGUF model file (`qwen3-resume-parser-Q5_K_M.gguf`) should be included in this Space repository (use Git LFS if >100MB).

## Performance

- **Previous (transformers)**: ~77 seconds per request on CPU
- **Current (GGUF Q5_K_M)**: ~5-10 seconds per request on CPU
- **Improvement**: 7-15x faster
- **Model Size**: ~400-500MB (Q5_K_M quantization)

## Links

- **🚀 Live Demo (CPU)**: [Try the Resume Parser](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) (this Space)
- **🚀 Live Demo (GPU)**: [Try the Resume Parser (GPU)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast) - Faster GPU version
- **📦 Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)
- **📊 Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)
- **💻 Repository**: [qwen3-resume-extraction](https://github.com/sandeeppanem/qwen3-resume-extraction)


