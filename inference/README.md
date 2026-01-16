# Inference & Deployment

This directory contains scripts for deploying the fine-tuned model for production use.

## Scripts

### `merge_lora.py`
Merges the LoRA adapter into the base model to create a standalone model.

**Usage:**
```bash
python inference/merge_lora.py
```

This creates a merged model in `merged_model/` directory that can be used without PEFT.

### `convert_to_gguf.py`
Converts the merged model to GGUF format and quantizes it for efficient CPU inference.

**Usage:**
```bash
python inference/convert_to_gguf.py \
    --input_dir merged_model \
    --quantization Q5_K_M
```

This will:
1. Clone llama.cpp repository (if needed)
2. Convert merged model to GGUF format
3. Build llama.cpp tools
4. Quantize to Q5_K_M format (~400-500MB)

The quantized model will be saved to `gguf/qwen3-resume-parser-Q5_K_M.gguf`.

## Workflow

1. **Merge LoRA**: Run `merge_lora.py` to create a merged model
2. **Convert to GGUF**: Run `convert_to_gguf.py` to create quantized GGUF model
3. **Deploy**: Copy GGUF model to `space/` directory for Hugging Face Spaces deployment

See the main [README.md](../README.md) for detailed instructions.

