# Qwen3 Resume Extraction Fine-Tuning

Fine-tune Qwen3-0.6B base model for resume parsing using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow.svg)](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-Dataset-yellow.svg)](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)
[![Hugging Face Spaces CPU](https://img.shields.io/badge/🤗%20Spaces-CPU%20Demo-blue.svg)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser)
[![Hugging Face Spaces GPU](https://img.shields.io/badge/🤗%20Spaces-GPU%20Demo-green.svg)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast)

## 🚀 Model & Dataset Links

- **🚀 Live Demos**:
  - **CPU-Optimized**: [Try the Resume Parser (CPU)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) - Free tier, GGUF optimized
  - **GPU-Accelerated**: [Try the Resume Parser (GPU)](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast) - NVIDIA T4, 8-bit quantization
- **Fine-tuned Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json) on Hugging Face
- **Training Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k) on Hugging Face

> **Note**: The fine-tuned model and training dataset are hosted on Hugging Face. You can download and use them directly from there. This repository contains the code and pipeline for reproducing the fine-tuning process. **Try the live demos** to see the model in action!

## Quick Start

### 🚀 Try the Live Demos

**The easiest way to try the model** - choose based on your needs:
- **CPU Version**: [**Try the Resume Parser (CPU)**](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) - Free tier, optimized with GGUF for fast CPU inference
- **GPU Version**: [**Try the Resume Parser (GPU)**](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast) - NVIDIA T4 GPU, 8-bit quantization for maximum speed

### 💻 Run Locally

Get started with the fine-tuned model in just a few steps:

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference with default example (quick check with LoRA adapter)
python notebooks/inference.py

# Parse your own resume file
python notebooks/inference.py --resume_file path/to/resume.txt
```

The inference script automatically downloads:
- Base model: `Qwen/Qwen3-0.6B` from Hugging Face
- LoRA adapter: `sandeeppanem/qwen3-0.6b-resume-json` from Hugging Face

## ✨ Key Features

- **📊 Training Dataset Creation**: Curated and processed 5K diverse resume samples with structured JSON extraction. 
**Key contribution** - the dataset is available on [Hugging Face Datasets](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k) for others to use and train their own models.
- **🚀 Zero API Costs**: Runs entirely locally with no external API dependencies. No OpenAI, Anthropic, or Gemini API keys required.
- **⚡ Dual Deployment Options**: 
  - **CPU-Optimized**: GGUF quantization for fast CPU inference (7-15x faster than standard Transformers)
  - **GPU-Accelerated**: 8-bit quantization on NVIDIA T4 for maximum performance
- **📦 Open Source**: Complete pipeline including training code, model weights, and dataset
- **🎯 Specialized Model**: Fine-tuned specifically for resume parsing with high accuracy
- **🔧 Easy Deployment**: Ready-to-use Gradio interfaces for both CPU and GPU deployments on Hugging Face Spaces
- **💾 Efficient Training**: LoRA fine-tuning enables domain adaptation with minimal computational resources
- **🔄 Streaming Output**: Real-time JSON generation with proper formatting for better user experience

## Project Structure

```
.
├── data/                              # Data files
│   ├── combined_resumes.json          # Raw resume data (merged from 2 sources)
│   ├── extracted_resumes.json        # Extracted structured data (resume → JSON pairs)
│   ├── qwen3_training_dataset.jsonl  # Training dataset (Qwen3 chat template format)
│   └── dataset_card.md                # Dataset documentation for Hugging Face
├── notebooks/                         # Training notebooks
│   ├── generate_training_data.ipynb  # Generate training data pairs (resume → JSON)
│   ├── train_qwen3_resume_parser.ipynb  # Fine-tuning notebook (for Google Colab)
│   └── inference.py                  # Quick inference script for testing (uses LoRA adapter)
├── scripts/                           # Data preprocessing scripts
│   ├── select_resumes_by_title.py    # Select resumes by job title
│   ├── combine_resume_datasets.py    # Merge datasets from multiple sources
│   └── convert_to_qwen3_dataset.py   # Convert extracted data to Qwen3 format
├── inference/                         # Model deployment scripts
│   ├── merge_lora.py                 # Merge LoRA adapter into base model
│   ├── convert_to_gguf.py            # Convert merged model to GGUF format
│   ├── test_local.py                 # Local testing script (mirrors app.py logic, uses GGUF from gguf/)
│   └── README.md                     # Inference documentation
├── deploy_hf_space_cpu/              # Hugging Face Space deployment (CPU)
│   ├── app.py                        # Gradio app using llama-cpp-python (GGUF)
│   ├── Dockerfile                    # Docker configuration for CPU Spaces
│   ├── requirements.txt              # Dependencies for CPU deployment
│   └── README.md                     # CPU Space documentation
├── deploy_hf_space_gpu/              # Hugging Face Space deployment (GPU)
│   ├── app.py                        # Gradio app using Transformers (8-bit quantization)
│   ├── Dockerfile                    # Docker configuration for GPU Spaces (optional)
│   ├── requirements.txt              # Dependencies for GPU deployment
│   └── README.md                     # GPU Space documentation
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Apache 2.0 License
└── README.md                         # This file
```

> **Note**: The fine-tuned model is available on [Hugging Face](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json), not in this repository. The `data/` directory contains example intermediate files for reference.

## Complete Pipeline

The project follows this workflow to create training data and fine-tune the model:

### Step 1: Data Preprocessing

**1.1 Select Resumes by Job Title** (from GitHub Resume Corpus):
```bash
python scripts/select_resumes_by_title.py \
    --corpus_dir data/resumes_corpus \
    --output_dir data/selected_resumes \
    --num_per_title 300
```
- Selects random resumes by job title from the GitHub corpus
- Outputs text files to `data/selected_resumes/`

**1.2 Combine Datasets** (merge from 2 sources):
```bash
python scripts/combine_resume_datasets.py \
    --dataset1 data/resume_dataset/data \
    --dataset2 data/selected_resumes \
    --output data/combined_resumes.json
```
- Extracts text from PDF files (Kaggle dataset)
- Reads text files (selected from GitHub corpus)
- Merges into single `combined_resumes.json` file

### Step 2: Generate Training Data Pairs

**2.1 Extract Structured Information**:
- Open `notebooks/generate_training_data.ipynb`
- Processes `combined_resumes.json` using LLM
- Extracts structured JSON for each resume
- Outputs `extracted_resumes.json` with (resume_text, extracted_json) pairs

### Step 3: Convert to Training Format

**3.1 Convert to Qwen3 Chat Template**:
```bash
python scripts/convert_to_qwen3_dataset.py
```
- Converts `extracted_resumes.json` to Qwen3 chat template format
- Outputs `data/qwen3_training_dataset.jsonl` ready for fine-tuning

### Step 4: Fine-Tune Model

**4.1 Train Model** (in Google Colab):
- Open `notebooks/train_qwen3_resume_parser.ipynb`
- Upload `qwen3_training_dataset.jsonl`
- Fine-tune Qwen3-0.6B with LoRA
- Download fine-tuned model

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install only conversion script dependencies (lighter install)
pip install transformers tqdm torch
```

**Note**: 
- For preprocessing: `pdfplumber` (for PDF extraction)
- For dataset conversion: `transformers`, `tqdm`, and `torch`
- For training: `peft`, `trl`, `datasets`, etc. (use Google Colab)

### 3. Data Preprocessing (Optional)

If starting from raw data sources, run these preprocessing steps:

```bash
# Step 1: Select resumes by job title (from GitHub corpus)
python scripts/select_resumes_by_title.py \
    --corpus_dir data/resumes_corpus \
    --output_dir data/selected_resumes

# Step 2: Combine datasets from both sources
python scripts/combine_resume_datasets.py \
    --dataset1 data/resume_dataset/data \
    --dataset2 data/selected_resumes \
    --output data/combined_resumes.json
```

**Note**: If you already have `combined_resumes.json`, you can skip this step.

### 4. Generate Training Data

Use `notebooks/generate_training_data.ipynb` to extract structured information:
- Processes `combined_resumes.json`
- Uses LLM to extract structured JSON for each resume
- Outputs `extracted_resumes.json` with training pairs

**Note**: The notebook uses placeholders for LLM API configuration. Update with your API settings.

### 5. Convert to Training Format

```bash
# Convert extracted_resumes.json to Qwen3 training format
python scripts/convert_to_qwen3_dataset.py

# Or specify custom paths
python scripts/convert_to_qwen3_dataset.py \
    --input data/extracted_resumes.json \
    --output data/qwen3_training_dataset.jsonl \
    --model Qwen/Qwen3-0.6B
```

This will:
- Load the Qwen3 tokenizer from Hugging Face
- Read `data/extracted_resumes.json` (or path specified with `--input`)
- Filter out examples with errors
- Convert to Qwen3 chat template format
- Save to `data/qwen3_training_dataset.jsonl` (or path specified with `--output`)

## Training

Training should be done in **Google Colab** for GPU access:

1. Open `notebooks/train_qwen3_resume_parser.ipynb` in Google Colab
2. Upload `data/qwen3_training_dataset.jsonl` to Colab (update `DATASET_PATH` in the notebook)
3. Enable GPU: Runtime → Change runtime type → GPU (T4 or better recommended)
4. Run all cells in order
5. The notebook will:
   - Check GPU availability and install required packages
   - Load Qwen3-0.6B base model with float16 precision
   - Configure LoRA with optimized parameters
   - Load and preprocess the training dataset
   - Fine-tune the model with memory-efficient settings
   - Save the fine-tuned LoRA adapter
   - Provide inference examples

**Note**: The notebook includes memory optimizations (gradient checkpointing, reduced batch size) suitable for GPUs with 14-22GB VRAM.

## Data Sources

The initial raw resume dataset was created by merging data from two sources:

1. **Resume Corpus** (GitHub): [florex/resume_corpus](https://github.com/florex/resume_corpus/blob/master/resumes_corpus.zip)
2. **Resume Dataset** (Kaggle): [snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

These datasets were combined and processed to create `data/combined_resumes.json`, which was then used to extract structured information and generate the training dataset.

## Dataset Format

> **📊 Dataset Available on Hugging Face**: The complete training dataset (5K samples) is available on [Hugging Face Datasets](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k). You can download and use it directly to train your own models or for research purposes.

The training dataset (`qwen3_training_dataset.jsonl`) contains examples in the following format:

```json
{"text": "<chat template formatted text>"}
```

Each example is formatted using Qwen3's chat template with:
- System message: Instructions for resume parsing
- User message: Resume text
- Assistant message: Structured JSON output

For detailed dataset information, see [`data/dataset_card.md`](data/dataset_card.md). This file contains comprehensive documentation about the dataset structure, fields, usage, and was used when uploading the dataset to Hugging Face.

## Model Configuration

- **Base Model**: Qwen/Qwen3-0.6B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Parameters**:
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Trainable parameters: ~2.3M (0.38% of total parameters)
- **Training Parameters** (for ~5k samples):
  - Batch size: 2
  - Gradient accumulation: 8 (effective batch size: 16)
  - Learning rate: 1e-4
  - Epochs: 3
  - Max sequence length: 1536
  - Optimizer: adamw_torch
  - Gradient checkpointing: Enabled (for memory efficiency)
  - Warmup steps: 100
  - Learning rate scheduler: cosine

## Inference

⚠️ **This repository contains ONLY the LoRA adapter weights.** You must load the base model `Qwen/Qwen3-0.6B` separately.

The training notebook includes inference examples. You can also run inference via the standalone script:

```bash
# Quick inference check (uses LoRA adapter, downloads from Hugging Face)
python notebooks/inference.py
```

Parse a custom resume text file:

```bash
python notebooks/inference.py --resume_file path/to/resume.txt
```

**Note**: For production deployment, see the [Deployment Options](#deployment-options) section below. Or **try the live demos**:
- **CPU Version**: [https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser)
- **GPU Version**: [https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast)

To use the fine-tuned model manually:

1. Load the base model and LoRA adapter
2. Use the Qwen3 chat template with `add_generation_prompt=True`
3. Generate with low temperature (0.1) for deterministic JSON output

See the inference section in `notebooks/train_qwen3_resume_parser.ipynb` for complete examples.

## Deployment Options

This project includes two optimized deployment implementations for different hardware configurations:

### Option 1: CPU-Optimized Deployment (GGUF)

> **🚀 Live Demo Available**: Try the CPU-optimized model at [https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) - no setup required!
> 
> **💡 For Best Performance**: Clone this repository and run locally on your machine. Local execution typically provides better performance than Hugging Face Spaces free-tier CPU instances.

For faster CPU inference (especially on Hugging Face Spaces), you can convert the merged model to GGUF format with quantization. This provides **7-15x faster inference** compared to using transformers directly.

**Best for**: Free-tier Hugging Face Spaces, local CPU inference, cost-effective deployment

> **💡 Tip**: For best performance, clone this repository and run the CPU version locally on your machine. Local execution typically provides better performance than the free-tier Hugging Face Spaces CPU instances.

### Performance Comparison

- **Transformers (CPU)**: ~77 seconds per request
- **GGUF Q5_K_M (CPU)**: ~5-10 seconds per request
- **Improvement**: 7-15x faster

### Conversion Process

**Step 1: Merge LoRA Adapter**

First, merge the LoRA adapter into the base model:

```bash
python inference/merge_lora.py
```

This creates a merged model in `merged_model/` directory.

**Step 2: Convert to GGUF and Quantize**

Convert the merged model to GGUF format and quantize:

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

### Step 3: Test Locally (Recommended)

> **💡 Recommended**: Running the CPU version locally on your machine provides better performance than Hugging Face Spaces free-tier CPU instances. Clone this repository and run locally for optimal results.

After creating the GGUF model, you can test it locally using `test_local.py` which runs a Gradio UI server:

```bash
# Start local Gradio server with streaming enabled
python inference/test_local.py
```

This will:
- Start a local Gradio server on `http://localhost:7860`
- Create a public link (via `share=True`) for easy access
- Enable streaming output so you can see JSON being generated in real-time
- Provide a web UI where you can paste resume text and see results

The script automatically finds the GGUF model in `gguf/qwen3-resume-parser-Q5_K_M.gguf` and mirrors the exact logic of `space/app.py` for local testing.

**Note**: Make sure you have the required dependencies installed:
```bash
pip install llama-cpp-python gradio
```

### Deployment to Hugging Face Spaces (CPU)

The `deploy_hf_space_cpu/` directory contains files optimized for GGUF CPU deployment:

- `deploy_hf_space_cpu/app.py` - Gradio app using llama-cpp-python
- `deploy_hf_space_cpu/Dockerfile` - Docker configuration for CPU Spaces
- `deploy_hf_space_cpu/requirements.txt` - Minimal dependencies (no transformers)
- `deploy_hf_space_cpu/README.md` - CPU Space-specific documentation

**Deployment Steps:**

1. Create a new Hugging Face Space (choose Gradio template)
2. Copy files from `deploy_hf_space_cpu/` to your Space repository
3. Copy the quantized GGUF file:
   ```bash
   cp gguf/qwen3-resume-parser-Q5_K_M.gguf /path/to/space/
   ```
4. If the GGUF file is >100MB, use Git LFS:
   ```bash
   git lfs track "*.gguf"
   git add qwen3-resume-parser-Q5_K_M.gguf
   ```
5. Push to your Space repository

### Option 2: GPU-Accelerated Deployment (8-bit Quantization)

> **🚀 Live Demo Available**: Try the GPU-accelerated model at [https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast) - no setup required!

For maximum performance on GPU hardware, use 8-bit quantization with Transformers. This provides **fastest inference** on NVIDIA T4 or compatible GPUs.

**Best for**: GPU-enabled Hugging Face Spaces, maximum performance, production deployments

### Deployment to Hugging Face Spaces (GPU)

The `deploy_hf_space_gpu/` directory contains files optimized for GPU deployment:

- `deploy_hf_space_gpu/app.py` - Gradio app using Transformers with 8-bit quantization
- `deploy_hf_space_gpu/requirements.txt` - Dependencies including transformers, bitsandbytes, peft
- `deploy_hf_space_gpu/README.md` - GPU Space-specific documentation

**Deployment Steps:**

1. Create a new Hugging Face Space (choose Gradio template, select NVIDIA T4 GPU)
2. Copy files from `deploy_hf_space_gpu/` to your Space repository
3. Push to your Space repository
4. The model will be automatically downloaded from Hugging Face at runtime

**Note**: GPU Spaces require selecting a GPU hardware option (NVIDIA T4 Small recommended) in the Space settings.

### Quantization Options

- **Q5_K_M** (recommended): Best balance for 16GB RAM, high quality, ~400-500MB
- **Q4_K_M**: Smaller size, slightly lower quality, ~300-400MB
- **Q6_K**: Higher quality, larger size, ~500-600MB

For Hugging Face Spaces with 16GB RAM, Q5_K_M is the recommended choice.

## Example Output

Here's what the model produces when given a resume:

**Input Resume:**
```
Senior IT Project Manager with 10+ years experience leading enterprise migrations. 
Skills: Python, SQL, AWS, Agile. Location: Chicago, IL. 
Experience: Project Manager at Acme Corp (2019-2024). 
Education: MS Computer Science.
```

**Model Output (JSON):**
```json
{
  "current_title": "Senior IT Project Manager",
  "years_experience": 10,
  "seniority": "senior",
  "primary_domain": "IT Project Management",
  "core_skills": ["Python", "SQL", "AWS", "Agile"],
  "location": "Chicago, IL",
  "current_company": "Acme Corp",
  "education": "MS Computer Science"
}
```

The model extracts structured information including:
- Job titles (current and previous)
- Companies
- Years of experience
- Skills (core and secondary)
- Education
- Location
- Industries
- And more...

## References

### Training Resources
- [Hugging Face SmolLM3 SFT Guide](https://huggingface.co/learn/smol-course/en/unit1/3)
- [LoRA Fine-Tuning Guide](https://huggingface.co/learn/smol-course/en/unit1/3a)
- [TRL SFTTrainer Documentation](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA Documentation](https://huggingface.co/docs/peft/conceptual_guides/lora)

### Data Sources
- [Resume Corpus (GitHub)](https://github.com/florex/resume_corpus/blob/master/resumes_corpus.zip)
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The fine-tuned model and dataset are also available under Apache 2.0 license on Hugging Face.
