# Qwen3 Resume Extraction Fine-Tuning

Fine-tune Qwen3-0.6B base model for resume parsing using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🚀 Model & Dataset Links

- **Fine-tuned Model**: [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json) on Hugging Face
- **Training Dataset**: [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k) on Hugging Face

> **Note**: The fine-tuned model and training dataset are hosted on Hugging Face. You can download and use them directly from there. This repository contains the code and pipeline for reproducing the fine-tuning process.

## Project Structure

```
.
├── data/
│   ├── combined_resumes.json          # Raw resume data (merged from 2 sources)
│   ├── extracted_resumes.json        # Extracted structured data (resume → JSON pairs)
│   ├── qwen3_training_dataset.jsonl  # Training dataset (Qwen3 chat template format)
│   └── dataset_card.md                # Dataset documentation for Hugging Face
├── notebooks/
│   ├── generate_training_data.ipynb  # Generate training data pairs (resume → JSON)
│   └── train_qwen3_resume_parser.ipynb  # Fine-tuning notebook (for Google Colab)
├── select_resumes_by_title.py        # Preprocessing: Select resumes by job title
├── combine_resume_datasets.py        # Preprocessing: Merge datasets from 2 sources
├── convert_to_qwen3_dataset.py       # Convert extracted data to Qwen3 format
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
python select_resumes_by_title.py \
    --corpus_dir data/resumes_corpus \
    --output_dir data/selected_resumes \
    --num_per_title 300
```
- Selects random resumes by job title from the GitHub corpus
- Outputs text files to `data/selected_resumes/`

**1.2 Combine Datasets** (merge from 2 sources):
```bash
python combine_resume_datasets.py \
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
python convert_to_qwen3_dataset.py
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
python select_resumes_by_title.py \
    --corpus_dir data/resumes_corpus \
    --output_dir data/selected_resumes

# Step 2: Combine datasets from both sources
python combine_resume_datasets.py \
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
python convert_to_qwen3_dataset.py

# Or specify custom paths
python convert_to_qwen3_dataset.py \
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
python inference.py
```

Parse a custom resume text file:

```bash
python inference.py --resume_file path/to/resume.txt
```

To use the fine-tuned model manually:

1. Load the base model and LoRA adapter
2. Use the Qwen3 chat template with `add_generation_prompt=True`
3. Generate with low temperature (0.1) for deterministic JSON output

See the inference section in `notebooks/train_qwen3_resume_parser.ipynb` for complete examples.

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
