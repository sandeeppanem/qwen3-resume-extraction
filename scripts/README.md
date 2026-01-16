# Data Preprocessing Scripts

This directory contains scripts for preprocessing raw resume data before training.

## Scripts

### `select_resumes_by_title.py`
Selects resumes from the GitHub Resume Corpus based on job titles.

**Usage:**
```bash
python scripts/select_resumes_by_title.py \
    --corpus_dir data/resumes_corpus \
    --output_dir data/selected_resumes \
    --num_per_title 300
```

### `combine_resume_datasets.py`
Combines resume data from multiple sources (e.g., Kaggle dataset + GitHub corpus).

**Usage:**
```bash
python scripts/combine_resume_datasets.py \
    --dataset1 data/resume_dataset/data \
    --dataset2 data/selected_resumes \
    --output data/combined_resumes.json
```

### `convert_to_qwen3_dataset.py`
Converts extracted resume data (resume → JSON pairs) to Qwen3 chat template format for training.

**Usage:**
```bash
python scripts/convert_to_qwen3_dataset.py
```

Or with custom paths:
```bash
python scripts/convert_to_qwen3_dataset.py \
    --input data/extracted_resumes.json \
    --output data/qwen3_training_dataset.jsonl \
    --model Qwen/Qwen3-0.6B
```

## Workflow

1. **Select Resumes**: Use `select_resumes_by_title.py` to filter resumes by job title
2. **Combine Datasets**: Use `combine_resume_datasets.py` to merge multiple data sources
3. **Extract Structured Data**: Use `notebooks/generate_training_data.ipynb` to extract JSON from resumes
4. **Convert Format**: Use `convert_to_qwen3_dataset.py` to create training dataset

See the main [README.md](../README.md) for the complete pipeline.

