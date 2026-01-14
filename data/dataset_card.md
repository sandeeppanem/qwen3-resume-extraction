---
license: apache-2.0
task_categories:
- text-generation
- information-extraction
tags:
- resumes
- json
- extraction
- structured-data
- nlp
language:
- en
size_categories:
- 1K<n<10K
---

# Dataset Card for resume-json-extraction-5k

## Dataset Description

This dataset contains 4,879 structured resume entries used for training language models to extract professional profile information from raw resumes. Each example contains a complete resume text and a corresponding structured JSON representation formatted using Qwen3's chat template.

### Dataset Summary

This dataset is designed for supervised fine-tuning of language models to perform information extraction tasks. Each training example consists of:
- **Input**: Raw resume text (unstructured)
- **Output**: Structured JSON representation with extracted professional information

The dataset consists of resume text paired with structured JSON outputs containing comprehensive professional profile information extracted from resumes.

### What This Dataset Contains

Each example in this dataset represents a complete conversation formatted for training:
1. **System Message**: Instructions for the model to act as an expert resume parser
2. **User Message**: The raw resume text to be parsed
3. **Assistant Message**: The structured JSON output with extracted information

The data is formatted using Qwen3's chat template (`<|im_start|>` and `<|im_end|>` tokens) and is ready for direct use in fine-tuning pipelines.

### Languages

The dataset is in English.

## Dataset Structure

### Data Instances

Each instance contains a single field `text` with the complete chat template formatted conversation:

```json
{
  "text": "<|im_start|>system\nYou are an expert resume parser. Extract structured information from resumes and return ONLY valid JSON. Do not include explanations or extra text.<|im_end|>\n<|im_start|>user\nResume:\n[Raw resume text here...]\n<|im_end|>\n<|im_start|>assistant\n{\"current_title\": \"...\", \"previous_titles\": [...], ...}<|im_end|>"
}
```

### Example Data Structure

Here's what a complete example looks like:

**Input (User Message)**:
```
Resume:
SOFTWARE ENGINEER
John Doe
Email: john@example.com

Summary:
Experienced software engineer with 5 years in Python and web development.

Experience:
Senior Software Engineer | Tech Corp | 2020 - Present
- Developed REST APIs using Python and Flask
- Led team of 3 junior developers
...
```

**Output (Assistant Message - Structured JSON)**:
```json
{
  "current_title": "Senior Software Engineer",
  "previous_titles": ["Software Engineer"],
  "current_company": "Tech Corp",
  "previous_companies": ["Previous Company"],
  "years_experience": 5,
  "seniority": "senior",
  "primary_domain": "Software Engineering",
  "industries": ["Technology", "SaaS"],
  "core_skills": ["Python", "REST APIs", "Web Development"],
  "secondary_skills": ["Team Leadership", "Flask", "API Design"],
  "tools": ["Python", "Flask", "Git"],
  "leadership_experience": true,
  "key_achievements": [
    "Led team of 3 junior developers",
    "Developed REST APIs serving 1M+ requests/day"
  ],
  "location": null,
  "summary": "Experienced software engineer with 5 years in Python and web development; led teams and built scalable REST APIs."
}
```

### Data Fields

- **text**: Complete chat template formatted text with system prompt, user resume, and assistant JSON response

### Structured JSON Schema

The assistant output in each example contains a structured JSON object with the following fields:

#### Core Profile Fields
- **`current_title`** (string): The most recent or current job title
- **`previous_titles`** (array of strings): List of prior job titles (up to 3 most recent)
- **`current_company`** (string): The most recent or current company name
- **`previous_companies`** (array of strings): List of prior companies (up to 3 most recent)
- **`years_experience`** (number): Total years of professional experience (e.g., 7.5)
- **`seniority`** (string): Seniority level - one of: `"junior"`, `"mid"`, `"senior"`, `"lead"`, or `"principal"`

#### Domain and Industry
- **`primary_domain`** (string): Primary professional domain or field (e.g., "Software Engineering", "Marketing")
- **`industries`** (array of strings): List of industries the person has worked in (up to 3 most prominent)

#### Skills and Tools
- **`core_skills`** (array of strings): Primary/core skills (up to 3 most prominent)
- **`secondary_skills`** (array of strings): Secondary or supporting skills (up to 3 most prominent)
- **`tools`** (array of strings or null): Tools, software, or technologies used (up to 3 most prominent)

#### Experience and Achievements
- **`leadership_experience`** (boolean): Whether the resume indicates leadership, management, or mentoring experience
- **`key_achievements`** (array of strings): Impact-oriented achievements with metrics or outcomes (up to 3 highest-impact)

#### Additional Information
- **`location`** (string or null): Geographic location if mentioned in the resume
- **`summary`** (string): Concise professional summary (maximum 300 characters)

#### Field Constraints
- Array fields are limited to a maximum of 3 items (most recent or most relevant)
- Fields may be `null` if information is not present or cannot be confidently determined
- All information is extracted only from what is explicitly stated in the resume (no inference)

### Data Splits

- **train**: 4,879 examples (100% of dataset)

## Dataset Creation

### Source Data

The dataset was created from extracted resume data where:
1. Raw resume text was collected
2. Structured information was extracted using LLM-based extraction
3. Examples were formatted using Qwen3 chat template
4. Only successful extractions (no errors) were included

### Preprocessing

1. Filtered out examples with extraction errors
2. Converted to Qwen3 chat template format
3. Applied tokenizer chat template with:
   - `add_generation_prompt=False` (for training)
   - `enable_thinking=False` (for JSON output)

### Annotations

Each example includes:
- System prompt: Instructions for resume parsing
- User input: Raw resume text
- Assistant output: Structured JSON with extracted information

## Uses

### Direct Use

This dataset is intended for fine-tuning language models to:
- Extract structured information from resumes
- Generate JSON output from unstructured text
- Perform information extraction tasks

### Out-of-Scope Use

This dataset should not be used for:
- Training models to generate fake resumes
- Extracting personal information for unauthorized purposes
- Any use case that violates privacy or data protection laws

## Bias, Risks, and Limitations

### Bias

- Dataset may reflect biases present in the source resume data
- Geographic and industry representation may be uneven
- May not represent all resume formats or styles

### Risks

- Models trained on this data may perpetuate biases
- Extracted information should be verified for accuracy
- Not suitable for automated hiring decisions without human review

### Limitations

- Limited to ~5k examples
- May not generalize to all resume formats
- JSON structure is fixed and may not capture all resume information

## Citation

```bibtex
@misc{resume_json_extraction_5k,
  title={resume-json-extraction-5k},
  author={Sandeep Panem},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k}}
}
```

## License

This dataset is licensed under Apache 2.0.

## Contact

For questions or issues, please open an issue on the dataset repository.
