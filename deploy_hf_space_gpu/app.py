#!/usr/bin/env python3
"""
Gradio app for Qwen3 Resume Parser - GPU-Accelerated Deployment.

This app is optimized for GPU inference using Transformers with 8-bit quantization.
Deployed on Hugging Face Spaces (NVIDIA T4 GPU) for maximum performance.

Features:
- Fast GPU inference using Transformers with 8-bit quantization
- Streaming output for real-time JSON generation
- Result caching for improved performance
- Base model + LoRA adapter loaded at runtime (efficient deployment)

Live Demo: https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser-fast
"""

import gradio as gr
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

# Model configuration
BASE_MODEL = "Qwen/Qwen3-0.6B"  # Base model
LORA_ADAPTER = "sandeeppanem/qwen3-0.6b-resume-json"  # LoRA adapter

# Global variables for model caching
_model = None
_tokenizer = None
_device = None

# Shared result cache (key: hash of resume text, value: (formatted_json, raw_output))
_result_cache = OrderedDict()
MAX_CACHE_SIZE = 100


def load_model():
    """Load base model + LoRA adapter with 8-bit quantization for GPU."""
    global _model, _tokenizer, _device
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "bitsandbytes not installed. "
            "Install with: pip install bitsandbytes"
        )
    
    print("Loading base model and LoRA adapter...")
    
    # Check for GPU
    if torch.cuda.is_available():
        _device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        _device = "cpu"
        print("⚠️ CUDA not available, using CPU")
    
    # 8-bit quantization config for T4 GPU (16GB)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Load tokenizer from base model
    print("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    print("✓ Tokenizer loaded")
    
    # Load base model with 8-bit quantization for GPU
    print("Loading base model...")
    if _device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Fallback to CPU (not recommended for production)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        base_model = base_model.to(_device)
    print("✓ Base model loaded")
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    _model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    _model.eval()
    print("✓ LoRA adapter loaded")
    print(f"✓ Model ready on {_device}")
    
    return _model, _tokenizer


def build_messages(resume_text: str) -> list[dict[str, str]]:
    """Build messages for Qwen3 chat template."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert resume parser. "
                "Extract structured information from resumes and return ONLY valid JSON. "
                "Do not include explanations or extra text."
            ),
        },
        {"role": "user", "content": f"Resume:\n{resume_text.strip()}"},
    ]


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
    """Parse resume with streaming output."""
    global _model, _tokenizer, _device
    
    if not resume_text or not resume_text.strip():
        yield "⚠️ Please provide resume text", ""
        return
    
    # Check cache
    normalized_text = resume_text.strip().lower()
    cache_key = hashlib.md5(normalized_text.encode()).hexdigest()
    
    if cache_key in _result_cache:
        cached_json, cached_raw = _result_cache[cache_key]
        yield cached_json, cached_raw
        return
    
    try:
        model, tokenizer = load_model()
        
        MAX_RESUME_CHARS = 4000
        if len(normalized_text) > MAX_RESUME_CHARS:
            truncated = normalized_text[:MAX_RESUME_CHARS]
            last_space = truncated.rfind(' ', MAX_RESUME_CHARS - 200, MAX_RESUME_CHARS)
            if last_space > MAX_RESUME_CHARS - 500:
                truncated = truncated[:last_space]
            normalized_text = truncated + "..."
        
        messages = build_messages(normalized_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,  # 5 minute timeout
        )
        
        # Generation parameters
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 350,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread with inference_mode
        def generate_with_inference_mode():
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        
        generation_thread = threading.Thread(
            target=generate_with_inference_mode,
            daemon=True,
        )
        generation_thread.start()
        
        # Process streamed tokens
        accumulated_text = ""
        final_json = None
        final_raw = None
        chunk_count = 0
        
        for new_text in streamer:
            if new_text:
                accumulated_text += new_text
                chunk_count += 1
                
                # Only do expensive operations every 5 chunks or if we have enough text
                # This reduces overhead during streaming
                if chunk_count % 5 == 0 or len(accumulated_text) > 50:
                    cleaned_text = accumulated_text
                    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
                    cleaned_text = re.sub(r'</?redacted_reasoning>', '', cleaned_text)
                    cleaned_text = re.sub(r'</?think>', '', cleaned_text)
                    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    
                    try:
                        parsed_json = json.loads(cleaned_text)
                        formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                        final_json = formatted_json
                        final_raw = cleaned_text
                        yield formatted_json, cleaned_text
                    except json.JSONDecodeError:
                        formatted_incomplete = _format_incomplete_json(cleaned_text)
                        yield formatted_incomplete, cleaned_text
        
        # Wait for generation thread to complete
        generation_thread.join()
        
        # Final processing after stream completes
        assistant_response = accumulated_text.strip()
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        assistant_response = re.sub(r'</?redacted_reasoning>', '', assistant_response)
        assistant_response = re.sub(r'</?think>', '', assistant_response)
        assistant_response = re.sub(r'\n\s*\n+', '\n', assistant_response)
        assistant_response = assistant_response.strip()
        
        # Final JSON parsing and caching
        try:
            parsed_json = json.loads(assistant_response)
            formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            final_json = formatted_json
            final_raw = assistant_response
            
            # Cache result
            if len(_result_cache) >= MAX_CACHE_SIZE:
                _result_cache.popitem(last=False)  # Remove oldest
            _result_cache[cache_key] = (formatted_json, assistant_response)
            
            yield formatted_json, assistant_response
        except json.JSONDecodeError:
            formatted_incomplete = _format_incomplete_json(assistant_response)
            yield formatted_incomplete, assistant_response
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        yield error_msg, ""


def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(title="Qwen3 Resume Structured Information Extraction ", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 Qwen3 Resume Structured Information Extraction
            
            Extract structured information from resumes using fine-tuned Qwen3-0.6B model.
            **Optimized with 8-bit quantization.**
            
            **How to use:**
            1. Paste your resume text in the text box below
            2. Click "Parse Resume" 
            3. View the extracted structured JSON output
            
            **Model:** [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)  
            **Dataset:** [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)  
            **Repository:** [qwen3-resume-extraction](https://github.com/sandeeppanem/qwen3-resume-extraction)  
            **Format:** Base model + LoRA adapter with 8-bit quantization (GPU)  
            **💻 CPU Version:** [Try the free CPU version](https://huggingface.co/spaces/sandeeppanem/qwen3-resume-parser) (GGUF optimized) 
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                resume_input = gr.Textbox(
                    label="Resume Text",
                    placeholder="Paste your resume text here...",
                    lines=15,
                    max_lines=20,
                )
                parse_btn = gr.Button("Parse Resume", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    **Example:**
                    ```
                    Senior IT Project Manager with 10+ years experience leading enterprise migrations. 
                    Skills: Python, SQL, AWS, Agile. Location: Chicago, IL. 
                    Experience: Project Manager at Acme Corp (2019-2024). 
                    Education: MS Computer Science.
                    ```
                    """
                )
            
            with gr.Column(scale=1):
                json_output = gr.Code(
                    label="Structured JSON Output",
                    language="json",
                    lines=20,
                )
                raw_output = gr.Textbox(
                    label="Raw Output",
                    lines=10,
                    visible=False,
                )
        
        # Examples - diverse resume samples (same as CPU version)
        example_resumes = [
            """Senior IT Project Manager with 10+ years experience leading enterprise migrations. 
Skills: Python, SQL, AWS, Agile. Location: Chicago, IL. 
Experience: Project Manager at Acme Corp (2019-2024). 
Education: MS Computer Science.""",
            
            """Software Engineer
John Smith
Email: john.smith@email.com | Phone: (555) 123-4567 | Location: San Francisco, CA

PROFESSIONAL SUMMARY
Full Stack Developer with 5 years of experience building scalable web applications. 
Expertise in React, Node.js, Python, and cloud technologies.

TECHNICAL SKILLS
Languages: JavaScript, Python, TypeScript, Java
Frameworks: React, Node.js, Express, Django, Spring Boot
Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis

PROFESSIONAL EXPERIENCE
Senior Software Engineer | TechCorp Inc. | San Francisco, CA | 2021 - Present
- Developed microservices architecture serving 1M+ users
- Led team of 3 junior developers
- Reduced API response time by 40% through optimization

Software Engineer | StartupXYZ | San Francisco, CA | 2019 - 2021
- Built customer-facing React applications
- Implemented CI/CD pipelines using Jenkins

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2019""",
            
            """Data Scientist
Sarah Johnson
sarah.johnson@email.com | (555) 987-6543 | New York, NY

SUMMARY
Data Scientist with 7 years of experience in machine learning, statistical analysis, and big data. 
Specialized in NLP and computer vision applications.

SKILLS
Programming: Python, R, SQL, Scala
ML/AI: TensorFlow, PyTorch, scikit-learn, XGBoost
Tools: Spark, Hadoop, Tableau, Jupyter
Cloud: AWS SageMaker, Azure ML

EXPERIENCE
Lead Data Scientist | DataTech Solutions | New York, NY | 2020 - Present
- Built recommendation system increasing user engagement by 35%
- Developed NLP models for sentiment analysis
- Managed team of 4 data scientists

Data Scientist | Analytics Pro | New York, NY | 2018 - 2020
- Created predictive models for customer churn
- Analyzed large datasets using Spark

EDUCATION
Master of Science in Data Science | Columbia University | 2018
Bachelor of Science in Statistics | NYU | 2016""",
            
            """Marketing Manager
Michael Chen
michael.chen@email.com | (555) 456-7890 | Los Angeles, CA

PROFESSIONAL PROFILE
Strategic Marketing Manager with 8+ years driving brand growth and digital marketing campaigns. 
Expert in SEO, content marketing, and social media strategy.

CORE COMPETENCIES
Digital Marketing, SEO/SEM, Content Strategy, Social Media Management, 
Google Analytics, HubSpot, Marketo, Brand Management

PROFESSIONAL EXPERIENCE
Marketing Manager | BrandCo | Los Angeles, CA | 2019 - Present
- Increased website traffic by 150% through SEO optimization
- Launched successful social media campaigns reaching 2M+ impressions
- Managed $500K annual marketing budget

Marketing Specialist | Growth Agency | Los Angeles, CA | 2016 - 2019
- Developed content marketing strategies
- Executed email marketing campaigns with 25% open rate

EDUCATION
Master of Business Administration (MBA) | UCLA | 2016
Bachelor of Arts in Communications | USC | 2014""",
            
            """Product Manager
Emily Rodriguez
emily.rodriguez@email.com | (555) 234-5678 | Seattle, WA

OVERVIEW
Product Manager with 6 years of experience in B2B SaaS products. 
Led product launches from concept to market, working with engineering and design teams.

KEY SKILLS
Product Strategy, Agile/Scrum, User Research, A/B Testing, 
Roadmap Planning, Stakeholder Management, JIRA, Figma

WORK EXPERIENCE
Senior Product Manager | CloudSoft | Seattle, WA | 2020 - Present
- Launched 3 major product features, increasing revenue by $2M annually
- Conducted user research and usability testing
- Managed product roadmap and prioritized features

Product Manager | StartupHub | Seattle, WA | 2018 - 2020
- Owned product lifecycle for mobile application
- Collaborated with cross-functional teams

EDUCATION
Master of Science in Product Management | University of Washington | 2018
Bachelor of Science in Business Administration | Washington State University | 2016""",
            
            """DevOps Engineer
David Kim
david.kim@email.com | (555) 345-6789 | Austin, TX

SUMMARY
DevOps Engineer with 4 years of experience in CI/CD, infrastructure automation, and cloud architecture. 
Proven track record of improving deployment efficiency and system reliability.

TECHNICAL SKILLS
Cloud Platforms: AWS, Azure, GCP
CI/CD: Jenkins, GitLab CI, GitHub Actions, CircleCI
Infrastructure: Terraform, Ansible, CloudFormation
Containers: Docker, Kubernetes, ECS
Monitoring: Prometheus, Grafana, ELK Stack
Scripting: Bash, Python, PowerShell

EXPERIENCE
DevOps Engineer | CloudInfra Inc. | Austin, TX | 2021 - Present
- Reduced deployment time from 2 hours to 15 minutes
- Implemented infrastructure as code using Terraform
- Set up monitoring and alerting systems

Junior DevOps Engineer | TechStart | Austin, TX | 2020 - 2021
- Maintained CI/CD pipelines
- Managed cloud infrastructure on AWS

EDUCATION
Bachelor of Science in Computer Engineering
University of Texas at Austin | 2020"""
        ]
        
        gr.Examples(
            examples=[[resume] for resume in example_resumes],
            inputs=resume_input,
            label="Select a sample resume:",
        )
        
        # Connect button with streaming
        parse_btn.click(
            fn=parse_resume_stream,
            inputs=resume_input,
            outputs=[json_output, raw_output],
        )
        
        gr.Examples(
            examples=[
                ["Senior Software Engineer with 10+ years of experience in Python, Java, and cloud technologies. Currently at Google, previously at Microsoft. MS in Computer Science from Stanford."],
                ["Data Scientist with expertise in machine learning, deep learning, and NLP. 5 years experience. PhD in Statistics. Proficient in Python, TensorFlow, PyTorch."],
                ["Product Manager with 8 years experience in tech startups. MBA from Wharton. Led product launches at 3 companies. Strong in Agile, Scrum, and product strategy."],
            ],
            inputs=resume_input,
            outputs=[json_output, raw_output],
        )
    
    return demo


if __name__ == "__main__":
    
    # Enable TF32 for faster GPU matrix operations (NVIDIA Ampere+ GPUs)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✓ TF32 enabled for faster GPU inference")
    
    # Load model at startup
    print("=" * 60)
    print("Application Startup")
    print("=" * 60)
    try:
        load_model()
        print("✓ Model loaded successfully at startup")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model at startup: {e}")
        print("Model will be loaded on first use.")
    
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

