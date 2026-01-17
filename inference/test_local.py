#!/usr/bin/env python3
"""
Local testing script for Qwen3 Resume Parser using llama.cpp (GGUF format).

This script mirrors the exact logic of app.py for local testing with Gradio UI.
Runs a local server where users can paste resume text and see streaming output.

The GGUF model file should be in the gguf/ directory at the project root.
"""

import gradio as gr
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

# Model configuration - GGUF file is in the gguf/ directory at project root
MODEL_PATH = "qwen3-resume-parser-Q5_K_M.gguf"

# Global variables for model caching
_model = None

# Shared result cache (key: hash of resume text, value: (formatted_json, raw_output))
# Using OrderedDict for FIFO eviction when cache is full
_result_cache = OrderedDict()
MAX_CACHE_SIZE = 100  # Keep last 100 results


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
    project_root = script_dir.parent  # inference/ -> project root
    possible_paths = [
        Path(MODEL_PATH),  # Current directory
        script_dir / MODEL_PATH,  # Same directory as test_local.py (inference/)
        project_root / "gguf" / MODEL_PATH,  # gguf/ directory at project root
        project_root / MODEL_PATH,  # Project root directory
        project_root.parent / "gguf" / MODEL_PATH,  # Fallback
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
        print(f"Project root: {project_root.absolute()}")
        print(f"Files in script directory: {list(script_dir.iterdir())}")
        if (project_root / "gguf").exists():
            print(f"Files in gguf directory: {list((project_root / 'gguf').iterdir())}")
        raise FileNotFoundError(
            f"GGUF model not found. Tried: {[str(p) for p in possible_paths]}\n"
            f"Make sure {MODEL_PATH} is in the gguf/ directory at project root."
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
            n_batch=64,  # Testing with reduced batch size
            n_gpu_layers=0,
            chat_format=None,  # Disable chat format parsing for speed
            verbose=False,
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
    
    # Normalize resume text for caching (strip whitespace)
    normalized_text = resume_text.strip()
    
    # Create hash key for cache lookup
    cache_key = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
    
    # Check cache first
    if cache_key in _result_cache:
        # Move to end (most recently used) for LRU-like behavior
        cached_json, cached_raw = _result_cache.pop(cache_key)
        _result_cache[cache_key] = (cached_json, cached_raw)
        yield cached_json, cached_raw
        return
    
    try:
        model = load_model()
        
        MAX_RESUME_CHARS = 4000
        if len(normalized_text) > MAX_RESUME_CHARS:
            truncated = normalized_text[:MAX_RESUME_CHARS]
            last_space = truncated.rfind(' ', MAX_RESUME_CHARS - 200, MAX_RESUME_CHARS)
            if last_space > MAX_RESUME_CHARS - 500:
                truncated = truncated[:last_space]
            normalized_text = truncated + "..."
        
        prompt = format_qwen3_prompt(normalized_text)
        accumulated_text = ""
        
        stream = model(
            prompt,
            max_tokens=350,
            temperature=0.1,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
            stream=True,
        )
        
        # Process streamed tokens
        final_json = None
        final_raw = None
        chunk_count = 0
        
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    accumulated_text += delta
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
        
        # Final processing after stream completes
        assistant_response = accumulated_text.strip()
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        assistant_response = re.sub(r'</?redacted_reasoning>', '', assistant_response)
        assistant_response = re.sub(r'</?think>', '', assistant_response)
        assistant_response = re.sub(r'\n\s*\n+', '\n', assistant_response)
        assistant_response = assistant_response.strip()
        
        try:
            parsed_json = json.loads(assistant_response)
            formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            final_json = formatted_json
            final_raw = assistant_response
            yield formatted_json, assistant_response
        except json.JSONDecodeError:
            yield (
                f"⚠️ Model output is not valid JSON:\n\n{assistant_response}",
                assistant_response,
            )
            return  # Don't cache invalid JSON
        
        # Cache the result for future users (only if we got valid JSON)
        if final_json and final_raw:
            # Enforce cache size limit (FIFO eviction)
            if len(_result_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry (first item in OrderedDict)
                _result_cache.popitem(last=False)
            
            # Add new result to cache
            _result_cache[cache_key] = (final_json, final_raw)
    
    except Exception as e:
        yield f"❌ Error: {str(e)}", ""


def parse_resume(resume_text: str) -> tuple[str, str]:
    """Parse resume text and return structured JSON (non-streaming version)."""
    result = None
    for result in parse_resume_stream(resume_text):
        pass
    return result if result else ("⚠️ No output generated", "")


# Load model at startup
try:
    # Load model at startup (will show error in logs if it fails)
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Warning: Could not load model at startup: {e}")
        print("Model will be loaded on first use.")
except Exception as e:
    print(f"Error loading model: {e}")


# Gradio Interface
def create_interface():
    """Create and return Gradio interface."""
    
    with gr.Blocks(title="Qwen3 Resume Structured Information Extraction (Local)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 Qwen3 Resume Structured Information Extraction
            
            Extract structured information from resumes using fine-tuned Qwen3-0.6B model.
            **Optimized for CPU inference using llama.cpp and Q5_K_M quantization.**
            
            **How to use:**
            1. Paste your resume text in the text box below
            2. Click "Parse Resume" 
            3. View the extracted structured JSON output (streaming enabled)
            
            **Model:** [sandeeppanem/qwen3-0.6b-resume-json](https://huggingface.co/sandeeppanem/qwen3-0.6b-resume-json)  
            **Dataset:** [sandeeppanem/resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k)  
            **Repository:** [qwen3-resume-extraction](https://github.com/sandeeppanem/qwen3-resume-extraction)  
            **Format:** GGUF Q5_K_M (optimized for CPU)
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
                    label="Extracted JSON",
                    language="json",
                    lines=20,
                )
                raw_output = gr.Textbox(
                    label="Raw Model Output",
                    lines=5,
                    visible=False,
                )
        
        # Examples - diverse resume samples
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
            fn=parse_resume_stream,  # Use streaming version
            inputs=resume_input,
            outputs=[json_output, raw_output],
        )
        
        # Also parse on Enter key with streaming
        resume_input.submit(
            fn=parse_resume_stream,  # Use streaming version
            inputs=resume_input,
            outputs=[json_output, raw_output],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
