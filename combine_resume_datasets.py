#!/usr/bin/env python3
"""
Combine two resume datasets into a single JSON file.

Dataset 1: PDF files from data/resume_dataset/data/
Dataset 2: Text files from selected_resumes/

Output: Single JSON file with each resume as {"resume_text": "..."}
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET1_ROOT = os.path.join(PROJECT_ROOT, "data", "resume_dataset", "data")
DATASET2_ROOT = os.path.join(PROJECT_ROOT, "data", "selected_resumes")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "combined_resumes.json")


def extract_text_from_pdf(path: str) -> str:
    """
    Extract normalized text from a PDF using pdfplumber.
    Returns a single string with newlines preserved between logical lines.
    """
    pages: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                pages.append(raw)
        
        text = "\n".join(pages)
        # Basic normalization: strip whitespace, drop empty lines
        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️  Error extracting text from {path}: {e}")
        return ""


def normalize_text(text: str) -> str:
    """Normalize text by cleaning whitespace and removing empty lines."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    return "\n".join(lines)


def iter_pdf_resumes(root: str):
    """
    Yield (domain_folder, absolute_pdf_path) for each resume PDF.
    """
    if not os.path.isdir(root):
        print(f"⚠️  Directory not found: {root}")
        return
    
    for domain in os.listdir(root):
        domain_path = os.path.join(root, domain)
        if not os.path.isdir(domain_path):
            continue
        
        for fname in os.listdir(domain_path):
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(domain_path, fname)
            yield domain, path


def extract_dataset1_pdfs(dataset_root: str) -> List[Dict[str, Any]]:
    """
    Extract resumes from PDF files in dataset 1.
    
    Returns:
        List of dictionaries with {"resume_text": "..."}
    """
    resumes = []
    pdf_count = 0
    error_count = 0
    
    print("="*80)
    print("Dataset 1: Extracting from PDF files")
    print("="*80)
    print(f"📁 Directory: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"❌ Directory not found: {dataset_root}")
        return resumes
    
    for domain, pdf_path in iter_pdf_resumes(dataset_root):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            error_count += 1
            continue
        
        text = normalize_text(text)
        if not text:
            error_count += 1
            continue
        
        resumes.append({
            "resume_text": text
        })
        pdf_count += 1
        
        if pdf_count % 100 == 0:
            print(f"  Extracted {pdf_count} resumes from PDFs...")
    
    print(f"\n✅ Dataset 1 Summary:")
    print(f"   Total PDF resumes extracted: {pdf_count}")
    if error_count > 0:
        print(f"   Skipped (empty/error): {error_count}")
    
    return resumes


def extract_dataset2_text_files(dataset_root: str) -> List[Dict[str, Any]]:
    """
    Extract resumes from text files in dataset 2.
    
    Returns:
        List of dictionaries with {"resume_text": "..."}
    """
    resumes = []
    txt_count = 0
    error_count = 0
    
    print("\n" + "="*80)
    print("Dataset 2: Reading from text files")
    print("="*80)
    print(f"📁 Directory: {dataset_root}")
    
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"❌ Directory not found: {dataset_root}")
        return resumes
    
    # Find all .txt files
    txt_files = list(dataset_path.glob("*.txt"))
    print(f"📄 Found {len(txt_files)} .txt files")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            text = normalize_text(text)
            if not text:
                error_count += 1
                continue
            
            resumes.append({
                "resume_text": text
            })
            txt_count += 1
            
            if txt_count % 100 == 0:
                print(f"  Read {txt_count} resumes from text files...")
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  ⚠️  Error reading {txt_file.name}: {e}")
    
    print(f"\n✅ Dataset 2 Summary:")
    print(f"   Total text resumes read: {txt_count}")
    if error_count > 0:
        print(f"   Skipped (empty/error): {error_count}")
    
    return resumes


def combine_datasets(
    dataset1_root: str,
    dataset2_root: str,
    output_file: str,
    append_mode: bool = False
):
    """
    Combine both datasets into a single JSON file.
    
    Args:
        dataset1_root: Path to PDF dataset
        dataset2_root: Path to text file dataset
        output_file: Path to output JSON file
        append_mode: If True, only process dataset 2 and append to existing file
    """
    print("="*80)
    print("Combining Resume Datasets")
    print("="*80)
    
    output_path = Path(output_file)
    
    # Load existing data if in append mode
    existing_resumes = []
    if append_mode and output_path.exists():
        print(f"\n📂 Loading existing resumes from: {output_path}")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_resumes = json.load(f)
            print(f"✅ Loaded {len(existing_resumes)} existing resumes")
        except Exception as e:
            print(f"⚠️  Error loading existing file: {e}")
            print("   Starting fresh...")
            existing_resumes = []
    
    # Extract from datasets
    dataset1_resumes = []
    if not append_mode:
        dataset1_resumes = extract_dataset1_pdfs(dataset1_root)
    else:
        print("\n⏭️  Skipping Dataset 1 (already processed)")
    
    dataset2_resumes = extract_dataset2_text_files(dataset2_root)
    
    # Combine
    all_resumes = existing_resumes + dataset1_resumes + dataset2_resumes
    
    print("\n" + "="*80)
    print("Combining Datasets")
    print("="*80)
    if append_mode:
        print(f"Existing resumes: {len(existing_resumes)}")
    else:
        print(f"Dataset 1 (PDFs): {len(dataset1_resumes)} resumes")
    print(f"Dataset 2 (Text): {len(dataset2_resumes)} resumes")
    print(f"Total combined: {len(all_resumes)} resumes")
    
    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {output_path}")
    
    # Write with pretty formatting for readability
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_resumes, f, indent=2, ensure_ascii=False)
    
    # Verify file was written correctly
    file_size = output_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Successfully saved {len(all_resumes)} resumes to {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Verify by reading back a sample
    print(f"\n🔍 Verifying file integrity...")
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        if len(loaded) == len(all_resumes):
            print(f"✅ Verification passed: {len(loaded)} resumes in file")
            print(f"   Sample resume length: {len(loaded[0]['resume_text'])} characters")
        else:
            print(f"⚠️  Warning: Expected {len(all_resumes)} resumes, found {len(loaded)}")
    except Exception as e:
        print(f"❌ Error verifying file: {e}")
    
    print("\n" + "="*80)
    print("✅ Combination complete!")
    print(f"📁 Output file: {output_path.absolute()}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine resume datasets into single JSON file")
    parser.add_argument(
        "--dataset1",
        type=str,
        default=DATASET1_ROOT,
        help=f"Path to PDF dataset (default: {DATASET1_ROOT})",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default=DATASET2_ROOT,
        help=f"Path to text file dataset (default: {DATASET2_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Output JSON file path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append mode: Only process dataset 2 and append to existing output file",
    )
    
    args = parser.parse_args()
    
    combine_datasets(
        dataset1_root=args.dataset1,
        dataset2_root=args.dataset2,
        output_file=args.output,
        append_mode=args.append
    )

