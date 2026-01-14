#!/usr/bin/env python3
"""
Select random resumes by job title.

For each unique job title, randomly selects 300 resumes and copies
the corresponding .txt files to a separate directory.
"""

from pathlib import Path
from collections import defaultdict
import random
import shutil

def select_resumes_by_title(
    corpus_dir: str = "data/resumes_corpus",
    output_dir: str = "selected_resumes",
    num_per_title: int = 300,
    seed: int = 42
):
    """
    Select random resumes by job title and copy to output directory.
    
    Args:
        corpus_dir: Path to resume corpus directory
        output_dir: Path to output directory for selected resumes
        num_per_title: Number of resumes to select per job title
        seed: Random seed for reproducibility
    """
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    
    if not corpus_path.exists():
        print(f"❌ Directory not found: {corpus_path}")
        return
    
    # Set random seed
    random.seed(seed)
    
    print("="*80)
    print("Selecting Resumes by Job Title")
    print("="*80)
    print(f"\n📁 Source directory: {corpus_path}")
    print(f"📁 Output directory: {output_path}")
    print(f"🎲 Random seed: {seed}")
    print(f"📊 Resumes per title: {num_per_title}")
    
    # Find all .lab files
    lab_files = list(corpus_path.glob("*.lab"))
    print(f"\n📄 Found {len(lab_files)} .lab files")
    
    # Group resumes by job title (using first title only)
    title_to_files = defaultdict(list)
    empty_files = 0
    error_files = 0
    
    print(f"\n📖 Reading job titles from .lab files...")
    for lab_file in lab_files:
        try:
            with open(lab_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    # Take only the first title from each file
                    first_line = content.split('\n')[0].strip()
                    if first_line:
                        # Get the corresponding .txt file
                        txt_file = lab_file.with_suffix('.txt')
                        if txt_file.exists():
                            title_to_files[first_line].append(txt_file)
                        else:
                            error_files += 1
                else:
                    empty_files += 1
        except Exception as e:
            error_files += 1
            if error_files <= 5:
                print(f"   ⚠️  Error reading {lab_file.name}: {e}")
    
    print(f"✅ Grouped resumes by job title")
    if empty_files > 0:
        print(f"   ⚠️  {empty_files} empty .lab files")
    if error_files > 0:
        print(f"   ⚠️  {error_files} files with missing .txt or errors")
    
    # Show statistics
    print("\n" + "="*80)
    print("Resumes by Job Title")
    print("="*80)
    for title, files in sorted(title_to_files.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {title}: {len(files)} resumes")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select and copy resumes
    print("\n" + "="*80)
    print("Selecting and Copying Resumes")
    print("="*80)
    
    total_copied = 0
    selection_stats = []
    
    for title, files in sorted(title_to_files.items()):
        # Randomly select resumes
        num_to_select = min(num_per_title, len(files))
        selected_files = random.sample(files, num_to_select)
        
        # Copy selected files
        copied_count = 0
        for txt_file in selected_files:
            try:
                # Copy to output directory with original filename
                dest_file = output_path / txt_file.name
                shutil.copy2(txt_file, dest_file)
                copied_count += 1
                total_copied += 1
            except Exception as e:
                print(f"   ⚠️  Error copying {txt_file.name}: {e}")
        
        selection_stats.append({
            "title": title,
            "available": len(files),
            "selected": num_to_select,
            "copied": copied_count
        })
        
        print(f"  {title}: Selected {num_to_select}/{len(files)} resumes, copied {copied_count} files")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total unique job titles: {len(title_to_files)}")
    print(f"Total resumes copied: {total_copied}")
    print(f"Output directory: {output_path.absolute()}")
    
    print("\nSelection Details:")
    print(f"{'Job Title':<30} {'Available':<12} {'Selected':<12} {'Copied':<12}")
    print("-" * 80)
    for stat in selection_stats:
        print(f"{stat['title']:<30} {stat['available']:<12} {stat['selected']:<12} {stat['copied']:<12}")
    
    print("\n" + "="*80)
    print(f"✅ Selection complete! {total_copied} resume files copied to {output_path}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Select random resumes by job title")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="data/resumes_corpus",
        help="Path to resume corpus directory (default: data/resumes_corpus)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="selected_resumes",
        help="Output directory for selected resumes (default: selected_resumes)",
    )
    parser.add_argument(
        "--num_per_title",
        type=int,
        default=300,
        help="Number of resumes to select per job title (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    select_resumes_by_title(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        num_per_title=args.num_per_title,
        seed=args.seed
    )

