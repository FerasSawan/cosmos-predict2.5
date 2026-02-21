#!/usr/bin/env python3
"""
Download the FlywheelAI Excavator Dataset from HuggingFace.

This script downloads only the front camera (D01) and joystick data (D05),
which is all that's needed for action-conditioned fine-tuning.

Usage:
    python scripts/excavator/download_dataset.py --output-dir datasets/excavator

    # To download all cameras (not recommended, 4x larger):
    python scripts/excavator/download_dataset.py --output-dir datasets/excavator --all-cameras

Requirements:
    pip install huggingface_hub tqdm
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_excavator_dataset(output_dir: str, token: str = None, all_cameras: bool = False):
    """
    Download the FlywheelAI excavator dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        token: HuggingFace token (optional, for private datasets)
        all_cameras: If True, download all 4 cameras. If False, only front camera + CSV.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if all_cameras:
        expected_size = "~84 GB"
        file_description = "all 4 camera angles + joystick data"
        allow_patterns = None  # Download everything
    else:
        expected_size = "~22 GB"
        file_description = "front camera (D01) + joystick data (D05) only"
        # Only download front camera videos and CSV files
        allow_patterns = [
            "*/D01_*.mp4",  # Front camera videos
            "*/D05_*.csv",  # Joystick CSV files
            "*.parquet",    # Index file
            "README.md",    # Documentation
        ]
    
    print("=" * 60)
    print("Downloading FlywheelAI Excavator Dataset")
    print("=" * 60)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Dataset: FlywheelAI/excavator-dataset")
    print(f"Downloading: {file_description}")
    print(f"Expected size: {expected_size}")
    print("=" * 60)
    print()
    
    # Download the dataset
    print("Starting download... This may take a while depending on your connection.")
    print("The dataset contains 176 sessions.")
    print()
    
    try:
        local_dir = snapshot_download(
            repo_id="FlywheelAI/excavator-dataset",
            repo_type="dataset",
            local_dir=str(output_path),
            token=token,
            resume_download=True,  # Resume if interrupted
            allow_patterns=allow_patterns,  # Filter to only needed files
        )
        
        print()
        print("=" * 60)
        print("Download complete!")
        print(f"Dataset saved to: {local_dir}")
        print("=" * 60)
        
        # Print dataset statistics
        print_dataset_stats(output_path)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you have huggingface_hub installed: pip install huggingface_hub")
        print("2. If the dataset requires authentication, run: huggingface-cli login")
        print("3. Check your internet connection")
        raise


def print_dataset_stats(dataset_path: Path):
    """Print statistics about the downloaded dataset."""
    print()
    print("Dataset Statistics:")
    print("-" * 40)
    
    # Count sessions (folders with timestamp names)
    sessions = [d for d in dataset_path.iterdir() 
                if d.is_dir() and d.name.isdigit() and len(d.name) == 14]
    print(f"Total sessions: {len(sessions)}")
    
    # Count video files
    front_video_count = 0
    all_video_count = 0
    csv_count = 0
    for session in sessions:
        front_video_count += len(list(session.glob("D01_*.mp4")))
        all_video_count += len(list(session.glob("D0[1-4]_*.mp4")))
        csv_count += len(list(session.glob("D05_*.csv")))
    
    print(f"Front camera videos (D01): {front_video_count}")
    if all_video_count > front_video_count:
        print(f"Total video files (all cameras): {all_video_count}")
    print(f"Joystick CSV files (D05): {csv_count}")
    
    # Check for parquet file
    parquet_file = dataset_path / "excavator_dataset.parquet"
    if parquet_file.exists():
        print(f"Parquet index file: Found")
    else:
        print(f"Parquet index file: Not found")
    
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Download the FlywheelAI Excavator Dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/excavator",
        help="Directory to save the dataset (default: datasets/excavator)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, use if dataset requires authentication)"
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Download all 4 camera angles (default: only front camera + CSV)"
    )
    
    args = parser.parse_args()
    
    download_excavator_dataset(args.output_dir, args.token, args.all_cameras)


if __name__ == "__main__":
    main()
