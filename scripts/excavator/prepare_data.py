#!/usr/bin/env python3
"""
Prepare the FlywheelAI Excavator Dataset for Cosmos Predict 2.5 action-conditioned training.

This script:
1. Parses joystick CSV files and combines marker_id 0 & 1 into 4D action vectors
2. Pads 4D actions to 7D for Cosmos compatibility
3. Resizes videos from 1920x1080 to 480x640
4. Creates JSON annotation files in Cosmos format
5. Splits data into train/val sets (90/10)

Usage:
    python scripts/excavator/prepare_data.py \
        --input-dir datasets/excavator \
        --output-dir datasets/excavator_cosmos \
        --camera front \
        --resolution 480,640

Requirements:
    pip install pandas numpy opencv-python tqdm
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# Camera mapping
CAMERA_MAP = {
    "front": "D01",
    "left": "D02", 
    "back": "D03",
    "right": "D04",
}


def parse_joystick_csv(csv_path: Path) -> np.ndarray:
    """
    Parse joystick CSV and combine marker_id 0 & 1 into 4D action vectors.
    
    The CSV has format:
        timestamp, filename, marker_id, normalized_x, normalized_y
    
    marker_id 0 (Right joystick): X=bucket, Y=boom
    marker_id 1 (Left joystick): X=cab rotation, Y=arm
    
    Returns:
        np.ndarray: Shape (num_frames, 4) with columns [bucket, boom, cab, arm]
    """
    df = pd.read_csv(csv_path)
    
    # Separate by marker_id
    right_joy = df[df['marker_id'] == 0.0].copy()
    left_joy = df[df['marker_id'] == 1.0].copy()
    
    # Sort by timestamp
    right_joy = right_joy.sort_values('timestamp').reset_index(drop=True)
    left_joy = left_joy.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure same length (should be, but handle edge cases)
    min_len = min(len(right_joy), len(left_joy))
    right_joy = right_joy.iloc[:min_len]
    left_joy = left_joy.iloc[:min_len]
    
    # Combine into 4D action vector: [bucket, boom, cab, arm]
    actions = np.column_stack([
        right_joy['normalized_x'].values,  # bucket (right X)
        right_joy['normalized_y'].values,  # boom (right Y)
        left_joy['normalized_x'].values,   # cab rotation (left X)
        left_joy['normalized_y'].values,   # arm (left Y)
    ])
    
    return actions


def pad_actions_to_7d(actions_4d: np.ndarray) -> np.ndarray:
    """
    Pad 4D excavator actions to 7D for Cosmos compatibility.
    
    Cosmos expects: [x, y, z, roll, pitch, yaw, gripper]
    We map: [bucket, boom, cab, arm, 0, 0, 0]
    
    Args:
        actions_4d: Shape (num_frames, 4)
    
    Returns:
        np.ndarray: Shape (num_frames, 7)
    """
    num_frames = actions_4d.shape[0]
    padding = np.zeros((num_frames, 3))
    return np.hstack([actions_4d, padding])


def compute_states_from_actions(actions: np.ndarray) -> np.ndarray:
    """
    Compute cumulative states from actions.
    
    For excavator, we treat the joystick positions as velocities and
    integrate to get pseudo-states. This is a simplification since we
    don't have actual kinematic state data.
    
    Args:
        actions: Shape (num_frames, 7)
    
    Returns:
        np.ndarray: Shape (num_frames + 1, 7) - states including initial state
    """
    # Start from zero state
    initial_state = np.zeros(7)
    
    # Cumulative sum of actions to get states
    # Scale down since actions are velocities
    state_changes = actions * 0.1
    states = np.vstack([initial_state, np.cumsum(state_changes, axis=0)])
    
    return states


def resize_video(input_path: Path, output_path: Path, target_height: int, target_width: int):
    """
    Resize video in a completely separate process so a crash doesn't kill the main script.
    """
    resize_script = Path(__file__).parent / "resize_single.py"
    result = subprocess.run(
        [sys.executable, str(resize_script), str(input_path), str(output_path), str(target_width), str(target_height)],
        timeout=1800,
    )
    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Video resize failed with exit code {result.returncode}")


def create_annotation_json(
    episode_id: str,
    video_path: str,
    actions: np.ndarray,
    states: np.ndarray,
    task_description: str = "excavator operation"
) -> Dict:
    """
    Create annotation JSON in Cosmos format.
    
    Args:
        episode_id: Unique identifier for this episode
        video_path: Relative path to video file
        actions: Action array of shape (num_frames, 7)
        states: State array of shape (num_frames + 1, 7)
        task_description: Text description of the task
    
    Returns:
        Dict: Annotation dictionary in Cosmos format
    """
    # Convert numpy arrays to lists for JSON serialization
    actions_list = actions.tolist()
    states_list = states.tolist()
    
    # Create gripper state (we use the last dimension, which is 0 for excavator)
    # But we'll create a plausible continuous value based on bucket position
    bucket_positions = actions[:, 0]  # bucket curl is first dimension
    continuous_gripper = ((bucket_positions + 1) / 2).tolist()  # normalize to [0, 1]
    # Add one more for the initial state (N+1 elements to match states array)
    continuous_gripper = [0.5] + continuous_gripper
    
    annotation = {
        "task": "excavator_trajectory_prediction",
        "texts": [task_description],
        "videos": [{"video_path": video_path}],
        "action": actions_list,
        "state": states_list,
        "continuous_gripper_state": continuous_gripper,
        "episode_id": episode_id,
    }
    
    return annotation


def process_session(
    session_path: Path,
    output_dir: Path,
    camera: str,
    target_height: int,
    target_width: int,
    split: str,
    skip_existing: bool = True,
) -> Tuple[str, Dict]:
    """
    Process a single session: resize video and create annotation.
    
    Args:
        session_path: Path to session folder
        output_dir: Base output directory
        camera: Camera to use (front, left, back, right)
        target_height: Target video height
        target_width: Target video width
        split: 'train' or 'val'
        skip_existing: If True, skip sessions that have already been processed
    
    Returns:
        Tuple of (episode_id, annotation_dict) or (None, None) if failed
    """
    session_id = session_path.name
    camera_prefix = CAMERA_MAP[camera]
    
    # Check if already processed (for resume capability)
    video_output_dir = output_dir / "videos" / split / session_id
    output_video_path = video_output_dir / "rgb.mp4"
    annotation_output_dir = output_dir / "annotations" / split
    annotation_path = annotation_output_dir / f"{session_id}.json"
    
    if skip_existing and output_video_path.exists() and annotation_path.exists():
        # Already processed, skip
        return session_id, "skipped"
    
    # Find video and CSV files
    video_files = list(session_path.glob(f"{camera_prefix}_*.mp4"))
    csv_files = list(session_path.glob("D05_*.csv"))
    
    if not video_files or not csv_files:
        return None, None
    
    video_path = video_files[0]
    csv_path = csv_files[0]
    
    # Parse joystick data
    try:
        actions_4d = parse_joystick_csv(csv_path)
    except Exception as e:
        print(f"Error parsing CSV {csv_path}: {e}")
        return None, None
    
    if len(actions_4d) < 13:  # Minimum frames for training
        print(f"Skipping {session_id}: too few frames ({len(actions_4d)})")
        return None, None
    
    # Pad to 7D
    actions_7d = pad_actions_to_7d(actions_4d)
    
    # Compute states
    states = compute_states_from_actions(actions_7d)
    
    # Create output paths
    video_output_dir.mkdir(parents=True, exist_ok=True)
    annotation_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resize video
    relative_video_path = f"videos/{split}/{session_id}/rgb.mp4"
    
    try:
        resize_video(video_path, output_video_path, target_height, target_width)
    except Exception as e:
        print(f"Error resizing video {video_path}: {e}")
        return None, None
    
    # Create annotation
    annotation = create_annotation_json(
        episode_id=session_id,
        video_path=relative_video_path,
        actions=actions_7d,
        states=states,
        task_description="excavator digging and moving operation"
    )
    
    # Save annotation JSON
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    return session_id, annotation


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    camera: str = "front",
    resolution: str = "480,640",
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Prepare the full dataset for Cosmos training.
    
    Args:
        input_dir: Path to downloaded excavator dataset
        output_dir: Path to save prepared dataset
        camera: Which camera to use (front, left, back, right)
        resolution: Target resolution as "height,width"
        train_ratio: Ratio of data for training (rest goes to validation)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Parse resolution
    target_height, target_width = map(int, resolution.split(','))
    
    print("=" * 60)
    print("Preparing Excavator Dataset for Cosmos Training")
    print("=" * 60)
    print(f"Input directory: {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Camera: {camera} ({CAMERA_MAP[camera]})")
    print(f"Target resolution: {target_height}x{target_width}")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print("=" * 60)
    print()
    
    # Find all session folders
    sessions = sorted([
        d for d in input_path.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 14
    ])
    
    print(f"Found {len(sessions)} sessions")
    
    if len(sessions) == 0:
        print("No sessions found! Make sure the dataset is downloaded correctly.")
        return
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(sessions)
    
    split_idx = int(len(sessions) * train_ratio)
    train_sessions = sessions[:split_idx]
    val_sessions = sessions[split_idx:]
    
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print()
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "annotations" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "annotations" / "val").mkdir(parents=True, exist_ok=True)
    
    # Process training sessions
    print("Processing training sessions...", flush=True)
    train_count = 0
    train_skipped = 0
    train_failed = 0
    for session_path in tqdm(train_sessions, desc="Train"):
        try:
            episode_id, result = process_session(
                session_path, output_path, camera, target_height, target_width, "train"
            )
            if episode_id:
                if result == "skipped":
                    train_skipped += 1
                train_count += 1
        except Exception as e:
            print(f"\nFailed on {session_path.name}: {e}", flush=True)
            train_failed += 1
    
    # Process validation sessions
    print("\nProcessing validation sessions...", flush=True)
    val_count = 0
    val_skipped = 0
    val_failed = 0
    for session_path in tqdm(val_sessions, desc="Val"):
        try:
            episode_id, result = process_session(
                session_path, output_path, camera, target_height, target_width, "val"
            )
            if episode_id:
                if result == "skipped":
                    val_skipped += 1
                val_count += 1
        except Exception as e:
            print(f"\nFailed on {session_path.name}: {e}", flush=True)
            val_failed += 1
    
    print()
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Successfully processed:")
    print(f"  Training episodes: {train_count} ({train_skipped} skipped/resumed, {train_failed} failed)")
    print(f"  Validation episodes: {val_count} ({val_skipped} skipped/resumed, {val_failed} failed)")
    print()
    print(f"Output structure:")
    print(f"  {output_path}/")
    print(f"    videos/train/ and videos/val/")
    print(f"    annotations/train/ and annotations/val/")
    print("=" * 60)
    
    # Save dataset info
    info = {
        "dataset": "FlywheelAI/excavator-dataset",
        "camera": camera,
        "resolution": f"{target_height}x{target_width}",
        "train_episodes": train_count,
        "val_episodes": val_count,
        "action_dimensions": 7,
        "action_mapping": {
            "0": "bucket_curl (right joystick X)",
            "1": "boom_up_down (right joystick Y)",
            "2": "cab_rotation (left joystick X)",
            "3": "arm_extend (left joystick Y)",
            "4": "unused (padding)",
            "5": "unused (padding)",
            "6": "unused (padding)",
        },
        "fps": 25,
        "seed": seed,
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDataset info saved to: {output_path / 'dataset_info.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FlywheelAI Excavator Dataset for Cosmos training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/excavator",
        help="Path to downloaded excavator dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/excavator_cosmos",
        help="Path to save prepared dataset"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=["front", "left", "back", "right"],
        help="Which camera angle to use (default: front)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480,640",
        help="Target resolution as 'height,width' (default: 480,640)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        camera=args.camera,
        resolution=args.resolution,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
