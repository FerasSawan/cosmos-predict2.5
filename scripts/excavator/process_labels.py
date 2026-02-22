#!/usr/bin/env python3
"""
Process video labels and add captions + segments to annotation JSONs.

This script reads a labeling CSV (filled out by the user) and:
1. Adds a 'caption' field to each annotation JSON (for T5 embedding generation)
2. Adds a 'segments' field with labeled time segments
3. Generates natural language captions from segment labels

Usage:
    python scripts/excavator/process_labels.py \
        --labels scripts/excavator/labeling_template.csv \
        --annotations-dir datasets/excavator_cosmos/annotations

Label format (CSV columns):
    session_id, split, segment_start_frame, segment_end_frame, label, notes

Valid labels:
    lower_boom, raise_boom, scoop, dump, extend_arm, retract_arm,
    rotate_left, rotate_right, idle, lower_boom+scoop, raise_boom+rotate_left, etc.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

LABEL_TO_TEXT = {
    "lower_boom": "lowering the boom",
    "raise_boom": "raising the boom",
    "scoop": "scooping with the bucket",
    "dump": "dumping the bucket",
    "extend_arm": "extending the arm",
    "retract_arm": "retracting the arm",
    "rotate_left": "rotating left",
    "rotate_right": "rotating right",
    "idle": "idle",
    "lower_boom+scoop": "lowering the boom and scooping",
    "raise_boom+scoop": "raising the boom while scooping",
    "lower_boom+extend_arm": "lowering the boom and extending the arm",
    "raise_boom+rotate_left": "raising the boom and rotating left",
    "raise_boom+rotate_right": "raising the boom and rotating right",
    "scoop+extend_arm": "scooping while extending the arm",
    "lower_boom+retract_arm": "lowering the boom and retracting the arm",
}

CAPTION_TEMPLATES = [
    "The excavator is {actions}.",
    "An excavator {actions} at a construction site.",
    "The excavator {actions}.",
    "An excavator performing operations: {actions}.",
]

WHOLE_VIDEO_TEMPLATES = [
    "An excavator performing a sequence of operations: {sequence}.",
    "The excavator is {sequence}.",
    "Excavator operations: {sequence}.",
]


def label_to_text(label: str) -> str:
    """Convert a label string to natural language."""
    label = label.strip().lower()
    if label in LABEL_TO_TEXT:
        return LABEL_TO_TEXT[label]
    parts = label.split("+")
    texts = [LABEL_TO_TEXT.get(p.strip(), p.strip()) for p in parts]
    return " and ".join(texts)


def generate_whole_video_caption(segments: list[dict]) -> str:
    """Generate a caption describing the whole video from its segments."""
    if not segments:
        return "An excavator at a construction site."

    action_texts = []
    for seg in segments:
        text = label_to_text(seg["label"])
        if text != "idle":
            action_texts.append(text)

    if not action_texts:
        return "An excavator idle at a construction site."

    unique_actions = list(dict.fromkeys(action_texts))
    sequence = ", then ".join(unique_actions)

    template = random.choice(WHOLE_VIDEO_TEMPLATES)
    return template.format(sequence=sequence)


def generate_segment_caption(label: str) -> str:
    """Generate a caption for a single segment."""
    text = label_to_text(label)
    template = random.choice(CAPTION_TEMPLATES)
    return template.format(actions=text)


def main():
    parser = argparse.ArgumentParser(description="Process video labels into annotation JSONs")
    parser.add_argument("--labels", type=str, required=True, help="Path to labeling CSV")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Base annotations directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for caption variation")
    args = parser.parse_args()

    random.seed(args.seed)

    labels_by_session = defaultdict(list)
    with open(args.labels, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row["session_id"].strip()
            split = row["split"].strip()

            start = row.get("segment_start_frame", "").strip()
            end = row.get("segment_end_frame", "").strip()
            label = row.get("label", "").strip()

            if not label:
                continue

            segment = {
                "start_frame": int(start) if start else 0,
                "end_frame": int(end) if end else -1,
                "label": label,
                "notes": row.get("notes", "").strip(),
            }
            labels_by_session[(session_id, split)].append(segment)

    updated = 0
    skipped = 0

    for (session_id, split), segments in labels_by_session.items():
        ann_path = Path(args.annotations_dir) / split / f"{session_id}.json"
        if not ann_path.exists():
            print(f"WARNING: Annotation not found: {ann_path}")
            skipped += 1
            continue

        with open(ann_path, "r") as f:
            annotation = json.load(f)

        annotation["segments"] = segments
        annotation["caption"] = generate_whole_video_caption(segments)

        with open(ann_path, "w") as f:
            json.dump(annotation, f, indent=2)

        updated += 1

    unlabeled = 0
    for split in ["train", "val"]:
        split_dir = Path(args.annotations_dir) / split
        if not split_dir.exists():
            continue
        for ann_file in split_dir.glob("*.json"):
            session_id = ann_file.stem
            if (session_id, split) not in labels_by_session:
                with open(ann_file, "r") as f:
                    annotation = json.load(f)
                if "caption" not in annotation:
                    annotation["caption"] = "An excavator at a construction site."
                    annotation["segments"] = []
                    with open(ann_file, "w") as f:
                        json.dump(annotation, f, indent=2)
                    unlabeled += 1

    print(f"\nDone!")
    print(f"  Updated with labels: {updated}")
    print(f"  Skipped (not found): {skipped}")
    print(f"  Default caption (unlabeled): {unlabeled}")
    print(f"\nNext step: Generate T5 embeddings:")
    print(f"  python scripts/excavator/generate_t5_embeddings.py \\")
    print(f"      --annotations-dir {args.annotations_dir}")


if __name__ == "__main__":
    main()
