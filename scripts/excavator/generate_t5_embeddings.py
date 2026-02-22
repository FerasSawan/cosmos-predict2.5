#!/usr/bin/env python3
"""
Generate T5-XXL embeddings for excavator annotation captions.

This script reads the 'caption' field from each annotation JSON and generates
a corresponding .npy file with the T5-XXL text embedding. These embeddings
are used during training for natural language conditioning.

Requirements:
    - GPU with 16GB+ VRAM (T5-XXL is ~11GB)
    - Run on VM (Brev H200), not local machine

Usage:
    python scripts/excavator/generate_t5_embeddings.py \
        --annotations-dir datasets/excavator_cosmos/annotations

    # Skip existing embeddings (for resume):
    python scripts/excavator/generate_t5_embeddings.py \
        --annotations-dir datasets/excavator_cosmos/annotations \
        --skip-existing
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate T5-XXL embeddings for excavator captions")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Base annotations directory")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for T5")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if .npy already exists")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding")
    args = parser.parse_args()

    print("Loading T5-XXL text encoder...")
    try:
        from cosmos_predict2._src.imaginaire.auxiliary.text_encoder import (
            CosmosT5TextEncoder,
            CosmosT5TextEncoderConfig,
        )
        config = CosmosT5TextEncoderConfig(max_length=args.max_length)
        encoder = CosmosT5TextEncoder(config)
        encoder.cuda()
        use_cosmos_encoder = True
        print("Using Cosmos T5 text encoder")
    except Exception as e:
        print(f"Cosmos encoder not available ({e}), falling back to HuggingFace T5")
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        model = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl", torch_dtype=torch.float16
        ).cuda()
        model.eval()
        use_cosmos_encoder = False
        print("Using HuggingFace T5-XXL encoder")

    print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    ann_files = []
    for split in ["train", "val"]:
        split_dir = Path(args.annotations_dir) / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob("*.json")):
            ann_files.append(f)

    print(f"Found {len(ann_files)} annotation files")

    captions = []
    paths = []
    for ann_file in ann_files:
        npy_path = ann_file.with_suffix(".npy")
        if args.skip_existing and npy_path.exists():
            continue

        with open(ann_file, "r") as f:
            data = json.load(f)

        caption = data.get("caption", "An excavator at a construction site.")
        captions.append(caption)
        paths.append(npy_path)

    if not captions:
        print("All embeddings already exist. Nothing to do.")
        return

    print(f"Generating embeddings for {len(captions)} captions...")

    for i in tqdm(range(0, len(captions), args.batch_size)):
        batch_captions = captions[i : i + args.batch_size]
        batch_paths = paths[i : i + args.batch_size]

        with torch.no_grad():
            if use_cosmos_encoder:
                for caption, npy_path in zip(batch_captions, batch_paths):
                    embedding = encoder.encode(caption)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    if embedding.ndim == 3:
                        embedding = embedding.squeeze(0)
                    if embedding.shape[0] < args.max_length:
                        pad = np.zeros(
                            (args.max_length - embedding.shape[0], embedding.shape[1]),
                            dtype=embedding.dtype,
                        )
                        embedding = np.concatenate([embedding, pad], axis=0)
                    elif embedding.shape[0] > args.max_length:
                        embedding = embedding[: args.max_length]
                    np.save(str(npy_path), embedding)
            else:
                inputs = tokenizer(
                    batch_captions,
                    return_tensors="pt",
                    max_length=args.max_length,
                    padding="max_length",
                    truncation=True,
                )
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.cpu().numpy()

                for j, npy_path in enumerate(batch_paths):
                    np.save(str(npy_path), embeddings[j])

    print(f"\nDone! Generated {len(captions)} T5 embeddings.")
    print(f"Embeddings saved alongside annotation JSONs as .npy files.")


if __name__ == "__main__":
    main()
