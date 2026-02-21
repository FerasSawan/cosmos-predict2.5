#!/bin/bash
# =============================================================================
# Excavator Action-Conditioned Training Script for Brev VM (H200)
# =============================================================================
#
# This script launches fine-tuning of Cosmos Predict 2.5 on the excavator dataset.
#
# Prerequisites:
#   1. Dataset prepared using prepare_data.py and uploaded to VM
#   2. Cosmos Predict 2.5 environment set up (see docs/setup.md)
#   3. HuggingFace login for model downloads: huggingface-cli login
#
# Usage:
#   ./scripts/excavator/train_excavator.sh
#
# Or with custom output directory:
#   IMAGINAIRE_OUTPUT_ROOT=/path/to/output ./scripts/excavator/train_excavator.sh
#
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set output directory for checkpoints (default: /tmp/imaginaire4-output)
# IMPORTANT: Change this to a location with sufficient disk space!
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/workspace/cosmos_output}"

# Set HuggingFace cache directory
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

# Create directories
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
mkdir -p "$HF_HOME"

echo "============================================================"
echo "Excavator Action-Conditioned Training"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $IMAGINAIRE_OUTPUT_ROOT"
echo "HF cache: $HF_HOME"
echo "============================================================"
echo ""

# Check if dataset exists
DATASET_PATH="datasets/excavator_cosmos"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please run prepare_data.py first and upload the prepared dataset."
    exit 1
fi

# Count training samples
TRAIN_COUNT=$(ls -1 "$DATASET_PATH/annotations/train" 2>/dev/null | wc -l)
VAL_COUNT=$(ls -1 "$DATASET_PATH/annotations/val" 2>/dev/null | wc -l)
echo "Dataset statistics:"
echo "  Training samples: $TRAIN_COUNT"
echo "  Validation samples: $VAL_COUNT"
echo ""

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "ERROR: No training samples found!"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Number of GPUs (H200 typically single GPU per instance)
NUM_GPUS="${NUM_GPUS:-1}"

# Training configuration
EXPERIMENT="excavator_action_conditioned_2b_480_640"
CONFIG_PATH="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"

echo "Starting training..."
echo "  Experiment: $EXPERIMENT"
echo "  Config: $CONFIG_PATH"
echo "  GPUs: $NUM_GPUS"
echo ""

# Launch training
# Note: We disable wandb by default. Remove job.wandb_mode=disabled to enable.
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=12341 \
    -m scripts.train \
    --config=$CONFIG_PATH \
    -- \
    experiment=$EXPERIMENT \
    job.wandb_mode=disabled \
    +dataloader_train.dataset.train_annotation_path="datasets/excavator_cosmos/annotations/train" \
    +dataloader_train.dataset.val_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_train.dataset.test_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_train.dataset.video_path="datasets/excavator_cosmos" \
    +dataloader_val.dataset.train_annotation_path="datasets/excavator_cosmos/annotations/train" \
    +dataloader_val.dataset.val_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_val.dataset.test_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_val.dataset.video_path="datasets/excavator_cosmos"

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "Checkpoints saved to: $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_excavator/"
echo ""
echo "Next steps:"
echo "1. Convert checkpoint for inference:"
echo "   python scripts/convert_distcp_to_pt.py <checkpoint_dir>/model <checkpoint_dir>"
echo ""
echo "2. Run inference:"
echo "   python scripts/excavator/inference_excavator.py --checkpoint <checkpoint_path>"
echo "============================================================"
