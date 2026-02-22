#!/bin/bash
# =============================================================================
# Excavator Action-Conditioned Training Script (v2 - Fixed Scaling + NL)
# =============================================================================
#
# Fine-tunes Cosmos Predict 2.5 action-conditioned model on excavator dataset
# with frame + natural language + joystick action conditioning.
#
# Key fixes from v1:
#   - Action scaling: 1.0 (was 20.0 which caused blue screens)
#   - Learning rate: 2^-16 (lower for stable fine-tuning)
#   - T5 text embeddings: enabled for natural language
#   - Uses ExcavatorDataset (reads actions directly from JSON)
#
# Prerequisites:
#   1. Dataset uploaded to VM at datasets/excavator_cosmos/
#   2. Labels processed: python scripts/excavator/process_labels.py ...
#   3. T5 embeddings generated: python scripts/excavator/generate_t5_embeddings.py ...
#   4. HuggingFace login: huggingface-cli login
#
# Usage:
#   ./scripts/excavator/train_excavator.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/home/ubuntu/cosmos_output}"
export HF_HOME="${HF_HOME:-/home/ubuntu/hf_cache}"

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
mkdir -p "$HF_HOME"

echo "============================================================"
echo "Excavator Training v2 (Frame + NL + Actions)"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output: $IMAGINAIRE_OUTPUT_ROOT"
echo "============================================================"

DATASET_PATH="datasets/excavator_cosmos"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Verify T5 embeddings exist
T5_COUNT=$(ls -1 "$DATASET_PATH/annotations/train/"*.npy 2>/dev/null | wc -l)
JSON_COUNT=$(ls -1 "$DATASET_PATH/annotations/train/"*.json 2>/dev/null | wc -l)
echo "Train: $JSON_COUNT annotations, $T5_COUNT T5 embeddings"

if [ "$T5_COUNT" -eq 0 ]; then
    echo ""
    echo "WARNING: No T5 embeddings found!"
    echo "Run this first:"
    echo "  python scripts/excavator/generate_t5_embeddings.py \\"
    echo "      --annotations-dir $DATASET_PATH/annotations"
    echo ""
    echo "Continuing without T5 (will use zero embeddings)..."
fi

VAL_COUNT=$(ls -1 "$DATASET_PATH/annotations/val/"*.json 2>/dev/null | wc -l)
echo "Val: $VAL_COUNT annotations"
echo ""

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

NUM_GPUS="${NUM_GPUS:-1}"
EXPERIMENT="excavator_action_conditioned_2b_480_640"
CONFIG_PATH="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"

echo "Starting training..."
echo "  Experiment: $EXPERIMENT"
echo "  GPUs: $NUM_GPUS"
echo "  LR: 2^-16 (~1.5e-05)"
echo "  Action scaling: 1.0"
echo "  Text conditioning: enabled"
echo ""

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
echo "Checkpoints: $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_excavator/"
echo ""
echo "Next steps:"
echo "1. Convert checkpoint:"
echo "   python scripts/convert_distcp_to_pt.py <ckpt_dir>/model <ckpt_dir>"
echo ""
echo "2. Inference:"
echo "   python scripts/excavator/inference_excavator.py --checkpoint <path>"
echo "============================================================"
