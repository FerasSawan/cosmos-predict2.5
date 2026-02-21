#!/bin/bash
# =============================================================================
# Convert Excavator Training Checkpoint for Inference
# =============================================================================
#
# After training, checkpoints are saved in DCP (Distributed Checkpoint) format.
# This script converts them to consolidated PyTorch format for inference.
#
# Usage:
#   ./scripts/excavator/convert_checkpoint.sh [checkpoint_iter]
#
# Example:
#   ./scripts/excavator/convert_checkpoint.sh 5000
#
# If no iteration is specified, uses the latest checkpoint.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Output root (should match training script)
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/workspace/cosmos_output}"

# Checkpoint directory
CHECKPOINTS_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_excavator/excavator_action_conditioned/2b_excavator_480_640/checkpoints"

echo "============================================================"
echo "Converting Excavator Checkpoint"
echo "============================================================"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo ""

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "ERROR: Checkpoints directory not found!"
    echo "Expected: $CHECKPOINTS_DIR"
    echo ""
    echo "Make sure training has completed and checkpoints were saved."
    exit 1
fi

# Get checkpoint iteration
if [ -n "$1" ]; then
    CHECKPOINT_ITER="iter_$1"
else
    # Use latest checkpoint
    if [ -f "$CHECKPOINTS_DIR/latest_checkpoint.txt" ]; then
        CHECKPOINT_ITER=$(cat "$CHECKPOINTS_DIR/latest_checkpoint.txt")
    else
        echo "ERROR: No latest_checkpoint.txt found and no iteration specified."
        exit 1
    fi
fi

CHECKPOINT_DIR="$CHECKPOINTS_DIR/$CHECKPOINT_ITER"

echo "Converting checkpoint: $CHECKPOINT_ITER"
echo "Full path: $CHECKPOINT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$CHECKPOINTS_DIR" | grep "iter_" || echo "  (none found)"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Convert checkpoint
echo "Running conversion..."
python ./scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"

echo ""
echo "============================================================"
echo "Conversion complete!"
echo "============================================================"
echo ""
echo "Generated files:"
ls -la "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  (no .pt files found)"
echo ""
echo "For inference, use: $CHECKPOINT_DIR/model_ema_bf16.pt"
echo ""
echo "Example inference command:"
echo "  python scripts/excavator/inference_excavator.py \\"
echo "    --checkpoint $CHECKPOINT_DIR/model_ema_bf16.pt \\"
echo "    --input-image your_excavator_image.jpg \\"
echo "    --output-dir outputs/excavator_inference"
echo "============================================================"
