#!/bin/bash
set -e

cd ~/cosmos-predict2.5
source .venv/bin/activate

echo "=== Starting Cosmos Predict 2.5 Image2World Inference ==="
echo "Image: excavator1.png"
echo "Model: 2B/post-trained"
echo "Inference type: image2world"
echo "Timestamp: $(date)"
echo ""

mkdir -p outputs/excavator

python examples/inference.py \
    -i assets/base/excavator_digging.json \
    -o outputs/excavator \
    --inference-type=image2world \
    --model=2B/post-trained \
    --offload-text-encoder \
    --offload-guardrail-models \
    2>&1 | tee outputs/excavator/inference_log.txt

echo ""
echo "=== Inference Complete ==="
echo "Output files:"
ls -la outputs/excavator/
