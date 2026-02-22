# Excavator Fine-Tuning Guide (v2 — Frame + NL + Actions)

## Overview
Fine-tune Cosmos Predict 2.5 2B action-conditioned model on excavator data with
**tri-modal conditioning**: initial frame + natural language caption + joystick actions.

## Key Fixes from v1
1. **Action scaling**: `c_act_scaler = 1.0` (was 20.0, which caused blue screen outputs)
2. **Learning rate**: `2^-16` (~1.5e-05) instead of `2^-14.5` for stable fine-tuning
3. **Natural language**: T5-XXL text embeddings enabled alongside action conditioning
4. **Custom dataset**: `ExcavatorDataset` reads actions directly from JSON (not recomputed from states)
5. **Starting checkpoint**: Action-conditioned checkpoint (not base)

## Dataset Structure
```
datasets/excavator_cosmos/
├── videos/
│   ├── train/{session_id}/rgb.mp4
│   └── val/{session_id}/rgb.mp4
└── annotations/
    ├── train/{session_id}.json   ← action + caption data
    ├── train/{session_id}.npy    ← T5 embedding of caption
    ├── val/{session_id}.json
    └── val/{session_id}.npy
```

## JSON Annotation Format
```json
{
  "task": "excavator_operation",
  "caption": "The excavator is lowering the boom and scooping dirt.",
  "segments": [
    {"start_frame": 0, "end_frame": 120, "label": "lower_boom"},
    {"start_frame": 120, "end_frame": 250, "label": "scoop"}
  ],
  "action": [[0.01, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], ...],
  "state": [[...], ...],
  "continuous_gripper_state": [0.5, ...],
  "episode_id": "20250828132109"
}
```

## Action Format (7D, joystick values in [-1, 1])
- Dim 0: bucket_curl (right joystick X)
- Dim 1: boom_up_down (right joystick Y)  
- Dim 2: cab_rotation (left joystick X)
- Dim 3: arm_extend (left joystick Y)
- Dim 4-6: padding (zeros)

**CRITICAL**: `c_act_scaler = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
Do NOT use 20.0 — that's for robot arm data with tiny delta values.

## Pipeline (On Brev VM)

### Step 1: Pull latest code
```bash
cd /home/ubuntu/cosmos-predict2.5-finetune  # or wherever the repo is
git pull origin excavator-finetune
```

### Step 2: Process labels (if updated JSONs aren't transferred yet)
```bash
python scripts/excavator/process_labels.py \
    --labels scripts/excavator/labeling_template.csv \
    --annotations-dir datasets/excavator_cosmos/annotations
```

### Step 3: Generate T5 embeddings
```bash
python scripts/excavator/generate_t5_embeddings.py \
    --annotations-dir datasets/excavator_cosmos/annotations \
    --skip-existing
```

### Step 4: Train
```bash
./scripts/excavator/train_excavator.sh
```

### Step 5: Convert checkpoint for inference
```bash
python scripts/convert_distcp_to_pt.py \
    /home/ubuntu/cosmos_output/cosmos_predict2_excavator/.../model \
    /home/ubuntu/cosmos_output/cosmos_predict2_excavator/...
```

### Step 6: Inference
```bash
python scripts/excavator/inference_excavator.py \
    --checkpoint /path/to/model_ema_bf16.pt \
    --input-json datasets/excavator_cosmos/annotations/val/20250829112715.json \
    --output-dir outputs/excavator
```

## Key Files
- `cosmos_predict2/_src/predict2/action/datasets/excavator_dataset.py` — Custom dataset (fixed scaling + T5)
- `cosmos_predict2/experiments/excavator/action.py` — Training experiment config
- `cosmos_predict2/experiments/excavator/data.py` — Dataset/dataloader config
- `cosmos_predict2/excavator_action_loader.py` — Inference action loader
- `scripts/excavator/train_excavator.sh` — Training launch script
- `scripts/excavator/process_labels.py` — Label CSV → JSON processor
- `scripts/excavator/generate_t5_embeddings.py` — T5 embedding generator
- `scripts/excavator/labeling_template.csv` — Template for video labeling
- `assets/excavator/inference_params.json` — Inference config (action_scaler=1.0)

## Troubleshooting
- **Blue screens**: Check `c_act_scaler` is 1.0, not 20.0
- **Loss spikes**: Lower learning rate further (try `2^-17`)
- **Missing T5 embeddings**: Run `generate_t5_embeddings.py` before training
- **OOM**: Reduce batch_size in `data.py` or use the small_batch experiment variant
