# Excavator Action-Conditioned Fine-Tuning Guide

This guide walks you through fine-tuning Cosmos Predict 2.5 on the FlywheelAI excavator dataset for action-conditioned video generation.

## Overview

After fine-tuning, you'll be able to:
- Provide an excavator image + joystick commands
- Generate a video showing what the excavator would look like executing those commands

**Action Format (4D joystick → 7D padded):**

| Dimension | Control | Joystick | Negative (-1) | Positive (+1) |
|-----------|---------|----------|---------------|---------------|
| 0 | Bucket curl | Right X | Close/Scoop | Open/Dump |
| 1 | Boom | Right Y | Down | Up |
| 2 | Cab rotation | Left X | Left | Right |
| 3 | Arm | Left Y | Retract | Extend |
| 4-6 | Unused | - | Padding (0) | Padding (0) |

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (H200 recommended for training)
- ~100GB disk space for dataset
- HuggingFace account (for model downloads)

## Step 1: Prepare Data (Run Locally)

### 1.1 Install Dependencies

```bash
cd cosmos-predict2.5
pip install huggingface_hub pandas numpy opencv-python tqdm
```

### 1.2 Download Dataset

```bash
python scripts/excavator/download_dataset.py --output-dir datasets/excavator
```

This downloads the FlywheelAI excavator dataset (~84GB) containing:
- 176 sessions of excavator operation
- 4 camera angles per session (we'll use front/D01)
- Synchronized joystick data at 25Hz

### 1.3 Prepare Data for Cosmos

```bash
python scripts/excavator/prepare_data.py \
    --input-dir datasets/excavator \
    --output-dir datasets/excavator_cosmos \
    --camera front \
    --resolution 480,640 \
    --train-ratio 0.9
```

This will:
- Parse joystick CSV files into 7D action vectors
- Resize videos from 1920x1080 to 480x640
- Create JSON annotations in Cosmos format
- Split into train (90%) and val (10%) sets

**Output structure:**
```
datasets/excavator_cosmos/
├── videos/
│   ├── train/{session_id}/rgb.mp4
│   └── val/{session_id}/rgb.mp4
├── annotations/
│   ├── train/{session_id}.json
│   └── val/{session_id}.json
└── dataset_info.json
```

## Step 2: Transfer to Brev VM

### 2.1 Upload Prepared Dataset

Upload the `datasets/excavator_cosmos/` directory to your Brev VM:

```bash
# Option 1: rsync (recommended for large files)
rsync -avz --progress datasets/excavator_cosmos/ user@brev-vm:/workspace/cosmos-predict2.5/datasets/excavator_cosmos/

# Option 2: scp
scp -r datasets/excavator_cosmos/ user@brev-vm:/workspace/cosmos-predict2.5/datasets/

# Option 3: Use cloud storage (S3, GCS) as intermediate
```

### 2.2 Upload Scripts and Configs

```bash
# Upload the excavator-specific files
rsync -avz scripts/excavator/ user@brev-vm:/workspace/cosmos-predict2.5/scripts/excavator/
rsync -avz cosmos_predict2/experiments/excavator/ user@brev-vm:/workspace/cosmos-predict2.5/cosmos_predict2/experiments/excavator/
rsync -avz cosmos_predict2/excavator_action_loader.py user@brev-vm:/workspace/cosmos-predict2.5/cosmos_predict2/
```

## Step 3: Setup on Brev VM

### 3.1 SSH into VM

```bash
ssh user@brev-vm
cd /workspace/cosmos-predict2.5
```

### 3.2 Setup Environment

Follow the standard Cosmos setup:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Login to HuggingFace (for model downloads)
huggingface-cli login
```

### 3.3 Set Output Directory

```bash
# Set where checkpoints will be saved (use a disk with sufficient space)
export IMAGINAIRE_OUTPUT_ROOT=/workspace/cosmos_output
export HF_HOME=/workspace/hf_cache

mkdir -p $IMAGINAIRE_OUTPUT_ROOT
mkdir -p $HF_HOME
```

## Step 4: Run Training

### 4.1 Verify Dataset

```bash
# Check dataset is in place
ls datasets/excavator_cosmos/annotations/train | wc -l  # Should show ~158
ls datasets/excavator_cosmos/annotations/val | wc -l    # Should show ~18
```

### 4.2 Launch Training

```bash
# Make script executable
chmod +x scripts/excavator/train_excavator.sh

# Run training
./scripts/excavator/train_excavator.sh
```

Or run manually:

```bash
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- \
    experiment=excavator_action_conditioned_2b_480_640 \
    job.wandb_mode=disabled \
    +dataloader_train.dataset.train_annotation_path="datasets/excavator_cosmos/annotations/train" \
    +dataloader_train.dataset.val_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_train.dataset.test_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_train.dataset.video_path="datasets/excavator_cosmos" \
    +dataloader_val.dataset.train_annotation_path="datasets/excavator_cosmos/annotations/train" \
    +dataloader_val.dataset.val_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_val.dataset.test_annotation_path="datasets/excavator_cosmos/annotations/val" \
    +dataloader_val.dataset.video_path="datasets/excavator_cosmos"
```

### 4.3 Monitor Training

Training progress is logged to the console. Key metrics to watch:
- `loss`: Should decrease over time
- `iter_speed`: Iterations per second

Checkpoints are saved every 1000 iterations to:
```
$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_excavator/excavator_action_conditioned/2b_excavator_480_640/checkpoints/
```

## Step 5: Convert Checkpoint for Inference

After training completes (or at any checkpoint):

```bash
chmod +x scripts/excavator/convert_checkpoint.sh

# Convert latest checkpoint
./scripts/excavator/convert_checkpoint.sh

# Or convert specific iteration
./scripts/excavator/convert_checkpoint.sh 5000
```

This creates:
- `model.pt`: Full checkpoint
- `model_ema_fp32.pt`: EMA weights (float32)
- `model_ema_bf16.pt`: EMA weights (bfloat16) - **use this for inference**

## Step 6: Run Inference

### 6.1 Generate with Predefined Actions

```bash
python scripts/excavator/inference_excavator.py \
    --checkpoint $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_excavator/excavator_action_conditioned/2b_excavator_480_640/checkpoints/iter_5000/model_ema_bf16.pt \
    --input-image your_excavator.jpg \
    --action-preset dig \
    --output-dir outputs/excavator_inference
```

Available presets:
- `dig`: Full digging sequence (lower, scoop, lift, dump)
- `boom_up` / `boom_down`: Raise/lower boom
- `bucket_scoop` / `bucket_dump`: Curl bucket in/out
- `rotate_left` / `rotate_right`: Swing cab
- `arm_extend` / `arm_retract`: Extend/retract arm
- `neutral`: No movement

### 6.2 Generate with Custom Actions

```bash
python scripts/excavator/inference_excavator.py \
    --checkpoint /path/to/model_ema_bf16.pt \
    --input-image excavator.jpg \
    --actions "[[0.0, -0.5, 0.0, 0.2], [0.0, -0.5, 0.0, 0.3], [0.0, -0.3, 0.0, 0.4]]" \
    --output-dir outputs/excavator_inference
```

Action format: `[bucket, boom, cab, arm]` where each value is in [-1, 1]

### 6.3 Generate from Dataset Sample

```bash
python scripts/excavator/inference_excavator.py \
    --checkpoint /path/to/model_ema_bf16.pt \
    --input-json datasets/excavator_cosmos/annotations/val/20250828132109.json \
    --output-dir outputs/excavator_inference
```

## Troubleshooting

### Out of Memory

If you get OOM errors, try:
1. Reduce batch size in training config
2. Use `excavator_action_conditioned_2b_480_640_small_batch` experiment
3. Enable gradient checkpointing

### Slow Training

- Ensure data is on fast storage (NVMe SSD)
- Check GPU utilization with `nvidia-smi`
- Increase `num_workers` in dataloader

### Poor Generation Quality

- Train for more iterations
- Try different guidance values (5-10)
- Ensure training data quality (check for corrupted videos)

## File Reference

| File | Purpose |
|------|---------|
| `scripts/excavator/download_dataset.py` | Download FlywheelAI dataset |
| `scripts/excavator/prepare_data.py` | Convert to Cosmos format |
| `scripts/excavator/train_excavator.sh` | Launch training |
| `scripts/excavator/convert_checkpoint.sh` | Convert checkpoint for inference |
| `scripts/excavator/inference_excavator.py` | Generate videos |
| `cosmos_predict2/excavator_action_loader.py` | Custom action loader |
| `cosmos_predict2/experiments/excavator/action.py` | Training config |
| `cosmos_predict2/experiments/excavator/data.py` | Dataset config |

## Next Steps

After fine-tuning, you can:
1. Generate synthetic training data for your VLA model
2. Test different action sequences to see model predictions
3. Fine-tune further on specific scenarios
4. Use multiview training for additional camera angles
