# Excavator Fine-Tuning: Full Context for Brev VM

This document contains EVERYTHING you need to know to set up and run fine-tuning of NVIDIA Cosmos Predict 2.5 on the excavator dataset. Read this completely before making any changes.

---

## 1. PROJECT OVERVIEW

**Goal**: Fine-tune Cosmos Predict 2.5 (2B model) for action-conditioned video generation of an excavator using joystick control data. The fine-tuned model will later generate synthetic training data for a Vision-Language-Action (VLA) model.

**Hardware**: Brev VM with NVIDIA H200 GPU (141GB VRAM)

**Repository**: `cosmos-predict2.5` — NVIDIA's official Cosmos Predict 2.5 codebase with custom excavator configs added.

---

## 2. DATASET — DOWNLOAD AND PREPARE ON THIS VM

The dataset needs to be downloaded from HuggingFace and prepared on this VM. Scripts for both steps are included in the repo.

### Source
- **HuggingFace**: `FlywheelAI/excavator-dataset`
- **Camera used**: Front-facing only (D01)
- **Original resolution**: 1920x1080
- **Target resolution**: 480x640 (height x width)

### Step 1: Install Dependencies
```bash
pip install huggingface_hub hf_xet tqdm pandas numpy opencv-python decord mediapy
```

### Step 2: Download the Dataset (~22GB, front camera + joystick CSV only)
```bash
python scripts/excavator/download_dataset.py --output-dir datasets/excavator
```
This downloads only D01 (front camera) videos and D05 (joystick CSV) files from HuggingFace. It should be fast on the VM's internet (~2-10 min). If it gets interrupted, just re-run — it resumes automatically via HuggingFace's `snapshot_download`.

### Step 3: Prepare the Dataset (resize videos, create annotations, split train/val)
```bash
python scripts/excavator/prepare_data.py \
    --input-dir datasets/excavator \
    --output-dir datasets/excavator_cosmos \
    --camera front \
    --resolution 480,640 \
    --train-ratio 0.8
```
This script:
1. Parses joystick CSVs and combines marker_id 0 & 1 into 4D action vectors
2. Pads 4D actions to 7D for Cosmos compatibility
3. Resizes videos from 1920x1080 to 480x640 (uses subprocess for crash isolation)
4. Creates JSON annotation files in Cosmos format
5. Splits data into train/val sets (80/20, seed=42)

If it gets interrupted, just re-run — it skips already-processed sessions automatically.

### Step 4: Verify the Output
```bash
ls datasets/excavator_cosmos/annotations/train/ | wc -l   # Should be 140
ls datasets/excavator_cosmos/annotations/val/ | wc -l     # Should be 35
cat datasets/excavator_cosmos/dataset_info.json
```

### Expected Output Structure
```
datasets/excavator_cosmos/
├── annotations/
│   ├── train/          # 140 JSON files (one per session)
│   │   ├── 20250829124537.json
│   │   ├── 20250829153125.json
│   │   └── ... (140 total)
│   └── val/            # 35 JSON files (one per session)
│       ├── 20250829154458.json
│       └── ... (35 total)
├── videos/
│   ├── train/
│   │   └── {session_id}/rgb.mp4    # 140 resized 480x640 videos
│   └── val/
│       └── {session_id}/rgb.mp4    # 35 resized 480x640 videos
└── dataset_info.json
```

### Expected Split
- **Training**: 140 sessions (80%)
- **Validation**: 35 sessions (20%)
- **Random seed**: 42

### Data Statistics
- Each session is a continuous excavator operation video (variable length, some are minutes long)
- FPS: ~25 fps
- Total: 175 sessions from 176 downloaded (1 session had no D01 video)
- Download size: ~22GB (front camera + CSVs only, not the full 84GB dataset)

---

## 3. ACTION DATA FORMAT

### Raw Joystick Data
The excavator has 2 joysticks with 4 axes total, all normalized to [-1, 1]:

| Dimension | Source | Control |
|-----------|--------|---------|
| 0 | Right Joystick X | Bucket curl |
| 1 | Right Joystick Y | Boom up/down |
| 2 | Left Joystick X | Cab rotation |
| 3 | Left Joystick Y | Arm extend/retract |

### Padded to 7D for Cosmos
Cosmos expects 7D actions: `[x, y, z, roll, pitch, yaw, gripper]`
We pad with zeros: `[bucket, boom, cab_rot, arm, 0, 0, 0]`

### JSON Annotation Structure
Each JSON annotation file has this exact structure:

```json
{
  "task": "excavator_trajectory_prediction",
  "texts": ["excavator digging and moving operation"],
  "videos": [{"video_path": "videos/train/{session_id}/rgb.mp4"}],
  "action": [
    [0.0013, 0.0103, 0.0125, 0.0115, 0.0, 0.0, 0.0],
    [0.0016, 0.0104, 0.0110, 0.0113, 0.0, 0.0, 0.0],
    ...
  ],
  "state": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.00013, 0.00103, 0.00125, 0.00115, 0.0, 0.0, 0.0],
    ...
  ],
  "continuous_gripper_state": [0.5, 0.5006, 0.5008, ...],
  "episode_id": "20250902153932"
}
```

**Key fields**:
- `action`: Per-frame 7D joystick values. Shape = (N, 7) where N = number of video frames. These are the RAW joystick readings padded to 7D.
- `state`: Cumulative sum of actions * 0.1. Shape = (N+1, 7). First element is always zeros. This is a pseudo-state, NOT real 3D robot poses.
- `continuous_gripper_state`: Shape = (N+1, ). Derived from bucket position, normalized to [0, 1]. First element is always 0.5.
- `action` has N elements, `state` and `continuous_gripper_state` have N+1 elements (includes initial state).

---

## 4. CRITICAL FIX NEEDED: CUSTOM DATALOADER

### The Problem

The default Cosmos training dataloader (`Dataset_3D` in `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py`) was built for robot arms with real 3D poses. It does NOT use the `action` field from the JSON. Instead, it:

1. Reads `state` (which it assumes are real 3D poses: xyz + euler angles)
2. Reads `continuous_gripper_state`
3. RECOMPUTES actions by calculating relative transforms between consecutive states using 3D rotation matrix math (`euler2rotm`, `rotm2euler`)

This is correct for the Bridge robot arm dataset (where states ARE real 3D poses), but **WRONG for our excavator data** because:
- Our `state` is a cumulative sum of joystick inputs, NOT real 3D positions/orientations
- When the cumulative state grows large (e.g., by frame 5000, `state[3]` could be ~50.0 radians = ~2865 degrees), the rotation matrix math produces completely distorted actions
- The recomputed "actions" would be scrambled garbage compared to the actual joystick values

### The Fix

Create a custom dataset class `ExcavatorDataset` that extends `Dataset_3D` and overrides `_get_actions()` to read the `action` field directly from the JSON annotation, bypassing the rotation matrix math entirely.

### Exact Code Changes Needed

**Step 1**: Create file `cosmos_predict2/_src/predict2/action/datasets/excavator_dataset.py`:

```python
"""
Custom dataset for excavator action-conditioned training.

Overrides the default Dataset_3D to read actions directly from the JSON
annotation instead of recomputing them from states using 3D rotation math.
This is necessary because excavator "states" are pseudo-states (cumulative
joystick sums), not real 3D robot poses.
"""

import json
import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class ExcavatorDataset(Dataset_3D):
    """Dataset that reads pre-computed actions directly from JSON annotations."""

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        """
        Override: read actions directly from the annotation JSON instead of
        recomputing from states via rotation matrix math.

        The parent class recomputes actions from consecutive states using
        euler2rotm/rotm2euler. For excavator data, the states are cumulative
        joystick sums (not real 3D poses), so the rotation math produces
        wrong results. Instead, we read the pre-computed action field.
        """
        # We need access to the current annotation data.
        # The parent __getitem__ loads the JSON and calls _get_robot_states,
        # then passes arm_states and gripper_states to _get_actions.
        # We store the current label in __getitem__ before this call.
        if hasattr(self, '_current_label') and self._current_label is not None:
            all_actions = np.array(self._current_label["action"])
            frame_ids = self._current_frame_ids

            # Get actions for the selected frames (N-1 actions for N frames)
            # frame_ids has self.sequence_length entries
            # We need actions at indices frame_ids[0] through frame_ids[-2]
            # (action[i] is the action AT frame i, not between i and i+1)
            action = np.zeros((self.sequence_length - 1, self.action_dim))
            for k in range(self.sequence_length - 1):
                fid = frame_ids[k]
                if fid < len(all_actions):
                    act = all_actions[fid]
                    action[k, :len(act)] = act[:self.action_dim]
                    action[k, 6] = gripper_states[k + 1]  # gripper from next state

            return torch.from_numpy(action)

        # Fallback to parent behavior if _current_label not set
        return super()._get_actions(arm_states, gripper_states, accumulate_action)

    def __getitem__(self, index, cam_id=None, return_video=False):
        """Override to store current label and frame_ids for _get_actions."""
        import random
        import warnings
        import traceback

        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            # Store for use in _get_actions
            self._current_label = label
            self._current_frame_ids = frame_ids

            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                try:
                    data["__key__"] = label["original_path"]
                except Exception:
                    try:
                        data["__key__"] = label["episode_metadata"]["episode_id"]
                    except Exception:
                        data["__key__"] = label["episode_metadata"]["segment_id"]

            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
                data["ai_caption"] = ""
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

            # Clean up
            self._current_label = None
            self._current_frame_ids = None

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            self._current_label = None
            self._current_frame_ids = None
            return self[np.random.randint(len(self.samples))]
```

**Step 2**: Update `cosmos_predict2/experiments/excavator/data.py` to use `ExcavatorDataset` instead of `Dataset_3D`:

Change the import from:
```python
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
```
To:
```python
from cosmos_predict2._src.predict2.action.datasets.excavator_dataset import ExcavatorDataset
```

And change all `L(Dataset_3D)(...)` calls to `L(ExcavatorDataset)(...)`.

**Step 3**: Also create a custom inference action loader. For inference, the default `load_default_action_fn` in `cosmos_predict2/action_conditioned.py` also recomputes actions from states. You need a custom `action_load_fn` that reads actions directly.

Create file `cosmos_predict2/excavator_action_loader.py`:

```python
"""
Custom action loader for excavator inference.
Reads action field directly from JSON instead of recomputing from states.
"""

import mediapy
import numpy as np
from loguru import logger


def load_excavator_action_fn():
    """Action loading function that reads pre-computed actions directly."""

    def load_fn(json_data, video_path, args):
        actions = np.array(json_data["action"])

        # Apply scaler (default 20.0) and gripper scale
        scaler = np.array([
            args.action_scaler, args.action_scaler, args.action_scaler,
            args.action_scaler, args.action_scaler, args.action_scaler,
            args.gripper_scale
        ])
        actions = actions * scaler

        # Downsample if needed
        if args.fps_downsample_ratio > 1:
            actions = actions[::args.fps_downsample_ratio]

        video_array = mediapy.read_video(video_path)
        img_array = video_array[args.start_frame_idx]

        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                img_array = mediapy.resize_image(img_array, (h, w))
            except Exception as e:
                logger.warning(f"Failed to resize image to {args.resolution}: {e}")

        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
        }

    return load_fn
```

---

## 5. ACTION SCALER CONSIDERATION

The default `c_act_scaler` is `[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, gripper_rescale_factor]`. This multiplies all actions by 20.

Our joystick values are normalized to [-1, 1], so after scaling they become [-20, 20]. This is the same range the model was pre-trained on with Bridge data, so **keep the default scaler of 20.0**. Do not change it.

---

## 6. EXISTING EXPERIMENT CONFIGS

### Training Config: `cosmos_predict2/experiments/excavator/action.py`

This file registers two experiment configurations:
- `excavator_action_conditioned_2b_480_640` — batch_size=4 (recommended for H200)
- `excavator_action_conditioned_2b_480_640_small_batch` — batch_size=2 (for testing/debugging)

Key settings:
- Model: 2B action-conditioned with rectified flow
- Resolution: 480x640
- Sequence: 13 frames (1 conditional + 12 action)
- action_dim: 7
- num_action_per_chunk: 12
- Learning rate: 2^(-14.5) ≈ 3.05e-05
- max_iter: 20,000
- save_iter: 1,000
- validation_iter: 500
- FPS: 25

### Data Config: `cosmos_predict2/experiments/excavator/data.py`

Defines dataset and dataloader configurations using the prepared data. Currently uses `Dataset_3D` — **this needs to be changed to `ExcavatorDataset`** (see Section 4).

---

## 7. HOW TO RUN TRAINING

### Training Command
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=excavator_action_conditioned_2b_480_640
```

For debugging with smaller batch:
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=excavator_action_conditioned_2b_480_640_small_batch
```

### What Happens During Training
1. The config system loads the experiment config from `excavator/action.py`
2. It imports all modules from `cosmos_predict2/experiments/` which registers the excavator configs
3. The data config from `excavator/data.py` creates `ExcavatorDataset` instances pointing to `datasets/excavator_cosmos/annotations/train/` and `val/`
4. The dataloader slides a 13-frame window across each video, creating training samples
5. For each sample: loads 13 consecutive video frames + reads 12 actions from the JSON
6. Actions are multiplied by `c_act_scaler` (20.0) before being fed to the model
7. The model learns to predict the next 12 frames given 1 conditional frame + 12 actions

### Checkpoints
- Saved every 1,000 iterations to `checkpoints/` in DCP (Distributed Checkpoint) format
- To convert DCP to PyTorch format for inference:
```bash
python scripts/convert_distcp_to_pt.py <dcp_checkpoint_path> <output_pt_path>
```

---

## 8. HOW TO RUN INFERENCE (AFTER TRAINING)

### Create Inference Parameter File
Create `assets/excavator/inference_params.json`:
```json
{
  "name": "excavator_inference",
  "input_root": "datasets/excavator_cosmos",
  "input_json_sub_folder": "annotations/val",
  "save_root": "outputs/excavator",
  "guidance": 7,
  "resolution": "480,640",
  "camera_id": 0,
  "start": 0,
  "end": 5,
  "fps_downsample_ratio": 1,
  "gripper_scale": 1.0,
  "gripper_key": "continuous_gripper_state",
  "state_key": "state",
  "reverse": false,
  "single_chunk": true,
  "start_frame_idx": 0,
  "save_fps": 25,
  "num_latent_conditional_frames": 1,
  "action_scaler": 20.0,
  "use_quat": false,
  "action_load_fn": "cosmos_predict2.excavator_action_loader.load_excavator_action_fn",
  "negative_prompt": "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering.",
  "seed": 0,
  "prompt": "excavator digging operation"
}
```

**Important**: Note that `action_load_fn` points to the custom excavator loader, NOT the default one.

### Run Inference
```bash
python examples/action_conditioned.py \
    -i assets/excavator/inference_params.json \
    --setup.checkpoint-path <path_to_converted_pt_checkpoint> \
    --setup.model Cosmos-Predict2.5-2B-Robot-Action-Cond
```

---

## 9. COMPLETE TASK LIST (DO THESE IN ORDER)

When you start working on this VM, do the following steps in order:

### Task 1: Environment Setup
Make sure Cosmos and all dependencies are installed:
```bash
pip install -e .
pip install huggingface_hub hf_xet tqdm pandas numpy opencv-python decord mediapy
```
Verify GPU is available: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### Task 2: Download the Dataset (~22GB, ~2-10 min on VM internet)
```bash
python scripts/excavator/download_dataset.py --output-dir datasets/excavator
```
This downloads only front camera (D01) videos and joystick CSVs (D05) from HuggingFace. If interrupted, re-run — it resumes automatically.

### Task 3: Prepare the Dataset (resize videos, create annotations, ~30-60 min)
```bash
python scripts/excavator/prepare_data.py \
    --input-dir datasets/excavator \
    --output-dir datasets/excavator_cosmos \
    --camera front \
    --resolution 480,640 \
    --train-ratio 0.8
```
If interrupted, re-run — it skips already-processed sessions. Verify the output:
```bash
ls datasets/excavator_cosmos/annotations/train/ | wc -l   # Should be 140
ls datasets/excavator_cosmos/annotations/val/ | wc -l     # Should be 35
cat datasets/excavator_cosmos/dataset_info.json
```

### Task 4: Create ExcavatorDataset (the critical dataloader fix)
Create `cosmos_predict2/_src/predict2/action/datasets/excavator_dataset.py` with the code from Section 4, Step 1. This overrides `_get_actions()` to read the `action` field directly from the JSON, bypassing the rotation matrix math that would produce wrong results for our data.

### Task 5: Update data.py to use ExcavatorDataset
Modify `cosmos_predict2/experiments/excavator/data.py`:
- Change import from `Dataset_3D` to `ExcavatorDataset`
- Change all `L(Dataset_3D)(...)` to `L(ExcavatorDataset)(...)`

### Task 6: Create excavator action loader for inference
Create `cosmos_predict2/excavator_action_loader.py` with the code from Section 4, Step 3.

### Task 7: Create inference params file
Create `assets/excavator/inference_params.json` from Section 8.

### Task 8: Test data loading (recommended before training)
```python
# Quick test to verify the dataloader works
import sys
sys.path.insert(0, '.')
from cosmos_predict2._src.predict2.action.datasets.excavator_dataset import ExcavatorDataset

dataset = ExcavatorDataset(
    train_annotation_path="datasets/excavator_cosmos/annotations/train",
    val_annotation_path="datasets/excavator_cosmos/annotations/val",
    test_annotation_path="datasets/excavator_cosmos/annotations/val",
    video_path="datasets/excavator_cosmos/",
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
print(f"Total samples: {len(dataset)}")
sample = dataset[0]
print(f"Video shape: {sample['video'].shape}")
print(f"Action shape: {sample['action'].shape}")
print(f"Action range: [{sample['action'].min():.4f}, {sample['action'].max():.4f}]")
```

### Task 9: Start Training
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=excavator_action_conditioned_2b_480_640
```
For debugging with smaller batch size first:
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=excavator_action_conditioned_2b_480_640_small_batch
```

---

## 10. KEY FILES REFERENCE

| File | Purpose |
|------|---------|
| `scripts/excavator/download_dataset.py` | Downloads dataset from HuggingFace (D01 + D05 only) |
| `scripts/excavator/prepare_data.py` | Resizes videos, creates annotations, splits train/val |
| `scripts/excavator/resize_single.py` | Helper script for crash-isolated video resizing |
| `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py` | Base `Dataset_3D` class (DO NOT MODIFY) |
| `cosmos_predict2/_src/predict2/action/datasets/excavator_dataset.py` | Custom `ExcavatorDataset` (TO CREATE on VM) |
| `cosmos_predict2/experiments/excavator/action.py` | Experiment configs (already created) |
| `cosmos_predict2/experiments/excavator/data.py` | Data/dataloader configs (needs update to use ExcavatorDataset) |
| `cosmos_predict2/experiments/excavator/__init__.py` | Module init (already created) |
| `cosmos_predict2/action_conditioned.py` | Default inference code (DO NOT MODIFY) |
| `cosmos_predict2/action_conditioned_config.py` | Inference config/args definitions |
| `cosmos_predict2/excavator_action_loader.py` | Custom inference action loader (TO CREATE on VM) |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py` | Main training config entry point |
| `cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py` | Bridge data configs (reference only) |
| `scripts/train.py` | Training entry point |
| `scripts/convert_distcp_to_pt.py` | DCP to PyTorch checkpoint converter |
| `examples/action_conditioned.py` | Inference entry point |
| `datasets/excavator_cosmos/dataset_info.json` | Dataset metadata (created by prepare_data.py) |

---

## 11. IMPORTANT IMPLEMENTATION DETAILS

### How Dataset_3D Creates Training Samples
In `dataset_local.py`, the `_init_sequences` method (line 175) creates training samples by:
1. Loading each JSON annotation
2. Getting `n_frames = len(ann["state"])` (which is N+1 for N action frames)
3. Sliding a window of `sequence_length` (13) frames across all n_frames
4. Each window becomes one training sample with `frame_ids` like [0,1,2,...,12], [1,2,3,...,13], etc.

### How Actions Are Used in Training
In `__getitem__` (line 320):
1. Load the JSON annotation
2. Call `_get_robot_states(label, frame_ids)` → extracts `state[frame_ids]` (6D) and `continuous_gripper_state[frame_ids]`
3. Call `_get_actions(arm_states, gripper_states, accumulate_action)` → **this is what we override**
4. Multiply by `c_act_scaler` (20.0)
5. The result is a (12, 7) tensor that gets fed to the model

### Our Custom _get_actions
Instead of the rotation matrix math, we directly index into the JSON `action` array using the same `frame_ids`. Since `action[i]` is the action at frame `i`, and we need 12 actions for frames [0→1, 1→2, ..., 11→12], we read `action[frame_ids[0]]` through `action[frame_ids[11]]`.

### `continuous_gripper_state` Field
For excavator data, this is derived from the bucket position: `(bucket_value + 1) / 2`, so it ranges [0, 1]. It's stored as N+1 elements (matching `state`). The first element is always 0.5 (neutral). This gets placed in dimension 6 of the action vector by `_get_actions`.

### accumulate_action = False
Our config sets `accumulate_action=False`. In the original `_get_actions`, this means frame-by-frame relative transforms (not accumulated from a base). In our override, this flag is ignored since we read actions directly.

---

## 12. TROUBLESHOOTING

### Download fails or times out
HuggingFace downloads can get "Read timed out" errors. Just re-run the download script — `snapshot_download` with `resume_download=True` handles resuming automatically. Install `hf_xet` for faster downloads: `pip install hf_xet`.

### prepare_data.py crashes during video resizing
The script uses a subprocess (`resize_single.py`) to isolate video resizing crashes. If a specific video causes a crash, that session is skipped and the script continues. If the main script itself crashes, just re-run — it skips already-processed sessions via the `--skip-existing` logic.

### Some sessions have CSV but no D01 video
This is normal — 1 out of 176 sessions has no front camera video. The prepare script handles this gracefully and skips those sessions.

### "No module named 'cosmos_predict2.experiments.excavator'"
Make sure `cosmos_predict2/experiments/excavator/__init__.py` exists.

### CUDA Out of Memory
Reduce `batch_size` in the experiment config. Try batch_size=2 or use the `_small_batch` variant.

### Videos fail to load / decord errors
Make sure `decord` is installed: `pip install decord`. The videos are in mp4v codec from OpenCV.

### Action values seem wrong
Print the actions from the dataloader and compare with the raw JSON. After the scaler, joystick values of ~0.01 become ~0.2 (0.01 * 20). Values should be in roughly [-20, 20] range.

### Training loss is NaN
Check if actions contain NaN values. Also verify the video frames load correctly (should be uint8 tensors).
