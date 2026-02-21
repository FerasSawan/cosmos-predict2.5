# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Excavator dataset configuration for action-conditioned training.

This module defines the dataset and dataloader configurations for the
FlywheelAI excavator dataset prepared by prepare_data.py.

Expected directory structure:
    datasets/excavator_cosmos/
    ├── videos/
    │   ├── train/
    │   │   └── {session_id}/rgb.mp4
    │   └── val/
    │       └── {session_id}/rgb.mp4
    └── annotations/
        ├── train/
        │   └── {session_id}.json
        └── val/
            └── {session_id}.json
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


# Excavator dataset paths
# These should match the output of prepare_data.py
EXCAVATOR_BASE_PATH = "datasets/excavator_cosmos/"

EXCAVATOR_TRAIN_ANNOTATION_PATH = os.path.join(EXCAVATOR_BASE_PATH, "annotations/train")
EXCAVATOR_VAL_ANNOTATION_PATH = os.path.join(EXCAVATOR_BASE_PATH, "annotations/val")


def get_sampler(dataset):
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# Excavator training dataset - 480x640 resolution, 13 frames (1 conditional + 12 action)
excavator_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=EXCAVATOR_TRAIN_ANNOTATION_PATH,
    val_annotation_path=EXCAVATOR_VAL_ANNOTATION_PATH,
    test_annotation_path=EXCAVATOR_VAL_ANNOTATION_PATH,  # Use val as test
    video_path=EXCAVATOR_BASE_PATH,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)

# Excavator validation dataset
excavator_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=EXCAVATOR_TRAIN_ANNOTATION_PATH,
    val_annotation_path=EXCAVATOR_VAL_ANNOTATION_PATH,
    test_annotation_path=EXCAVATOR_VAL_ANNOTATION_PATH,
    video_path=EXCAVATOR_BASE_PATH,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)


# Dataloaders
excavator_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=excavator_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=excavator_13frame_480_640_train_dataset),
    batch_size=4,  # Adjust based on GPU memory
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

excavator_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=excavator_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=excavator_13frame_480_640_val_dataset),
    batch_size=2,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)


def register_excavator_data():
    """Register excavator datasets with the config store."""
    cs = ConfigStore.instance()
    
    # Register training data
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="excavator_13frame_480_640_train",
        node=excavator_13frame_480_640_train_dataloader,
    )
    
    # Register validation data
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="excavator_13frame_480_640_val",
        node=excavator_13frame_480_640_val_dataloader,
    )


# Auto-register when module is imported
register_excavator_data()
