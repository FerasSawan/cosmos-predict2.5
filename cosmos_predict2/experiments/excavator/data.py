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
Excavator dataset configuration for action-conditioned training with NL support.

Uses ExcavatorDataset which:
- Reads actions directly from JSON (no 3D rotation recomputation)
- Uses correct action scaling (1.0 instead of 20.0)
- Loads T5 text embeddings for natural language conditioning

Expected directory structure:
    datasets/excavator_cosmos/
    +-- videos/
    |   +-- train/{session_id}/rgb.mp4
    |   +-- val/{session_id}/rgb.mp4
    +-- annotations/
        +-- train/{session_id}.json
        +-- train/{session_id}.npy  (T5 embeddings)
        +-- val/{session_id}.json
        +-- val/{session_id}.npy    (T5 embeddings)
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.excavator_dataset import ExcavatorDataset


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


excavator_13frame_480_640_train_dataset = L(ExcavatorDataset)(
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
    load_t5_embeddings=True,
    mode="train",
)

excavator_13frame_480_640_val_dataset = L(ExcavatorDataset)(
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
    load_t5_embeddings=True,
    mode="val",
)


excavator_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=excavator_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=excavator_13frame_480_640_train_dataset),
    batch_size=4,
    drop_last=True,
)

excavator_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=excavator_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=excavator_13frame_480_640_val_dataset),
    batch_size=2,
    drop_last=True,
)


def register_excavator_data():
    """Register excavator datasets with the config store."""
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="excavator_13frame_480_640_train",
        node=excavator_13frame_480_640_train_dataloader,
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="excavator_13frame_480_640_val",
        node=excavator_13frame_480_640_val_dataloader,
    )


register_excavator_data()
