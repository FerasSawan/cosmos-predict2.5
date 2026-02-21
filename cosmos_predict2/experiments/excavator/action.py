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
Excavator action-conditioned training experiment configuration.

This experiment fine-tunes Cosmos Predict 2.5 2B model on the FlywheelAI
excavator dataset for action-conditioned video generation.

Action format (7D, padded from 4D joystick):
    - Dim 0: bucket_curl (right joystick X)
    - Dim 1: boom_up_down (right joystick Y)
    - Dim 2: cab_rotation (left joystick X)
    - Dim 3: arm_extend (left joystick Y)
    - Dim 4-6: unused (padding)

Usage:
    torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
        -- experiment=excavator_action_conditioned_2b_480_640

For H200 with more VRAM, you can increase batch_size in dataloader_train.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint as base
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]


# Excavator dataset configuration
# This will be registered as a data config that can be used with the action-conditioned model
excavator_action_conditioned_2b_480_640 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            # We'll override the data config inline since excavator data isn't registered
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_excavator",
            group="excavator_action_conditioned",
            name="2b_excavator_480_640",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # ~3.05e-05, same as bridge experiment
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=1_000,  # Save more frequently for excavator
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            max_iter=20_000,  # Adjust based on dataset size and convergence
            logging_iter=10,
            validation_iter=500,
            run_validation=True,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=2000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=25,  # Excavator data is 25 FPS
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=2000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=25,
                    save_s3=False,
                ),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,  # 1 conditional frame + 12 action frames / 4
                net=dict(
                    action_dim=7,  # 4D joystick + 3D padding
                    num_action_per_chunk=12,  # 12 actions per chunk (0.48s at 25fps)
                ),
            ),
        ),
        # Dataloader configuration for excavator
        # Note: The actual dataset paths are configured in the data.py file
        # This overrides batch_size and video_size for H200
        dataloader_train=dict(
            batch_size=4,  # H200 has 141GB VRAM, can handle larger batches
            sampler=dict(
                dataset=dict(
                    fps_downsample_ratio=1,
                    video_size=[480, 640],
                ),
            ),
            dataset=dict(
                fps_downsample_ratio=1,
                video_size=[480, 640],
            ),
        ),
        dataloader_val=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(
                    fps_downsample_ratio=1,
                    video_size=[480, 640],
                ),
            ),
            dataset=dict(
                fps_downsample_ratio=1,
                video_size=[480, 640],
            ),
        ),
    ),
    flags={"allow_objects": True},
)


# Smaller batch size variant for testing or lower VRAM GPUs
excavator_action_conditioned_2b_480_640_small_batch = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_excavator",
            group="excavator_action_conditioned",
            name="2b_excavator_480_640_small",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=1_000,
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=10,
            validation_iter=500,
            run_validation=True,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, do_x0_prediction=False, guidance=[0, 3, 7], fps=25, save_s3=False),
                every_n_sample_ema=dict(every_n=2000, do_x0_prediction=False, guidance=[0, 3, 7], fps=25, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(action_dim=7, num_action_per_chunk=12),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,  # Smaller batch for testing
            sampler=dict(dataset=dict(fps_downsample_ratio=1, video_size=[480, 640])),
            dataset=dict(fps_downsample_ratio=1, video_size=[480, 640]),
        ),
        dataloader_val=dict(
            batch_size=1,
            sampler=dict(dataset=dict(fps_downsample_ratio=1, video_size=[480, 640])),
            dataset=dict(fps_downsample_ratio=1, video_size=[480, 640]),
        ),
    ),
    flags={"allow_objects": True},
)


# Register experiments
cs = ConfigStore.instance()

for _item in [
    excavator_action_conditioned_2b_480_640,
    excavator_action_conditioned_2b_480_640_small_batch,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
