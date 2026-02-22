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

Fine-tunes Cosmos Predict 2.5 2B action-conditioned model on the FlywheelAI
excavator dataset with frame + natural language + joystick action conditioning.

Key fixes from v1:
- Action scaling: 1.0 (was 20.0, which caused blue screen outputs)
- Lower learning rate: 2^-16 for more stable fine-tuning
- T5 text embeddings: enabled for natural language conditioning
- Uses ExcavatorDataset which reads actions directly from JSON

Usage:
    torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
        -- experiment=excavator_action_conditioned_2b_480_640
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelVariant

ACTION_COND_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)]


excavator_action_conditioned_2b_480_640 = LazyDict(
    dict(
        defaults=[
            ACTION_COND_CHECKPOINT.experiment,
            {"override /data_train": "excavator_13frame_480_640_train"},
            {"override /data_val": "excavator_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_excavator",
            group="excavator_action_conditioned",
            name="2b_excavator_480_640",
        ),
        optimizer=dict(
            lr=2 ** (-16),  # ~1.5e-05, lower than default for stable fine-tuning
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=500,
            load_path=get_checkpoint_path(ACTION_COND_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=10,
            validation_iter=1_000,
            run_validation=True,
            max_val_iter=10,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=25,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
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
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()

for _item in [
    excavator_action_conditioned_2b_480_640,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
