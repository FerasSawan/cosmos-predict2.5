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
Custom action loader for excavator joystick data.

This module provides a custom action loading function for the FlywheelAI
excavator dataset, which uses 4D joystick actions padded to 7D for Cosmos
compatibility.

Action mapping:
    - Dim 0: bucket_curl (right joystick X) - bucket open/close
    - Dim 1: boom_up_down (right joystick Y) - boom raise/lower  
    - Dim 2: cab_rotation (left joystick X) - cab swing left/right
    - Dim 3: arm_extend (left joystick Y) - arm in/out
    - Dim 4-6: unused (padding zeros)

Usage:
    In inference_params.json, set:
    "action_load_fn": "cosmos_predict2.excavator_action_loader.load_excavator_action_fn"
"""

import numpy as np
import mediapy
from loguru import logger

from cosmos_predict2.action_conditioned_config import ActionConditionedInferenceArguments


def load_excavator_action_fn():
    """
    Custom action loading function for excavator dataset.
    
    This function handles the excavator-specific data format where actions
    are stored as 7D vectors (4D joystick + 3D padding).
    
    Returns:
        Callable: A function that loads action data from JSON and video.
    """
    
    def load_fn(
        json_data: dict,
        video_path: str,
        args: ActionConditionedInferenceArguments,
    ) -> dict:
        """
        Load excavator action data from JSON and prepare it for inference.
        
        Args:
            json_data: JSON data containing excavator actions and states
            video_path: Path to the video file
            args: Inference arguments
            
        Returns:
            Dictionary containing actions, video data, and metadata
        """
        # Get actions directly from JSON (already in 7D format from prepare_data.py)
        actions = np.array(json_data["action"], dtype=np.float32)
        
        # Apply action scaling
        # For excavator, we scale all dimensions uniformly
        action_scaler = getattr(args, 'action_scaler', 1.0)
        actions = actions * action_scaler
        
        # Handle FPS downsampling if specified
        fps_downsample_ratio = getattr(args, 'fps_downsample_ratio', 1)
        if fps_downsample_ratio > 1:
            actions = actions[::fps_downsample_ratio]
        
        # Load video
        video_array = mediapy.read_video(video_path)
        
        # Get starting frame
        start_frame_idx = getattr(args, 'start_frame_idx', 0)
        img_array = video_array[start_frame_idx]
        
        # Resize if specified
        resolution = getattr(args, 'resolution', 'none')
        if resolution != 'none':
            try:
                h, w = map(int, resolution.split(','))
                img_array = mediapy.resize_image(img_array, (h, w))
            except Exception as e:
                logger.warning(f"Failed to resize image to {resolution}: {e}")
        
        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
        }
    
    return load_fn


def create_excavator_action_sequence(
    bucket: float = 0.0,
    boom: float = 0.0,
    cab: float = 0.0,
    arm: float = 0.0,
    num_frames: int = 12,
) -> np.ndarray:
    """
    Helper function to create a constant action sequence for inference.
    
    This is useful for testing the model with specific joystick positions.
    
    Args:
        bucket: Bucket curl value [-1, 1]. Negative=close/scoop, Positive=open/dump
        boom: Boom value [-1, 1]. Negative=down, Positive=up
        cab: Cab rotation value [-1, 1]. Negative=left, Positive=right
        arm: Arm value [-1, 1]. Negative=retract, Positive=extend
        num_frames: Number of frames to generate
        
    Returns:
        np.ndarray: Action sequence of shape (num_frames, 7)
    """
    action = np.array([bucket, boom, cab, arm, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.tile(action, (num_frames, 1))


def create_digging_sequence(num_frames: int = 48) -> np.ndarray:
    """
    Create a sample digging action sequence.
    
    This creates a realistic digging motion:
    1. Lower boom, extend arm (approach ground)
    2. Curl bucket in (scoop)
    3. Raise boom (lift load)
    4. Rotate cab (move to dump location)
    5. Open bucket (dump)
    
    Args:
        num_frames: Total number of frames (should be divisible by 4)
        
    Returns:
        np.ndarray: Action sequence of shape (num_frames, 7)
    """
    phase_length = num_frames // 4
    
    sequences = []
    
    # Phase 1: Lower boom, extend arm (approach)
    phase1 = create_excavator_action_sequence(
        bucket=0.0, boom=-0.6, cab=0.0, arm=0.4, num_frames=phase_length
    )
    sequences.append(phase1)
    
    # Phase 2: Curl bucket in (scoop)
    phase2 = create_excavator_action_sequence(
        bucket=-0.8, boom=-0.2, cab=0.0, arm=0.2, num_frames=phase_length
    )
    sequences.append(phase2)
    
    # Phase 3: Raise boom (lift)
    phase3 = create_excavator_action_sequence(
        bucket=-0.2, boom=0.7, cab=0.3, arm=-0.2, num_frames=phase_length
    )
    sequences.append(phase3)
    
    # Phase 4: Open bucket (dump)
    phase4 = create_excavator_action_sequence(
        bucket=0.6, boom=0.2, cab=0.0, arm=0.0, num_frames=phase_length
    )
    sequences.append(phase4)
    
    return np.vstack(sequences)


# Action dimension info for reference
EXCAVATOR_ACTION_INFO = {
    "dimensions": 7,
    "mapping": {
        0: {"name": "bucket_curl", "control": "right_joystick_x", "negative": "close/scoop", "positive": "open/dump"},
        1: {"name": "boom", "control": "right_joystick_y", "negative": "down", "positive": "up"},
        2: {"name": "cab_rotation", "control": "left_joystick_x", "negative": "left", "positive": "right"},
        3: {"name": "arm", "control": "left_joystick_y", "negative": "retract", "positive": "extend"},
        4: {"name": "unused", "control": "padding", "negative": "n/a", "positive": "n/a"},
        5: {"name": "unused", "control": "padding", "negative": "n/a", "positive": "n/a"},
        6: {"name": "unused", "control": "padding", "negative": "n/a", "positive": "n/a"},
    },
    "range": [-1.0, 1.0],
    "fps": 25,
}
