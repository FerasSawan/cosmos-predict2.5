"""
Custom dataset for excavator action-conditioned training with natural language.

Overrides the default Dataset_3D to:
1. Read actions directly from JSON annotations (not recomputed from states)
2. Use correct action scaling (1.0 instead of 20.0)
3. Support T5 text embeddings for natural language conditioning
"""

import json
import random
import traceback
import warnings

import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class ExcavatorDataset(Dataset_3D):
    """Dataset that reads pre-computed actions directly from JSON annotations
    and supports natural language conditioning via T5 embeddings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c_act_scaler = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        """
        Override: read actions directly from the annotation JSON instead of
        recomputing from states via rotation matrix math.

        The parent class recomputes actions from consecutive states using
        euler2rotm/rotm2euler. For excavator data, the states are cumulative
        joystick sums (not real 3D poses), so the rotation math produces
        wrong results. Instead, we read the pre-computed action field.
        """
        if hasattr(self, '_current_label') and self._current_label is not None:
            all_actions = np.array(self._current_label["action"])
            frame_ids = self._current_frame_ids

            action = np.zeros((self.sequence_length - 1, self.action_dim))
            for k in range(self.sequence_length - 1):
                fid = frame_ids[k]
                if fid < len(all_actions):
                    act = all_actions[fid]
                    action[k, :len(act)] = act[:self.action_dim]
                    action[k, 6] = gripper_states[k + 1]

            return torch.from_numpy(action)

        return super()._get_actions(arm_states, gripper_states, accumulate_action)

    def __getitem__(self, index, cam_id=None, return_video=False):
        """Override to store current label/frame_ids and load T5 embeddings."""
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            self._current_label = label
            self._current_frame_ids = frame_ids

            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions = actions * torch.from_numpy(self.c_act_scaler).float()

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
                npy_path = ann_file.replace(".json", ".npy")
                t5_embeddings = np.squeeze(np.load(npy_path))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
                data["ai_caption"] = label.get("caption", "")

            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

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
