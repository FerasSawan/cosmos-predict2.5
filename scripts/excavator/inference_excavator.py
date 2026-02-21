#!/usr/bin/env python3
"""
Excavator Action-Conditioned Inference Script

Generate videos of excavator motion conditioned on joystick actions.

This script allows you to:
1. Provide an initial excavator image
2. Specify joystick actions (or use predefined sequences like "dig")
3. Generate a video showing the predicted excavator motion

Usage:
    # Generate with custom actions
    python scripts/excavator/inference_excavator.py \
        --checkpoint /path/to/model_ema_bf16.pt \
        --input-image excavator.jpg \
        --actions "[[0.0, -0.5, 0.0, 0.2], [0.0, -0.5, 0.0, 0.3], ...]" \
        --output-dir outputs/excavator

    # Generate with predefined digging sequence
    python scripts/excavator/inference_excavator.py \
        --checkpoint /path/to/model_ema_bf16.pt \
        --input-image excavator.jpg \
        --action-preset dig \
        --output-dir outputs/excavator

    # Generate from dataset sample
    python scripts/excavator/inference_excavator.py \
        --checkpoint /path/to/model_ema_bf16.pt \
        --input-json datasets/excavator_cosmos/annotations/val/20250828132109.json \
        --output-dir outputs/excavator

Requirements:
    - Fine-tuned excavator checkpoint (converted to .pt format)
    - Input image or JSON annotation file
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mediapy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cosmos_predict2.excavator_action_loader import (
    create_excavator_action_sequence,
    create_digging_sequence,
    EXCAVATOR_ACTION_INFO,
)


def parse_actions(actions_str: str) -> np.ndarray:
    """Parse action string into numpy array."""
    try:
        actions = json.loads(actions_str)
        actions = np.array(actions, dtype=np.float32)
        
        # Pad to 7D if needed
        if actions.shape[1] == 4:
            padding = np.zeros((actions.shape[0], 3), dtype=np.float32)
            actions = np.hstack([actions, padding])
        
        return actions
    except Exception as e:
        raise ValueError(f"Failed to parse actions: {e}")


def get_preset_actions(preset: str, num_frames: int = 48) -> np.ndarray:
    """Get predefined action sequences."""
    presets = {
        "dig": create_digging_sequence(num_frames),
        "boom_up": create_excavator_action_sequence(boom=0.6, num_frames=num_frames),
        "boom_down": create_excavator_action_sequence(boom=-0.6, num_frames=num_frames),
        "bucket_scoop": create_excavator_action_sequence(bucket=-0.7, num_frames=num_frames),
        "bucket_dump": create_excavator_action_sequence(bucket=0.7, num_frames=num_frames),
        "rotate_left": create_excavator_action_sequence(cab=-0.5, num_frames=num_frames),
        "rotate_right": create_excavator_action_sequence(cab=0.5, num_frames=num_frames),
        "arm_extend": create_excavator_action_sequence(arm=0.5, num_frames=num_frames),
        "arm_retract": create_excavator_action_sequence(arm=-0.5, num_frames=num_frames),
        "neutral": create_excavator_action_sequence(num_frames=num_frames),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]


def run_inference(
    checkpoint_path: str,
    input_image: str = None,
    input_json: str = None,
    actions: np.ndarray = None,
    output_dir: str = "outputs/excavator_inference",
    guidance: float = 7.0,
    seed: int = 42,
    resolution: str = "480,640",
    num_frames: int = 13,
    experiment: str = "excavator_action_conditioned_2b_480_640",
):
    """
    Run action-conditioned inference to generate excavator video.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint (.pt file)
        input_image: Path to input image (optional if input_json provided)
        input_json: Path to annotation JSON (optional if input_image provided)
        actions: Action sequence array of shape (num_frames-1, 7)
        output_dir: Directory to save output videos
        guidance: Guidance scale for generation
        seed: Random seed
        resolution: Output resolution as "height,width"
        num_frames: Number of frames to generate (including initial frame)
        experiment: Experiment name for model config
    """
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    import torchvision.transforms.functional as TF
    
    torch.enable_grad(False)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Excavator Action-Conditioned Inference")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_path}")
    print(f"Resolution: {resolution}")
    print(f"Guidance: {guidance}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Load input
    if input_json:
        print(f"\nLoading from JSON: {input_json}")
        with open(input_json, 'r') as f:
            json_data = json.load(f)
        
        # Get video path and load first frame
        video_rel_path = json_data["videos"][0]["video_path"]
        video_base = Path(input_json).parent.parent.parent
        video_path = video_base / video_rel_path
        
        print(f"Loading video: {video_path}")
        video = mediapy.read_video(str(video_path))
        img_array = video[0]
        
        # Get actions from JSON if not provided
        if actions is None:
            actions = np.array(json_data["action"], dtype=np.float32)
            print(f"Loaded {len(actions)} actions from JSON")
    
    elif input_image:
        print(f"\nLoading image: {input_image}")
        img_array = mediapy.read_image(input_image)
    
    else:
        raise ValueError("Must provide either --input-image or --input-json")
    
    # Resize image if needed
    h, w = map(int, resolution.split(','))
    if img_array.shape[0] != h or img_array.shape[1] != w:
        print(f"Resizing image from {img_array.shape[:2]} to ({h}, {w})")
        img_array = mediapy.resize_image(img_array, (h, w))
    
    # Validate actions
    if actions is None:
        raise ValueError("No actions provided. Use --actions, --action-preset, or --input-json")
    
    # Truncate or pad actions to match num_frames - 1
    num_actions_needed = num_frames - 1
    if len(actions) < num_actions_needed:
        # Pad with last action
        padding = np.tile(actions[-1:], (num_actions_needed - len(actions), 1))
        actions = np.vstack([actions, padding])
    elif len(actions) > num_actions_needed:
        actions = actions[:num_actions_needed]
    
    print(f"\nUsing {len(actions)} actions for {num_frames} frames")
    print(f"Action shape: {actions.shape}")
    
    # Initialize inference handler
    print("\nInitializing model...")
    video2world = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
    )
    
    print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Prepare input tensor
    img_tensor = TF.to_tensor(img_array).unsqueeze(0)
    vid_input = torch.cat([
        img_tensor,
        torch.zeros_like(img_tensor).repeat(num_frames - 1, 1, 1, 1)
    ], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    
    # Convert actions to tensor
    actions_tensor = torch.from_numpy(actions).float()
    
    print("\nGenerating video...")
    
    # Generate video
    video_output = video2world.generate_vid2world(
        prompt="",
        input_path=vid_input,
        action=actions_tensor,
        guidance=guidance,
        num_video_frames=num_frames,
        num_latent_conditional_frames=1,
        resolution=resolution,
        seed=seed,
        negative_prompt="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
    )
    
    # Process output
    video_normalized = (video_output - (-1)) / (1 - (-1))
    video_clamped = (
        (torch.clamp(video_normalized[0], 0, 1) * 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    
    # Save video
    output_name = f"excavator_seed{seed}_guidance{guidance}.mp4"
    output_video_path = output_path / output_name
    mediapy.write_video(str(output_video_path), video_clamped, fps=25)
    
    print(f"\nSaved video to: {output_video_path}")
    
    # Save actions used
    actions_path = output_path / f"excavator_seed{seed}_actions.json"
    with open(actions_path, 'w') as f:
        json.dump({
            "actions": actions.tolist(),
            "action_info": EXCAVATOR_ACTION_INFO,
            "guidance": guidance,
            "seed": seed,
            "resolution": resolution,
            "num_frames": num_frames,
        }, f, indent=2)
    
    print(f"Saved actions to: {actions_path}")
    
    # Cleanup
    video2world.cleanup()
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate excavator videos conditioned on joystick actions"
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint (.pt file)"
    )
    
    # Input options (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-image",
        type=str,
        help="Path to input excavator image"
    )
    input_group.add_argument(
        "--input-json",
        type=str,
        help="Path to annotation JSON file (will use first frame and actions)"
    )
    
    # Action options
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--actions",
        type=str,
        help="JSON string of actions: [[bucket, boom, cab, arm], ...]"
    )
    action_group.add_argument(
        "--action-preset",
        type=str,
        choices=["dig", "boom_up", "boom_down", "bucket_scoop", "bucket_dump",
                 "rotate_left", "rotate_right", "arm_extend", "arm_retract", "neutral"],
        help="Use predefined action sequence"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/excavator_inference",
        help="Directory to save output videos"
    )
    
    # Generation options
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="Guidance scale (default: 7.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480,640",
        help="Output resolution as 'height,width' (default: 480,640)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=13,
        help="Number of frames to generate (default: 13)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="excavator_action_conditioned_2b_480_640",
        help="Experiment name for model config"
    )
    
    args = parser.parse_args()
    
    # Parse or get actions
    actions = None
    if args.actions:
        actions = parse_actions(args.actions)
    elif args.action_preset:
        actions = get_preset_actions(args.action_preset, num_frames=args.num_frames * 4)
    # If using input_json without explicit actions, they'll be loaded from JSON
    
    run_inference(
        checkpoint_path=args.checkpoint,
        input_image=args.input_image,
        input_json=args.input_json,
        actions=actions,
        output_dir=args.output_dir,
        guidance=args.guidance,
        seed=args.seed,
        resolution=args.resolution,
        num_frames=args.num_frames,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
