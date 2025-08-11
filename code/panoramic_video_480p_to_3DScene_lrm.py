#!/usr/bin/env python3
"""
Pano-LRM Inference Script

This script provides a minimal inference interface for pano LRM model.
It loads a trained model, processes input data, and generates video outputs.

"""

import torch
from omegaconf import OmegaConf
from Pano_LRM.pano_infer import SATVideoDiffusionEngine
import os
import logging
import sys
import argparse
from typing import Dict, Any
from datetime import datetime
from Pano_LRM.dataset.panorama import PanoraScene

# Configure logging
def setup_logging():
    """Setup logging configuration."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Default configuration - these will be overridden by command line arguments
DEFAULT_CONFIG = {
    'config_paths': ["code/Pano_LRM/lrm_gs_clear.yaml", "code/Pano_LRM/sft_paro_gs.yaml"],
    'ckpt_path': "./checkpoints/pano_lrm/pano_lrm.pt",
    'sample_idx': 0,
    'data_root': 'data',
    'device': None  # Will be auto-detected
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LRM-GS Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python script_pano_lrm.py
        """
    )
    
    parser.add_argument("--config_paths", nargs="+", 
                       default=DEFAULT_CONFIG['config_paths'],
                       help="Paths to configuration files (default: %(default)s)")
    parser.add_argument("--ckpt_path", type=str, 
                       default=DEFAULT_CONFIG['ckpt_path'],
                       help="Path to model checkpoint (default: %(default)s)")
    parser.add_argument("--sample_idx", type=int, 
                       default=DEFAULT_CONFIG['sample_idx'],
                       help="Sample index to process (default: %(default)s)")
    parser.add_argument("--video_path", type=str, 
                       help="Directory to panorama video)")
    parser.add_argument("--pose_path", type=str, 
                       help="Directory to panorama video pose)")
    parser.add_argument("--out_path", type=str, 
                       help="Directory to output_path)")                     
    parser.add_argument("--device", type=str, 
                       default=DEFAULT_CONFIG['device'],
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--wan_model_path", type=str,
                       default="./checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
                       help="Path to Wan model checkpoint (default: %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def load_configs(config_paths: list) -> Dict[str, Any]:
    """Load and merge configuration files."""
    configs = []
    for cfg_path in config_paths:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        configs.append(OmegaConf.load(cfg_path))
    return OmegaConf.merge(*configs)

def load_model_weights(model, ckpt_path: str, device: str) -> None:
    """Load model weights from checkpoint file."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading model weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['module'], strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {str(e)}")

def collate_fn(batch):
    """
    Collate function for batching data samples.
    
    Args:
        batch: List of data samples
        
    Returns:
        Dictionary containing batched tensors
    """
    B = len(batch)
    T = len(batch[0])
    frames = 81

    C, H, W = batch[0][0]['pano_img'].shape
    _, H_org, W_org = batch[0][0]['img'].shape

    device = torch.device("cpu")  

    x = torch.empty(B, T, 3, H_org, W_org, dtype=torch.bfloat16)
    pano_imgs = torch.empty(B, T, 3, H, W, dtype=torch.bfloat16)

    camera_pose = torch.empty(B, T, 4, 4, dtype=torch.float32)
    pers_pose = torch.empty(B, T, 4, 4, dtype=torch.float32)
    K = torch.empty(B, T, 3, 3, dtype=torch.float32)
    depthmap = torch.empty(B, T, 1, H, W, dtype=torch.bfloat16)
    depth_mask = torch.empty(B, T, 1, H, W, dtype=torch.bool)
    fps = torch.full((B, T, 1), 8, dtype=torch.uint8)
    num_frames = torch.full((B, T, 1), 49, dtype=torch.uint8)
  
    for b in range(B):
        for t in range(T):
            x[b, t] = batch[b][t]['img'].to(dtype=torch.bfloat16)
            pano_imgs[b, t] = batch[b][t]['pano_img'].to(dtype=torch.bfloat16) if 'pano_img' in batch[b][t] else torch.zeros(3, H, W, dtype=torch.bfloat16)
            camera_pose[b, t] = torch.from_numpy(batch[b][t]['camera_pose']).float()
            pers_pose[b, t] = torch.from_numpy(batch[b][t]['pers_pose']).float()
            K[b, t] = torch.from_numpy(batch[b][t]['camera_intrinsics']).float()
            depthmap[b, t] = torch.from_numpy(batch[b][t]['depthmap']).float()
            depth_mask[b, t] = torch.from_numpy(batch[b][t]['depth_mask']).bool()
            
    caption = batch[0][0].get('caption', 'a scene with camera movement')

    x = x.bfloat16().contiguous()
    x = x[:,frames:]
    pano_imgs = pano_imgs[:,:frames]

    depthmap= depthmap[:,:frames]
    depth_mask = depth_mask[:,:frames]
    camera_pose = camera_pose.float().contiguous()
    pers_pose = pers_pose.float().contiguous()
  
    K = K.float().contiguous()

    K[:,:,0,0]  = 0.5
    K[:,:,1,1]  = 0.5
    K[:,:,0,2] = 0.5
    K[:,:,1,2] = 0.5

    depthmap = depthmap.clamp(0, 30.)

    if camera_pose[:,0][0,0,0] != 1.0:
        # Convert to relative camera coordinates
        camera_anchor = torch.linalg.inv(camera_pose[:,0])

        # Convert pose to relative pose
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            ]], dtype=torch.float32, device='cpu')

        camera_norm_matrix = canonical_camera_extrinsics @ camera_anchor
        camera_pose = (camera_norm_matrix[None].repeat(1, T, 1, 1) @ camera_pose)

        pers_pose = (camera_norm_matrix[None].repeat(1, T, 1, 1) @ pers_pose)

    camera_pose_context, camera_pose = camera_pose[:,:frames], camera_pose[:,frames:]
    pers_camera_pose_context, pers_camera_pose = pers_pose[:,:frames], pers_pose[:,frames:]
    K_context, K = K[:,:frames], K[:,frames:]

    near = torch.tensor(0.0001).repeat(camera_pose.shape[1]) 
    far = torch.tensor(30.).repeat(camera_pose.shape[1]) 

    scene = batch[0][0]['instance']
    out_dir = batch[0][0]['out_dir']
    
    return {
        "mp4": x,
        "pano_imgs": pano_imgs,
        'K_context': K_context,
        'depth': depthmap,
        'pose_context': camera_pose_context,
        'K': K,
        "fps": fps,
        "num_frames": num_frames,
        "pose": pers_camera_pose,
        'txt': scene,
        'save_dir': out_dir,
        'near': near,
        'far': far,
        'depth_mask':depth_mask,
        }

def move_to_device(batch, device):
    """Move all tensors in batch to specified device."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def main():
    """Main inference function."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Starting Pano-LRM inference")
    logger.info("=" * 60)
    logger.info(f"Configuration files: {args.config_paths}")
    logger.info(f"Checkpoint path: {args.ckpt_path}")
    logger.info(f"Wan model path: {args.wan_model_path}")
    logger.info(f"Sample index: {args.sample_idx}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # try:
    # Load configurations
    logger.info("Loading configurations...")
    config = load_configs(args.config_paths)

    # Create model arguments
    class DummyArgs:
        pass
    model_args = DummyArgs()
    model_args.model_config = config["model"] if "model" in config else config
    # Add wan_model_path to model config
    model_args.model_config["wan_model_path"] = args.wan_model_path
    model_args.device = args.device
    model_args.fp16 = False
    model_args.bf16 = True
    model_args.batch_from_same_dataset=True
    model_args.data_config = config["data"]

    logger.info(f"Using device: {model_args.device}")

    # Initialize model
    logger.info("Initializing model...")
    model = SATVideoDiffusionEngine(model_args)
    model.eval()
    model.to(model_args.device)

    # Load model weights
    load_model_weights(model, args.ckpt_path, model_args.device)

    # Load dataset and get sample
    logger.info("Loading dataset and sample...")
    dataset = PanoraScene(mp4_path=args.video_path, pose_path=args.pose_path, out_path = args.out_path, resolution=(960,480), num_seq=100, num_frames=49, max_thresh=100, train_lrm=True) 

    # Get sample and process
    sample = dataset[args.sample_idx]
    batch = collate_fn([sample])
    batch = move_to_device(batch, model_args.device)

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        loss, loss_dict = model.shared_step(batch)
    
    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("=" * 60)
    
    return 0
    
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
