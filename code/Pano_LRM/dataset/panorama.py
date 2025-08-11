import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from typing import *
import sys

from .base_many_view_dataset import BaseManyViewDataset
from decord import VideoReader
from tqdm import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation as R
from ..sgm.pipeline_utils import generate_panovideo_data
from scipy.ndimage import sobel
from shims.crop_shim import apply_crop_shim
import json
GLOBAL_DEPTH_SCALE = 1000.0

def image_uv_torch_batch(height: int, width: int, B: int, T: int, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Generate (B, T, H, W, 2) UV grid in [0, 1] range."""
    u = torch.linspace(0.5 / width, 1 - 0.5 / width, width, device=device, dtype=dtype)
    v = torch.linspace(0.5 / height, 1 - 0.5 / height, height, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    uv = uv.unsqueeze(0).unsqueeze(0).expand(B, T, height, width, 2)
    return uv

def spherical_uv_to_directions_torch_batch(uv: torch.Tensor) -> torch.Tensor:
    """Convert UV grid to direction vectors on unit sphere."""
    theta = (1 - uv[..., 0]) * (2 * torch.pi)
    phi = uv[..., 1] * torch.pi
    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)
    return directions

def get_pano_pcs_torch_batch(pano_depth: torch.Tensor) -> torch.Tensor:
    """Convert panoramic depth (B, T, H, W) to 3D point cloud (B, T, H, W, 3)"""
    B, T, H, W = pano_depth.shape
    device = pano_depth.device
    dtype = pano_depth.dtype

    uv = image_uv_torch_batch(height=H, width=W, B=B, T=T, device=device, dtype=dtype)
    directions = spherical_uv_to_directions_torch_batch(uv)
    points = pano_depth.unsqueeze(-1) * directions
    return points

def generate_depth_mask(depth, depth_thresh=650, grad_threshold=0.1):
    """Generate depth mask based on depth threshold and gradient."""
    valid_mask = np.isfinite(depth)
    depth_mask = depth < depth_thresh

    dx = sobel(depth, axis=1)
    dy = sobel(depth, axis=0)
    grad_magnitude = np.hypot(dx, dy)

    grad_norm = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
    grad_mask = grad_norm < grad_threshold

    mask = valid_mask & depth_mask & grad_mask
    return mask

def parse_rotation(rot_str):
    """Parse rotation string to pitch, yaw, roll values."""
    parts = rot_str.split()
    pitch = float(parts[0].split('=')[1])
    yaw = float(parts[1].split('=')[1])
    roll = float(parts[2].split('=')[1])
    return pitch, yaw, roll

def get_csv_and_framecount(folder_path):
    """Get CSV file path and frame count from folder."""
    folder_name = os.path.basename(folder_path.rstrip('/'))
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            return os.path.join(folder_path, fname), 0
    return None, 0

def unreal_to_opencv_w2c(pitch, yaw, roll, x, y, z):
    """Convert Unreal Engine coordinates to OpenCV world-to-camera transformation."""
    r_unreal = R.from_euler('YXZ', [yaw, pitch, roll], degrees=True)
    R_unreal = r_unreal.as_matrix()

    R_cv = R_unreal
    t_cv = np.array([y, -z, x])

    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_cv
    T_c2w[:3, 3] = t_cv[:,0]
    T_w2c = np.linalg.inv(T_c2w)

    return T_w2c

class PanoraScene(BaseManyViewDataset):
    """Panorama scene dataset for pano-lrm training and inference."""
    
    def __init__(self, mp4_path, pose_path, num_frames=49, 
                 out_hw=(480,720), fov=90, *args, **kwargs):
        self.mp4_path = mp4_path
        self.pose_path = pose_path
        # Remove unsupported parameters from kwargs
        kwargs.pop('num_seq', None)
        kwargs.pop('min_thresh', None)
        kwargs.pop('max_thresh', None)
        kwargs.pop('train_lrm', None)
        
        super().__init__(*args, **kwargs)
        self.out_hw = out_hw
        
        focal = 256
        self.K = np.array([
            [focal, 0, 256.],
            [0., focal, 256.],
            [0., 0, 1.],
        ])

        self.num_frames = num_frames
    
        
        print('Found dataset:',  1)
        self.to_tensor = tf.ToTensor()
    
    def __len__(self):
        return 1
    
    def sample_frames(self, sorted_list, base_samples=81, additional_samples=32):
        """Sample frames with boundary constraints."""

        n = len(sorted_list)
        if n < base_samples:
            return list(range(n)), []

        stride = 1
        max_possible_stride = max(1, (n - 1) // (base_samples - 1))
        stride = min(stride, max_possible_stride)

        L = 0
        R = L + (base_samples - 1) * stride
        base_indices = [L + i * stride for i in range(base_samples)]
        base_indices_vals = [sorted_list[i] for i in base_indices]

        additional_indices_vals = base_indices_vals
        return base_indices_vals, additional_indices_vals

    def _get_views(self, idx_input, resolution, rng, attempts=0): 
        """Get views for training/inference."""
        using_pano = True

        video_path = self.mp4_path
        video = VideoReader(video_path)

        with open(self.pose_path, 'r') as f:
            w2c_poses = json.load(f) 

        # 转换为 NumPy 数组 (形状: [N, 4, 4])
        pose_all_w2c = np.array(w2c_poses, dtype=np.float32)
        pose_all = np.linalg.inv(pose_all_w2c) 

        img_idxs, extra_idxs = self.sample_frames(range(pose_all.shape[0]))
        extra_idxs.sort()
            
        views = []
        instance = video_path.split('/')[-1].replace('.mp4', '')
        for idx in img_idxs:
        
            pano = video[idx].asnumpy()
            depth = np.zeros_like(pano)
            
            cam_pose = pose_all_w2c[idx]
            
            pano = np.clip(pano, 0, None)
            pano = pano / (np.max(pano) + 1e-6)

            # Generate perspective views
            splitted_imgs, splitted_poses = generate_panovideo_data(
                images=np.expand_dims(pano, 0),
                K=self.K,
                Rts=np.expand_dims(cam_pose, 0),
                resolution=(512,512),
                frame_interval=1,
                correct_pano_depth_=True)

            idx_ = 6
            selected_img = ((splitted_imgs[idx_])* 255).astype(np.uint8)
            selected_pose = splitted_poses[idx_]

            pano = (pano * 255).astype(np.uint8)
            pano = self.to_tensor(pano)
            depth = self.to_tensor(depth)
            selected_img = self.to_tensor(selected_img)

            pano, _, depth = apply_crop_shim(pano, torch.from_numpy(self.K), resolution, depth, flag_no_crop=True)
            depth = depth[0].numpy()

            pano_org = F.interpolate(pano[None], size=(512, 1024), mode='bilinear')[0]

            mask = generate_depth_mask(depth)
            
            if using_pano:
                selected_img = pano_org
                #selected_pose = np.linalg.inv(cam_pose)

            views.append(dict(
                pano_img=pano,
                img=selected_img,
                depthmap=depth,
                depth_mask=mask,
                caption='The camera pans from left to right, capturing a wide-angle view of the road and sidewalk.',
                camera_pose=np.linalg.inv(cam_pose),
                pers_pose=selected_pose,
                camera_intrinsics=self.K,
                dataset='panorama',
                label='',
                instance=instance.split('/')[-1],
            ))

        for idx in extra_idxs:
            pano = video[idx].asnumpy()
            depth = np.zeros_like(pano)
    
            if len(pano.shape) == 2 or pano.shape[2] == 1:
                pano = np.stack([pano]*3, axis=-1)
            
            cam_pose = pose_all_w2c[idx]

            pano = np.clip(pano, 0, None)
            pano = pano / (np.max(pano) + 1e-6)

            splitted_imgs, splitted_poses = generate_panovideo_data(
                images=np.expand_dims(pano, 0),
                K=self.K,
                Rts=np.expand_dims(cam_pose, 0),
                resolution=(512,512),
                frame_interval=1,
                correct_pano_depth_=True)

            idx_ = 6
            selected_img = ((splitted_imgs[idx_])* 255).astype(np.uint8)
            selected_pose = splitted_poses[idx_]

            pano = (pano * 255).astype(np.uint8)
            pano = self.to_tensor(pano)
            depth = self.to_tensor(depth)
            selected_img = self.to_tensor(selected_img)
            
            pano_org = F.interpolate(pano[None], size=(512, 1024), mode='bilinear')[0]

            pano, _, depth = apply_crop_shim(pano, torch.from_numpy(self.K), resolution, depth, flag_no_crop=True)
            depth = depth[0].numpy()
            mask = generate_depth_mask(depth) 
            assert np.isfinite(depth).all(), f"Depth contains NaN or Inf values at index {idx}"

            if using_pano:
                selected_img = pano_org
                #selected_pose = np.linalg.inv(cam_pose)

            views.append(dict(
                pano_img=pano,
                img=selected_img,
                depthmap=depth,
                depth_mask=mask,
                caption='The camera pans from left to right, capturing a wide-angle view of the road and sidewalk.',
                camera_pose=np.linalg.inv(cam_pose),
                pers_pose=selected_pose,
                camera_intrinsics=self.K,
                dataset='panorama',
                label='',
                instance=instance.split('/')[-1],
            ))
        del video
        return views

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(split='train', ROOT=path, **kwargs)



        

      
