#!/usr/bin/env python3

import math
from typing import Any, Dict
import torch.nn.functional as F
import sys
import os

import torch
from torch import nn
from einops import rearrange

# Add DiffSynth-Studio to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DiffSynth-Studio'))
from diffsynth import WanVideoPipeline, ModelManager
import gc
import numpy as np
from Pano_LRM.sgm.export_ply import export_ply
from Pano_LRM.sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
from Pano_LRM.sgm.gs_decoder import DecoderSplattingCUDA
from Pano_LRM.sgm.pano_render import Pano_DecoderSplattingCUDA
from Pano_LRM.sgm.gsrenderer import GS_decoder
from Pano_LRM.sgm.gs_adapter.gaussian_adapter import GaussianAdapter_pano
from Pano_LRM.utils import get_pano_pcs_torch_batch
from torchvision.utils import save_image
import imageio
from pathlib import Path


class SATVideoDiffusionEngine(nn.Module):
    """
    SAT Video Diffusion Engine with Gaussian Splatting.
    
    This class implements a video diffusion model that combines video generation
    with 3D Gaussian Splatting rendering for high-quality video synthesis.
    
    Attributes:
        log_keys: Keys for logging
        input_key: Key for input data
        not_trainable_prefixes: Prefixes of modules that should not be trained
        en_and_decode_n_samples_a_time: Number of samples to encode/decode at once
        lr_scale: Learning rate scale factor
        lora_train: Whether to use LoRA training
        dtype: Data type for computations
        dtype_str: String representation of data type
        renderer: Main renderer for standard resolution
        pano_renderer: Panoramic renderer for high resolution
        gs_decoder1: Gaussian Splatting decoder
        up_sampler: Upsampling module
        pipe: Video pipeline
        tiler_kwargs: Arguments for tiling
        device: Device to run on
    """
    
    def __init__(self, args, **kwargs):
        """
        Initialize the SATVideoDiffusionEngine.
        
        Args:
            args: Arguments object containing model configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        model_config = args.model_config
        # Model args preprocessing
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        renderer_config = model_config.get("renderer_config", None)
        renderer_pano_config = model_config.get("renderer_pano_config", None)
        up_sampler_config = model_config.get("up_sampler_config", None)
        gs_decoder_config = model_config.get("gs_decoder_config", None)

        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)  

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        if network_config:
            network_config["params"]["dtype"] = dtype_str

        # Initialize components
        self.renderer = DecoderSplattingCUDA(make_scale_invariant=True)
        self.pano_renderer = Pano_DecoderSplattingCUDA(make_scale_invariant=True)
        self.gs_decoder1 = GS_decoder(gs_dim=14)
        self.up_sampler = GaussianAdapter_pano(gaussian_scale_min=0.5, gaussian_scale_max=25.0, sh_degree=0)

        # Initialize video pipeline
        wan_model_path = model_config.get("wan_model_path", "Wan2.1_VAE.pth")
        if wan_model_path is None:
            raise ValueError("wan_model_path must be specified in config or command line arguments")
        model_path = [wan_model_path]
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        # Tiling configuration
        self.tiler_kwargs = {"tiled": False, "tile_size": (34, 34), "tile_stride": (18, 16)}

        # Store other parameters
        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def forward(self, x, batch):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            batch: Batch dictionary containing all necessary data
            
        Returns:
            Tuple of (loss_mean, loss_dict)
        """
        self.infer(x, batch)
        # Return dummy values for now - replace with actual loss computation
        dummy_loss = torch.tensor(0.0, device=x.device)
        loss_dict = {"loss": dummy_loss, "loss_mse": dummy_loss, "loss_lpips": dummy_loss, "loss_depth": dummy_loss}
        return dummy_loss, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        """
        Shared step for training/inference.
        
        Args:
            batch: Batch dictionary containing all necessary data
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        x = self.get_input(batch)
        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x, batch)
            batch["lr_input"] = lr_z

        # Prepare video for encoding
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.pipe.encode_video(x, **self.tiler_kwargs)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Clean up memory
        gc.collect()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def get_input(self, batch):
        """
        Extract input tensor from batch.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Input tensor
        """
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        """
        Encode input using first stage model.
        
        Args:
            x: Input tensor
            batch: Batch dictionary
            
        Returns:
            Encoded tensor
        """
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return x * self.scale_factor  # already encoded

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def infer(self, input, batch):
        """
        Main inference function.
        
        Args:
            input: Input tensor
            batch: Batch dictionary containing all necessary data
        """
        frames = 81
        
        def spherical_ray_condition(K, c2w, H, W, device, flip_flag=None):
            """
            Generate Plücker coordinates for spherical images, interface aligned with pinhole ray_condition.
            Args:
                K: [B, V, 3, 3] (not used, but passed for interface alignment)
                c2w: [B, V, 4, 4]
                H, W: height, width
                device: device
                flip_flag: [B, V] bool, whether to flip left-right (for meshgrid)
            Returns:
                plucker: [B, V, H, W, 6]
            """
            B, V = c2w.shape[:2]

            # Generate grid
            j, i = torch.meshgrid(
                torch.linspace(0, H-1, H, device=device, dtype=c2w.dtype),
                torch.linspace(0, W-1, W, device=device, dtype=c2w.dtype),
                indexing='ij'
            )  # [H, W]
            i = i.reshape(1, 1, H*W).expand(B, V, H*W) + 0.5  # [B, V, HxW]
            j = j.reshape(1, 1, H*W).expand(B, V, H*W) + 0.5

            # Normalize to [0,1] -> [-pi, pi] / [-pi/2, pi/2]
            theta = (2 * (i / W) - 1) * torch.pi        # horizontal angle
            phi = (2 * (j / H) - 1) * (torch.pi / 2)    # vertical angle

            # Direction vectors (spherical -> 3D unit vectors)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            xs = cos_phi * cos_theta
            ys = cos_phi * sin_theta
            zs = sin_phi

            rays_d = torch.stack([xs, ys, zs], dim=-1)  # [B, V, HxW, 3]
            rays_d = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)

            # Apply rotation
            rays_d = torch.matmul(rays_d, c2w[..., :3, :3].transpose(-1, -2))  # [B, V, HxW, 3]

            # Camera position
            rays_o = c2w[..., :3, 3]                     # [B, V, 3]
            rays_o = rays_o.unsqueeze(2).expand_as(rays_d)  # [B, V, HxW, 3]

            # Plücker encoding
            rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # [B, V, HxW, 3]
            plucker = torch.cat([rays_dxo, rays_d], dim=-1) # [B, V, HxW, 6]

            plucker = plucker.view(B, V, H, W, 6)  # reshape to [B, V, H, W, 6]

            return plucker

        # Generate spherical ray conditions
        plucker = spherical_ray_condition(batch["K_context"], batch["pose_context"], batch['pano_imgs'].shape[-2], batch['pano_imgs'].shape[-1], device=batch['pano_imgs'].device)
        input = input.to(plucker.dtype)
        gs_outputs, gs_depth = self.gs_decoder1(input, plucker) #[49, 120, 180]

        gs_dim = gs_outputs.shape[-1]
        gs_outputs = gs_outputs.reshape(1, frames, -1, gs_dim).unsqueeze(-2)
        densities = gs_outputs[..., -1].sigmoid().unsqueeze(-1)

        gs_depth = gs_depth.reshape(1, frames, -1, 3).mean(-1, keepdim=True).unsqueeze(-2)

        # Process ground truth depth
        gt_depth = batch['depth']
        gt_depth = F.interpolate(gt_depth.reshape(frames,1,480,960), size=(240, 480), mode='nearest')
        gt_depth = gt_depth[None]
        gt_depth = gt_depth.permute(0,1,3,4,2)
        gt_depth = gt_depth.reshape(1, frames, -1 ,1,1)

        # Process ground truth mask
        gt_mask = batch['depth_mask']
        gt_mask = gt_mask.float()
        gt_mask = F.interpolate(gt_mask[0], size=(240, 480), mode='nearest')
        gt_mask = gt_mask.bool()

        # Process videos
        videos = batch['pano_imgs']
        videos = F.interpolate(videos[0], size=(240, 480), mode='bilinear')
        videos = videos[None]
        videos = videos.permute(0,1,3,4,2)
        videos = (videos+1)/2
        videos = videos.reshape(1, frames, -1 , 3)

        # Upsampling configuration
        upsampler_size = 2
        stride = 1
        depths = 30. * torch.sigmoid(gs_depth) 
        
        def sample_image_grid(
            shape: tuple[int, ...],
            device: torch.device = torch.device("cpu")): 
            """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

            # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
            # (row, col) coordinate.
            indices = [torch.arange(length, device=device) for length in shape]
            stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

            # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
            # each entry is an (x, y) coordinate.
            coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
            coordinates = reversed(coordinates)
            coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

            return coordinates, stacked_indices

        # Sample image grid
        xy_ray, _ = sample_image_grid((batch['pano_imgs'].shape[-2]//upsampler_size, batch['pano_imgs'].shape[-1]//upsampler_size), batch['pano_imgs'].device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        xy_ray = xy_ray[None, None, ...].expand(1, batch['pano_imgs'].shape[1], -1, -1, -1) #not used here

        def map_pdf_to_opacity(pdf, global_step):
            """Map PDF to opacity values."""
            x = 0.0 + min(global_step / 1., 1) * (0. - 0.)
            exponent = 2**x
            return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

        # Forward pass through upsampler
        gaussians = self.up_sampler.forward(
            rearrange(batch["pose_context"], "b v i j -> b v () () () i j"),
            rearrange(batch["K_context"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            map_pdf_to_opacity(densities, 0),
            rearrange(gs_outputs[..., 3:-1].to(torch.float), "b v r srf c -> b v r srf () c"),
            (batch['pano_imgs'].shape[-2]//upsampler_size, batch['pano_imgs'].shape[-1]//upsampler_size)
        )

        # Rotation matrix for point cloud transformation
        rot_matrix = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=np.float32)  # shape (3, 3)

        rot_matrix_t = torch.tensor(rot_matrix, dtype=depths.dtype, device=depths.device)  # shape (3, 3)

        depths = depths.reshape(frames, 1, 240, 480)
        gt_depth = gt_depth.reshape(frames, 1, 240, 480)
        rotated_points = torch.matmul(get_pano_pcs_torch_batch(depths.squeeze(2)).reshape(1, frames, -1, 3), rot_matrix_t.T) 

        # Apply camera transformation
        # c2w: [1, 49, 4, 4]
        c2w = batch["pose_context"]     # [1, 49, 4, 4]
        R = c2w[..., :3, :3]            # [1, 49, 3, 3]
        T = c2w[..., :3, 3]             # [1, 49, 3]

        # Rotation: pc @ R^T (note the last two dimensions)
        #rotated_points = rotated_points.squeeze(-2)  # remove the size=1 dimension
        rotated_points = torch.matmul(rotated_points.to(torch.float), R.transpose(-1, -2)) + T.unsqueeze(2)

        gaussians.means = rotated_points.reshape(1, frames, -1, 1, 1, 3) #

        # Rearrange gaussians
        # gaussians.harmonics = gt_harmonics
        gaussians.means = rearrange(gaussians.means[:,::2], "b v r srf spp xyz -> b (v r srf spp) xyz").to(torch.float)
        gaussians.covariances = rearrange(gaussians.covariances[:,::2], "b v r srf spp i j -> b (v r srf spp) i j").to(torch.float)
        gaussians.harmonics = rearrange(gaussians.harmonics[:,::2], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh").to(torch.float)
        gaussians.opacities = rearrange(gaussians.opacities[:,::2], "b v r srf spp -> b (v r srf spp)").to(torch.float)

        # Render based on resolution
        #if batch['mp4'].shape[-1] != 512:
        with torch.cuda.amp.autocast(dtype=torch.float):
            pano_render_output = self.pano_renderer.forward(
                gaussians,
                batch["pose"][:,::stride],
                batch["K"][:,::stride],
                batch["near"][None][:,::stride],
                batch["far"][None][:,::stride],
                (batch['mp4'].shape[-2], batch['mp4'].shape[-1]),
                depth_mode=None,
            )
        # perpsective rendering
        with torch.cuda.amp.autocast(dtype=torch.float):
            render_output = self.renderer.forward(
                gaussians,
                batch["pose"][:,::stride],
                batch["K"][:,::stride],
                batch["near"][None][:,::stride],
                batch["far"][None][:,::stride],
                (512, 512),
                depth_mode=None,
            )
        render_output = render_output.color
        frames_ = render_output[0]  # [T, 3, H, W]

        pano_render_output = pano_render_output.color

        save_dir = batch['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save predicted panorama as video
        pred_frames = (pano_render_output[0].clamp(0, 1).mul(255).byte().cpu().numpy())
        pred_frames = pred_frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
        imageio.mimwrite(f'{save_dir}/enerated_3dgs_lrm_render_pano.mp4', pred_frames, fps=12, quality=10, macro_block_size=None)
        
        # # Save ground truth panorama as video
        # gt_frames = ((batch['mp4'][:,::stride][0]+1)/2).clamp(0, 1).mul(255).byte().cpu().numpy()
        # gt_frames = gt_frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
        # imageio.mimwrite(f'{save_dir}/generated_3dgs_lrm_render_pano.mp4', gt_frames, fps=12, quality=10, macro_block_size=None)

        # Generate 360-degree view
        with torch.cuda.amp.autocast(dtype=torch.float):
            # Get last pose
            last_pose = batch["pose"][:, -1:]          # [B,1,4,4]
            B = last_pose.size(0)
            num_views = 72

            eye3 = torch.eye(3, device=last_pose.device, dtype=last_pose.dtype).view(1,1,3,3).expand_as(last_pose[..., :3, :3])
            last_pose[..., :3, :3] = eye3         

            # Angles (no repetition of start and end)
            angles = torch.arange(num_views, device=last_pose.device, dtype=last_pose.dtype) / num_views * 2 * math.pi
            if num_views > 1:
                angles = angles[:-1]                   # [N]
            N = angles.numel()

            # Rotation matrix around world Y-axis (4x4)
            c, s = torch.cos(angles), torch.sin(angles)
            rot_mats = torch.zeros(N, 4, 4, device=last_pose.device, dtype=last_pose.dtype)
            rot_mats[:, 0, 0] =  c;  rot_mats[:, 0, 2] =  s
            rot_mats[:, 1, 1] =  1
            rot_mats[:, 2, 0] = -s;  rot_mats[:, 2, 2] =  c
            rot_mats[:, 3, 3] =  1
            rot_mats = rot_mats.unsqueeze(0)           # [1,N,4,4]

            # Apply rotations (keep original position, only rotate orientation; if want to rotate around camera's own axis/orbit, change multiplication order or translation)
            new_poses = last_pose @ rot_mats           # [B,N,4,4]

            # Extend intrinsics / near / far
            extended_K = batch["K"][:, -1:].expand(-1, N, -1, -1)  # [B,N,3,3]

            near = batch["near"][:71]
            if near.dim() == 1:            # [B]
                near = near.unsqueeze(0)   # -> [B,1]

            far = batch["far"][:71]
            if far.dim() == 1:
                far = far.unsqueeze(0)

            # Render 360-degree view
            render_output1 = self.renderer.forward(
                gaussians,
                new_poses,
                extended_K,
                near,
                far,
                (512, 512),
                depth_mode=None,
            )
            render_output1 = render_output1.color
    
        # Combine frames
        frame1 = render_output1[0]
        frames_ = torch.cat([frames_, frame1], dim=0)

        # Convert to imageio format [T, H, W, C] RGB
        video_frames = (frames_.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy())                   

        # Save as MP4 (auto-select encoder)
        imageio.mimwrite(f'{save_dir}/generated_3dgs_lrm_render_persp.mp4',  video_frames, fps=12, quality=10, macro_block_size=None)  # avoid size alignment issues
        
        # save gs for viewer
        xyz = gaussians.means.detach()[0]
  
        opacities = gaussians.opacities.detach()[0]
        harmonics = gaussians.harmonics[0]

        scale = gaussians.scales[:,::2].reshape(1,-1, 3).detach()[0]
        rotation = gaussians.rotations[:,::2].reshape(1,-1, 3,3).detach()[0]

        extrinsic = batch["pose_context"][0, 0]

        export_ply(extrinsics=extrinsic, K=batch["K_context"][0,0], means=xyz, scales=scale, rotations=rotation, harmonics=harmonics, opacities=opacities, path=Path(save_dir+'/generated_3dgs_lrm.ply'),covariances=None)
        return

