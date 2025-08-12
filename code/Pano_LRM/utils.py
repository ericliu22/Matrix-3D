
import torch
import functools
import importlib
import os
from functools import partial
from inspect import isfunction

def image_uv_torch_batch(height: int, width: int, B: int, T: int, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Generate (B, T, H, W, 2) UV grid in [0, 1] range.
    """
    u = torch.linspace(0.5 / width, 1 - 0.5 / width, width, device=device, dtype=dtype)
    v = torch.linspace(0.5 / height, 1 - 0.5 / height, height, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')  # [W, H]
    uv = torch.stack([u, v], dim=-1)  # [H, W, 2]

    # expand to (B, T, H, W, 2)
    uv = uv.unsqueeze(0).unsqueeze(0).expand(B, T, height, width, 2)
    return uv



def spherical_uv_to_directions_torch_batch(uv: torch.Tensor) -> torch.Tensor:
    """
    Convert UV grid to direction vectors on unit sphere.
    
    Args:
        uv: Tensor of shape (B, T, H, W, 2)
    
    Returns:
        directions: Tensor of shape (B, T, H, W, 3)
    """
    theta = (1 - uv[..., 0]) * (2 * torch.pi)
    phi = uv[..., 1] * torch.pi

    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)  # (B, T, H, W, 3)

    return directions


def get_pano_pcs_torch_batch(pano_depth: torch.Tensor) -> torch.Tensor:
    """
    From panoramic depth (B, T, H, W) to 3D point cloud (B, T, H, W, 3)
    """
    B, T, H, W = pano_depth.shape
    device = pano_depth.device
    dtype = pano_depth.dtype

    uv = image_uv_torch_batch(height=H, width=W, B=B, T=T, device=device, dtype=dtype)  # (B, T, H, W, 2)
    directions = spherical_uv_to_directions_torch_batch(uv)  # (B, T, H, W, 3)

    points = pano_depth.unsqueeze(-1) * directions  # (B, T, H, W, 3)
    return points

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **extra_kwargs):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **extra_kwargs)


