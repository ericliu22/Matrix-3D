import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch.nn.functional as F

def rescale_depth(
    depth: Float[Tensor, "1 h w"],
    shape: tuple[int, int],
) -> Float[Tensor, "1 h_out w_out"]:
    depth = depth.unsqueeze(0)  # [1, 1, h, w]
    depth_resized = F.interpolate(depth, size=shape, mode='nearest')
    return depth_resized.squeeze(0)  # [1, h_out, w_out]

def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:

    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth: Float[Tensor, "*#batch 1 h w"] = None,
    flag_no_crop: bool = False,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch 1 h_out w_out"] | None,  # updated depth
]:
 
    *_, h_in, w_in = images.shape
    w_out, h_out = shape
    # assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images and depth to the correct size
    if flag_no_crop:
        images = rescale(images, (h_out, w_out))
    else:
        images = rescale(images, (h_scaled, w_scaled))
    images, intrinsics = center_crop(images, intrinsics, [h_out, w_out])
    if depth is not None:
        if flag_no_crop:
            depth = rescale_depth(depth, (h_out, w_out))
        else:
            depth = rescale_depth(depth, (h_scaled, w_scaled))
            depth, _ = center_crop(depth, intrinsics, [h_out, w_out])
        return images, intrinsics, depth
    return images, intrinsics, None


def apply_crop_shim(images, intrinsics, shape, depth=None, flag_no_crop=False):
    images, intrinsics, depth = rescale_and_crop(images, intrinsics, shape, depth, flag_no_crop)
    return images, intrinsics, depth




