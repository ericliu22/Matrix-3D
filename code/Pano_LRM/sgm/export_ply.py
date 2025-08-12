from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor, unsqueeze
import json

def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    K: Float[Tensor, "3 3"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 3 3"],
    covariances: Float[Tensor, "gaussian 3 3"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # Shift the scene so that the median Gaussian is at the origin.
    means = means - means.median(dim=0).values

    # Rescale the scene so that most Gaussians are within range [-1, 1].
    # scale_factor = means.abs().quantile(0.95, dim=0).max()
    # scale_factor = 1
    # means = means / scale_factor
    # scales = scales / scale_factor

    # # Define a rotation that makes +Z be the world up vector.
    # rotation = [
    #     [0, 0, 1],
    #     [-1, 0, 0],
    #     [0, -1, 0],
    # ]
    # rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # # The Polycam viewer seems to start at a 45 degree angle. Since we want to be
    # # looking directly at the object, we compose a 45 degree rotation onto the above
    # # rotation.
    # adjustment = torch.tensor(
    #     R.from_rotvec([0, 0, -45], True).as_matrix(),
    #     dtype=torch.float32,
    #     device=means.device,
    # )
    # rotation = adjustment @ rotation
    # # We also want to see the scene in camera space (as the default view). We therefore
    # # compose the w2c rotation onto the above rotation.
    # rotation = rotation @ extrinsics[:3, :3].inverse()

    # # Apply the rotation to the means (Gaussian positions).
    # means = einsum(rotation, means, "i j, ... j -> ... i")

    # # Apply the rotation to the Gaussian rotations.
    # rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    # rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations.detach().cpu().numpy()).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    # harmonics_view_invariant = harmonics[..., 0]
    harmonics_view_invariant = harmonics
    harmonics_view_invariant_dc = harmonics_view_invariant[..., 0]
    harmonics_view_invariant_extra = harmonics_view_invariant[..., 1:]
    mask = torch.where(opacities > 0)[0].cpu().numpy()
    distances = torch.norm(means, dim=1)  # [N]

    # 2. 设置一个半径阈值 r（单位应与你的坐标单位一致）
    r = 1.0  # 举例，保留距离原点小于 1 的高斯

    # 3. 创建掩码
    # mask = distances < r

    # mask = mask.cpu().numpy()
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.detach().cpu().numpy()[mask].shape[0], dtype=dtype_full)
    # attributes = (
    #     means.detach().cpu().numpy()[mask],
    #     torch.zeros_like(means).detach().cpu().numpy()[mask],
    #     harmonics_view_invariant.detach().cpu().contiguous().numpy()[mask],
    #     opacities[..., None].detach().cpu().numpy()[mask],
    #     scales.log().detach().cpu().numpy()[mask],
    #     rotations[mask],
    # )
    attributes = (
        means.detach().cpu().numpy()[mask],
        torch.zeros_like(means).detach().cpu().numpy()[mask],
        harmonics_view_invariant_dc.unsqueeze(2).flatten(start_dim=1).detach().cpu().contiguous().numpy()[mask],
        # harmonics_view_invariant_extra.flatten(start_dim=1).detach().cpu().contiguous().numpy()[mask],
        opacities[..., None].detach().cpu().numpy()[mask],
        scales.log().detach().cpu().numpy()[mask],
        rotations[mask],
    )
    
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    return 
