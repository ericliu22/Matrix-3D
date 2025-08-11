###加载gs用来可视化
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, network_gui, render_new
from scene.cameras import Camera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

import torch
import numpy as np
import cv2
import os
ply_path = "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output_step1/valid/a_anime_city,_anime,_4k_resolution,_ultra_detail,_best_quality/geom_optim/output/point_cloud/iteration_3000/point_cloud.ply"
name = ply_path.split('/')[-6]
print(name)
#step1:load ply
gaussian= GaussianModel(sh_degree=0)
gaussian.load_ply(ply_path)
print("load gs well")




def generate_pose_matrix(rotation_angle, position):
    # 绕Y轴的旋转矩阵
    R = np.array([
        [np.cos(rotation_angle),  0, np.sin(rotation_angle)],
        [0,                       1, 0                     ],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
    ])
    
    # 构建位姿矩阵
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = np.array(position)  # 确保position为数组
    return pose

def generate_poses(radius=3, num_poses=36):
    """
    生成往返扫描的相机姿态轨迹：
    1. 第一段：0° → +180°（顺时针）
    2. 第二段：0° → -180°（逆时针）
    """
    poses = []
    half_poses = num_poses // 2  # 每段轨迹的姿势数量
    half_poses = num_poses
    # 第一段：0° → +180°（顺时针）
    for angle_deg in np.linspace(0, 361, half_poses, endpoint=False):
        angle_rad = np.radians(angle_deg)
        x = radius * np.sin(angle_rad)
        z = radius * np.cos(angle_rad)
        poses.append(generate_pose_matrix(
            rotation_angle=angle_rad,
            position=[0, 0, 0]  # 固定y=0
        ))
    return np.stack(poses)  # 形状为 (num_poses, 4, 4)

#step2: generate camera_pose
poses = generate_poses()
print(f"len(poses)={len(poses)}")
#step3: render image
for i, pose in enumerate(poses):
    # rot_z_180 = np.array([
    #     [-1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]])
    # pose=(rot_z_180@pose)
    # pose_flatten=pose.T.flatten().tolist()
    # view_matrix=pose_flatten
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    focal_length_y=960
    focal_length_x=960
    height=512
    width=512
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
    viewpoint_cam=Camera(colmap_id=0, R=pose[:3,:3], T=pose[:3, 3], 
                  FoVx=FovX, FoVy=FovY, 
                  image=torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8), gt_alpha_mask=None,
                  image_name=None, uid=0, data_device="cuda")
    
    render_pkg = render_new(
        viewpoint_cam, 
        gaussian,
        pipe=None,
        bg_color=background,  # 改为关键字形式
        kernel_size=1
    )
    rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
    image = rendering[:3].detach().cpu()
    image_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    save_path=f"/ai-video-sh/haoyuan.li/AIGC/matrix3d_inference/debug/gs_img/{name}/img_{i}.png"
    os.makedirs(f"/ai-video-sh/haoyuan.li/AIGC/matrix3d_inference/debug/gs_img/{name}/",exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"图像已保存至 {save_path}")
    print("render_pkg success")

    

