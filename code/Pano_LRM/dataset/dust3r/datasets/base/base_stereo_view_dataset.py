# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL
import numpy as np
import torch
import sys
sys.path.append('code/Pano_LRM/dataset')
from dust3r.datasets.base.easy_dataset import EasyDataset

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping

import torchvision.transforms as tvf
from dust3r.utils.image import ImgNorm

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 3 ,3

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    #fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1
    breakpoint()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def compute_plucker_coordinates_for_image(K, R, t, image_shape):
    """
    计算图像中每个像素的普朗克坐标。

    :param K: 相机内参矩阵 (3x3)
    :param R: 相机旋转矩阵 (3x3)
    :param t: 相机平移向量 (3x1)
    :param image_shape: 图像的分辨率 (height, width)
    :return: 普朗克坐标数组 (d, m)，与图像分辨率相同
    """
    height, width = image_shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建像素坐标网格
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # 将像素坐标转换为相机坐标系中的射线
    X_cam = np.stack([(x_grid - cx) / fx, (y_grid - cy) / fy, np.ones_like(x_grid)], axis=-1)
    X_cam_norm = np.linalg.norm(X_cam, axis=-1, keepdims=True)
    X_cam = X_cam / X_cam_norm  # 归一化

    # 计算方向向量 d
    d_array = X_cam @ R.T

    # 计算位置向量 m
    t_cross_d = np.cross(t.flatten(), d_array, axis=-1)
    m_array = t_cross_d
    pluk_coor = np.concatenate([m_array, d_array],axis=-1)
    return np.transpose(pluk_coor, (2, 0, 1))



class BaseStereoViewDataset (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 train_lrm=False,
                 aug_crop=False,
                 seed=None):
        self.num_views = 2
        self.split = split
        self._set_resolutions(resolution)

        self.transform = transform
        
        self.transform = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
        self.train_lrm = train_lrm

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for w, h in self._resolutions)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            #if self.train_lrm:
            height, width = view['img'].shape[1:]
            # else:
            #     width, height = view['img'].size

            view['true_shape'] = np.int32((height, width))

            #if self.train_lrm:
            view['img'] = np.array(view['img'].permute(1,2,0)) #/ 255.
            # else:
            #     view['img'] = np.array(view['img']) / 255.
            #view['img_org'] = torch.from_numpy(view['img']).float()
            view['img'] = ImgNorm(view['img']).float()

            if 'pano_img' in view.keys():
                view['pano_img'] = np.array(view['pano_img'].permute(1,2,0))
                view['pano_img'] = ImgNorm(view['pano_img']).float()
            
            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            # view['pts3d'] = pts3d
            
            # view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
    
            # check all datatypes
            # for key, val in view.items():
            #     res, err_msg = is_good_type(key, val)
            #     assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

    def _set_resolutions(self, resolutions):
        ''' Set the resolution(s) of the dataset.
        Params:
            - resolutions: int or tuple or list of tuples
        '''
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]
            
            
        self._resolutions = []

        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        
        # calculate min distance to margin
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        
        ## Center crop
        # Crop on the principal point, make it always centered
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        
        ## Recale with max factor, so  one of width or height might be larger than target_resolution
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]
