from plyfile import PlyData, PlyElement
import numpy as np
import os
from os import makedirs, path
from errno import EEXIST

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

class GaussianModel:
    def __init__(self, max_sh_degree=3):
        self.max_sh_degree = max_sh_degree

    def load_ply_gof(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        if len(extra_f_names) == 0:
            features_extra = np.zeros((xyz.shape[0], 3, (3 + 1) ** 2 - 1))
        else:
            import pdb; pdb.set_trace()
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_extra
        self._opacity = opacities
        self._scaling = scales
        self._rotation = rots

        self.active_sh_degree = self.max_sh_degree
    
    def save_ply_3dgs(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.transpose(0, 2, 1).reshape(self._features_dc.shape[0], -1)
        f_rest = self._features_rest.transpose(0, 2, 1).reshape(self._features_dc.shape[0], -1)
        opacities = self._opacity
        scale = self._scaling
        rotation = self._rotation

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

if __name__ == '__main__':
    model = GaussianModel(max_sh_degree=0)
    base_dir = "/mnt/workspace/zhongqi.yang/VideoInpainting_new/worldgen/output/scene_test"
    output_dir = "/mnt/workspace/zhongqi.yang/tmp_gs"
    cs = os.listdir(base_dir)
    for c in cs:
        model.load_ply_gof(f'{base_dir}/{c}/point_cloud/iteration_6000/point_cloud.ply')
        model.save_ply_3dgs(f'{output_dir}/{c}.ply')