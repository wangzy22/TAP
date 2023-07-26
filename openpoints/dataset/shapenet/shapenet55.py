import os
import math
import torch
import random
import numpy as np
import open3d as o3d
from PIL import Image
import torch.utils.data as data
from ..build import DATASETS
from ..data_util import rotate_theta_phi
import logging


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self,
                 data_dir,
                 n_views,
                 num_points=1024,
                 split='train',
                 gravity_dim=2,
                 transform=None,
                 random_view=False
                 ):
        self.data_root = data_dir
        self.subset = ['train', 'val'] if split == 'train' else ['test']
        if num_points == 2048:
            self.pc_dir = [os.path.join(data_dir, 'pointclouds_p2048', subset) for subset in self.subset]
        elif num_points == 1024:
            self.pc_dir = [os.path.join(data_dir, 'pointclouds', subset) for subset in self.subset]
        else:
            raise NotImplementedError('num_points must be either 1024 or 2048')
        self.nviews = n_views
        self.num_points = num_points
        self.total_views = 12
        self.gravity_dim = gravity_dim
        self.transform = transform
        self.rotation_matrixs = self.get_rotation_matrix()
        self.random_view = random_view

        self.file_list = []
        for pc_dir in self.pc_dir:
            file_list_subset = sorted(os.listdir(pc_dir))
            file_list_full = list(map(lambda x: os.path.join(pc_dir, x), file_list_subset))
            self.file_list.extend(file_list_full)
        logging.info(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_rotation_matrix(self):
        phi = -1/2 + 1/6
        theta = np.linspace(0, 2, self.total_views+1)
        v_theta, v_phi = np.meshgrid(theta[:self.total_views], phi)
        angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs
    
    def get_random_rotation_matrix(self):
        phi = np.random.rand() - 0.5
        theta = np.random.rand() * 2
        angles = np.array([[phi, theta]])
        angles = torch.from_numpy(angles) * math.pi
        rotation_matrixs = rotate_theta_phi(angles)
        return rotation_matrixs

    def __getitem__(self, idx):
        sample_path = self.file_list[idx]

        points = np.asarray(o3d.io.read_point_cloud(sample_path).points).astype(np.float32)
        points = points[:, [2, 0, 1]]
        points_norm = self.pc_norm(points).astype(np.float32)

        # data = self.random_sample(data, self.sample_points)
        data = {
            'pos': points_norm
        }

        if self.transform is not None:
            data = self.transform(data)

        random_view = np.random.choice(self.total_views, self.nviews, replace=False)
        view_matrix = self.rotation_matrixs[random_view]
        image_list = []
        for v in random_view:
            if self.num_points == 2048: 
                image_path = sample_path.replace('pointclouds_p2048', 'shapenet55v1').replace('.ply', '_{}.jpg'.format(str(v+1).zfill(3)))
            elif self.num_points == 1024:
                image_path = sample_path.replace('pointclouds', 'shapenet55v1').replace('.ply', '_{}.jpg'.format(str(v+1).zfill(3)))
            else:
                raise NotImplementedError('num_points must be either 1024 or 2048')
            image = Image.open(image_path).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            image_list.append(torch.from_numpy(image))
        
        data['x'] = torch.cat((data['pos'],
                               torch.from_numpy(points_norm[:, self.gravity_dim:self.gravity_dim+1] - points_norm[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        if self.random_view:
            assert self.nviews == 1
            data['views'] = self.get_random_rotation_matrix()
        else:
            data['views'] = view_matrix
        data['imgs'] = torch.stack(image_list, dim=0)

        return data

    def __len__(self):
        return len(self.file_list)
