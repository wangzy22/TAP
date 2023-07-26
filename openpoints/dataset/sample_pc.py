import os
import open3d as o3d
from tqdm import tqdm


def sample_pc(data_dir, num_points):
    save_dir = os.path.join(data_dir, 'pointclouds')
    for split in ['train', 'val', 'test']:
        print(split + "==>")
        split_dir = os.path.join(data_dir, split)
        save_split_dir = os.path.join(save_dir, split)
        if not os.path.exists(save_split_dir):
            os.makedirs(save_split_dir)
        samples = sorted(os.listdir(split_dir))
        for sample in tqdm(samples):
            if 'off' in sample:
                sample_name = os.path.join(split_dir, sample)
                save_name = os.path.join(save_split_dir, sample.replace('off', 'ply'))
                mesh = o3d.io.read_triangle_mesh(sample_name)
                pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=4)
                o3d.io.write_point_cloud(save_name, pcd)


if __name__ == '__main__':
    data_dir = 'data/ShapeNet55/shapenet55v1png'
    num_points = 1024
    sample_pc(data_dir, num_points)