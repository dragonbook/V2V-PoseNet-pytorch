import sys
import numpy as np

sys.path.append('/home/maiqi/yalong/project/KeyPoint/Code/V2V-PoseNet-pytorch/lib/pyigllib')
import pyigl as igl
from iglhelpers import e2p
from igl_mesh_io import read_mesh_eigen


# TODO:
# 1. check boundary cases


class PointsVoxelization(object):
    def __init__(self, cubic_size, original_size, cropped_size, pool_factor, std):
        self.cubic_size = cubic_size
        self.original_size = original_size
        self.cropped_size = cropped_size  # input size
        self.pool_factor = pool_factor
        self.std = std

        output_size = self.cropped_size / self.pool_factor

        self.d3output_x, self.d3output_y, self.d3output_z = \
            np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')

    def __call__(self, sample):
        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']

        # augmentation
        # resize
        new_size = np.random.rand() * 40 + 80
        # rotation
        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi
        # translation
        trans = np.random.randint(1, self.original_size-self.cropped_size+1+1, size=3)

        input = self._generate_cubic_input(points, refpoint, new_size, angle, trans)
        heatmap = self._generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans)

        return input, heatmap

    def _generate_coord(self, points, refpoint, new_size, angle, trans):
        # points shape: (n, 3)
        coord = points

        # normalize
        coord = (coord - refpoint) / (self.cubic_size/2)

        # discretize
        coord = self._discretize(coord)
        coord += (self.original_size / 2 - self.cropped_size / 2) 

        # resize
        if new_size < 100:
            coord = coord / self.original_size * np.floor(self.original_size*new_size/100) + \
                    np.floor(self.original_size/2 - self.original_size/2*new_size/100)
        else:
            coord = coord / self.original_size * np.floor(self.original_size*new_size/100) - \
                    np.floor(self.original_size/2*new_size/100 - self.original_size/2)

        # rotation
        if angle != 0:
            original_coord = coord.copy()
            original_coord[:,1] = self.original_size-1 - original_coord[:,1]
            original_coord[:,0] -= (self.original_size-1)/2
            original_coord[:,1] -= (self.original_size-1)/2
            coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
            coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
            coord[:,0] += (self.original_size-1)/2
            coord[:,1] += (self.original_size-1)/2
            coord[:,1] = self.original_size-1 - coord[:,1]

        # translation
        coord -= trans - 1

        return coord


    def _generate_cubic_input(self, points, refpoint, new_size, angle, trans):
        coord = self._generate_coord(points, refpoint, new_size, angle, trans)

        # scattering
        coord = np.floor(coord + 0.5) + 1
        cubic = self._scattering(coord)

        return cubic
    
    def _generate_heatmap_gt(self, keypoints, refpoint, new_size, angle, trans):
        coord = self._generate_coord(keypoints, refpoint, new_size, angle, trans)

        coord /= self.pool_factor
        coord += 1  

        # heatmap generation
        heatmap = np.zeros((keypoints.shape[0], self.cropped_size, self.cropped_size, self.cropped_size))
        for i in range(coord.shape[0]):
            xi, yi, zi= coord[i] - 1
            heatmap[i] = np.exp(-(np.power((self.d3output_x-xi)/self.std, 2)/2 + \
                np.power((self.d3output_y-yi)/self.std, 2)/2 + \
                np.power((self.d3output_z-zi)/self.std, 2)/2))
        return heatmap

    def _discretize(self, coord):
        min_normalized = -1
        max_normalized = 1
        scale = (max_normalized - min_normalized) / self.cropped_size
        return np.floor((coord - min_normalized) / scale) 

    def _scattering(self, coord):
        mask = (coord[:,0] >= 1) & (coord[:,0] <= self.cropped_size) & \
               (coord[:,1] >= 1) & (coord[:,1] <= self.cropped_size) & \
               (coord[:,2] >= 1) & (coord[:,2] <= self.cropped_size)
        coord = coord[mask,:]
        cubic = np.zeros((self.cropped_size, self.cropped_size, self.cropped_size))
        cubic[coord[:,0] - 1, coord[:,1] - 1, coord[:,2] - 1] = 1
        return cubic
    