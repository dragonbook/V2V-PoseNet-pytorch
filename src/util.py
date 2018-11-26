import sys
import numpy as np

sys.path.append('/home/maiqi/yalong/project/KeyPoint/Code/V2V-PoseNet-pytorch/lib/pyigllib')
import pyigl as igl
from iglhelpers import e2p
from igl_mesh_io import read_mesh_eigen


class PointsVoxelization(object):
    def __init__(self, cubic_size, original_size, cropped_size, pool_factor):
        self.cubic_size = cubic_size
        self.original_size = original_size
        self.cropped_size = cropped_size
        self.pool_factor = pool_factor

    def __call__(self, sample):
        points, keypoints = sample['points'], sample['keypoints']
        # TODO:


    def _generate_cubic_input(self, points, refpoint, new_size, angle, trans):
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

        # scattering
        coord = np.floor(coord + 0.5) + 1
        cubic = self._scattering(coord)
    
    def _generate_heatmap_gt(self, keypoints, refpoint, new_size, angle, trans):
        pass

    def _discretize(self, coord):
        min_normalized = -1
        max_normalized = 1
        scale = (max_normalized - min_normalized) / self.cropped_size
        return np.floor((coord - min_normalized) / scale) 

    def _scattering(self, coord):
        cubic = np.zeros((self.cubic_size, self.cubic_size, self.cubic_size))
        # TODO: