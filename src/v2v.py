import sys
import numpy as np


# TODO:
# 1. check class PointsVoxelization
# 2. np or torch?


def discretize(coord, cropped_size):
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return np.floor((coord - min_normalized) / scale) 


def scattering(coord, cropped_size):
    mask = (coord[:, 0] >= 1) & (coord[:, 0] <= cropped_size) & \
           (coord[:, 1] >= 1) & (coord[:, 1] <= cropped_size) & \
           (coord[:, 2] >= 1) & (coord[:, 2] <= cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))
    cubic[coord[:, 0] - 1, coord[:, 1] - 1, coord[:, 2] - 1] = 1

    return cubic


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized + scale / 2

    coord = coord * cubic_size / 2 + refpoint

    return coord


def extract_coord_from_output(output):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    return: shape (batch, jointNum, 3)
    '''
    assert(len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T
    
    xyz_output = max_index.reshape([*output.shape[:-3], *vsize]) - 1

    return xyz_output


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # normalize
    coord = (coord - refpoint) / (cubic_size/2)

    # discretize
    coord = discretize(coord, cropped_size)
    coord += (original_size / 2 - cropped_size / 2) 

    # resize
    if new_size < 100:
        coord = coord / original_size * np.floor(original_size*new_size/100) + \
                np.floor(original_size/2 - original_size/2*new_size/100)
    else:
        coord = coord / original_size * np.floor(original_size*new_size/100) - \
                np.floor(original_size/2*new_size/100 - original_size/2)

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,1] = original_size-1 - original_coord[:,1]
        original_coord[:,0] -= (original_size-1)/2
        original_coord[:,1] -= (original_size-1)/2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += (original_size-1)/2
        coord[:,1] += (original_size-1)/2
        coord[:,1] = original_size-1 - coord[:,1]

    # translation
    coord -= trans - 1

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    coord = np.floor(coord + 0.5) + 1
    cubic = scattering(coord, cropped_size)

    return cubic


def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    cubic_size, cropped_size, original_size = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)
    coord /= pool_factor
    coord += 1  

    # heatmap generation
    heatmap = np.zeros((keypoints.shape[0], cropped_size, cropped_size, cropped_size))
    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i] - 1
        heatmap[i] = np.exp(-(np.power((d3output_x-xi)/std, 2)/2 + \
            np.power((d3output_y-yi)/std, 2)/2 + \
            np.power((d3output_z-zi)/std, 2)/2))

    return heatmap


class InputOutputVoxelization(object):
    def __init__(self, sizes, pool_factor, std):
        self.sizes = sizes
        self.cubic_size, self.cropped_size, self.original_size = self.sizes
        self.pool_factor = pool_factor
        self.std = std

        output_size = self.cropped_size / self.pool_factor
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')

    def __call__(self, sample):
        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']

        ## Augmentations
        # Resize
        new_size = np.random.rand() * 40 + 80

        # Rotation
        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi

        # Translation
        trans = np.random.randint(1, self.original_size-self.cropped_size+1+1, size=3)

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)

        return input, heatmap


class InputVoxelization(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.cubic_size, self.cropped_size, self.original_size = self.sizes

    def __call__(self, sample):
        points, refpoint = sample['points'], sample['refpoint']

        ## Cancel data augmentations
        new_size = 100
        angle = 0
        trans = self.original_size/2 - self.cropped_size/2 + 1

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)

        return input
