from torch.utils.data import Dataset
import os
import numpy as np
import sys

#sys.path.append('/home/maiqi/yalong/project/KeyPoint/Code/V2V-PoseNet-pytorch/lib/pyigllib')
#import pyigl as igl
#from iglhelpers import *
#from igl_mesh_io import *


class Tooth13Dataset(Dataset):
    def __init__(self, root, mode, transform=None):
        if not mode in ['train', 'test']: raise ValueError('Invalid mode')
        
        self.mesh_names_file = os.path.join(root, mode + '_names.txt')
        self.keypoints_file = os.path.join(root, mode + '_keypoints.txt')
        self.refpoints_file = os.path.join(root, mode + '_refpoints.txt')
        self.transform = transform

        if not self._check_exists(): raise RuntimeError('Required dataset files do not exist')

        self._load()
    
    def __getitem__(self, index):
        mesh_name = self.mesh_names[index]
        sample = {'mesh_name': mesh_name, 'keypoints': self.keypoints[index], 'refpoint': self.refpoints[index]}

        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.mesh_names)

    def _load(self):
        with open(self.mesh_names_file) as f:
            self.mesh_names = [l.rstrip() for l in f.readlines()]

        self.keypoints = np.loadtxt(self.keypoints_file)
        self.keypoints = self.keypoints.reshape(self.keypoints.shape[0], -1, 3)
        self.refpoints = np.loadtxt(self.refpoints_file)

        assert(len(self.mesh_names) == self.keypoints.shape[0])
        assert(len(self.mesh_names) == self.refpoints.shape[0])

        
    def _check_exists(self):
        return os.path.exists(self.mesh_names_file) and os.path.exists(self.keypoints_file) and os.path.exists(self.refpoints_file)
