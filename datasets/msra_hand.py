from torch.utils.data import Dataset
import os
import numpy as np
import sys


# TODO: implement MSRAHandDataset class
# (1) test
# (2) load depth image, pixel2world and world to pixel etc
#


class MARAHandDataset(Dataset):
    def __init__(self, root, center_dir, mode, test_subject_id, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 241.42
        self.fy = 241.42
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        self.subject_num = 9

        self.root = root
        self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')
        
        self._load()
    
    def __getitem__(self, index):
        sample = {
            'name': self.names[index],
            'joint_world': self.joints_world[index],
            'ref_pt': self.ref_pts[index]
        }

        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train': ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else: ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
                ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.subject_num):
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
        
            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')

                lines = []
                with open(annot_file) as f:
                    lines = [line.rstrip() for line in f]

                # skip first line
                for i in range(1, len(lines)):
                    # referece point
                    splitted = ref_pt_str[file_id].split()
                    if splitted[0] == 'invalid':
                        print('Warning: found invalid reference frame')
                        file_id += 1
                        continue
                    else:
                        self.ref_pts[frame_id, 0] = float(splitted[0])
                        self.ref_pts[frame_id, 1] = float(splitted[1])
                        self.ref_pts[frame_id, 2] = float(splitted[2])

                    # joint point
                    splitted = lines[i].split()
                    for jid in range(self.joint_num):
                        self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                        self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                        self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])
                    
                    filename = os.path.join(self.root, 'P'+str(mid), fd, '{:0>6d}'.format(i-1) + '_depth.bin')
                    self.names.append(filename)

                    frame_id += 1
                    file_id += 1

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id: self.test_size += num
                else: self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                if not os.path.exists(annot_file):
                    print('Error: annotation file {} does not exist'.format(annot_file))
                    return False

        # Check precomputed centers by v2v-hand model's author
        for subject_id in range(self.subject_num):
            center_train = os.path.join(self.center_dir, 'center_train_' + str(subject_id) + '_refined.txt')
            center_test = os.path.join(self.center_dir, 'center_test_' + str(subject_id) + '_refined.txt')
            if not os.path.exists(center_train) or not os.path.exists(center_test):
                print('Error: precomputed center files do not exist')
                return False

        return True




