import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import sampler

from lib.solver import train_epoch, val_epoch
from lib.mesh_util import read_mesh_vertices
from datasets.tooth13_dataset import Tooth13Dataset

from v2v_util import V2VVoxelization
from v2v_model import V2VModel

import numpy as np


print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')


#torch.random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


# Basic configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
#dtype = torch.double


# Data configuration
print('==> Preparing data ..')
data_dir = r'/home/yalong/yalong/project/KeyPointsEstimation/V2V-PoseNet-pytorch/experiments/tooth/exp1/split1/'
dataset_scale = 10
keypoints_num = 7


# Transformation
def apply_dataset_scale(x):
    if isinstance(x, tuple):
        for e in x: e *= dataset_scale
    else: x *= dataset_scale

    return x


def to_tensor(x):
    return torch.from_numpy(x)


voxelization_train = V2VVoxelization(augmentation=False)
voxelization_val = V2VVoxelization(augmentation=False)


class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def transform_train(sample):
    vertices, keypoints, refpoint = sample['vertices'].copy(), sample['keypoints'].copy(), sample['refpoint'].copy()
    assert(keypoints.shape[0] == keypoints_num)

    vertices, keypoints, refpoint = apply_dataset_scale((vertices, keypoints, refpoint))
    input, heatmap = voxelization_train({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})

    return (to_tensor(input), to_tensor(heatmap))


def transform_val(sample):
    #vertices, keypoints, refpoint = sample['vertices'], sample['keypoints'], sample['refpoint']
    vertices, keypoints, refpoint = sample['vertices'].copy(), sample['keypoints'].copy(), sample['refpoint'].copy()
    assert(keypoints.shape[0] == keypoints_num)

    vertices, keypoints, refpoint = apply_dataset_scale((vertices, keypoints, refpoint))
    input, heatmap = voxelization_val({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})

    return (to_tensor(input), to_tensor(heatmap))


# Datasets
train_set = Tooth13Dataset(root=data_dir, mode='train', transform=transform_train)
train_num = 1
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=6,sampler=ChunkSampler(train_num, 0))
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=6)

val_set = Tooth13Dataset(root=data_dir, mode='val', transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)


# Model, criterion and optimizer
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device, dtype)
if device == 'cuda':
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    print('cudnn.backends: ', torch.backends.cudnn.enabled)


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, outputs, targets):
        # Assume batch = 1
        return ((outputs - targets)**2).mean()


criterion = nn.MSELoss()
#criterion = Criterion()
#optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


## Train and validate
print('Start train ..')
for epoch in range(200):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
    #val_epoch(net, criterion, val_loader, device=device, dtype=dtype)


# Test
# def test(model, test_loader, output_transform, device=torch.device('cuda'), dtype=torch.float):
#     model.eval()

#     samples_num = len(test_loader)
#     keypoints = None
#     idx = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, refpoints) in enumerate(test_loader):
#             outputs = model(inputs.to(device, dtype))

#             outputs = outputs.cpu().numpy()
#             refpoints = refpoints.cpu().numpy()

#             # (batch, keypoints_num, 3)
#             keypoints_batch = output_transform((outputs, refpoints))

#             if keypoints is None:
#                 # Initialize keypoints until dimensions awailable now
#                 keypoints = np.zeros((samples_num, *keypoints_batch.shape[1:]))

#             batch_size = keypoints_batch.shape[0]
#             keypoints[idx:idx+batch_size] = keypoints_batch
#             idx += batch_size


#     return keypoints
   

# def remove_dataset_scale(x):
#     if isinstance(x, tuple):
#         for e in x: e /= dataset_scale
#     else: x /= dataset_scale

#     return x


# voxelization_test = voxelization_train

# def output_transform(x):
#     heatmaps, refpoints = x
#     keypoints = voxelization_test.evaluate(heatmaps, refpoints)
#     return remove_dataset_scale(keypoints)


# def transform_test(sample):
#     vertices, refpoint = sample['vertices'], sample['refpoint']
#     vertices, refpoint = apply_dataset_scale((vertices, refpoint))
#     input = voxelization_test.voxelize(vertices, refpoint)
#     return to_tensor(input), to_tensor(refpoint.reshape((1, -1)))


# test_set = Tooth13Dataset(root=data_dir, mode='test', transform=transform_test)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6)

# print('Start test ..')
# keypoints_estimate = test(net, test_loader, output_transform, device, dtype)

# test_res_filename = r'./test_res.txt'
# print('Write result to ', test_res_filename)
# # Reshape one sample keypoints in one line
# result = keypoints_estimate.reshape(keypoints_estimate.shape[0], -1)
# np.savetxt(test_res_filename, result, fmt='%0.4f')


# print('Start save fit ..')
# fit_set = Tooth13Dataset(root=data_dir, mode='train', transform=transform_test)
# fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=1, shuffle=False, num_workers=6)
# keypoints_fit = test(net, fit_loader, output_transform, device=device, dtype=dtype)
# fit_res_filename = r'./fit_res.txt'
# print('Write fit result to ', fit_res_filename)
# fit_result = keypoints_fit.reshape(keypoints_fit.shape[0], -1)
# np.savetxt(fit_res_filename, fit_result, fmt='%0.4f')

# print('All done ..')
