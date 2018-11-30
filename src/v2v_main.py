import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from lib.solver import train_epoch, val_epoch
from lib.mesh_util import read_mesh_vertices
from datasets.tooth13_dataset import Tooth13Dataset

from v2v_util import V2VVoxelization
from v2v_model import V2VModel

import numpy as np


# Basic configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float


# Data configuration
print('==> Preparing data ..')
data_dir = r'/home/maiqi/yalong/dataset/cases-tooth-keypoints/D-11-15-aug/v2v-23-same-ori/split1'
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


voxelization_train = V2VVoxelization(augmentation=True)
voxelization_val = voxelization_train

def transform_train(sample):
    vertices, keypoints, refpoint = sample['vertices'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices, keypoints, refpoint = apply_dataset_scale((vertices, keypoints, refpoint))
    input, heatmap = voxelization_train({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})

    return (to_tensor(input), to_tensor(heatmap))


def transform_val(sample):
    vertices, keypoints, refpoint = sample['vertices'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices, keypoints, refpoint = apply_dataset_scale((vertices, keypoints, refpoint))
    input, heatmap = voxelization_val({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})

    return (to_tensor(input), to_tensor(heatmap))


train_set = Tooth13Dataset(root=data_dir, mode='train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=6)

val_set = Tooth13Dataset(root=data_dir, mode='val', transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=6)


# Model, criterion and optimizer
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)


## Train and validate
print('Start train ..')
for epoch in range(2):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
    val_epoch(net, criterion, val_loader, device=device, dtype=dtype)


## Test
def test(model, test_loader, output_transform, device=torch.device('cuda'), dtype=torch.float):
    model.eval()

    samples_num = len(test_loader)
    keypoints = None
    idx = 0

    with torch.no_grad():
        for batch_idx, (inputs, refpoints) in enumerate(test_loader):
            outputs = model(inputs.to(device, dtype))

            outputs = outputs.cpu().numpy()
            refpoints = refpoints.cpu().numpy()

            # (batch, keypoints_num, 3)
            keypoints_batch = output_transform(outputs, refpoints)

            if keypoints is None:
                # Initialize keypoints until dimensions awailable now
                keypoints = np.zeros((samples_num, *keypoints_batch.shape[1:]))

            batch_size = keypoints_batch.shape[0]
            keypoints[idx:idx+batch_size] = keypoints_batch
            idx += batch_size


    return keypoints
   

voxelization_test = voxelization_train

def transform_test(sample):
    vertices, refpoint = sample['vertices'], sample['refpoint']
    vertices, refpoint = apply_dataset_scale((vertices, refpoint))
    input = voxelization_test.voxelize(vertices, refpoint)
    return to_tensor(input), to_tensor(refpoint.reshape((1, -1)))


test_set = Tooth13Dataset(root=data_dir, mode='test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6)

output_transform = voxelization_test.evaluate
         
print('Start test ..')
keypoints_estimate = test(net, test_loader, output_transform, device, dtype)

test_res_filename = r'./test_res.txt'
print('Write result to ', test_res_filename)
# Reshape one sample keypoints in one line
result = keypoints_estimate.reshape(keypoints_estimate.shape[0], -1)
np.savetxt(test_res_filename, result, fmt='%0.4f')

print('All done ..')
