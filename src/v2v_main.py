import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from solver import train_epoch, val_epoch

from datasets.tooth13_dataset import Tooth13Dataset
from mesh_util import read_mesh_vertices
from v2v_util import V2VVoxelization
from v2v_model import V2VModel


# Basic configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float


# Data configuration
print('==> Preparing data ..')
data_dir = r''
keypoints_num = 7


# Transformation
def to_tensor(x):
    return torch.from_numpy(x)


def transform_train(sample):
    mesh_name, keypoints, refpoint = sample['mesh_name'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices = read_mesh_vertices(mesh_name)

    voxelization_train = V2VVoxelization()
    input, heatmap = voxelization_train({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})

    return (to_tensor(input), to_tensor(heatmap))


def transform_val(sample):
    mesh_name, keypoints, refpoint = sample['mesh_name'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices = read_mesh_vertices(mesh_name)

    voxelization_val = V2VVoxelization()
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


## Train and test
print('Start train ..')
for epoch in range(200):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
    val_epoch(net, criterion, val_loader, device=device, dtype=dtype)
