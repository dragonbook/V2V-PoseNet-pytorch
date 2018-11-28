import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from solver import train_epoch, test_epoch

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

cubic_size, cropped_size, original_size = 140, 88, 96
data_sizes = (cubic_size, cropped_size, original_size)


# Transformation
voxelization_train = V2VVoxelization(data_sizes, pool_factor=2, std=1.7, augmentation=True)
voxelization_test = V2VVoxelization(data_sizes, pool_factor=2, std=1.7, augmentation=False)

def to_tensor(x):
    return torch.from_numpy(x).to('cpu', torch.float)

def transform_train(sample):
    mesh_name, keypoints, refpoint = sample['mesh_name'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices = read_mesh_vertices(mesh_name)
    input, heatmap = voxelization_train({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})
    return (to_tensor(input), to_tensor(heatmap))

def transform_test(sample):
    mesh_name, keypoints, refpoint = sample['mesh_name'], sample['keypoints'], sample['refpoint']
    vertices = read_mesh_vertices(mesh_name)
    input, heatmap = voxelization_test({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})
    return (to_tensor(input), to_tensor(heatmap))
 

trainset = Tooth13Dataset(root=data_dir, mode='train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=6)

testset = Tooth13Dataset(root=data_dir, mode='test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=6)


# Model, criterion and optimizer
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)


for epoch in range(200):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, trainloader, device=device, dtype=dtype)
    test_epoch(net, criterion, testloader, device=device, dtype=dtype)

