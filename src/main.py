import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os

from datasets.tooth13_dataset import Tooth13Dataset
from mesh_util import read_mesh_vertices
from v2v import InputOutputVoxelization, InputVoxelization

from model import V2VModel
from util import progress_bar


# Get outer configurations
parser = argparse.ArgumentParser(description='V2V Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
num_workers = 6
lr = 2.5e-4
start_epoch = 0


# Data
print('==> Preparing data ..')
data_dir = r''
keypoints_num = 7

cubic_size, cropped_size, original_size = 140, 88, 96
pool_factor = 2
std = 1.7

data_sizes = (cubic_size, cropped_size, original_size)


# Transformation
voxelization_train = InputOutputVoxelization(data_sizes, pool_factor, std)
voxelization_test = InputVoxelization(data_sizes)

def to_tensor(x):
    return torch.from_numpy(x).to('cpu', torch.float)


def transform_train(sample):
    mesh_name, keypoints, refpoint = sample['mesh_name'], sample['keypoints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    vertices = read_mesh_vertices(mesh_name)
    input, heatmap = voxelization_train({'points': vertices, 'keypoints': keypoints, 'refpoint': refpoint})
    return (to_tensor(input), to_tensor(heatmap))


def transform_test(sample):
    mesh_name, refpoint = sample['mesh_name'], sample['refpoint']
    vertices = read_mesh_vertices(mesh_name)
    input = voxelization_test({'points': vertices, 'refpoint': refpoint})
    return to_tensor(input)
 

trainset = Tooth13Dataset(root=data_dir, mode='train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = Tooth13Dataset(root=data_dir, mode='test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Model
input_channels, output_channel = 1, keypoints_num 
net = V2VModel(input_channels, output_channel)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# TODO: Resume?


# 
criterion = nn.MSELoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr)


# Training
def train(epoch):
    print('\nEpoch {}'.format(epoch))
    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: {0:.3f}'.format(train_loss/(batch_idx+1)))


# Testing
def test(epoch):
    net.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            progress_bar(batch_idx, len(testloader), 'Loss: {0:.3f}'.format(test_loss/(batch_idx+1)))

    # TODO: save checkpoint


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)



### TODO ###
# (1) add augmentation to test samples for monitoring loss
# (2) separate training and testing logic into a solver
