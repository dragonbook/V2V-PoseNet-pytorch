import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os

from solver import train_epoch, val_epoch, test_epoch
from sampler import ChunkSampler
from model import Model
from loss import MixedLoss
from v2v_util import V2VVoxelization
from msra_hand import MARAHandDataset


#######################################################################################
# Note,
# Run in project root direcotry(ROOT_DIR) with:
# PYTHONPATH=./ python experiments/msra-subject3/main.py
# 
# This script will train model on MSRA hand datasets, save checkpoints to ROOT_DIR/checkpoint,
# and save test results(test_res.txt) and fit results(fit_res.txt) to ROOT_DIR.
#


#######################################################################################
## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    #parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args


#######################################################################################
## Configurations
print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')

# When we need to resume training, enable randomness to avoid seeing the determinstic
# (agumented) samples many times.
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

#
args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

save_checkpoint = True
checkpoint_per_epochs = 1
checkpoint_dir = r'./checkpoint'

start_epoch = 0
epochs_num = 15

batch_size = 12


#######################################################################################
## Data, transform, dataset and loader
# Data
print('==> Preparing data ..')
data_dir = r'/home/maiqi/yalong/dataset/cvpr15_MSRAHandGestureDB'
center_dir = r'/home/maiqi/yalong/project/KeyPoint/Code/V2V-PoseNet-Rlease-Codes/V2V-PoseNet_RELEASE-hand/data-result/MSRA-result/center'
keypoints_num = 21
test_subject_id = 3
cubic_size = 200


# Transform
voxelization_train = V2VVoxelization(cubic_size=200, augmentation=True)
voxelization_val = V2VVoxelization(cubic_size=200, augmentation=False)


def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    input, heatmap, coord = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})

    sample = {
        'input': input,
        'target': {
            'heatmap': heatmap,
            'coord': coord
        },
        'extra': {
            'refpoint': refpoint.reshape((1, -1))
        }
    }

    return sample



def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    input, heatmap, coord = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})

    sample = {
        'input': input,
        'target': {
            'heatmap': heatmap,
            'coord': coord
        },
        'extra': {
            'refpoint': refpoint.reshape((1, -1))
        }
    }

    return sample


# Dataset and loader
train_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
# train_num = 24
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=6,sampler=ChunkSampler(train_num, 0))

# No separate validation dataset, just use test dataset instead
val_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)
# val_num = 24
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6, sampler=ChunkSampler(val_num))


#######################################################################################
## Model, criterion and optimizer
print('==> Constructing model ..')
output_res = 44
net = Model(in_channels=1, out_channels=keypoints_num, output_res=output_res)

net = net.to(device, dtype)
if device == torch.device('cuda'):
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    print('cudnn.enabled: ', torch.backends.cudnn.enabled)


criterion = MixedLoss()
optimizer = optim.Adam(net.parameters())


#######################################################################################
## Resume
if resume_train:
    # Load checkpoint
    epoch = resume_after_epoch
    checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')

    print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


#######################################################################################
## Train and Validate
print('==> Training ..')
for epoch in range(start_epoch, start_epoch + epochs_num):
    print('Epoch: {}'.format(epoch))
    train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
    val_epoch(net, criterion, val_loader, device=device, dtype=dtype)

    if save_checkpoint and epoch % checkpoint_per_epochs == 0:
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)


#######################################################################################
## Test
print('==> Testing ..')

def transform_test(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)

    input, heatmap, coord = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})

    sample = {
        'input': input,
        'target': {
            'heatmap': heatmap,
            'coord': coord
        },
        'extra': {
            'refpoint': refpoint.reshape((1, -1))
        }
    }

    return sample


def transform_coord(coords, refpoints):
    keypoints = voxelization_val.warp2continuous_raw(coords, refpoints)
    return keypoints

transform_output = transform_coord


class BatchResultCollector():
    def __init__(self, samples_num, transform_output):
        self.samples_num = samples_num
        self.transform_output = transform_output
        self.keypoints = None
        self.idx = 0
    
    def __call__(self, data_batch):
        outputs_batch = data_batch['output']['coord']
        refpoints_batch = data_batch['extra']['refpoint']

        outputs_batch = outputs_batch.cpu().numpy()
        refpoints_batch = refpoints_batch.numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.keypoints is None:
            # Initialize keypoints until dimensions awailable now
            self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0] 
        self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
        self.idx += batch_size

    def get_result(self):
        return self.keypoints


print('Test on test dataset ..')
def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')


test_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)
test_res_collector = BatchResultCollector(len(test_set), transform_output)

test_epoch(net, test_loader, test_res_collector, device, dtype)
keypoints_test = test_res_collector.get_result()
save_keypoints('./test_res.txt', keypoints_test)


print('Fit on train dataset ..')
fit_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_test)
fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=batch_size, shuffle=False, num_workers=6)
fit_res_collector = BatchResultCollector(len(fit_set), transform_output)

test_epoch(net, fit_loader, fit_res_collector, device, dtype)
keypoints_fit = fit_res_collector.get_result()
save_keypoints('./fit_res.txt', keypoints_fit)

print('All done ..')
