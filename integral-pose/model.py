# from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from v2v_model import V2VModel
import numpy as np


class VolumetricSoftmax(nn.Module):
    '''
    TODO: soft-argmax: norm coord to [-1, 1], instead of [0, N]

    ref: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    '''

    def __init__(self, channel, sizes):
        super(VolumetricSoftmax, self).__init__()
        self.channel = channel
        self.xsize, self.ysize, self.zsize = sizes[0], sizes[1], sizes[2]
        self.volume_size = self.xsize * self.ysize * self.zsize

        # TODO: optimize, compute x, y, z together
        # pos = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), np.arange(self.zsize), indexing='ij')
        # pos = np.array(pos).reshape((3, -1))
        # pos = torch.from_numpy(pos)
        # self.register_buffer('pos', pos)

        pos_x, pos_y, pos_z = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), np.arange(self.zsize), indexing='ij')

        pos_x = torch.from_numpy(pos_x.reshape((-1))).float()
        pos_y = torch.from_numpy(pos_y.reshape((-1))).float()
        pos_z = torch.from_numpy(pos_z.reshape((-1))).float()

        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, x):
        # x: (N, C, X, Y, Z)
        x = x.view(-1, self.volume_size)
        p = F.softmax(x, dim=1)

        #print('self.pos_x: {}, device: {}, dtype: {}'.format(type(self.pos_x), self.pos_x.device, self.pos_x.dtype))
        #print('p: {}, device: {}, dtype: {}'.format(type(p), p.device, p.dtype))

        expected_x = torch.sum(self.pos_x * p, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * p, dim=1, keepdim=True)
        expected_z = torch.sum(self.pos_z * p, dim=1, keepdim=True)

        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        out = expected_xyz.view(-1, self.channel, 3)

        return out



class Model(nn.Module):
    def __init__(self, in_channels, out_channels, output_res=44):
        super(Model, self).__init__()
        self.output_res = output_res
        self.basic_model = V2VModel(in_channels, out_channels)
        self.spatial_softmax = VolumetricSoftmax(out_channels, (self.output_res, self.output_res, self.output_res))

    def forward(self, x):
        heatmap = self.basic_model(x)
        coord = self.spatial_softmax(heatmap)

        #print('model heatmap: {}'.format(heatmap.dtype))

        output = {
            'heatmap': heatmap,
            'coord': coord
        }

        return output
