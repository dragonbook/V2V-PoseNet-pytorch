import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from lib.accuracy import *
from vis.plot import *


gt_file = r'./test_s3_gt.txt'
pred_file = r'./test_res.txt'


gt = np.loadtxt(gt_file)
gt = gt.reshape(gt.shape[0], -1, 3)

pred = np.loadtxt(pred_file)
pred = pred.reshape(pred.shape[0], -1, 3)

print('gt: ', gt.shape)
print('pred: ', pred.shape)


keypoints_num = 21
names = ['joint'+str(i+1) for i in range(keypoints_num)]


dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=100, num=100)

fig, ax = plt.subplots()
plot_acc(ax, dist, acc, names)
fig.savefig('msra_s3_joint_acc.png')
plt.show()


mean_err = compute_mean_err(pred, gt)
fig, ax = plt.subplots()
plot_mean_err(ax, mean_err, names)
fig.savefig('msra_s3_joint_acc.png')
plt.show()


print('mean_err: {}'.format(mean_err))
mean_err_all = compute_mean_err(pred.reshape((-1, 1, 3)), gt.reshape((-1, 1,3)))
print('mean_err_all: ', mean_err_all)
