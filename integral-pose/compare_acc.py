import sys
import numpy as np
import matplotlib.pyplot as plt
from accuracy import *
from plot import *


gt_file = r'./test_s3_gt.txt'
pred_file = r'./v2vposenet-loss_test_res.txt'  # copied from ./experiments/msra-subject3/test_res.txt
pred_file1 = r'./test_res.txt'  # one-hot heatmap loss + L1 coord loss


gt = np.loadtxt(gt_file)
gt = gt.reshape(gt.shape[0], -1, 3)

pred = np.loadtxt(pred_file)
pred = pred.reshape(pred.shape[0], -1, 3)

pred1 = np.loadtxt(pred_file1)
pred1 = pred1.reshape(pred1.shape[0], -1, 3)

print('gt: ', gt.shape)
print('pred: ', pred.shape)
print('pred1: ', pred1.shape)


names = ['kp'+str(i+1) for i in range(gt.shape[1]) ]


##
dist, acc = compute_dist_acc_wrapper(pred, gt, 100, 100)

fig, ax = plt.subplots()
plot_acc(ax, dist, acc, names)
plt.show()


##
_, acc1 = compute_dist_acc_wrapper(pred1, gt, 100, 100)

fig, ax = plt.subplots()
plot_acc(ax, dist, acc1, names)
plt.show()

##
mean_err = compute_mean_err(pred, gt)
mean_err1 = compute_mean_err(pred1, gt)

fig, ax = plt.subplots()
_pos = np.arange(len(names))
ax.bar(_pos, mean_err, width=0.1, label='gaussian-heatmap-loss(V2VPoseNet-loss)')
ax.bar(_pos+0.1, mean_err1, width=0.1, label='one_hot-heamap-loss + L1 coord loss')
ax.set_xticks(_pos)
ax.set_xticklabels(names)
ax.set_xlabel('keypoints categories')
ax.set_ylabel('distance mean error (mm)')
ax.legend(loc='upper right')

plt.show()


all_mean_err = np.mean(mean_err)
all_mean_err1 = np.mean(mean_err1)
print('all_mean_error: ', all_mean_err)
print('all_mean_error1: ', all_mean_err1)


## histogram
dist_err = compute_dist_err(pred, gt)
dist_err1 = compute_dist_err(pred1, gt)

fig, ax = plt.subplots()
bins = np.linspace(0, 10, 200)
ax.hist([dist_err[:].ravel(), dist_err1[:].ravel()], bins=bins, label=['gaussian-heatmap-loss(V2VPoseNet-loss)', 'one_hot-heamap-loss + L1 coord loss'])
ax.set_xlabel('distance error (mm)')
ax.set_ylabel('keypoints number')
plt.legend(loc='upper right')

plt.show()