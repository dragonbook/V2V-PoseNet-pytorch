import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCrossEntropyWithLogits(nn.Module):
    '''
    Similar to tensorflow's tf.nn.softmax_cross_entropy_with_logits
    ref: https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f

    The 'input' is unnormalized scores.
    The 'target' is a probability distribution.

    Shape:
        Input: (N, C), batch size N, with C classes
        Target: (N, C), batch size N, with C classes
    '''
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()

    def forward(self, input, target):
        loss = torch.sum(-target * F.log_softmax(input, -1), -1)
        mean_loss = torch.mean(loss)
        return mean_loss


class MixedLoss(nn.Module):
    '''
    ref: https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/model.py

    input: {
        'heatmap': (N, C, X, Y, Z), unnormalized
        'coord': (N, C, 3)
    }

    target: {
        'heatmap': (N, C, X, Y, Z), normalized
        'coord': (N, C, 3)
    }

    '''
    def __init__(self, heatmap_weight=0.5):
    # def __init__(self, heatmap_weight=0.05):
        super(MixedLoss, self).__init__()
        self.w1 = heatmap_weight
        self.w2 = 1 - self.w1
        self.cross_entropy_loss = SoftmaxCrossEntropyWithLogits()

    def forward(self, input, target):
        pred_heatmap, pred_coord = input['heatmap'], input['coord']
        gt_heatmap, gt_coord = target['heatmap'], target['coord']

        # Heatmap loss
        N, C = pred_heatmap.shape[0:2]
        pred_heatmap = pred_heatmap.view(N*C, -1)
        gt_heatmap = gt_heatmap.view(N*C, -1)

        # Note, averaged over N*C
        hm_loss = self.cross_entropy_loss(pred_heatmap, gt_heatmap)

        # Coord L1 loss
        l1_loss = torch.mean(torch.abs(pred_coord - gt_coord))

        return self.w1 * hm_loss + self.w2 * l1_loss
