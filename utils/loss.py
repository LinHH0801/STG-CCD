
import torch
import torch.nn.functional as F

import torch.nn as nn

# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input1, target1):
        if input1.dim()>2:
            input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)
            input1 = input1.transpose(1,2)
            input1 = input1.contiguous().view(-1, input1.size(2)).squeeze()
        if target1.dim()==4:
            target1 = target1.contiguous().view(target1.size(0), target1.size(1), -1)
            target1 = target1.transpose(1,2)
            target1 = target1.contiguous().view(-1, target1.size(2)).squeeze()
        elif target1.dim()==3:
            target1 = target1.view(-1)
        else:
            target1 = target1.view(-1, 1)

        logpt = -F.cross_entropy(input1, target1, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma)*logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()








