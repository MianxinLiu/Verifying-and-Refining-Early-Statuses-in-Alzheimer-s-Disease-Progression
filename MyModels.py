import torch.nn.functional as F
import torch
import torch.nn as nn
import scipy.io as scio

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from Modules import FilterLinear
import math
import numpy as np
import GNET
import torch
from collections import OrderedDict

gpu = 1

class GCN_base(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_base, self).__init__()

        self.gcn = GNET.GCN(ROInum, 1, nn.ReLU(),0.3)

        self.bn1 = torch.nn.BatchNorm1d(ROInum)
        self.fl1 = nn.Linear(ROInum,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, ROInum)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out
