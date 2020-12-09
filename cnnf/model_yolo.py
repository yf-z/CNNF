import torch
import torch.nn as nn
import torch.nn.functional as F
import cnnf.layers as layers
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil
from tensorboardX import SummaryWriter


class YoloPart(nn.Module):
    """ part of yolo head """

    def __init__(self, cls=2, anchor=[[116,90], [156,198], [373,326]], ind=0, cycles=1, res_param=0.1):
        super(YoloPart, self).__init__()

        self.ind = ind
        self.res_param = res_param
        self.cycles = cycles
        self.cls = cls
        self.anchor = anchor
        self.channel = len(self.anchor) * (5 + self.cls)

        # 1st block: input = 1024 * 13 * 13
        self.conv1 = layers.ConvCom(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = layers.ConvCom(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv3 = layers.ConvCom(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = layers.ConvCom(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = layers.ConvCom(1024, self.channel, kernel_size=1, stride=1, padding=0, bn=False, act=False)

    def forward(self, out, step='forward', first=True, inter=False, inter_recon=False):
        if ('forward' in step):
            if (self.ind == 0) or (first == True):
                if (self.ind==0) and (first == True):
                    orig_feature = out
                out = self.conv1(out)
            if (self.ind <= 1) or (first == True):
                if (self.ind==1) and (first == True):
                    orig_feature = out
                out = self.conv2(out)
            if (self.ind == 2) and (first == True):
                orig_feature = out
            out = self.conv3(out)
            block1 = out
            out = self.conv4(out)
            block2 = out
            out = self.conv5(out)
        elif ('backward' in step):
            out = self.conv5(out, step='backward')
            block2_recon = out
            out = self.conv4(out, step='backward')
            block1_recon = out
            out = self.conv3(out, step='backward')
            if (self.ind <= 1):
                out = self.conv2(out, step='backward')
            if (self.ind == 0):
                out = self.conv1(out, step='backward')

        if (inter == True) and ('forward' in step):
            return out, orig_feature, block1, block2
        elif (inter_recon == True) and ('backward' in step):
            return out, block1_recon, block2_recon
        else:
            return out

    def reset(self):
        """
        Resets the pooling and activation states
        """
        self.conv1.reset()
        self.conv2.reset()
        self.conv3.reset()
        self.conv4.reset()
        self.conv5.reset()

    def run_cycles(self, data):  # without gradient track
        # evaluate with all the iterations
        # in each iteration: reconstruct input and recompute input to next forward
        # for each iteration: input moves forward 1 layer
        with torch.no_grad():
            data = data.cuda()
            self.reset()
            output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
            ff_prev = orig_feature
            for i_cycle in range(self.cycles):
                reconstruct = self.forward(output, step='backward')
                ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
                output = self.forward(ff_current, first=False)
                ff_prev = ff_current

        return output

    def run_cycles_adv(self, data):  # with gradient track
        data = data.cuda()
        self.reset()
        output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
        ff_prev = orig_feature
        for i_cycle in range(self.cycles):
            reconstruct = self.forward(output, step='backward')
            ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
            output = self.forward(ff_current, first=False)
            ff_prev = ff_current
        return output

    def run_average(self, data):
        # return averaged logits  (average for different iterations)
        data = data.cuda()
        self.reset()
        output_list = []
        output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
        ff_prev = orig_feature
        output_list.append(output)
        totaloutput = torch.zeros(output.shape).cuda()
        for i_cycle in range(self.cycles):
            reconstruct = self.forward(output, step='backward')
            ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
            ff_prev = ff_current
            output = self.forward(ff_current, first=False)
            output_list.append(output)
        for i in range(len(output_list)):
            totaloutput += output_list[i]
        return totaloutput / (self.cycles + 1)

    def forward_adv(self, data):
        # run the first forward pass
        data = data.cuda()
        self.reset()
        output = self.forward(data)
        return output
