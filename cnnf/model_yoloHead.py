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

    def __init__(self, outch=18, ind=0, cycles=1, res_param=0.1):
        super(YoloPart, self).__init__()

        self.ind = ind
        self.res_param = res_param
        self.cycles = cycles

        # 1st block: input = 1024 * 13 * 13
        self.conv1_1 = layers.ConvCom(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = layers.ConvCom(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = layers.ConvCom(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv1_4 = layers.ConvCom(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = layers.ConvCom(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv1_6 = layers.ConvCom(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv1_7 = layers.ConvCom(1024, outch, kernel_size=1, stride=1, padding=0, bn=False, act=False)

        # 2nd block: input = 512 * 13 * 13 from 1_5
        self.conv2_1 = layers.ConvCom(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample2 = layers.UpSampling()
        self.concat2 = layers.Concat([256, 26, 26])
        self.conv2_2 = layers.ConvCom(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = layers.ConvCom(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_4 = layers.ConvCom(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2_5 = layers.ConvCom(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_6 = layers.ConvCom(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2_7 = layers.ConvCom(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_8 = layers.ConvCom(512, outch, kernel_size=1, stride=1, padding=0, bn=False, act=False)

        # 3rd block: input = 26 * 26 * 256 from 2_6
        self.conv3_1 = layers.ConvCom(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample3 = layers.UpSampling()
        self.concat3 = layers.Concat([128, 52, 52])
        self.conv3_2 = layers.ConvCom(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = layers.ConvCom(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = layers.ConvCom(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_5 = layers.ConvCom(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_6 = layers.ConvCom(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_7 = layers.ConvCom(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_8 = layers.ConvCom(256, outch, kernel_size=1, stride=1, padding=0, bn=False, act=False)


    def forward(self, out, concat, step='forward', first=True, inter=False, inter_recon=False):
        if ('forward' in step):
            # total_out = []
            block1 = None
            block2 = None

            # block 1
            if (self.ind == 0) or (first == True):
                if (self.ind==0) and (first == True):
                    orig_feature = out
                out = self.conv1_1(out)
                out = self.conv1_2(out)
                out = self.conv1_3(out)
                out = self.conv1_4(out)
                out = self.conv1_5(out)
                block1 = self.conv1_7(self.conv1_6(out))
                # total_out.append(self.conv1_7(self.conv1_6(out)))
            # block 2
            if (self.ind <= 1) or (first == True):
                if (self.ind==1) and (first == True):
                    orig_feature = out
                out = self.conv2_1(out)
                out = self.upsample2(out)
                out = self.concat2(out, concat[0])
                out = self.conv2_2(out)
                out = self.conv2_3(out)
                out = self.conv2_4(out)
                out = self.conv2_5(out)
                out = self.conv2_6(out)
                block2 = self.conv2_8(self.conv2_7(out))
                # total_out.append(self.conv2_8(self.conv2_7(out)))
                # if (self.ind == 1) and (first == True):
                #     orig_feature = total_out[-1]
            out = self.conv3_1(out)
            out = self.upsample3(out)
            out = self.concat3(out, concat[1])
            out = self.conv3_2(out)
            out = self.conv3_3(out)
            out = self.conv3_4(out)
            out = self.conv3_5(out)
            out = self.conv3_6(out)
            out = self.conv3_8(self.conv3_7(out))
            # total_out.append(self.conv3_8(self.conv3_7(out)))
            # out = total_out
        elif ('backward' in step):
            block1_recon = None
            block2_recon = None

            out = self.conv3_8(out, step='backward')
            out = self.conv3_7(out, step='backward')
            out = self.conv3_6(out, step='backward')
            out = self.conv3_5(out, step='backward')
            out = self.conv3_4(out, step='backward')
            out = self.conv3_3(out, step='backward')
            out = self.conv3_2(out, step='backward')
            out = self.concat3(out, step='backward')
            out = self.upsample3(out, step='backward')
            out = self.conv3_1(out, step='backward')
            if (self.ind <= 1):
                block2_recon = self.conv2_8(self.conv2_7(out))
                out = self.conv2_6(out, step='backward')
                out = self.conv2_5(out, step='backward')
                out = self.conv2_4(out, step='backward')
                out = self.conv2_3(out, step='backward')
                out = self.conv2_2(out, step='backward')
                out = self.concat2(out, step='backward')
                out = self.upsample2(out, step='backward')
                out = self.conv2_1(out, step='backward')
            if (self.ind == 0):
                block1_recon = self.conv1_7(self.conv1_6(out))
                out = self.ins1(out)
                out = self.conv1_5(out, step='backward')
                out = self.conv1_4(out, step='backward')
                out = self.conv1_3(out, step='backward')
                out = self.conv1_2(out, step='backward')
                out = self.conv1_1(out, step='backward')

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
        self.conv1_1.reset()
        self.conv1_2.reset()
        self.conv1_3.reset()
        self.conv1_4.reset()
        self.conv1_5.reset()

        # 2nd block: input = 512 * 13 * 13 from 1_5
        self.conv2_1.reset()
        self.conv2_2.reset()
        self.conv2_3.reset()
        self.conv2_4.reset()
        self.conv2_5.reset()
        self.conv2_6.reset()

        # 3rd block: input = 26 * 26 * 256 from 2_6
        self.conv3_1.reset()
        self.conv3_2.reset()
        self.conv3_3.reset()
        self.conv3_4.reset()
        self.conv3_5.reset()
        self.conv3_6.reset()

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
