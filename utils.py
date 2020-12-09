from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime
import os
import logging
import numpy as np
import math
import torchvision
import pdb
import shutil
from tensorboardX import SummaryWriter
import skimage as sk
import random
    
def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def generate_yolo_output(x, cls=2, anchor=[[116,90], [156,198], [373,326]], H=13, W=13):
    """
    x: N * list_n
        list_n : m * [bx, by, bw, bh, cls]
    y : N * (len(anchor) * (5 + cls)) * H * W
    """
    N = len(x)
    C = len(anchor) * (5 + cls)
    x_org = torch.zeros(N, C, H, W)
    for n in range(N):
        for box in x[n]:
            xx, xy, xw, xh, xconf, xcls = box
            cx = int(torch.floor(xx))
            cy = int(torch.floor(xy))
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                raise ValueError("processor: cx, cy out of bound")
            tx = torch.logit(xx - cx)
            ty = torch.logit(xy - cy)
            for k in range(len(anchor)):
                st = k * (5 + cls)
                # coord-recon
                x_org[n, st, cy, cx] = tx
                x_org[n, st + 1, cy, cx] = ty
                # size-recon
                x_org[n, st + 2, cy, cx] = torch.log(xw / anchor[k][0])
                x_org[n, st + 3, cy, cx] = torch.log(xh / anchor[k][1])
                # conf-recon
                x_org[n, st + 4, cy, cx] = xconf
                x_org[n, st + 5 + xcls, cy, cx] = 1.0
    return x_org


def generate_yolo_box_list(x, cls=2, anchor=[[116,90], [156,198], [373,326]]):
    """
    inverse function of generate_yolo_output()
    """
    N, C, H, W = [*x.shape]
    ch = len(anchor) * (5 + cls)
    if C != ch:
        raise ValueError("expected channel {} != input channel {}".format(ch, C))
    x_list = []
    for n in range(N):
        box_list = []
        for k in range(len(anchor)):
            st = k * (5 + cls)
            # precompute
            x[:, st:st + 2, :, :] = torch.sigmoid(x[:, st:st + 2, :, :])
            x[:, st + 2:st + 4, :, :] = torch.exp(x[:, st + 2:st + 4, :, :])
            # size
            x[:, st + 2, :, :] = x[:, st + 2, :, :] * anchor[k, 0]
            x[:, st + 3, :, :] = x[:, st + 3, :, :] * anchor[k, 1]
            # compute for each cell
            for cy in range(H):
                for cx in range(W):
                    # score
                    en = (k + 1) * (5 + cls)
                    score = x[n, st + 5:en, cy, cx] * x[n, st + 4, cy, cx]
                    # coordinate
                    x[n, st, cy, cx] += cx
                    x[n, st + 1, cy, cx] += cy
                    # class decision
                    x[n, st + 4, cy, cx] = torch.max(score)
                    x[n, st + 5, cy, cx] = torch.argmax(score)
                    box_list.append(x[n, st:st + 6])
        x_list.append(box_list)
    return x_list