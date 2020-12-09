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
import cv2
    
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

def convert_scale(data, original_size, scale=np.array([13, 13])):
    data = data.reshape(-1, 4)
    p = original_size/scale

    # convert to center
    data[:, 0] += data[:, 2]/2
    data[:, 1] += data[:, 3]/2

    # convert with scales
    data[:, 0] = data[:, 0]/p[0]
    data[:, 1] = data[:, 1]/p[1]
    data[:, 2] = data[:, 2]/p[0]
    data[:, 3] = data[:, 3]/p[1]

    # uncomment if need h and w to be integers
    # data[:, 2] = np.round(data[:, 2], 0)
    # data[:, 3] = np.round(data[:, 3], 0)

    return data

def load_ground_truth(imgs:str):
    # load grouth truth data
    # file list: list of file names:
    # imgs: imgs files

    data = []
    for f in imgs:
        # fix on colab
        # npyf = f[:-4].rsplit('/')[-1] 
        npyf = f[:-4]+".npy"
        size = cv2.imread(f, 0).shape
        cur_data = np.load(npyf)
        cur_data = convert_scale(cur_data, original_size=size)
        data.append(torch.from_numpy(cur_data))

    ground_truth = generate_yolo_output(data, cls=1)

    return ground_truth

def load_data(files:str):
    # load data: N*C*H*W
    # return type: tensor or numpy data
    data =[]
    for f in files:
        cur_data = np.load(f)
        # data.append(torch.from_numpy(cur_data))
        data.append(cur_data)

    data = np.array(data)
    data = np.squeeze(data)

    return data


def generate_yolo_output(x, cls=2, anchor=[[116,90], [156,198], [373,326]], H=13, W=13):
    """
    x: N * list_n
        list_n : m * [bx, by, bw, bh, xconf, cls]
    y : N * (len(anchor) * (5 + cls)) * H * W
    """
    N = len(x)
    C = len(anchor) * (5 + cls)
    x_org = torch.zeros(N, C, H, W)
    for n in range(N):
        for box in x[n]:
            # problematic
            # xx, xy, xw, xh, xconf, xcls = box
            xx, xy, xw, xh = box
            xcls = 0
            xconf = 1.0

##########################################################

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
                x_org[n, st + 5 + xcls, cy, cx] = 1.0 # x_org[n, st + 5 + xcls-1, cy, cx]
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
                    box_list.append(x[n, st:st + 6, cy, cx])
        x_list.append(box_list)
    return x_list

# example of use
# if __name__ == "__main__":
# #     f = ["/Users/zhangyifan/Desktop/grad/2020 Fall/CSCI 5561 Computer Vision/project/1_Handshaking_Handshaking_1_9.jpg"]
# #     d = load_ground_truth(f)
#     f = ["/Users/zhangyifan/Desktop/grad/2020 Fall/CSCI 5561 Computer Vision/project/0_Parade_marchingband_1_5_yoloface.npy", "/Users/zhangyifan/Desktop/grad/2020 Fall/CSCI 5561 Computer Vision/project/0_Parade_marchingband_1_6_yoloface.npy"]
#     print(load_data(f).shape)