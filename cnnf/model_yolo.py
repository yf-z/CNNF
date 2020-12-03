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

class YoloHead(nn.Module):
    """
    Yolo Head part (input features, output regression results)
    """
    def __init__(self, out, step='forward', first=True, inter=False, inter_recon=False):
        if 'forward' in step:
