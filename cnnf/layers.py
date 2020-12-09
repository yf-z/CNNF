import torch
from torch import nn
import math
import numpy as np

class InsNorm(nn.Module):
    """
    Normalize the the input by the norm of each channel. 
    """
    def forward(self, x, std=False, returnnorm=False):
        if (std==True):
            normx = torch.sqrt(torch.var(x, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        else:
            normx = torch.sqrt(torch.mean(x**2, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        if returnnorm==True:
            return x/normx, normx
        else:
            return x/normx

class Flatten(nn.Module):
    """
    Flattens the input into vector.
    """
    def forward(self, x, step='forward'):
        if 'forward' in step:
            self.size = x.size()
            batch_size = x.size(0)
            return x.view(batch_size, -1)

        elif 'backward' in step:
            batch_size = x.size(0)
            return x.view(batch_size, *self.size[1:])

        else:
            raise ValueError("step must be 'forward' or 'backward'")

class Conv2d(nn.Module):
    """
    Applies a 2D convolution over the input. In the feedback step,
    a transposed 2D convolution is applied to the input with the same weights
    as the 2D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=bias, **kwargs)

        if self.conv.stride[0] > 1:
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                         kernel_size, bias=bias, output_padding=1, **kwargs)
        else:
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                         kernel_size, bias=bias, **kwargs)

        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.out_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization

        # Tie the weights between the convolution and the transposed convolution
        self.conv_t.weight = self.conv.weight

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.conv(x)

        elif 'backward' in step:
            return self.conv_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class Linear(nn.Module):
    """
    Applies a linear transform over the input. In the feedback step,
    a transposed linear transform is applied to the input with the transposed
    weights of the linear transform.
    """
    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_t = nn.Linear(out_features, in_features, bias=bias)
        
        if(bias==True):
            self.linear.bias.data.zero_()
        # Tie the weight of the transposed linear transform with the
        # transposed weights of the linear transform
        self.linear_t.weight.data = self.linear.weight.data.t()

        # Make the weight publicly accessible
        self.weight = self.linear.weight

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.linear(x)

        elif 'backward' in step:
            # Tie the weight of the transposed linear transform with the
            # transposed weights of the linear transform
            self.linear_t.weight.data = self.linear.weight.data.t()

            return self.linear_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class ReLU(nn.Module):
    """
    AdaReLU: Applies AdaReLU activation over the input. In the feedback step,
    the input is pointwise multiplied through the units that were activated
    in the forward step---which are save during the forward step.
    """
    def __init__(self):
        super().__init__()
        self.state = None
        self.hidden = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated
            if self.hidden is None:
                self.state = (x > 0)                
            else:
                self.state = (x * self.hidden) > 0
            result = x * self.state.float()
                
            return result

        elif 'backward' in step:
            # Units that were activated in the forward step are passed through
            self.hidden = x
            masked_hidden = x * self.state.float()
            if unit_space is not None:
                return masked_hidden, unit_space
            else:
                return masked_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.state = None


class LeakyReLU(nn.Module):
    """
    LeakyReLU
    """

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.mask = None
        self.negative_slope = negative_slope

    def forward(self, x, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated
            self.mask = (x > 0).float() + (x < 0).float() * self.negative_slope
            result = x * self.mask

            return result

        elif 'backward' in step:
            recon_hidden = x / self.mask
            if unit_space is not None:
                return recon_hidden, unit_space
            else:
                return recon_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.mask = None


class resReLU(nn.Module):
    """
    AdaReLU with residual updates.
    """
    def __init__(self, res_param = 0.1):
        super().__init__()
        self.state = None
        self.hidden = None
        self.res_param = res_param

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated
            if self.hidden is None:
                self.state = (x > 0)                
            else:
                x = x + self.res_param * (self.hidden - x)
                self.state = (x * self.hidden) > 0
            result = x * self.state.float()
                
            return result

        elif 'backward' in step:
            # Units that were activated in the forward step are passed through
            self.hidden = x
            masked_hidden = x * self.state.float()
            if unit_space is not None:
                return masked_hidden, unit_space
            else:
                return masked_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.state = None


class AvgPool2d(nn.Module):
    
    def __init__(self, kernel_size, scale_factor=10, **kwargs):
        super().__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size, **kwargs)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor, **kwargs)  # feedforward, before avgpool, size is 10

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.avgpool(x)

        elif 'backward' in step:
            return self.upsample(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class UpSampling(nn.Module):

    def __init__(self, kernel_size=2, scale_factor=2, **kwargs):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, **kwargs)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor, **kwargs)

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.upsample(x)

        elif 'backward' in step:
            return self.avgpool(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")
        
        
class MaxPool2d(nn.Module):
    """
    AdaPool:
    In the feedforward pass, if the pixel in g that governs this grid is > 0, do maxpool on this grid
    In the feedforward pass, if the pixel in g that governs this grid is < 0, do min pool on this grid
    In the feedback pass, if the pixel comes from a max value, put it back to the position of the max value
    In the feedback pass, if the pixel comes from a min value, put it back to the position of the min value
    """
    def __init__(self, kernel_size, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, return_indices=True, **kwargs)
        self.unpool = nn.MaxUnpool2d(kernel_size, **kwargs)
        
        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None
        self.unpoolsize = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            if self.hidden is None:
                self.unpoolsize = x.shape
                max_pool, self.pos_ind = self.maxpool(x)   # self.state is the indices of maxpool
                return max_pool
                
            else:
                max_pool, self.pos_ind = self.maxpool(x)
                min_pool, self.neg_ind = self.maxpool(-x)
                min_pool = -min_pool
                self.pos_state = (self.hidden > 0).float()
                self.neg_state = (self.hidden < 0).float()
                out = self.pos_state * max_pool + self.neg_state * min_pool
                return out

        elif 'backward' in step:
            self.hidden = x
            if ((self.pos_state is None) or (self.neg_state is None)):  # reconstruction in the first iteration, using maxpool states
                max_unpool = self.unpool(x, self.pos_ind, output_size=self.unpoolsize)
                return max_unpool
            else:            
                max_unpool = self.unpool(x * self.pos_state, self.pos_ind, output_size=self.unpoolsize)
                min_unpool = self.unpool(x * self.neg_state, self.neg_ind, output_size=self.unpoolsize)
                return max_unpool + min_unpool

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None


class Bias(nn.Module):
    """
    Add a bias to the input. In the feedback step, the bias is subtracted
    from the input.
    """
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(*size))
        self.bias.data.zero_()

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return x + self.bias

        # no bias in backwards pass
        elif 'backward' in step:
            self.x = x
            return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def path_norm_loss(self, unit_space):
        return torch.mean((self.x * self.bias - unit_space * self.bias)**2)


class Dropout(nn.Module):
    """
    Performs dropout regularization to the input. In the feedback step, the 
    same dropout transformation is applied in the backwards step.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, step='forward'):
        if 'forward' in step:
            if self.training:
                self.dropout = (torch.rand_like(x) > self.p).float() / self.p
                return x * self.dropout
            else:
                return x

        elif 'backward' in step:
            if self.training:
                return x * self.dropout
            else:
                return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class ConvCom(nn.Module):
    """
    a compound conv layer including: conv + bn + activation
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, bn=True, act=True, **kwargs):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        if bias:
            self.bias = Bias((in_channels, out_channels, kernel_size ** 2))
        else:
            self.bias = None
        if bn:
            self.bn = InsNorm()
        else:
            self.bn = None
        if act:
            self.act = LeakyReLU()
        else:
            self.act = None


    def forward(self, x, step='forward'):
        if 'forward' in step:
            x = self.conv(x)
            if self.bias is not None:
                x = self.bias(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
            return x

        elif 'backward' in step:
            if self.act is not None:
                x = self.act(x, step='backward')
            if self.bn is not None:
                x = self.bn(x)
            if self.bias is not None:
                x = self.bias(x, step='backward')
            x = self.conv(x, step='backward')
            return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        if self.act is not None:
            self.act.reset()


class Concat(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.cat = torch.empty(*size)

    def forward(self, x, step='forward', bias=None):
        if bias is not None:
            self.cat = bias
        if 'forward' in step:
            return torch.cat((x,self.cat), dim=1)
        elif 'backward' in step:
            chunks = torch.split(x, chunks=2, dim=1)
            self.cat = chunks[1]
            return chunks[0]
        else:
            raise ValueError("step must be 'forward' or 'backward'")


class AnchorProcessor(nn.Module):
    def __init__(self, anchor=[[116,90], [156,198], [373,326]], cls=1):
        super().__init__()
        self.cls = cls
        if any(anchor<=0):
            raise ValueError("all anchors are expected to be positive! Input Anchor = {}".format(anchor))
        self.anchor = anchor
        self.sig = nn.Sigmoid()
        self.vec_in = len(self.anchor)*(5+self.cls)
        self.vec_o = len(self.anchor)*6

    def forward(self, x, step='forward'):
        if 'forward' in step:
            N, C, H, W = [*x.shape]
            if C != self.vec_in:
                raise ValueError("expected channel {} != input channel {}".format(self.vec_in, C))
            x_box = torch.zeros(N, self.vec_o, H, W)
            for k in range(len(self.anchor)):
                st_in = k * (5 + self.cls)
                st_o = k * 6
                # precompute
                x_box[:, st_o:st_o+2, :, :] = self.sig(x[:, st_in:st_in+2, :, :])
                x_box[:, st_o+2:st_o+4, :, :] = torch.exp(x[:, st_in+2:st_in+4, :, :])
                # size
                x_box[:, st_o + 2, :, :] = x[:, st_in + 2, :, :] * self.anchor[k, 0]
                x_box[:, st_o + 3, :, :] = x[:, st_in + 3, :, :] * self.anchor[k, 1]
                # compute for each
                for cy in range(H):
                    for cx in range(W):
                        # coordinate
                        x_box[:, st_o, cy, cx] += cx
                        x_box[:, st_o+1, cy, cx] += cy
                        # class
                        en = (k + 1) * (5 + self.cls)
                        score = x[:, st_in+5:en, cy, cx] * x[:, st_in+4, cy, cx]
                        x_box[:, st_o+4, cy, cx] = torch.max(score)
                        x_box[:, st_o+5, cy, cx] = torch.argmax(score)
            return x_box
        elif 'backward' in step:
            N, C, H, W = [*x.shape]
            if C != self.vec_o:
                raise ValueError("expected channel {} != input channel {}".format(self.vec_o, C))
            x_org = torch.zeros(N, self.vec_in, H, W)
            for n in range(N):
                for k in range(len(self.anchor)):
                    st_in = k * (5 + self.cls)
                    st_o = k * 6
                    # compute for each window
                    for cy in range(H):
                        for cx in range(W):
                            if all(x[n, st_o+2:st_o+4, cy, cx] > 0):
                                # coordinate_recon
                                x_org[n, st_in, cy, cx] = torch.logit(x[:, st_o, cy, cx] - cx)
                                x_org[n, st_in+1, cy, cx] = torch.logit(x[:, st_o+1, cy, cx] - cy)
                                # size_recon
                                x_org[n, st_in + 2, cy, cx] = torch.log(x[:, st_o + 2, cy, cx] / self.anchor[k, 0])
                                x_org[n, st_in + 3, cy, cx] = torch.log(x[:, st_o + 3, cy, cx] / self.anchor[k, 1])
                                # class_recon
                                x_org[n, st_in + 4, cy, cx] = 1.0
                                cls = int(x[n, st_o + 5, cy, cx])
                                if cls < 0 or cls >= self.cls:
                                    raise ValueError("all class number expected to be integers between 0 to {}".format(self.cls-1))
                                x_org[n, st_in + 5 + cls, cy, cx] = 1.0
            return x_org


class BoxProcessor(nn.Module):
    def __init__(self, anchor=[[116,90], [156,198], [373,326]], cls=1, conf_thr=0.5):
        super().__init__()
        self.cls = cls
        if any(anchor<=0):
            raise ValueError("all anchors are expected to be positive! Input Anchor = {}".format(anchor))
        self.anchor = anchor
        self.sig = nn.Sigmoid()
        self.vec_in = len(self.anchor)*(5+self.cls)
        self.conf_thr = conf_thr

    def forward(self, x, step='forward', grid_size=[13, 13]):
        if 'forward' in step:
            N, C, H, W = [*x.shape]
            if C != self.vec_in:
                raise ValueError("expected channel {} != input channel {}".format(self.vec_in, C))
            x_list = []
            for n in range(N):
                box_list = []
                for k in range(len(self.anchor)):
                    st = k * (5 + self.cls)
                    # precompute
                    x[:, st:st+2, :, :] = self.sig(x[:, st:st+2, :, :])
                    x[:, st+2:st+4, :, :] = torch.exp(x[:, st+2:st+4, :, :])
                    # size
                    x[:, st + 2, :, :] = x[:, st + 2, :, :] * self.anchor[k, 0]
                    x[:, st + 3, :, :] = x[:, st + 3, :, :] * self.anchor[k, 1]
                    # compute for each cell
                    for cy in range(H):
                        for cx in range(W):
                            # score
                            en = (k + 1) * (5 + self.cls)
                            score = x[n, st + 5:en, cy, cx] * x[n, st + 4, cy, cx]
                            if torch.max(score) > self.conf_thr:
                                # coordinate
                                x[n, st, cy, cx] += cx
                                x[n, st+1, cy, cx] += cy
                                # class decision
                                x[n, st + 4, cy, cx] = torch.max(score)
                                x[n, st + 5, cy, cx] = torch.argmax(score)
                                box_list.append(x[n, st:st+6])
                x_list.append(box_list)
            return x_list
        elif 'backward' in step:
            N = len(x)
            C = self.vec_in
            H, W = grid_size
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
                    for k in range(len(self.anchor)):
                        st = k * (5 + self.cls)
                        # coord-recon
                        x_org[n, st, cy, cx] = tx
                        x_org[n, st+1, cy, cx] = ty
                        # size-recon
                        x_org[n, st+2, cy, cx] = torch.log(xw / self.anchor[k][0])
                        x_org[n, st+3, cy, cx] = torch.log(xh / self.anchor[k][1])
                        # conf-recon
                        x_org[n, st+4, cy, cx] = xconf
                        x_org[n, st+5+xcls, cy, cx] = 1.0
            return x_org