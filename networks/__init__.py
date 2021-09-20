# Add your custom network here
from .DRNet import DRNet
from .Densenet import RRDBNet

import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)

def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

def errnetnp(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=False, **kwargs)


def densenet(in_channels, out_channels):
    return RRDBNet(in_channels, out_channels, nf=32, nb=4, gc=32, bias=True, issub = False)

def errnetna(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=None, bottom_kernel_size=1, pyramid=True, **kwargs)
