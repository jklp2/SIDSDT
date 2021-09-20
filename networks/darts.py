import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
PRIMITIVES = [
    'skip_connect',
    'multi_scale',
    'channel_attention',
    'spatial_attention'
]
OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    # nn.BatchNorm2d(C, affine=affine)
    ),
    'multi_scale': lambda C, stride, affine: multi_scale3(C),
    'channel_attention':lambda C, stride, affine: channel_attention(C, C, 3, stride, 1, affine=affine),
    'spatial_attention':lambda C, stride, affine: spatial_attention(C, C, 3, stride, 1, affine=affine),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class channel_attention(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(channel_attention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.ca = nn.Sequential(
      nn.Conv2d(C_in, C_in // 8, 1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in // 8, C_in, 1, padding=0, bias=True),
      nn.Sigmoid()
    )

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.ca(y)

    return y

class multi_scale(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(multi_scale, self).__init__()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv_small = nn.Conv2d(C_in,C_out,3,padding=1)
    self.conv_big = nn.Conv2d(C_in,C_out,3,padding=1)
    self.conv_fussion = nn.Conv2d(C_out*2,C_out,3,padding=1)
    self.ac=nn.ReLU()
  def forward(self, x):
    fea_big = self.ac(self.conv_big(x))
    fea_small = self.ac(self.conv_small(self.pool(x)))
    return self.ac(self.conv_fussion(torch.cat([fea_big,self.Up(fea_small)],dim=1)))

class multi_scale2(nn.Module):
    def __init__(self,C):
        super().__init__()
        def getconv(inchannel, outchannel):
            return nn.Sequential(nn.Conv2d(inchannel,outchannel,3,padding=1),nn.ReLU())
        self.conv1_1 = getconv(C,C)
        self.conv1_2 = getconv(C,C)
        self.conv2_1 = getconv(2*C,C)
        self.conv2_2 = getconv(2*C,C)
        self.conv3 = getconv(2*C,C)
        self.up = partial(F.interpolate,scale_factor=2)
        self.down = partial(F.interpolate,scale_factor=0.5)
    def forward(self, x):
        self.fea1_1 = self.conv1_1(x)
        self.fea1_2 = self.conv1_2(self.down(x))
        self.fea2_1 = self.conv2_1(torch.cat([self.fea1_1, self.up(self.fea1_2)],dim=1))
        self.fea2_2 = self.conv2_2(torch.cat([self.down(self.fea1_1),self.fea1_2],dim=1))
        self.fea3 = self.conv3(torch.cat([self.fea2_1, self.up(self.fea2_2)], dim=1))
        return self.fea3


class multi_scale3(nn.Module):
    def __init__(self, C_in):
        super(multi_scale3, self).__init__()
        self.up = torch.nn.PixelShuffle(2)
        self.down = torch.nn.PixelUnshuffle(2)
        # self.ds = torch.nn.PixelUnshuffle(2)
        self.conv0 = nn.Conv2d(2*C_in, C_in,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(4*C_in, 4*C_in,kernel_size=3,padding=1)
        # self.conv2 = nn.Conv2d(16*C_in,  16*C_in, kernel_size=3, padding=1)
        # self.up = partial(F.interpolate, scale_factor=2)
        # self.down = partial(F.interpolate, scale_factor=0.5)
        self.ac = nn.ReLU()
    def forward(self, x):
        inp1 = self.down(x)
        # inp2 = self.down(inp1)
        # import ipdb;ipdb.set_trace()
        # fea2 = self.up(self.ac(self.conv2(inp2)))
        # fea1 = self.up(self.ac(self.conv1(torch.cat([inp1,0.1*fea2],dim=1))))
        fea1 = self.up(self.ac(self.conv1(inp1)))
        return self.ac(self.conv0(torch.cat([x,fea1],dim=1)))

class spatial_attention(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(spatial_attention, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in // 8, 1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in // 8, 1, 1, padding=0, bias=True),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):  # shape/2

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        # out = self.bn(out)
        return out

class coderBlock(nn.Module):
    def __init__(self,C,oplist):
        super(coderBlock,self).__init__()
        self.size = int(math.sqrt(2*len(oplist)))
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(C, C, 3, padding=1),
                                                  nn.InstanceNorm2d(C),
                                                  nn.ReLU()) for _ in range(self.size)])
        self.ops = nn.ModuleList()
        self.addormul=[]
        for i in oplist:
            if i in [2,3]:
            # if i in [3,4]:
                self.addormul.append(1)
            else:
                self.addormul.append(0)
            self.ops.append(OPS[PRIMITIVES[i]](C,1,True))

    def forward(self,input):
        s0=input
        states = [s0]
        offset = 0
        for i in range(self.size):
            ori = self.convs[i](states[-1])
            s = 0
            s+=ori
            for j, h in enumerate(states):
                if self.addormul[j+offset]:
                    s+=ori*self.ops[j+offset](h)
                else:
                    s+=self.ops[j+offset](h)
            # s=self.convs[i](states[-1])+sum(self.ops[j+offset](h) for j,h in enumerate(states))
            offset += len(states)
            states.append(s)
        return states[-1]

################################# BEST ###################################################
# class encoder(nn.Module):
#     def __init__(self,C,steps,oplist):
#         super(encoder,self).__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, C, 3, padding=1, bias=True),
#             nn.ReLU(),
#         )
#         self.blocks = nn.Sequential(*[coderBlock(C,oplist) for _ in range(steps)])
#
#     def forward(self, input):
#         return self.blocks(self.stem(input))

def hook_fn(a,b,c):
    with open("src.py","a") as f:
        f.write("abc\n")
class encoder(nn.Module):
    def __init__(self,C,steps,oplist):
        super(encoder,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 2, padding=0, stride =2, bias=True),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([coderBlock(C,oplist) for _ in range(steps)])


    def forward(self, input):
        fea=[self.stem(input)]
        for block in self.blocks:
            fea.append(block(fea[-1]))
        output = torch.cat(fea[1:],dim=1)

        return output


class decoder(nn.Module):
    def __init__(self, C, steps, oplist):
        super(decoder,self).__init__()
        self.blocks = nn.Sequential(*[coderBlock(C,oplist) for _ in range(steps)])
        self.finalconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(C, 3, 3, padding=1, bias=True),
            nn.ReLU(),
        )
    def forward(self, input):
        return self.finalconv(self.blocks(input))


class conv(nn.Sequential):
    def __init__(self, C):
        super(conv, self).__init__()
        self.add_module('conv2d', nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1))
        self.add_module('IN', nn.InstanceNorm2d(C))
        self.add_module('act', nn.ReLU())



class cell(nn.Module):
    def __init__(self, C):
        super(cell,self).__init__()
        self.convs = nn.ModuleList([conv(C) for _ in range(3)])
    def forward(self, x):
        h=[x]
        for f in self.convs:
            cur = f(h[-1])
            h.append(cur)
        return h[-1]+x


class encoder_test(nn.Module):
    def __init__(self,C,steps):
        super(encoder_test,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=True),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[cell(C) for _ in range(steps)])
    def forward(self, input):
        return self.blocks(self.stem(input))



class decoder_test(nn.Module):
    def __init__(self, C, steps):
        super(decoder_test, self).__init__()
        self.blocks = nn.Sequential(*[cell(C) for _ in range(steps)])
        self.finalconv = nn.Sequential(
            nn.Conv2d(C, 3, 3, padding=1, bias=True),
            nn.ReLU()
        )

    def forward(self, input):
        return self.finalconv(self.blocks(input))

class encoder_test_scale(nn.Module):
    def __init__(self,C,steps):
        super(encoder_test_scale,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=True),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[cell(C) for _ in range(steps)])
    def forward(self, input):
        return self.blocks(self.stem(input))

class decoder_test_scale(nn.Module):
    def __init__(self, C, steps):
        super(decoder_test_scale, self).__init__()
        self.blocks = nn.Sequential(*[cell(C) for _ in range(steps)])
        self.finalconv = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=True),
            nn.ReLU()
        )

    def forward(self, input):
        return self.finalconv(self.blocks(input))

if __name__=='__main__':
    syn_encoder = encoder(128, steps=3, oplist=[1, 3, 3, 3, 2, 1]).cuda()
    print(syn_encoder)


