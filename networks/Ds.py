import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5,
    norm_layer=nn.BatchNorm2d, use_sigmoid=False,
    branch=1, bias=True, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc*branch, ndf*branch, kernel_size=kw, stride=2, padding=padw, groups=branch, bias=True), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=2, padding=padw, bias=bias),
                norm_layer(nf*branch), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=bias),
            norm_layer(nf*branch),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf*branch, 1*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=True)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class gfnd(nn.Module):
    def __init__(self):
        super(gfnd, self).__init__()
        self.D=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1, stride=2,dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1,dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=1, stride=4, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, padding=1, stride=1, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=1, stride=4, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, padding=1, stride=1, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=4, dilation=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, padding=1, stride=1, dilation=1, bias=True)
        )
    def forward(self, input):
        fea=self.D(input)
        return fea


class conv(nn.Module):
    def __init__(self,input_channel,output_channel,stride):
        super(conv, self).__init__()
        self.conv=nn.Conv2d(input_channel, output_channel,stride=stride)
        nn.init.kaiming_normal(self.conv.weight)
    def forward(self,input):
        padded_input=F.pad(input,[1,1,1,1],mode='constant')
        fea=self.conv(padded_input)
        return fea

class TIPD(nn.Module):

    def __init__(self):
        self.sd=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1=conv(6,64,2)
        self.conv2=conv(64,128,2)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3 = conv(128, 256, 2)
        self.bn3=nn.BatchNorm2d(256)
        self.conv4 = conv(256,512,1)
        self.bn4=nn.BatchNorm2d(512)
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        self.final_conv=conv(512,1,1)
        sgm= nn.Sigmoid()
        self.D=nn.ModuleList([self.conv1,
                              self.lrelu,
                              self.conv2,
                              self.bn2,
                              self.lrelu,
                              self.conv3,
                              self.bn3,
                              self.lrelu,
                              self.conv4,
                              self.bn4,
                              self.lrelu,
                              self.final_conv,
                              sgm])

    def forward(self,img,t):
        input=torch.cat(img,t)
        results=[torch.mean(self.D(input,[1,2,3]))]
        for i in range(2):
            input=self.ds(input)
            results.append(torch.mean(self.D(input,[1,2,3])))
        return torch.min([results[0],results[1],results[2]])


