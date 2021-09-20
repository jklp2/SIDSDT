import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, bias=True, issub = True):
        super(RRDBNet, self).__init__()
        self.issub = issub
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB = nn.ModuleList([])
        self.SUB = nn.ModuleList([])
        for _ in range(nb):
            self.RRDB.append(self.create_rrdb(nf, gc, bias))
            self.SUB.append(self.create_sub(nf, gc, bias))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.sub_conv1 = nn.Conv2d(gc, nf, 3, 2, 1, bias=True)
        self.sub_conv2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.sub_conv3 = nn.Conv2d(nf, 1, 3, 2, 1, bias=True)
        self.sub_upconv1 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.sub_upconv2 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.sub_upconv3 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.pyramid_module = PyramidPooling(32, 32, scales=(4, 8, 16, 32), ct_channels=8)

    @staticmethod
    def create_rrdb(nf, gc, bias):
        rrdb = nn.ModuleList([])
        for _ in range(3):
            rrdb.append(nn.Sequential(
                nn.Conv2d(nf, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias),
            ))
        return rrdb

    @staticmethod
    def create_sub(nf, gc, bias):
        sub = nn.ModuleList([])
        for _ in range(3):
            sub.append(nn.Sequential(
                nn.Conv2d(2*gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(2*gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(2*gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(2*gc, gc, 3, 1, 1, bias=bias),
                nn.Conv2d(gc+nf, gc, 3, 1, 1, bias=bias),
            ))
        return sub

    def forward(self, x):
        g = [[[0]*6]*3] * len(self.RRDB)
        g_sub = [[[0]*5]*3] * len(self.RRDB)
        fea = self.conv_first(x)
        #  main
        for i in range(len(g)):
            if i == 0:
                for j in range(3):
                    if j == 0:
                        for k in range(5):
                            if k == 0:
                                g[i][j][k] = fea
                            else:
                                g[i][j][k] = self.lrelu(self.RRDB[i][j][k - 1](torch.cat(g[i][j][:k], 1)))
                        g[i][j][5] = self.RRDB[i][j][4](torch.cat(g[i][j][:5], 1)) * 0.2 + g[i][j][0]
                    else:
                        for k in range(5):
                            if k == 0:
                                g[i][j][k] = g[i][j - 1][5]
                            else:
                                g[i][j][k] = self.lrelu(self.RRDB[i][j][k - 1](torch.cat(g[i][j][:k], 1)))
                        g[i][j][5] = self.RRDB[i][j][4](torch.cat(g[i][j][:5], 1)) * 0.2 + g[i][j][0]
            else:
                for j in range(3):
                    if j == 0:
                        for k in range(5):
                            if k == 0:
                                g[i][j][k] = g[i - 1][2][5] * 0.2 + g[i - 1][0][0]
                            else:
                                g[i][j][k] = self.lrelu(self.RRDB[i][j][k - 1](torch.cat(g[i][j][:k], 1)))
                        g[i][j][5] = self.RRDB[i][j][4](torch.cat(g[i][j][:5], 1)) * 0.2 + g[i][j][0]
                    else:
                        for k in range(5):
                            if k == 0:
                                g[i][j][k] = g[i][j - 1][5]
                            else:
                                g[i][j][k] = self.lrelu(self.RRDB[i][j][k - 1](torch.cat(g[i][j][:k], 1)))
                        g[i][j][5] = self.RRDB[i][j][4](torch.cat(g[i][j][:5], 1)) * 0.2 + g[i][j][0]
        trunk = self.trunk_conv(g[-1][2][5] * 0.2 + g[-1][0][0])
        fea = fea + trunk
        fea = self.pyramid_module(fea)
        fea = self.lrelu(self.conv1(fea))
        fea = self.lrelu(self.conv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        #  sub
        if self.issub:
            for i in range(len(g)):
                if i == 0:
                    for j in range(3):
                        if j == 0:
                            for k in range(5):
                                if k == 0:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g[i][j][k+1]), 1)))
                                else:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j][k-1]), 1)))
                        else:
                            for k in range(5):
                                if k == 0:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j-1][4]), 1)))
                                else:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j][k-1]), 1)))
                else:
                    for j in range(3):
                        if j == 0:
                            for k in range(5):
                                if k == 0:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i-1][2][4]), 1)))
                                else:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j][k-1]), 1)))
                        else:
                            for k in range(5):
                                if k == 0:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j-1][4]), 1)))
                                else:
                                    g_sub[i][j][k] = \
                                        self.lrelu(self.SUB[i][j][k](torch.cat((g[i][j][k+1], g_sub[i][j][k-1]), 1)))

            out_sub = self.lrelu(self.sub_conv1(g_sub[-1][-1][-1]))
            out_sub = self.lrelu(self.sub_conv2(out_sub))
            out_sub = self.lrelu(self.sub_conv3(out_sub))
            out_sub = self.lrelu(self.sub_upconv1(F.interpolate(out_sub, scale_factor=2, mode='nearest')))
            out_sub = self.lrelu(self.sub_upconv2(F.interpolate(out_sub, scale_factor=2, mode='nearest')))
            out_sub = self.lrelu(self.sub_upconv3(F.interpolate(out_sub, scale_factor=2, mode='nearest')))

            return out, out_sub
        else:
            return out
