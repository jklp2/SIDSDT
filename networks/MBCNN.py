import pdb

import torch
import torch.nn as nn
import math

class IDCT(nn.Module):
    def __init__(self, shape):
        super(IDCT,self).__init__()

        self.H = torch.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == 0:
                    self.H[i][j] = torch.tensor(math.sqrt(1/shape[0]) * math.cos(i * (2 * j + 1)*math.pi / (2. * shape[0])))
                else:
                    self.H[i][j] = torch.tensor(math.sqrt(2/shape[0])*math.cos(i*(2*j+1)*math.pi/(2.*shape[0])))
        print(self.H)

    def forward(self,x):
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                x[b,c,:,:] = self.H.mm(x[b,c,:,:])
                x[b, c, :, :] = x[b, c, :, :].mm(self.H.permute(1,0))
        return x

class DenseBLock(nn.Module):
    def __init__(self, nf=32, gc=32, bias=True):
        super(DenseBLock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 2, bias=bias,dilation = 2)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 2, bias=bias,dilation = 2)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 2, bias=bias,dilation = 2)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 2, bias=bias,dilation = 2)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 2, bias=bias,dilation = 2)
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        # pdb.set_trace()
        x2 = torch.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = torch.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = torch.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = torch.relu(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class PreBlock(nn.Module):
    def __init__(self,shape):
        super(PreBlock, self).__init__()
        self.db=DenseBLock()
        self.idct = IDCT(shape)
        self.conv = nn.Conv2d(160, 32, 3, 1, 1, bias=True)

    def forward(self,x):
        fea = self.db(x)
        fea = self.idct(fea)
        fea = self.conv(fea)
        return fea+x




if __name__=='__main__':
    x = torch.tensor([[[[4.,5.,6.,7.],[4.,5.,6.,7.],[4.,5.,6.,7.],[4.,5.,6.,7.]]]])
    print(x.shape)
    idct = IDCT([4,4])
    y = idct(x)
    print(y)
    # pb = PreBlock([128,128])
    # y = pb(x)
    # print(y)
