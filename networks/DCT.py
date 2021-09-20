import pdb

import torch
import torch.nn as nn
import math

class IDCT(nn.Module):
    def __init__(self, ):
        super(IDCT,self).__init__()
        self.H = torch.zeros([8,8]).cuda()
        for i in range(8):
            for j in range(8):
                if i == 0:
                    self.H[i][j] = torch.tensor(math.sqrt(1/8) * math.cos(i * (2 * j + 1)*math.pi / (16)))
                else:
                    self.H[i][j] = torch.tensor(math.sqrt(2/8)*math.cos(i*(2*j+1)*math.pi/(16)))
        self.Ht=self.H.permute(1,0).cuda()
    def forward(self,x,opt):
        if opt:
            y=torch.zeros(x.shape).cuda()
            for b in range(x.shape[0]):
                for c in range(x.shape[1]):
                    for i in range(x.shape[2]//8):
                        for j in range(x.shape[3]//8):
                            y[b,c,i:i+8,j:j+8] = (self.H.mm(x[b,c,i:i+8,j:j+8])).mm(self.Ht)
            ret = torch.zeros([x.shape[0], x.shape[1] * 64, x.shape[2] // 8, x.shape[3] // 8]).cuda()
            for b in range(y.shape[0]):
                for c in range(y.shape[1]):
                    for i in range(y.shape[2] // 8):
                        for j in range(y.shape[3] // 8):
                            ret[b,c*64:c*64+64,i,j]=y[b, c, i:i + 8, j:j + 8].reshape((64))
            return ret
        else:
            ret = torch.zeros([x.shape[0], x.shape[1]//64, x.shape[2] * 8, x.shape[3] * 8]).cuda()
            ret2 = torch.zeros([x.shape[0], x.shape[1] // 64, x.shape[2] * 8, x.shape[3] * 8]).cuda()
            for b in range(x.shape[0]):
                for c in range(x.shape[1]//64):
                    for i in range(x.shape[2]):
                        for j in range(x.shape[3]):
                            ret[b, c, i*8:i*8+8, j*8:j*8+8]=x[b, c*64:c*64+64, i, j].reshape((8,8))
            for b in range(ret.shape[0]):
                for c in range(ret.shape[1]):
                    for i in range(ret.shape[2] // 8):
                        for j in range(ret.shape[3] // 8):
                            ret2[b, c, i:i + 8, j:j + 8] = (self.Ht.mm(ret[b, c, i:i + 8, j:j + 8])).mm(self.H)
            return ret2
if __name__=='__main__':
    x = torch.tensor([[[[4.,5.,6.,7.,4.,5.,6.,7.],[4.,5.,6.,7.,4.,5.,6.,7.],[4.,5.,6.,7.,4.,5.,6.,7.],
                        [4.,5.,6.,7.,4.,5.,6.,7.],[4.,5.,6.,7.,4.,5.,6.,7.],[4.,5.,6.,7.,4.,5.,6.,7.],
                        [4.,5.,6.,7.,4.,5.,6.,7.],[4.,5.,6.,7.,4.,5.,6.,7.]]]]).cuda()
    print(x.shape)
    idct = IDCT().cuda()
    y = idct(x,1)
    print(y)
    y = idct(y,0)
    print(y)
    # pb = PreBlock([128,128])
    # y = pb(x)
    # print(y)
