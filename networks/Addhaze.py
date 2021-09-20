import torch
import torch.nn as nn
class Addhaze(nn.Module):
    def __init__(self,A):
        super().__init__()
        self.A = A
    def forward(self, img, t):
        T= torch.cat((t,t,t),dim =1)
        img  = img * T
        img +=  0.5*T
        # for i in range(3):
        #     img[:,i,:,:] = img[:,i,:,:] *t[0]
        #     img[:, i, :, :] = img[:, i, :, :]+self.A[i]*(1-t[0])
        return img

