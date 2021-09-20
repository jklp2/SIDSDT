import pdb

import torch
import torch.nn as nn


class GRUnet(torch.nn.Module):
    def __init__(self, x_c,h_c,depth):
        self.x_c=x_c
        self.h_c=h_c
        self.depth= depth
        super(GRUnet,self).__init__()
        # self.gru=nn.GRU(input_size=10,hidden_size=10,num_layer=3)
        self.gate0=nn.Sequential(nn.Conv2d(h_c+x_c,h_c,kernel_size=3,padding=1),
                                 nn.Sigmoid()
                                 )
        self.gate1 = nn.Sequential(nn.Conv2d(h_c + x_c, h_c, kernel_size=3, padding=1),
                                   nn.Sigmoid()
                                   )
        self.gate2 = nn.Sequential(nn.Conv2d(h_c+x_c, h_c, kernel_size=3, padding=1),
                                   nn.Tanh()
                                   )

    def forward(self,x):
        b,_,hs,ws=x.shape
        # pdb.set_trace()
        h=torch.zeros([b,self.h_c,hs,ws]).to(x.device)
        for i in range(self.depth):
            fea0=torch.cat([h,x],dim=1)
            fea1=torch.cat([self.gate0(fea0)*h,x],dim=1)
            fea2=self.gate1(fea0)
            h=h*(1-fea2)+self.gate2(fea1)*fea2
        return h


if __name__=="__main__":
    grunet=GRUnet(20,10,10)
    x=torch.randn([1,20,100,100])
    h=torch.randn([1,10,100,100])
    output=grunet(x)
    print(output.data)
