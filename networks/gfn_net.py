import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class SCALE1(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(SCALE1, self).__init__()
        self.s1_pool1=nn.MaxPool2d(kernel_size=4, stride=4)
        self.encoder1=nn.Sequential( nn.Conv2d(12, 32, kernel_size=3, padding=1, dilation=1,bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)
                                     )
        self.encoder2 = nn.Sequential(
                                       nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.encoder3 = nn.Sequential(
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder1=nn.Sequential(nn.ConvTranspose2d(32,32,kernel_size=3,stride=1,padding=1),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)
                                     )
        self.decoder2 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder3 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.s1_d3conv3=nn.Sequential(nn.Conv2d(192, 3, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
    def forward(self, x):
        x=self.s1_pool1(x)
        # pdb.set_trace()
        haze,whiteb,contrp,gamma=x.split(3,1)
        # pdb.set_trace()
        e1=self.encoder1(x)
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        d1=self.decoder1(e3)
        d2=self.decoder2(d1)
        d3=self.decoder3(d2)
        concat_features=torch.cat([e1,e2,e3,d1,d2,d3],1)
        cm_whiteb,cm_contrp,cm_gamma=self.s1_d3conv3(concat_features).split(1,1)
        cm_whiteb=torch.cat([cm_whiteb for i in range(3)],1)
        cm_contrp=torch.cat([cm_contrp for i in range(3)],1)
        cm_gamma=torch.cat([cm_gamma for i in range(3)],1)
        return whiteb.mul(cm_whiteb)+contrp.mul(cm_contrp)+cm_gamma.mul(gamma)

class SCALE2(nn.Module):
    def __init__(self):
        super(SCALE2, self).__init__()
        self.s1_pool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.s2_up1=nn.ConvTranspose2d(3,3,kernel_size=4,stride=2,padding=1)
        self.encoder1=nn.Sequential(nn.Conv2d(15, 32, kernel_size=3, padding=1, dilation=1,bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)

                                     )
        self.encoder2 = nn.Sequential(
                                    nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True)
                                )
        self.encoder3 = nn.Sequential(
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder1=nn.Sequential(nn.ConvTranspose2d(32,32,kernel_size=3,stride=1,padding=1),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)
                                     )
        self.decoder2 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder3 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.s1_d3conv3=nn.Sequential(nn.Conv2d(192, 3, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
    def forward(self, x,pre):
        x=self.s1_pool1(x)
        haze,whiteb,contrp,gamma=x.split(3,1)

        # pre = F.interpolate(pre, size=[h, w])
        pre=self.s2_up1(pre)
        _, _, h, w = x.shape
        _,_,hp,wp=pre.shape

        if hp!=h:
            pre=F.pad(pre,(0,0,0,1),mode="replicate")
        if wp!=w:
            pre=F.pad(pre,(0,1,0,0),mode="replicate")
        e1=self.encoder1(torch.cat([pre,x],1))
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        d1=self.decoder1(e3)
        d2=self.decoder2(d1)
        d3=self.decoder3(d2)
        concat_features=torch.cat([e1,e2,e3,d1,d2,d3],1)
        cm_whiteb,cm_contrp,cm_gamma=self.s1_d3conv3(concat_features).split(1,1)
        cm_whiteb=torch.cat([cm_whiteb for i in range(3)],1)
        cm_contrp=torch.cat([cm_contrp for i in range(3)],1)
        cm_gamma=torch.cat([cm_gamma for i in range(3)],1)
        return whiteb.mul(cm_whiteb)+contrp.mul(cm_contrp)+cm_gamma.mul(gamma)


class SCALE3(nn.Module):
    def __init__(self):
        super(SCALE3, self).__init__()
        self.s3_up2=nn.ConvTranspose2d(3,3,kernel_size=4,stride=2,padding=1)
        self.encoder1=nn.Sequential(nn.Conv2d(15, 32, kernel_size=3, padding=1, dilation=1,bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)

                                     )
        self.encoder2 = nn.Sequential(
                                    nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(0.1, inplace=True)
                                    )
        self.encoder3 = nn.Sequential(
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder1=nn.Sequential(nn.ConvTranspose2d(32,32,kernel_size=3,stride=1,padding=1),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.LeakyReLU(0.1, inplace=True)
                                     )
        self.decoder2 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.decoder3 = nn.Sequential(
                                       nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.s1_d3conv3=nn.Sequential(nn.Conv2d(192, 3, kernel_size=3, padding=1, dilation=1, bias=True),
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
    def forward(self, x,pre):
        haze,whiteb,contrp,gamma=x.split(3,1)
        # pre = F.interpolate(pre, size=[h, w])
        pre=self.s3_up2(pre)
        _, _, h, w = x.shape
        _, _, hp, wp = pre.shape
        if hp != h:
            pre = F.pad(pre, (0, 0, 0, 1), mode="replicate")
        if wp != w:
            pre = F.pad(pre, (0, 1,0,0), mode="replicate")
        e1=self.encoder1(torch.cat([pre,x],1))
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        d1=self.decoder1(e3)
        d2=self.decoder2(d1)
        d3=self.decoder3(d2)
        concat_features=torch.cat([e1,e2,e3,d1,d2,d3],1)
        cm_whiteb,cm_contrp,cm_gamma=self.s1_d3conv3(concat_features).split(1,1)
        cm_whiteb=torch.cat([cm_whiteb for i in range(3)],1)
        cm_contrp=torch.cat([cm_contrp for i in range(3)],1)
        cm_gamma=torch.cat([cm_gamma for i in range(3)],1)
        return whiteb.mul(cm_whiteb)+contrp.mul(cm_contrp)+cm_gamma.mul(gamma)

class GFNNet(nn.Module):

    def __init__(self):
        super(GFNNet, self).__init__()
        self.scale1=SCALE1()
        self.scale2=SCALE2()
        self.scale3=SCALE3()

    def forward(self, x):
        output1=self.scale1(x)
        output2=self.scale2(x,output1)
        output3 = self.scale3(x, output2)
        return output1,output2,output3

if __name__=='__main__':
    x = torch.ones([1,12,127,127])
    model=GFNNet()
    y=model(x)
    pdb.set_trace()
    print(y)

