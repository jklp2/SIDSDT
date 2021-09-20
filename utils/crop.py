# encoding=utf-8
"""
    TTA plugin into test data_loader loop,
        containing overlap and data_aug (8×)

    Author:   KuangShi Zhang(15010225399@126.com)
    Refactor: xuhaoyu@tju.edu.cn
"""

import pdb

import torch
from torch.autograd import Variable
from torchvision.transforms import transforms


import pdb

import torch
import torch.nn.functional as F2

from torchvision.transforms import functional as F

class OverlapTTA(object):
    """overlap TTA
            Args:
                nw(int): num of patches (in width direction)
                nh(int):  num of patches (in height direction)
                patch_w(int): width of a patch.
                patch_h(int): height of a patch.
                norm_patch(bool): if norm each patch or not.
                flip_aug(bool): not used yet.
                device(str): device string, default 'cuda:0'.
            Usage Example

    """
    def __init__(self, img, nw, nh, patch_w=256, patch_h=256, norm_patch=False, flip_aug=False, device='cuda:0'):

        self.img = img
        self.nw = nw
        self.nh = nh
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.N, self.C, self.H, self.W = img.shape
        self.norm_patch = norm_patch
        self.flip_aug = flip_aug
        self.transforms = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))]) if norm_patch else None
        self.device = img.device

        #####################################
        #                 步长
        #####################################
        stride_h = (self.H - 256) // (nh - 1)
        stride_w = (self.W - 256) // (nw - 1)

        self.overlap_times = torch.zeros((self.C, self.H, self.W)).cpu()
        self.slice_h = []
        self.slice_w = []

        #####################################
        #   除了最后一个patch, 都按照固定步长取块
        # 将位置信息先保存在slice_h和slice_w数组中
        #####################################
        for i in range(nh - 1):
            self.slice_h.append([i * stride_h, i * stride_h + 256])
        self.slice_h.append([self.H - 256, self.H])
        for i in range(nw - 1):
            self.slice_w.append([i * stride_w, i * stride_w + 256])
        self.slice_w.append([self.W - 256, self.W])

        #####################################
        #             保存结果的数组
        #####################################
        self.result = torch.zeros((self.C, self.H, self.W)).cpu()

    def collect(self, x, cur):
        x = x.detach().cpu()

        j = cur % self.nw
        i = cur// self.nw

        #####################################
        #         分别记录图像和重复次数
        #####################################
        self.result[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += x
        self.overlap_times[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += 1

    def combine(self):
        if self.flip_aug:
            pass
        else:
            return self.result / self.overlap_times

    def __getitem__(self, index):
        """
            获取tta patch作为网络输入
            :param index:
            :return:
        """
        if self.flip_aug:
            pass

        else:
            j = index % self.nw
            i = index // self.nw
            img = self.img[:, :, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]]
            if self.norm_patch:
                img = self.transforms(img[0]).unsqueeze(dim=0)

            img_var = Variable(img, requires_grad=False).to(self.device)
            return img_var

    def __len__(self):
        return self.nw * self.nh


class InflateCrop(object):
    """overlap TTA
            Args:
                nw(int): num of patches (in width direction)
                nh(int):  num of patches (in height direction)
                inflate_width(int): inflate size for each patch
            Usage Example
                see the main of this file
    """
    def __init__(self, img, nw=10, nh=10, inflate_width=20):
        self.img = img
        self.nw = nw
        self.nh = nh
        self.N, self.C, self.H, self.W = img.shape
        self.inflate_width=inflate_width
        self.device = img.device
        self.slice_h=[[i*self.H//nh,(i+1)*self.H//nh if i!=nh-1 else self.H] for i in range(nh)]
        self.slice_w = [[i * self.W // nw, (i + 1) * self.W// nw if i != nw - 1 else self.W] for i in range(nw)]
        self.result = torch.zeros((self.C, self.H, self.W)).cpu()

    def collect(self, x, cur):
        x = x.detach().cpu()
        j = cur % self.nw
        i = cur // self.nw

        # self.result[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += \
        #     x[:,self.inflate_width:-self.inflate_width,self.inflate_width:-self.inflate_width]
        self.result[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += \
            x[:, self.inflate_width if i!=0 else 0:-self.inflate_width  if i!=self.nh-1 else self.H,
            self.inflate_width if j!=0 else 0:-self.inflate_width if j!=self.nw-1 else self.W]

    def combine(self):
        return self.result

    def __getitem__(self, index):
        j = index % self.nw
        i = index // self.nw
        img = self.img[:, :, max(self.slice_h[i][0]-self.inflate_width,0):min(self.slice_h[i][1]+self.inflate_width,self.H),
        max(self.slice_w[j][0]-self.inflate_width,0):min(self.slice_w[j][1]+self.inflate_width,self.W)]
        # if i==0:
        #     img=F2.pad(img,[0,0,self.inflate_width,0], mode="constant",value=0)
        # elif i==self.nh-1:
        #     img=F2.pad(img,[0,0,0,self.inflate_width], mode="constant",value=0)
        # if j==0:
        #     img=F2.pad(img,[self.inflate_width,0,0,0], mode="constant",value=0)
        # elif j==self.nw-1:
        #     img = F2.pad(img,[0,self.inflate_width,0,0], mode="constant",value=0)
        img_var = Variable(img, requires_grad=False).to(self.device)
        return img_var

    def __len__(self):
        return self.nw * self.nh


class TTA(object):
    def __init__(self, img):
        """
            8 derections aug
            Usage Example:
                >>> for i, data in enumerate(dataset):
                >>>     tta = TTA(img)
                >>>     for x in tta:  # 获取每个patch输入
                >>>         generated = model(x)
                >>>         torch.cuda.empty_cache()
                >>>         tta.collect(generated[0])  # 收集inference结果
                >>>     output = tta.combine()

        """
        self.img = img
        self.device = img.device
        self.N, self.C, self.H, self.W = img.shape
        self.r = torch.zeros((self.C, self.H, self.W))
        self.c = []
        self.len=8
        self.operations=[
            lambda x: x,
            lambda x: torch.unsqueeze(F.to_tensor(F.vflip(F.to_pil_image(x[0].cpu()))),0).to(self.device),
            lambda x: torch.unsqueeze(F.to_tensor(F.hflip(F.to_pil_image(x[0].cpu()))),0).to(self.device),
            lambda x: torch.unsqueeze(F.to_tensor(F.vflip(F.hflip(F.to_pil_image(x[0].cpu())))),0).to(self.device),
            lambda x: x.permute(0,1,3,2),
            lambda x: torch.unsqueeze(F.to_tensor(F.vflip(F.to_pil_image(x[0].permute(0,2,1).cpu()))),0).to(self.device),
            lambda x: torch.unsqueeze(F.to_tensor(F.hflip(F.to_pil_image(x[0].permute(0,2,1).cpu()))),0).to(self.device),
            lambda x: torch.unsqueeze(F.to_tensor(F.hflip(F.vflip(F.to_pil_image(x[0].permute(0,2,1).cpu())))),0).to(self.device)
        ]

    def collect(self, x):
        t = x.detach().cpu()
        self.c.append(t)
        # else:
        #     raise NotImplementedError

    def combine(self):
        self.r += self.c[0]
        self.r += F.to_tensor(F.vflip(F.to_pil_image(self.c[1])))
        self.r += F.to_tensor(F.hflip(F.to_pil_image(self.c[2])))
        self.r += F.to_tensor(F.vflip(F.hflip(F.to_pil_image(self.c[3]))))
        self.r += self.c[4].permute(0,2,1)
        self.r += F.to_tensor(F.vflip(F.to_pil_image(self.c[5]))).permute(0,2,1)
        self.r += F.to_tensor(F.hflip(F.to_pil_image(self.c[6]))).permute(0,2,1)
        self.r += F.to_tensor(F.vflip(F.hflip(F.to_pil_image(self.c[7])))).permute(0,2,1)
        self.c.clear()
        return self.r/8

    def __getitem__(self, index):
        return self.operations[index](self.img)


    def __len__(self):
        return 8

if __name__ =='__main__':
    data=torch.ones((1,3,2500,1000)).cuda()
    overlap = OverlapTTA(data, 10, 10, 256, 256, norm_patch=False, flip_aug=False)
    for j,patch in enumerate(overlap):
        print(j)
        tta=TTA(patch)
        for x in tta:
            generated=x
            tta.collect(generated[0])
        output_tta=tta.combine()
        overlap.collect(output_tta,j)
    output=overlap.combine()
    print(output)
    #
    # data = torch.ones((1, 3, 1397, 4138)).cuda()/8
    # crop = InflateCrop(data,15,7,16)
    # print(crop.slice_h)
    # print(crop.slice_w)
    # for j, patch in enumerate(crop):
    #     output_tta = patch
    #     crop.collect(output_tta[0], j)
    # output = crop.combine()
    # print(output)