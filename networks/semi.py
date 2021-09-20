import torch
import torch.nn as nn
import torchvision.models.resnet
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += identity

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=1, padding=3),
                                *[BasicBlock(64,64) for _ in range(3)],
                                )
        self.b1 = nn.Sequential(nn.Conv2d(64,128,5,stride = 2, padding=2),
                                *[BasicBlock(128,128) for _ in range(3)],
                                )
        self.b2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                *[BasicBlock(256, 256) for _ in range(6)],
                                )
        self.Upconv16 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.b3 = nn.Sequential(*[BasicBlock(128, 128) for _ in range(3)])
        self.Upconv20 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.b4 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(3)],
                                nn.Conv2d(64,3,7,stride=1,padding=3),
                                nn.Tanh()
                                )


    def forward(self,x):
        fea0 = self.b0(x)
        fea1 = self.b1(fea0)
        fea2 = self.b2(fea1)
        fea3 = self.Upconv16(fea2) + fea1
        fea4 = self.b3(fea3)
        fea5 = self.Upconv20(fea4) + fea0
        return self.b4(fea5)

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.d = nn.Sequential(nn.Conv2d(3,64,kernel_size=4,stride=2,padding=2),
                             nn.Conv2d(64,128,kernel_size=4,stride=2,padding=2),
                             nn.InstanceNorm2d(128),
                             nn.Conv2d(128,256,kernel_size=4,stride=2,padding=2),
                             nn.InstanceNorm2d(256),
                             nn.Conv2d(256,512,kernel_size=4,stride=1,padding=2),
                             nn.InstanceNorm2d(512),
                             nn.Conv2d(512,1,kernel_size=4,stride=1,padding=2)
                             )
    def forward(self,x):
        return self.d(x)


class DCloss(nn.Module):
    def __init__(self):
        super(DCloss,self).__init__()
        self.dc = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.loss = nn.L1Loss()
        self.target = torch.zeros([1,3,400,400]).cuda()
    def forward(self,x):
        return self.loss(torch.min(-self.dc(-x), dim=1).values,self.target)

class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False
        self.target = torch.zeros([1, 3, 400, 400]).cuda()
        self.loss = nn.L1Loss()
    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return self.loss(edge,self.target)



class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGLoss(nn.Module):
    def __init__(self, normalize=True):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        # import pdb;pdb.set_trace()
        self.criterion = nn.MSELoss()
        self.weights = [1.0]
        self.indices = [16]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features.cuda()
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        # indices = sorted(indices)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out

if __name__=='__main__':

    vgg_loss = VGGLoss()
    x = torch.ones([4,3,400,400]).cuda()
    y = torch.ones([4,3,400,400]).cuda()
    print(vgg_loss(x, y))


