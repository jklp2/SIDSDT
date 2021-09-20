import torch.nn as nn
# from networks.DRNet import DRNet

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.conv4=nn.Conv2d(64,64,kernel_size=3, padding=1)


    def forward(self, x):
        fea=self.conv1(x)
        fea=self.relu(fea)
        # fea=self.conv2(fea)
        # fea=self.relu(fea)
        # fea = self.conv3(fea)
        # fea = self.relu(fea)
        # fea = self.conv4(fea)
        # fea = self.relu(fea)
        return fea

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)


    def forward(self, x):
        # output = self.conv1(x)
        # output = self.relu(output)
        # output = self.conv2(output)
        # output = self.relu(output)
        # output = self.conv3(output)
        # output = self.relu(output)
        output = self.conv4(x)
        output = self.relu(output)

        return output