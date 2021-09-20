import pdb
from collections import OrderedDict

import torch
from torch import nn
import os
import numpy as np
from utils import  losses
from .base_model import BaseModel
from utils import op
from networks.vae import encoder,decoder
from networks.Ds import NLayerDiscriminator
from utils import nw
import itertools
from networks.DRNet import DRNet
import copy

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy

class Base(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            op.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            op.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        self.H1 = data['H1'].cuda()
        self.C1 = data['C1'].cuda()
        self.H2 = data['H2'].cuda()
        self.C2 = data['C2'].cuda()

class Model(Base):
    def name(self):
        return 'demo'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train(self):
        for net in self.nets.values():
            if net is not None:
                net.train()

    def _eval(self):
        for net in self.nets.values():
            net.eval()

    def initialize(self, opt):



        BaseModel.initialize(self,opt)

        # fea_disc=NLayerDiscriminator(64).cuda()
        disc=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()
        self.nets={  'lastG':None,
                     'G':DRNet(3, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False).cuda(),
                     'D':disc}
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(self.nets['G'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D = torch.optim.Adam(self.nets['D'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        # self.optimizer_feaD = torch.optim.Adam(self.nets['fea_D'].parameters(),
        #                                     lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


    def forward(self):
        self.output=self.nets['G'](self.H1)
        with torch.no_grad():
            if self.nets['lastG'] is not None:
                self.target=self.nets['lastG'](self.H1).detach()
            else:
                self.target=self.H1

        self.score_real=self.nets['D'](self.C2)
        self.score_fake=self.nets['D'](self.output)


    def get_loss(self):
        self.loss_G = 0
        self.loss_D = 0
        self.loss_mse=self.loss_dic['mse'](self.output,self.target)
        self.loss_G += self.loss_mse
        self.loss_gan_G = self.loss_dic['gan'](self.score_fake, 1)
        self.loss_gan_D = (self.loss_dic['gan'](self.score_real, 1) + self.loss_dic['gan'](self.score_fake, 0)) * 0.5
        if self.epoch>1:
            self.loss_G+=self.loss_gan_G*0.001
        if self.epoch > 0:
            self.loss_D+=self.loss_gan_D*0.001




    def optimize_parameters(self):
        self._train()
        self.forward()
        self.get_loss()
        if self.epoch>0:
            for p in self.nets['D'].parameters():
                p.requires_grad = True
            self.optimizer_D.zero_grad()
            self.loss_D.backward(retain_graph=True)
            self.optimizer_D.step()
        for p in self.nets['D'].parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def upgrade(self):
        print("upgrade")
        self.nets['lastG']=copy.deepcopy(self.nets['G'])
        self.nets['lastG'].eval()
        print(id(self.nets['lastG']))
        print(id(self.nets['G']))
    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            self.nets[key].load_state_dict(state_dict[key])
        self.epoch = state_dict['epoch']
        self.optimizer_G.load_state_dict(state_dict['opt_g'])
        self.optimizer_D.load_state_dict(state_dict['opt_d'])
        # self.optimizer_feaD.load_state_dict(state_dict['opt_fead'])

        return state_dict

    def state_dict(self):
        state_dict = {
        }
        for key in self.nets:
            state_dict[key]=self.nets[key].state_dict()
        state_dict.update({'opt_g':self.optimizer_G.state_dict(),
                           'opt_d':self.optimizer_D.state_dict(),
                           # 'opt_fead':self.optimizer_feaD.state_dict(),
                           'epoch':self.epoch})

        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_mse']=self.loss_mse.item()
        ret_errors['loss_gan_G'] = self.loss_gan_G.item()
        ret_errors['loss_gan_D'] = self.loss_gan_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.H1).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals





