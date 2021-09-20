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
        self.input_syn = data['input_syn']
        self.target_syn = data['target_syn']
        self.input_real = data['input_real']
        self.target_real = data['target_real']

class Model(Base):
    def name(self):
        return 'prototype'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train(self):
        for net in self.nets.values():
            net.train()

    def initialize(self, opt):



        BaseModel.initialize(self,opt)
        syn_encoder=DRNet(3, 64, 256, 3, nn.InstanceNorm2d, None, 1, 3, True)
        syn_decoder=DRNet(64, 3, 256, 3, nn.InstanceNorm2d, None, 1, 3, True)
        real_encoder=DRNet(3, 64, 256, 3, nn.InstanceNorm2d, None, 1, 3, True)
        real_decoder=DRNet(64, 3, 256, 3, nn.InstanceNorm2d, None, 1, 3, True)
        fea_disc=NLayerDiscriminator(64)
        disc=NLayerDiscriminator(3)
        self.nets={  'syn_encoder':syn_encoder,
                     'syn_decoder':syn_decoder,
                     'real_encoder':real_encoder,
                     'real_decoder':real_decoder,
                     'fea_D':fea_disc,
                     'D':disc}
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['syn_decoder'].parameters(),
                                                            self.nets['real_encoder'].parameters(),
                                                            self.nets['real_decoder'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D = torch.optim.Adam(self.nets['D'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_feaD = torch.optim.Adam(self.nets['fea_D'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


    def forward(self):
        self.fea_syn=self.nets['syn_encoder'](self.input_syn)
        self.fea_real=self.nets['real_encoder'](self.input_real)
        self.output_syn=self.nets['syn_decoder'](self.fea_syn)
        self.output_real=self.nets['real_decoder'](self.fea_real)
        self.score_fake=self.nets['D'](self.output_real)
        self.score_real=self.nets['D'](self.target_real)
        self.score_real_fea=self.nets['fea_D'](self.fea_real)
        self.score_syn_fea=self.nets['fea_D'](self.fea_syn)

    def get_loss(self):
        self.loss_G = 0
        self.loss_syn=self.loss_dic['mse'](self.output_syn,self.target_syn)
        self.loss_gan_G=self.loss_dic['gan'](self.score_fake,1)
        self.loss_gan_D = (self.loss_dic['gan'](self.score_real, 1)+self.loss_dic['gan'](self.score_fake,0))*0.5
        self.loss_gan_fea_G=self.loss_dic['gan'](self.score_syn_fea,1)
        self.loss_gan_fea_D = (self.loss_dic['gan'](self.score_real_fea, 1) + self.loss_dic['gan'](self.score_syn_fea, 0)) * 0.5
        self.loss_G+=self.loss_syn
        self.loss_G+=self.loss_gan_G
        self.loss_G+=self.loss_gan_fea_G


    def optimize_parameters(self):
        self._train()
        self.forward()
        self.get_loss()
        for p in self.nets['D'].parameters():
            p.requires_grad = False
        for p in self.nets['fea_D'].parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_G.step()
        for p in self.nets['D'].parameters():
            p.requires_grad = True
        for p in self.nets['fea_D'].parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad()
        self.loss_gan_D.backward(retain_graph=True)
        self.optimizer_D.step()
        self.optimizer_feaD.zero_grad()
        self.loss_gan_fea_D.backward()
        self.optimizer_feaD.step()

    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            self.nets[key].load_state_dict(state_dict[key])
        self.epoch = state_dict['epoch']
        self.optimizer_G.load_state_dict(state_dict['opt_g'])
        self.optimizer_D.load_state_dict(state_dict['opt_d'])
        self.optimizer_feaD.load_state_dict(state_dict['opt_fead'])

        return state_dict

    def state_dict(self):
        state_dict = {
        }
        for key in self.nets:
            state_dict[key]=self.nets[key].state_dict()
        state_dict.update({'opt_g':self.optimizer_G.state_dict(),
                           'opt_d':self.optimizer_D.state_dict(),
                           'opt_fead':self.optimizer_feaD.state_dict(),})

        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_syn']=self.loss_syn.item()
        ret_errors['loss_gan_G'] = self.loss_gan_G.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals





