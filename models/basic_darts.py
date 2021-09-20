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
from networks.darts import encoder,decoder
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
        self.input_syn = data['input_syn'].cuda()
        self.target_syn = data['target_syn'].cuda()
        self.input_real = data['input_real'].cuda()
        self.target_real = data['target_real'].cuda()

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

    def _eval(self):
        for net in self.nets.values():
            net.eval()

    def initialize(self, opt):
        BaseModel.initialize(self,opt)
        syn_encoder=encoder(48).cuda()
        if opt.skip:
            syn_decoder = decoder(48).cuda()
        else:
            syn_decoder=decoder(48).cuda()
        real_encoder=encoder(48).cuda()
        if opt.skip:
            real_decoder = decoder(48).cuda()
        else:
            real_decoder=decoder(48).cuda()
        # fea_disc=NLayerDiscriminator(64).cuda()
        disc=NLayerDiscriminator(input_nc=6,n_layers=opt.Dlayers).cuda()
        self.nets={  'syn_encoder':syn_encoder,
                     'syn_decoder':syn_decoder,
                     'real_encoder':real_encoder,
                     'real_decoder':real_decoder,
                     # 'fea_D':fea_disc,
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
        # self.optimizer_feaD = torch.optim.Adam(self.nets['fea_D'].parameters(),
        #                                     lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self._init_optimizer([self.optimizer_G ,self.optimizer_D ])


    # def forward(self):
    #     self.fea_syn=self.nets['syn_encoder'](self.input_syn)
    #     self.fea_real=self.nets['real_encoder'](self.input_real)
    #     self.output_syn=self.nets['syn_decoder'](self.fea_syn)
    #     self.output_real=self.nets['real_decoder'](self.fea_real)
    #     self.s2r=self.nets['real_decoder'](self.fea_syn)
    #     self.s2r_fea=self.nets['real_encoder'](self.s2r)
    #     self.output_s2r=self.nets['syn_decoder'](self.s2r_fea)
    #     self.clear_sefea=self.nets['syn_encoder'](self.target_syn)
    #     self.clear_refea=self.nets['real_encoder'](self.target_syn)
    #     self.serd=self.nets['real_decoder'](self.clear_sefea)
    #     self.resd=self.nets['syn_decoder'](self.clear_sefea)
    #     self.real_dehaze=self.nets['syn_decoder'](self.fea_real)
    #     self.score_fake=self.nets['D'](self.real_dehaze)
    #     self.score_real=self.nets['D'](self.target_syn)


    def forward_G(self):
        self.fea_syn=self.nets['syn_encoder'](self.input_syn)
        self.fea_real=self.nets['real_encoder'](self.input_real)
        self.output_syn=self.nets['syn_decoder'](self.fea_syn)
        self.output_real=self.nets['real_decoder'](self.fea_real)
        self.s2r=self.nets['real_decoder'](self.fea_syn)
        self.s2r_fea=self.nets['real_encoder'](self.s2r)
        self.output_s2r=self.nets['syn_decoder'](self.s2r_fea)
        self.clear_sefea=self.nets['syn_encoder'](self.target_syn)
        self.clear_refea=self.nets['real_encoder'](self.target_syn)
        self.serd=self.nets['real_decoder'](self.clear_sefea)
        self.resd=self.nets['syn_decoder'](self.clear_refea)
        self.real_dehaze=self.nets['syn_decoder'](self.fea_real)
        self.score_fake=self.nets['D'](torch.cat([self.input_real,self.real_dehaze],dim=1))
        # self.score_real=self.nets['D'](self.target_syn)

    def forward_D(self):
        # self.fea_syn=self.nets['syn_encoder'](self.input_syn)
        # self.fea_real=self.nets['real_encoder'](self.input_real)
        # self.output_syn=self.nets['syn_decoder'](self.fea_syn)
        # self.output_real=self.nets['real_decoder'](self.fea_real)
        # self.s2r=self.nets['real_decoder'](self.fea_syn)
        # self.s2r_fea=self.nets['real_encoder'](self.s2r)
        # self.output_s2r=self.nets['syn_decoder'](self.s2r_fea)
        # self.clear_sefea=self.nets['syn_encoder'](self.target_syn)
        # self.clear_refea=self.nets['real_encoder'](self.target_syn)
        # self.serd=self.nets['real_decoder'](self.clear_sefea)
        # self.resd=self.nets['syn_decoder'](self.clear_sefea)
        # self.real_dehaze=self.nets['syn_decoder'](self.fea_real)
        self.score_fake=self.nets['D'](torch.cat([self.input_real, self.real_dehaze.detach()], dim=1))
        self.score_real=self.nets['D'](torch.cat([self.input_syn, self.target_syn], dim=1))

    def get_Gloss(self):
        self.loss_G = 0
        # self.loss_D = 0
        self.loss_syn=self.loss_dic['mse'](self.output_syn,self.target_syn)
        self.loss_real=self.loss_dic['mse'](self.output_real,self.input_real)
        self.loss_cycle=self.loss_dic['mse'](self.output_s2r,self.target_syn)
        self.loss_serd = self.loss_dic['mse'](self.serd, self.target_syn)
        self.loss_resd = self.loss_dic['mse'](self.resd, self.target_syn)
        self.loss_gan_G = self.loss_dic['gan'](self.score_fake, 1)

        self.loss_G+=self.loss_syn
        self.loss_G+=self.loss_real
        self.loss_G+=self.loss_cycle*2
        self.loss_G += self.loss_serd
        self.loss_G += self.loss_resd
        if self.epoch>=1:
            self.loss_G+=self.loss_gan_G*0.001
        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001

    def get_Dloss(self):
        # self.loss_G = 0
        self.loss_D = 0

        self.loss_gan_D = (self.loss_dic['gan'](self.score_real, 1)+self.loss_dic['gan'](self.score_fake,0))*0.5

        # self.loss_G+=self.loss_syn
        # self.loss_G+=self.loss_real
        # self.loss_G+=self.loss_cycle*2
        # self.loss_G += self.loss_serd
        # self.loss_G += self.loss_resd
        # self.loss_G+=self.loss_gan_G*0.0001
        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001
        self.loss_D+=self.loss_gan_D*0.001


    def optimize_parameters(self):
        self._train()
        self.forward_G()
        self.get_Gloss()

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.forward_D()
        self.get_Dloss()

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()




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
        ret_errors['loss_syn'] = self.loss_syn.item()
        ret_errors['loss_real'] = self.loss_real.item()
        ret_errors['loss_cycle'] = self.loss_cycle.item()
        ret_errors['loss_serd'] = self.loss_serd.item()
        ret_errors['loss_resd'] = self.loss_resd.item()
        ret_errors['loss_gan_G'] = self.loss_gan_G.item()
        ret_errors['loss_gan_D'] = self.loss_gan_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals





