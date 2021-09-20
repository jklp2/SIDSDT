import pdb
from collections import OrderedDict
import torch


from torch import nn
import os
import numpy as np
from utils import  losses
from .base_model import BaseModel
from utils import op
from networks.darts import encoder,decoder
from networks.Ds import NLayerDiscriminator
from utils import nw
import itertools
from networks.DRNet import DRNet,SELayer
from networks.GRU import GRUnet
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

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class Model(Base):
    def name(self):
        return 'Composite_Reuse_Autocoder'

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
        if opt.darts:
            syn_encoder = encoder(64,steps=4,oplist=[3,1,2,3,4,2]).cuda()
            dehaze_decoder = decoder(64,steps=1,oplist=[3,1,2,3,4,2]).cuda()

            real_encoder = encoder(64,steps=4,oplist=[2,1,2,3,4,2]).cuda()
            enhance_decoder = decoder(64,steps=1,oplist=[1,1,2,3,4,2]).cuda()
        else:
            syn_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
            if opt.skip:
                dehaze_decoder = DRNet(35, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
            else:
                dehaze_decoder=DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()

            real_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
            if opt.skip:
                enhance_decoder = DRNet(35, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
            else:
                enhance_decoder=DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()

        # fea_disc=NLayerDiscriminator(64).cuda()
        disc_HorC=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()
        disc_SorR=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()
        self.nets={  'syn_encoder':syn_encoder,
                     'dehaze_decoder':dehaze_decoder,
                     'real_encoder':real_encoder,
                     'enhance_decoder':enhance_decoder,
                     'D_HorC':disc_HorC,
                     'D_SorR':disc_SorR}

        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['dehaze_decoder'].parameters(),
                                                            self.nets['real_encoder'].parameters(),
                                                            self.nets['enhance_decoder'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D_HorC = torch.optim.Adam(self.nets['D_HorC'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D_SorR= torch.optim.Adam(self.nets['D_SorR'].parameters(),
                                                 lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

        self._init_optimizer([self.optimizer_G, self.optimizer_D_HorC, self.optimizer_D_SorR])

    def forward_G(self):
        self.fea_syn = self.nets['syn_encoder'](self.input_syn)
        self.fea_real = self.nets['real_encoder'](self.input_real)
        self.dehaze_syn = self.nets['dehaze_decoder'](self.fea_syn)
        self.dehaze_real = self.nets['dehaze_decoder'](self.fea_real)
        self.enhance_syn = self.nets['enhance_decoder'](self.fea_syn)
        self.enhance_real = self.nets['enhance_decoder'](self.fea_real)
        self.freal = self.enhance_syn.detach()
        self.fea_freal = self.nets['real_encoder'](self.freal)
        self.dehaze_freal = self.nets['dehaze_decoder'](self.fea_freal)
        self.fea_clear = self.nets['real_encoder'](self.target_syn)
        self.dehaze_clear = self.nets['dehaze_decoder'](self.fea_clear)
        # self.score_HorC_dreal = self.nets['D_HorC'](self.dehaze_real)
        # self.score_SorR_freal = self.nets['D_SorR'](self.enhance_syn)

    def forward_G2(self):
        self.fea_freal = self.nets['real_encoder'](self.freal)
        self.dehaze_freal = self.nets['dehaze_decoder'](self.fea_freal)

    def get_G2loss(self):
        self.loss_G2 = 0
        self.loss_dehaze_freal = self.loss_dic['l1'](self.dehaze_freal, self.target_syn)
        self.loss_G2 += self.loss_dehaze_freal*10


    def forward_D(self):
        self.score_HorC_dreal = self.nets['D_HorC'](self.dehaze_real.detach())
        self.score_SorR_freal = self.nets['D_SorR'](self.freal)
        self.score_HorC_clear = self.nets['D_HorC'](self.target_real)
        self.score_SorR_real = self.nets['D_SorR'](self.input_real)

    def get_Gloss(self):
        self.loss_G = 0
        # self.loss_D = 0

        self.loss_dehaze_syn = self.loss_dic['l1'](self.dehaze_syn, self.target_syn)
        self.loss_dehaze_freal = self.loss_dic['l1'](self.dehaze_freal, self.target_syn)
        self.loss_enhance_real = self.loss_dic['l1'](self.enhance_real, self.input_real)
        self.loss_dehaze_clear = self.loss_dic['l1'](self.dehaze_clear, self.target_syn)

        self.loss_G += self.loss_dehaze_syn*10
        self.loss_G += self.loss_dehaze_freal*10
        self.loss_G += self.loss_enhance_real*10
        self.loss_G += self.loss_dehaze_clear * 10
        if self.epoch>30:
            self.score_HorC_dreal = self.nets['D_HorC'](self.dehaze_real)
            self.score_SorR_freal = self.nets['D_SorR'](self.enhance_syn)
            self.loss_dehaze_real = self.loss_dic['gan'](self.score_HorC_dreal, 1)
            self.loss_enhance_syn = self.loss_dic['gan'](self.score_SorR_freal, 1)
            self.loss_G += self.loss_enhance_syn
            self.loss_G += self.loss_dehaze_real

        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001

    def get_Dloss(self):
        # self.loss_G = 0
        self.loss_D = 0

        self.loss_D_HorC = (self.loss_dic['gan'](self.score_HorC_dreal, 0) + self.loss_dic['gan'](self.score_HorC_clear,1))
        self.loss_D_SorR = (self.loss_dic['gan'](self.score_SorR_freal, 0) + self.loss_dic['gan'](self.score_SorR_real,1))

        self.loss_D += self.loss_D_HorC+self.loss_D_SorR


    def optimize_parameters(self):
        self._train()
        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], False)
        self.forward_G()
        self.get_Gloss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], True)
        self.forward_D()
        self.get_Dloss()
        #
        self.optimizer_D_HorC.zero_grad()
        self.optimizer_D_SorR.zero_grad()

        self.loss_D.backward()

        self.optimizer_D_HorC.step()
        self.optimizer_D_SorR.step()




    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            try:
                self.nets[key].load_state_dict(state_dict[key])
            except:
                print("there is no "+key)
        self.epoch = state_dict['epoch']
        self.optimizer_G.load_state_dict(state_dict['opt_g'])
        self.optimizer_D_HorC.load_state_dict(state_dict['opt_d_horc'])
        self.optimizer_D_SorR.load_state_dict(state_dict['opt_d_sorr'])
        return state_dict

    def state_dict(self):
        state_dict = {
        }
        for key in self.nets:
            state_dict[key] = self.nets[key].state_dict()
        state_dict.update({'opt_g': self.optimizer_G.state_dict(),
                           'opt_d_horc': self.optimizer_D_HorC.state_dict(),
                           'opt_d_sorr': self.optimizer_D_SorR.state_dict(),
                           'epoch': self.epoch})
        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_dehaze_syn'] = self.loss_dehaze_syn.item()
        ret_errors['loss_dehaze_freal'] = self.loss_dehaze_freal.item()
        ret_errors['loss_dehaze_clear'] = self.loss_dehaze_clear.item()

        ret_errors['loss_D_HorC'] = self.loss_D_HorC.item()
        ret_errors['loss_D_SorR'] = self.loss_D_SorR.item()
        if self.epoch>30:
            ret_errors['loss_dehaze_real'] = self.loss_dehaze_real.item()
            ret_errors['loss_enhance_real'] = self.loss_enhance_real.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals

class Model_final_stage(Base):
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
        syn_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
        if opt.skip:
            syn_decoder = DRNet(105, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
        else:
            syn_decoder=DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
        real_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
        if opt.skip:
            # refineNet = nn.Sequential(SELayer(70), DRNet(70, 35, 256, 3, nn.InstanceNorm2d, None, 1, 3, False, opt.skip)).cuda()
            refineNet = nn.Sequential(SELayer(70), GRUnet(70,105,5)).cuda()
        else:
            # refineNet = nn.Sequential(SELayer(64), DRNet(64, 32, 256, 3, nn.InstanceNorm2d, None, 1, 3, False, opt.skip)).cuda()
            refineNet = nn.Sequential(SELayer(64), GRUnet(64,32,5)).cuda()


        disc=NLayerDiscriminator(input_nc=6,n_layers=opt.Dlayers).cuda()
        self.nets={  'syn_encoder':syn_encoder,
                     'syn_decoder':syn_decoder,
                     'real_encoder':real_encoder,
                     'refinenet':refineNet,
                     'D':disc}
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['syn_decoder'].parameters(),
                                                            self.nets['real_encoder'].parameters(),
                                                            self.nets['refinenet'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_G0 = torch.optim.Adam(itertools.chain(
                                                            self.nets['refinenet'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D = torch.optim.Adam(self.nets['D'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

        self._init_optimizer([self.optimizer_G ,self.optimizer_D ])
    def forward_G(self):
        self.fea_syn_syn=self.nets['syn_encoder'](self.input_syn)
        self.fea_real_syn = self.nets['real_encoder'](self.input_syn)
        self.fea_syn_real=self.nets['syn_encoder'](self.input_real)
        self.fea_real_real = self.nets['real_encoder'](self.input_real)
        self.fea_refine_syn=self.nets['refinenet'](torch.cat([self.fea_syn_syn,self.fea_real_syn],dim=1))
        self.fea_refine_real = self.nets['refinenet'](torch.cat([self.fea_syn_real, self.fea_real_real], dim=1))
        self.output_syn = self.nets['syn_decoder'](self.fea_refine_syn)
        self.output_real = self.nets['syn_decoder'](self.fea_refine_real)
        self.score_fake = self.nets['D'](torch.cat([self.input_real, self.output_real], dim=1))

    def forward_D(self):
        self.score_fake=self.nets['D'](torch.cat([self.input_real, self.output_real.detach()], dim=1))
        self.score_real=self.nets['D'](torch.cat([self.input_syn, self.target_syn], dim=1))

    def get_Gloss(self):
        self.loss_G = 0
        # self.loss_D = 0
        self.loss_syn=self.loss_dic['l1'](self.output_syn,self.target_syn)
        self.loss_gan_G = self.loss_dic['gan'](self.score_fake, 1)
        self.loss_G+=self.loss_syn
        if self.epoch>=100:
            self.loss_G+=self.loss_gan_G*0.001

    def get_Dloss(self):
        self.loss_D = 0

        self.loss_gan_D = (self.loss_dic['gan'](self.score_real, 1)+self.loss_dic['gan'](self.score_fake,0))*0.5
        self.loss_D+=self.loss_gan_D*0.001


    def optimize_parameters(self):
        self._train()
        self.forward_G()
        self.get_Gloss()
        # if(self.epoch>1):
        #     self.optimizer_G.zero_grad()
        # else:
        #     self.optimizer_G0.zero_grad()
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
            if key in state_dict.keys():
                self.nets[key].load_state_dict(state_dict[key])
        self.epoch = state_dict['epoch']
        try:
            self.optimizer_G.load_state_dict(state_dict['opt_g'])
            self.optimizer_D.load_state_dict(state_dict['opt_d'])
        except:
            print("optimizer load fail!")
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
        # ret_errors['loss_real'] = self.loss_real.item()
        # ret_errors['loss_cycle'] = self.loss_cycle.item()
        # ret_errors['loss_serd'] = self.loss_serd.item()
        # ret_errors['loss_resd'] = self.loss_resd.item()
        ret_errors['loss_gan_G'] = self.loss_gan_G.item()
        ret_errors['loss_gan_D'] = self.loss_gan_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals




