import pdb
from collections import OrderedDict
import torch


from torch import nn
import os
import numpy as np
from utils import losses
from .base_model import BaseModel
from utils import op
from networks.semi import *
# from networks.Ds import NLayerDiscriminator
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
        return 'Semi'

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
        self.G = Net()
        self.D = D()

        self.nets={  'G':self.G.cuda(),
                     'D':self.D.cuda()}
        self.DCloss = DCloss().cuda()
        self.edgeloss = EdgeMap().cuda()
        self.vggloss = VGGLoss().cuda()
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(self.nets['G'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D = torch.optim.Adam(self.nets['D'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


        self._init_optimizer([self.optimizer_G, self.optimizer_D])
    def forward_G0(self):
        self.dehaze_syn = self.nets['G'](self.input_syn)
        self.dehaze_real = self.nets['G'](self.input_real)


    def get_G0loss(self):
        self.loss_G = 0
        self.loss_dehaze_syn = self.loss_dic['mse'](self.dehaze_syn, self.target_syn)
        self.score_g = self.nets['D'](self.dehaze_syn)
        self.loss_Ggan = self.loss_dic['gan'](self.score_g,1)
        self.loss_t = self.edgeloss(self.dehaze_real)
        self.loss_d = self.DCloss(self.dehaze_real)
        self.loss_p = self.vggloss(self.dehaze_syn, self.target_syn)
        self.loss_G += self.loss_dehaze_syn + 0.01*self.loss_p+0.001*self.loss_Ggan + 0.00001*self.loss_t + 0.00001*self.loss_d

    def forward_D(self):
        self.score_dehaze = self.nets['D'](self.dehaze_syn.detach())
        self.score_target = self.nets['D'](self.target_syn)



        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001

    def get_Dloss(self):
        # self.loss_G = 0
        self.loss_D = 0
        self.loss_D= (self.loss_dic['gan'](self.score_dehaze, 0) + self.loss_dic['gan'](self.score_target,1))
        self.loss_D += self.loss_D


    def optimize_parameters(self):
        self._train()
        self.set_requires_grad([self.nets['D']], False)
        self.forward_G0()
        self.get_G0loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.set_requires_grad([self.nets['D']], True)
        self.forward_D()
        self.get_Dloss()
        #
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()




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
        self.optimizer_D.load_state_dict(state_dict['opt_d'])
        return state_dict

    def state_dict(self):
        state_dict = {
        }
        for key in self.nets:
            state_dict[key] = self.nets[key].state_dict()
        state_dict.update({'opt_g': self.optimizer_G.state_dict(),
                           'opt_d': self.optimizer_D.state_dict(),
                           'epoch': self.epoch})
        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_dehaze_syn'] = self.loss_dehaze_syn.item()
        ret_errors['loss_Ggan'] = self.loss_Ggan.item()
        ret_errors['loss_t'] = self.loss_t.item()
        ret_errors['loss_p'] = self.loss_p.item()

        ret_errors['loss_d'] = self.loss_d.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals





