import pdb
from collections import OrderedDict
import torch

import torchvision.transforms as transforms
from torch import nn
import os
import numpy as np
from utils import  losses
from .base_model import BaseModel
from utils import op
from networks.darts import encoder,decoder,encoder_test,decoder_test, encoder_test_scale,decoder_test_scale
from networks.Ds import NLayerDiscriminator
from utils import nw
import itertools
from networks.DRNet import DRNet,SELayer,SELayer_old
from networks.danetwork import define_Gen
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
    """
        Task       Round I        Round II
        S2C         {S,C}          {S,C}
        R2R         {R,R}          {R,R}
        S2R         {S,S}        GAN + {R，R}
        R2C         {S,C}        GAN + {P, C}
    """
    def name(self):
        return 'Composite_Reuse_Autocoder_unrolled'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ret_errors = OrderedDict()
    def _train(self):
        for net in self.nets.values():
            net.train()

    def _eval(self):
        for net in self.nets.values():
            net.eval()

    def initialize(self, opt):
        BaseModel.initialize(self,opt)
        if opt.darts:
            # tensor([[0.1751, 0.3163, 0.2247, 0.2839],
            #         [0.6133, 0.1376, 0.1220, 0.1271],
            #         [0.1778, 0.2504, 0.2640, 0.3078],
            #         [0.6567, 0.1209, 0.1071, 0.1153],
            #         [0.1223, 0.1682, 0.2490, 0.4605],
            #         [0.1711, 0.4207, 0.1724, 0.2358]], device='cuda:0',
            #        grad_fn= < SoftmaxBackward >)
            # tensor([[0.1056, 0.3204, 0.2692, 0.3048],
            #         [0.3196, 0.1840, 0.2384, 0.2580],
            #         [0.2734, 0.2271, 0.2425, 0.2570],
            #         [0.2634, 0.2783, 0.1912, 0.2671],
            #         [0.0797, 0.3039, 0.1298, 0.4867],
            #         [0.0296, 0.5089, 0.0895, 0.3720]], device='cuda:0',
            #        grad_fn= < SoftmaxBackward >)


            syn_encoder = encoder(64,steps=3,oplist=[1,0,3,0,3,1]).cuda()
            dehaze_decoder = decoder(192,steps=1,oplist=[1,0,0,1,3,1]).cuda()
            
            real_encoder = encoder(64,steps=3,oplist=[1,0,3,0,3,1]).cuda()
            enhance_decoder = decoder(192,steps=1,oplist=[1,0,0,1,3,1]).cuda()
            # syn_encoder = encoder_test(128, steps=3).cuda()
            # real_encoder = encoder_test(128, steps=3).cuda()
            # dehaze_decoder = decoder_test(128, steps=1).cuda()
            # enhance_decoder = decoder_test(128, steps=1).cuda()

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

        # self.optimizer_G = torch.optim.SGD(itertools.chain(self.nets['syn_encoder'].parameters(),
        #                                                     self.nets['dehaze_decoder'].parameters(),
        #                                                     self.nets['real_encoder'].parameters(),
        #                                                     self.nets['enhance_decoder'].parameters()),
        #                                     lr=opt.lr)

        self.optimizer_D_HorC = torch.optim.Adam(self.nets['D_HorC'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D_SorR= torch.optim.Adam(self.nets['D_SorR'].parameters(),
                                                 lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

        self._init_optimizer([self.optimizer_G, self.optimizer_D_HorC, self.optimizer_D_SorR])
        self.freal = [None]
        self.dehaze_freal_t = [None]
        self.dehaze_real_t = [None]


    def inference_syn(self, x):
        self.fea_syn = self.nets['syn_encoder'](x)
        self.dehaze_syn = self.nets['dehaze_decoder'](self.fea_syn)
        return self.dehaze_syn.detach() 

    def inference_real(self,x):
        self.fea_real = self.nets['real_encoder'](x)
        self.dehaze_real = self.nets['dehaze_decoder'](self.fea_real)
        return self.dehaze_real.detach()

    def inference_preal(self,x):
        self.fea_syn = self.nets['syn_encoder'](x)
        self.freal = self.nets['enhance_decoder'](self.fea_syn)
        return self.freal.detach()

    def forward_G0(self):   #S2C {S,C}  self.dehaze_syn_t
        self.fea_syn = self.nets['syn_encoder'](self.input_syn)
        self.dehaze_syn = self.nets['dehaze_decoder'](self.fea_syn)
        self.dehaze_syn_t = self.dehaze_syn.detach() 
    def get_G0loss(self):
        self.loss_G = 0
        self.loss_dehaze_syn = self.loss_dic['l1'](self.dehaze_syn, self.target_syn)
        self.loss_G += self.loss_dehaze_syn
        self.ret_errors["loss_dehaze_syn"] = self.loss_dehaze_syn.item()

    def forward_G1(self):   #R2R  {R,R} self.rec_real_t
        self.fea_real = self.nets['real_encoder'](self.input_real)
        self.rec_real = self.nets['enhance_decoder'](self.fea_real)
        self.rec_real_t = self.rec_real.detach()

    def get_G1loss(self):
        self.loss_G = 0
        self.loss_rec_real = self.loss_dic['l1'](self.rec_real, self.input_real)
        self.loss_G += self.loss_rec_real
        self.ret_errors["loss_rec_real"] = self.loss_rec_real.item()

    def forward_G2(self):  #S2R gan self.freal
        self.fea_syn = self.nets['syn_encoder'](self.input_syn)
        self.enhance_syn = self.nets['enhance_decoder'](self.fea_syn)
        self.freal = self.enhance_syn.detach()

    def get_G2loss(self):
        self.loss_G = 0
        self.score_SorR_freal = self.nets['D_SorR'](self.enhance_syn)
        self.loss_enhance_syn = self.loss_dic['gan'](self.score_SorR_freal, 1)
        self.loss_G += 0.01*self.loss_enhance_syn
        self.ret_errors["loss_enhance_syn"] = self.loss_enhance_syn.item()


    def forward_G3(self):  #R2C gan self.dehaze_real_t
        self.fea_real = self.nets['real_encoder'](self.input_real)
        self.dehaze_real = self.nets['dehaze_decoder'](self.fea_real)
        self.dehaze_real_t = self.dehaze_real.detach()

    def get_G3loss(self):
        self.loss_G = 0
        self.score_HorC_dreal = self.nets['D_HorC'](self.dehaze_real)
        self.loss_dehaze_real = self.loss_dic['gan'](self.score_HorC_dreal, 1)
        self.loss_G += 0.01*self.loss_dehaze_real
        self.ret_errors["loss_dehaze_real"] = self.loss_dehaze_real.item()


    def forward_G4(self):  #R2C  {C,C} self.dehaze_clear_t
        self.fea_clear = self.nets['real_encoder'](self.target_syn)
        self.dehaze_clear = self.nets['dehaze_decoder'](self.fea_clear)
        self.dehaze_clear_t=self.dehaze_clear.detach()

    def get_G4loss(self):
        self.loss_G = 0
        self.loss_dehaze_clear = self.loss_dic['l1'](self.dehaze_clear, self.target_syn)
        self.loss_G += self.loss_dehaze_clear
        self.ret_errors["loss_dehaze_clear"] = self.loss_dehaze_clear.item()



    def forward_G5(self):  #S2R   {R,R} self.enhance_real_t
        self.fea_real = self.nets['syn_encoder'](self.input_real)
        self.enhance_real = self.nets['enhance_decoder'](self.fea_real)
        self.enhance_real_t=self.enhance_real.detach()

    def get_G5loss(self):
        self.loss_G = 0
        self.loss_enhance_real = self.loss_dic['l1'](self.enhance_real, self.input_real)
        self.loss_G += self.loss_enhance_real
        self.ret_errors["loss_enhance_real"] = self.loss_enhance_real.item()


    def forward_G6(self):   #R2C {P,C} self.dehaze_freal_t
        self.fea_freal = self.nets['real_encoder'](self.freal)
        self.dehaze_freal = self.nets['dehaze_decoder'](self.fea_freal)
        self.dehaze_freal_t = self.dehaze_freal.detach()
        

    def get_G6loss(self):
        self.loss_G = 0
        self.loss_dehaze_freal = self.loss_dic['l1'](self.dehaze_freal, self.target_syn)
        self.loss_G += self.loss_dehaze_freal
        self.ret_errors["loss_dehaze_freal"] = self.loss_dehaze_freal.item()


    def forward_G7(self):  #S2R   {S,S}  self.enhance_real_t
        self.fea_real = self.nets['syn_encoder'](self.input_syn)
        self.enhance_real = self.nets['enhance_decoder'](self.fea_real)
        self.enhance_real_t=self.enhance_real.detach()

    def get_G7loss(self):
        self.loss_G = 0
        self.loss_enhance_real = self.loss_dic['l1'](self.enhance_real, self.input_syn)
        self.loss_G += self.loss_enhance_real
        self.ret_errors["loss_enhance_real"] = self.loss_enhance_real.item()



    def forward_G8(self):  #R2C  {S,C} self.dehaze_clear_t
        self.fea_clear = self.nets['real_encoder'](self.input_syn)
        self.dehaze_clear = self.nets['dehaze_decoder'](self.fea_clear)
        self.dehaze_clear_t=self.dehaze_clear.detach()

    def get_G8loss(self):
        self.loss_G = 0
        self.loss_dehaze_clear = self.loss_dic['l1'](self.dehaze_clear, self.target_syn)
        self.loss_G += self.loss_dehaze_clear
        self.ret_errors["loss_dehaze_clear"] = self.loss_dehaze_clear.item()






    def forward_D(self):
        self.score_HorC_dreal = self.nets['D_HorC'](self.dehaze_real_t)
        self.score_SorR_freal = self.nets['D_SorR'](self.freal)
        self.score_HorC_clear = self.nets['D_HorC'](self.target_real)
        self.score_SorR_real = self.nets['D_SorR'](self.input_real)


        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001

    def get_Dloss(self):
        # self.loss_G = 0
        self.loss_D = 0

        self.loss_D_HorC = (self.loss_dic['gan'](self.score_HorC_dreal, 0) + self.loss_dic['gan'](self.score_HorC_clear,1))
        self.loss_D_SorR = (self.loss_dic['gan'](self.score_SorR_freal, 0) + self.loss_dic['gan'](self.score_SorR_real,1))

        self.loss_D += self.loss_D_HorC+self.loss_D_SorR

    def forward_D0(self,x): #HorC
        self.score_HorC = self.nets['D_HorC'](x)

    def get_D0loss(self,target):
        self.loss_D = 0
        self.loss_D_HorC = self.loss_dic['gan'](self.score_HorC, target)
        self.loss_D += self.loss_D_HorC
        self.ret_errors["loss_D_HorC"] = self.loss_D_HorC.item()

    def forward_D1(self, x):  # HorC
        self.score_SorR = self.nets['D_SorR'](x)

    def get_D1loss(self, target):
        self.loss_D = 0
        self.loss_D_SorR = self.loss_dic['gan'](self.score_SorR, target)
        self.loss_D += self.loss_D_SorR
        self.ret_errors["loss_D_SorR"] = self.loss_D_SorR.item()




    def op1(self):
        """
            Task       Round I        Round II
            S2C         {S,C}          {S,C}
            R2R         {R,R}          {R,R}
            S2R         {S,S}        GAN + {R，R}
            R2C         {S,C}        GAN + {P, C}
        """
        self._train()
        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], False)
        self.forward_G0()  #S2C {S,C}  self.dehaze_syn_t
        self.get_G0loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G1()  #R2R  {R,R} self.rec_real_t
        self.get_G1loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G7()  #S2R   {S,S}  self.enhance_real_t
        self.get_G7loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G8()  #R2C  {S,C} self.dehaze_clear_t
        self.get_G8loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], True)

        self.forward_D0(self.dehaze_syn_t)  # HorC
        self.get_D0loss(0)
        self.optimizer_D_HorC.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_HorC.step()

        self.forward_D0(self.target_syn)  # HorC
        self.get_D0loss(1)
        self.optimizer_D_HorC.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_HorC.step()

        self.forward_D1(self.target_syn)  # SorC
        self.get_D1loss(0)
        self.optimizer_D_SorR.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_SorR.step()

        self.forward_D1(self.target_real)  # SorC
        self.get_D1loss(1)
        self.optimizer_D_SorR.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_SorR.step()


        # self.forward_D()
        # self.get_Dloss()
        # #
        # self.optimizer_D_HorC.zero_grad()
        # self.optimizer_D_SorR.zero_grad()
        #
        # self.loss_D.backward()
        #
        # self.optimizer_D_HorC.step()
        # self.optimizer_D_SorR.step()

    def op2(self):
        """
            Task       Round I        Round II
            S2C         {S,C}          {S,C}
            R2R         {R,R}          {R,R}
            S2R         {S,S}        GAN + {R，R}
            R2C         {S,C}        GAN + {P, C}
        """
        self._train()
        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], False)
        self.forward_G0()  #S2C {S,C}  self.dehaze_syn_t
        self.get_G0loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G1()  #R2R  {R,R} self.rec_real_t
        self.get_G1loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G5()   #S2R   {R,R} self.enhance_real_t
        self.get_G5loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G2()   #S2R gan self.freal
        self.get_G2loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G6()  #R2C {P,C} self.dehaze_freal_t
        self.get_G6loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G3()   #R2C gan self.dehaze_real_t
        self.get_G3loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], True)

        self.forward_D0(self.dehaze_real_t)  # HorC
        self.get_D0loss(0)
        self.optimizer_D_HorC.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_HorC.step()

        self.forward_D0(self.target_syn)  # HorC
        self.get_D0loss(1)
        self.optimizer_D_HorC.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_HorC.step()

        self.forward_D1(self.freal)  # SorC
        self.get_D1loss(0)
        self.optimizer_D_SorR.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_SorR.step()

        self.forward_D1(self.target_real)  # SorC
        self.get_D1loss(1)
        self.optimizer_D_SorR.zero_grad()
        self.loss_D.backward()
        self.optimizer_D_SorR.step()



    def optimize_parameters(self):
        if self.epoch<80:
            self.op1()
        else:
            self.op2()
        # self._train()
        # self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], False)
        # self.forward_G0()  #S2C
        # self.get_G0loss()
        # self.optimizer_G.zero_grad()
        # self.loss_G.backward()
        # self.optimizer_G.step()
        #
        # self.forward_G1()  #R2R
        # self.get_G1loss()
        # self.optimizer_G.zero_grad()
        # self.loss_G.backward()
        # self.optimizer_G.step()
        #
        # self.forward_G2()   #S2R
        # if self.epoch>30:
        #     self.get_G2loss()
        #     self.optimizer_G.zero_grad()
        #     self.loss_G.backward()
        #     self.optimizer_G.step()
        #
        # self.forward_G3()   #R2C
        # if self.epoch>30:
        #     self.get_G3loss()
        #     self.optimizer_G.zero_grad()
        #     self.loss_G.backward()
        #     self.optimizer_G.step()
        #
        # self.forward_G4()  #R2C  c2c
        # self.get_G4loss()
        # self.optimizer_G.zero_grad()
        # # self.loss_G.backward()
        # # self.optimizer_G.step()
        #
        # self.forward_G5()  #S2R   r2r
        # self.get_G5loss()
        # self.optimizer_G.zero_grad()
        # # self.loss_G.backward()
        # # self.optimizer_G.step()
        #
        # self.forward_G6()  #R'2C
        # self.get_G6loss()
        # self.optimizer_G.zero_grad()
        # # self.loss_G.backward()
        # # self.optimizer_G.step()
        #
        #
        #
        # self.set_requires_grad([self.nets['D_HorC'], self.nets['D_SorR']], True)
        # self.forward_D()
        # self.get_Dloss()
        # #
        # self.optimizer_D_HorC.zero_grad()
        # self.optimizer_D_SorR.zero_grad()
        #
        # self.loss_D.backward()
        #
        # self.optimizer_D_HorC.step()
        # self.optimizer_D_SorR.step()




    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')

        # import ipdb;ipdb.set_trace()
        for key in self.nets:
            try:
                self.nets[key].load_state_dict(state_dict[key])
            except:
                print("there is no "+key)
        self.epoch = state_dict['epoch']
        try:
            self.optimizer_G.load_state_dict(state_dict['opt_g'])
            self.optimizer_D_HorC.load_state_dict(state_dict['opt_d_horc'])
            self.optimizer_D_SorR.load_state_dict(state_dict['opt_d_sorr'])
        except:
            print("load optimizer failed")


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
        # ret_errors = OrderedDict()
        # ret_errors['loss_dehaze_syn'] = self.loss_dehaze_syn.item()
        #
        # #ret_errors['loss_dehaze_freal'] = self.loss_dehaze_freal.item()
        # ret_errors['loss_dehaze_clear'] = self.loss_dehaze_clear.item()

        # ret_errors['loss_D_HorC'] = self.loss_D_HorC.item()
        # ret_errors['loss_D_SorR'] = self.loss_D_SorR.item()
        # if self.epoch>30:
        #     ret_errors['loss_dehaze_real'] = self.loss_dehaze_real.item()
        #     ret_errors['loss_enhance_real'] = self.loss_enhance_real.item()
        return self.ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals

class Model_final_stage(Base):
    def name(self):
        return 'CRA_urolled_final'

    def set_input(self, data, mode='train'):
        self.input_syn = data['input_syn'].cuda()
        self.target_syn = data['target_syn'].cuda()
        # self.input_real = data['input_real'].cuda()
        # self.target_real = data['target_real'].cuda()
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
            # syn_encoder = encoder(128, steps=3, oplist=[1, 3, 3, 3, 2, 1]).cuda()
            # dehaze_decoder = decoder(128, steps=3, oplist=[0, 2, 2, 0, 1, 1]).cuda()

            syn_encoder = encoder(64, steps=3, oplist=[1, 3, 3, 3, 2, 1]).cuda()
            dehaze_decoder = decoder(192, steps=1, oplist=[0, 2, 2, 0, 1, 1]).cuda()
            real_encoder = encoder(64,steps=3,oplist=[1,3,3,3,2,1]).cuda()
            # enhance_decoder = decoder(64,steps=3,oplist=[0,2,2,0,1,1]).cuda()
            # syn_encoder = encoder_test(128, steps=3).cuda()
            # real_encoder = encoder_test(128, steps=3).cuda()
            # dehaze_decoder = decoder_test(128, steps=1).cuda()
        else:
            syn_encoder = DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
            if opt.skip:
                dehaze_decoder = DRNet(35, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip, False).cuda()
            else:
                dehaze_decoder = DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip, False).cuda()

        if opt.skip:
            # refineNet = nn.Sequential(SELayer(70), DRNet(70, 35, 256, 3, nn.InstanceNorm2d, None, 1, 3, False, opt.skip)).cuda()
            refineNet = nn.Sequential(SELayer(70), GRUnet(70,105,5)).cuda()
        else:
            # refineNet = nn.Sequential(SELayer(64), DRNet(64, 32, 256, 3, nn.InstanceNorm2d, None, 1, 3, False, opt.skip)).cuda()
            refineNet = nn.Sequential(SELayer(384), GRUnet(192,192,3)).cuda()

        disc_HorC=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()

        self.nets={  'syn_encoder': syn_encoder,
                     'dehaze_decoder': dehaze_decoder,
                     'real_encoder': real_encoder,
                     'refinenet': refineNet,
                     'D_HorC': disc_HorC,
                     }
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['dehaze_decoder'].parameters(),
                                                            self.nets['real_encoder'].parameters(),
                                                            self.nets['refinenet'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_G0 = torch.optim.Adam(itertools.chain(
                                                            self.nets['refinenet'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
        self.optimizer_D = torch.optim.Adam(self.nets['D_HorC'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

        self._init_optimizer([self.optimizer_G, self.optimizer_D ])
    def inference(self,x):
        self.fea_syn_syn  = self.nets['syn_encoder'](x)
        self.fea_real_syn = self.nets['real_encoder'](x)
        self.fea_refine_syn = self.nets['refinenet'](torch.cat([self.fea_syn_syn,self.fea_real_syn],dim=1))
        return self.nets['dehaze_decoder'](self.fea_refine_syn)

    def forward_G0(self):  # synthetic or peudo real
        self.fea_syn_syn  = self.nets['syn_encoder'](self.input_syn)
        self.fea_real_syn = self.nets['real_encoder'](self.input_syn)
        self.fea_refine_syn = self.nets['refinenet'](torch.cat([self.fea_syn_syn,self.fea_real_syn],dim=1))
        self.output_syn = self.nets['dehaze_decoder'](self.fea_refine_syn)


    def forward_G1(self):  # real
        self.fea_syn_real = self.nets['syn_encoder'](self.input_real)
        self.fea_real_real = self.nets['real_encoder'](self.input_real)
        self.fea_refine_real = self.nets['refinenet'](torch.cat([self.fea_syn_real, self.fea_real_real], dim=1))
        self.output_real = self.nets['dehaze_decoder'](self.fea_refine_real)
        self.score_fake = self.nets['D_HorC'](self.output_real)

    def forward_D(self):
        self.score_fake=self.nets['D_HorC'](self.output_real.detach())
        self.score_real=self.nets['D_HorC'](self.target_syn)

    def get_G0loss(self):
        self.loss_G = 0
        self.loss_syn=self.loss_dic['l1'](self.output_syn,self.target_syn)
        self.loss_G+=self.loss_syn


    def get_G1loss(self):
        self.loss_G = 0
        self.loss_gan_G = self.loss_dic['gan'](self.score_fake, 1)
        self.loss_G+=self.loss_gan_G*0.01

    def get_Dloss(self):
        self.loss_D = 0
        self.loss_gan_D = (self.loss_dic['gan'](self.score_real, 1) + self.loss_dic['gan'](self.score_fake,0))*0.5
        self.loss_D+=self.loss_gan_D*0.01


    def optimize_parameters(self):
        self._train()
        self.forward_G0()
        self.get_G0loss()
        # if(self.epoch>1):
        #     self.optimizer_G.zero_grad()
        # else:
        #     self.optimizer_G0.zero_grad()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        # self.forward_G1()
        # self.get_G1loss()
        # self.optimizer_G.zero_grad()
        # self.loss_G.backward()
        # self.optimizer_G.step()


        # self.forward_D()
        # self.get_Dloss()
        #
        # self.optimizer_D.zero_grad()
        # self.loss_D.backward()
        # self.optimizer_D.step()




    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            if key in state_dict.keys():
                self.nets[key].load_state_dict(state_dict[key])
        # torch.nn.init.constant_(self.nets['syn_encoder'].stem[0].bias, 0)
        # torch.nn.init.constant_(self.nets['real_encoder'].stem[0].bias, 0)
        # torch.nn.init.constant_(self.nets['dehaze_decoder'].finalconv[0].bias, 0)


        self.epoch = state_dict['epoch']+1
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
        # ret_errors['loss_gan_G'] = self.loss_gan_G.item()
        # ret_errors['loss_gan_D'] = self.loss_gan_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals




class Model_S2S(Base):
    def name(self):
        return 'Composite_Reuse_Autocoder_unrolled'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_input(self, data, mode='train'):
        self.input_syn = data['input_syn'].cuda()
        self.target_syn = data['target_syn'].cuda()
        # self.input_real = data['input_real'].cuda()
        # self.target_real = data['target_real'].cuda()

    def _train(self):
        for net in self.nets.values():
            net.train()

    def _eval(self):
        for net in self.nets.values():
            net.eval()

    def initialize(self, opt):
        BaseModel.initialize(self,opt)
        if opt.darts:
            syn_encoder = encoder(64,steps=3,oplist=[1,3,3,3,2,1]).cuda()
            dehaze_decoder = decoder(192,steps=1,oplist=[0,2,2,0,1,1]).cuda()
            
            # real_encoder = encoder(64,steps=3,oplist=[1,3,3,3,2,1]).cuda()
            # enhance_decoder = decoder(192,steps=1,oplist=[0,2,2,0,1,1]).cuda()
            # syn_encoder = encoder(128,steps=3,oplist=[2,2,2,2,2,2]).cuda()
            # dehaze_decoder = decoder(128,steps=1,oplist=[2,2,2,2,2,2]).cuda()
            # syn_encoder = encoder(64, steps=4, oplist=[1,1,1,1,1,1]).cuda()
            # dehaze_decoder = decoder(64, steps=2, oplist=[1, 1, 1, 1, 1, 1]).cuda()
            # real_encoder = encoder(64,steps=6,oplist=[1,3,3,3,2,1]).cuda()
            # enhance_decoder = decoder(64,steps=3,oplist=[0,2,2,0,1,1]).cuda()
            # syn_encoder = encoder_test(128, steps=3).cuda()
            # dehaze_decoder = decoder_test(128, steps=1).cuda()
        else:
            syn_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
            if opt.skip:
                dehaze_decoder = DRNet(35, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
            else:
                dehaze_decoder=DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()

            # real_encoder=DRNet(3, 32, 256, 2, nn.InstanceNorm2d, None, 1, 3, False, opt.skip).cuda()
            # if opt.skip:
            #     enhance_decoder = DRNet(35, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()
            # else:
            #     enhance_decoder=DRNet(32, 3, 256, 1, nn.InstanceNorm2d, None, 1, 3, False, opt.skip,False).cuda()

        # fea_disc=NLayerDiscriminator(64).cuda()
        disc_HorC=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()
        # disc_SorR=NLayerDiscriminator(input_nc=3,n_layers=opt.Dlayers).cuda()
        self.nets={  'syn_encoder':syn_encoder,
                     'dehaze_decoder':dehaze_decoder,
                     'D_HorC':disc_HorC,
                     }

        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['dehaze_decoder'].parameters(),),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

        self.optimizer_D_HorC = torch.optim.Adam(self.nets['D_HorC'].parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


        self._init_optimizer([self.optimizer_G,self.optimizer_D_HorC ])
    def forward_G0(self):
        self.fea_syn = self.nets['syn_encoder'](self.input_syn)
        self.dehaze_syn = self.nets['dehaze_decoder'](self.fea_syn)
        self.dehaze_syn_t = self.dehaze_syn.detach()
    def get_G0loss(self):
        self.loss_G = 0
        self.loss_dehaze_syn = self.loss_dic['l1'](self.dehaze_syn, self.target_syn)
        self.loss_G += self.loss_dehaze_syn

    def forward_D(self):
        self.score_HorC_dsyn= self.nets['D_HorC'](self.dehaze_syn.detach())
        self.score_HorC_clear = self.nets['D_HorC'](self.target_syn)


        # self.loss_G=self.loss_syn+self.loss_real+self.loss_cycle*2+self.loss_serd+self.loss_resd+self.loss_gan_G*0.0001

    def get_Dloss(self):
        # self.loss_G = 0
        self.loss_D = 0

        self.loss_D_HorC = (self.loss_dic['gan'](self.score_HorC_dsyn, 0) + self.loss_dic['gan'](self.score_HorC_clear,1))

        self.loss_D += self.loss_D_HorC

    def inference_syn(self, x):
        self.fea_syn = self.nets['syn_encoder'](x)
        self.dehaze_syn = self.nets['dehaze_decoder'](self.fea_syn)
        return self.dehaze_syn.detach() 


    def optimize_parameters(self):
        self._train()
        # self.set_requires_grad([self.nets['D_HorC']], False)
        self.forward_G0()
        self.get_G0loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        # self.set_requires_grad([self.nets['D_HorC']], True)
        # self.forward_D()
        # self.get_Dloss()
        #
        # self.optimizer_D_HorC.zero_grad()
        # self.loss_D.backward()
        # self.optimizer_D_HorC.step()




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
        # import ipdb;ipdb.set_trace()
        try:
            self.optimizer_G.load_state_dict(state_dict['opt_g'])
            self.optimizer_D_HorC.load_state_dict(state_dict['opt_d_horc'])
        except:
            print("optimizer load fail!")
        return state_dict

    def state_dict(self):
        state_dict = {
        }
        for key in self.nets:
            state_dict[key] = self.nets[key].state_dict()
        state_dict.update({'opt_g': self.optimizer_G.state_dict(),
                           'opt_d_horc': self.optimizer_D_HorC.state_dict(),
                           'epoch': self.epoch})
        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_dehaze_syn'] = self.loss_dehaze_syn.item()

        # ret_errors['loss_D_HorC'] = self.loss_D_HorC.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals



class Model_final_distill(Base):
    def name(self):
        return 'Model_final_distill'

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
        class DAopt():
            pass

        daopt = DAopt()
        daopt.input_nc = 3
        daopt.output_nc = 3
        daopt.ngf = 64
        daopt.task_layers = 4
        daopt.norm = 'batch'
        daopt.activation = 'PReLU'
        daopt.task_model_type = 'UNet'
        daopt.init_type = 'kaiming'
        daopt.drop_rate = 0
        daopt.gpu_ids = [0]
        daopt.U_weight = 0.1
        BaseModel.initialize(self,opt)
        if opt.darts:
            # syn_encoder = encoder(128, steps=3, oplist=[1, 3, 3, 3, 2, 1]).cuda()
            # dehaze_decoder = decoder(128, steps=1, oplist=[0, 2, 2, 0, 1, 1]).cuda()
            # syn_encoder = encoder(64, steps=4, oplist=[1,1,1,1,1,1]).cuda()
            # dehaze_decoder = decoder(64, steps=2, oplist=[1, 1, 1, 1, 1, 1]).cuda()
            # real_encoder = encoder(128,steps=3,oplist=[1,3,3,3,2,1]).cuda()
            # enhance_decoder = decoder(64,steps=3,oplist=[0,2,2,0,1,1]).cuda()
            syn_encoder = encoder_test(128, steps=3).cuda()
            real_encoder = encoder_test(128, steps=3).cuda()
            dehaze_decoder = decoder_test(128, steps=1).cuda()
            refineNet = nn.Sequential(SELayer(256), GRUnet(128,128,3)).cuda()

        else:
            syn_encoder = define_Gen(3, 32, daopt.ngf, daopt.task_layers, daopt.norm,
                          daopt.activation, daopt.task_model_type, daopt.init_type, daopt.drop_rate,
                          False, daopt.gpu_ids, daopt.U_weight).cuda()
            real_encoder = define_Gen(3, 32, daopt.ngf, daopt.task_layers, daopt.norm,
                                     daopt.activation, daopt.task_model_type, daopt.init_type, daopt.drop_rate,
                                     False, daopt.gpu_ids, daopt.U_weight).cuda()
            dehaze_decoder = define_Gen(32, 3, daopt.ngf, daopt.task_layers, daopt.norm,
                                     daopt.activation, daopt.task_model_type, daopt.init_type, daopt.drop_rate,
                                     False, daopt.gpu_ids, daopt.U_weight).cuda().cuda()

            refineNet = nn.Sequential(SELayer(64), GRUnet(32,32,5)).cuda()




        self.netR = define_Gen(daopt.input_nc, daopt.output_nc, daopt.ngf, daopt.task_layers, daopt.norm,
                          daopt.activation, daopt.task_model_type, daopt.init_type, daopt.drop_rate,
                          False, daopt.gpu_ids, daopt.U_weight)
        self.netR.load_state_dict(torch.load('/media/raid/vae_dehaze/checkpoints/da/30_netR_Dehazing.pth'))

        self.nets={  'syn_encoder': syn_encoder,
                     'dehaze_decoder': dehaze_decoder,
                     'real_encoder': real_encoder,
                     'refinenet': refineNet,
                     }
        for net in self.nets.values():
            nw.init_weights(net, init_type=opt.init_type)
        self.loss_dic = losses.init_loss(opt, self.Tensor)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.nets['syn_encoder'].parameters(),
                                                            self.nets['dehaze_decoder'].parameters(),
                                                            self.nets['real_encoder'].parameters(),
                                                            self.nets['refinenet'].parameters()),
                                            lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)



        self._init_optimizer([self.optimizer_G ])
        transform_list = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def forward_G0(self):
        # import pdb;pdb.set_trace()
        self.fea_syn_syn=self.nets['syn_encoder'](self.input_syn)
        self.fea_real_syn = self.nets['real_encoder'](self.input_syn)
        self.fea_refine_syn=self.nets['refinenet'](torch.cat([self.fea_syn_syn,self.fea_real_syn],dim=1))
        self.output_syn = self.nets['dehaze_decoder'](self.fea_refine_syn)


    def forward_G1(self):
        self.fea_syn_real = self.nets['syn_encoder'](self.input_real)
        self.fea_real_real = self.nets['real_encoder'](self.input_real)
        self.fea_refine_real = self.nets['refinenet'](torch.cat([self.fea_syn_real, self.fea_real_real], dim=1))
        self.output_real = self.nets['dehaze_decoder'](self.fea_refine_real)

    def get_G0loss(self):
        self.loss_G = 0
        self.loss_syn = self.loss_dic['l1'](self.output_syn, self.target_syn)
        self.loss_G += self.loss_syn


    def get_G1loss(self):
        self.loss_G = 0
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()

            self.target_real = ((self.netR(self.input_real)+1)/2).detach()
        self.loss_real = self.loss_dic['l1'](self.output_real, self.target_real)
        self.loss_G += self.loss_real




    def optimize_parameters(self):
        self._train()
        self.forward_G0()
        self.get_G0loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.forward_G1()
        self.get_G1loss()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()


        # self.forward_D()
        # self.get_Dloss()
        #
        # self.optimizer_D.zero_grad()
        # self.loss_D.backward()
        # self.optimizer_D.step()




    def load(self,resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            if key in state_dict.keys():
                self.nets[key].load_state_dict(state_dict[key],strict=False)
        # torch.nn.init.constant_(self.nets['syn_encoder'].stem[0].bias, 0)
        # torch.nn.init.constant_(self.nets['real_encoder'].stem[0].bias, 0)
        # torch.nn.init.constant_(self.nets['dehaze_decoder'].finalconv[0].bias, 0)


        self.epoch = state_dict['epoch']+1
        try:
            self.optimizer_G.load_state_dict(state_dict['opt_g'])
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
                           # 'opt_fead':self.optimizer_feaD.state_dict(),
                           'epoch':self.epoch})

        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_syn'] = self.loss_syn.item()
        ret_errors['loss_real'] = self.loss_real.item()
        # ret_errors['loss_real'] = self.loss_real.item()
        # ret_errors['loss_cycle'] = self.loss_cycle.item()
        # ret_errors['loss_serd'] = self.loss_serd.item()
        # ret_errors['loss_resd'] = self.loss_resd.item()
        # ret_errors['loss_gan_G'] = self.loss_gan_G.item()
        # ret_errors['loss_gan_D'] = self.loss_gan_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        # ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)


        return ret_visuals