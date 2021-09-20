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
from networks.DRNet import DRNet,SELayer
from networks.GRU import GRUnet
from networks.fhanet import define_G,define_D,aestimate,testimate,eq2,fussioncnn
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
        if 'target_t' in data.keys():
            self.target_t = data['target_t'].cuda()
            self.target_A = data['target_A'].cuda()

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

class Model_cyclegan(Base):
    def name(self):
        return 'FHA_cyclegan'

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
        BaseModel.initialize(self, opt)
        opt.lambda_A = 10
        opt.lambda_B = 10
        opt.lambda_identity = 0.5
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.netG = 'resnet_9blocks'
        opt.norm = 'instance'
        opt.no_dropout = False
        opt.init_type = 'normal'
        opt.init_gain = 0.02
        opt.ndf = 64
        opt.netD = 'basic'
        opt.n_layers_D = 3

        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
        self.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
        self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])
        self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])

        self.nets = {'netG_A': self.netG_A, 'netG_B': self.netG_B, 'netD_A': self.netD_A, 'netD_B': self.netD_B}
        if self.isTrain:
            for net in self.nets.values():
                nw.init_weights(net, init_type=opt.init_type)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(),
                                                                self.netG_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G, self.optimizer_D])

    def forward(self):
        self.fake_clear = self.netG_A(self.input_real)
        self.rec_hazy = self.netG_B(self.fake_clear)
        self.fake_hazy = self.netG_B(self.target_real)
        self.rec_clear = self.netG_A(self.fake_hazy)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_dic['gan'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_dic['gan'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.target_real, self.fake_clear)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.input_real, self.fake_hazy)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.target_real)
            self.loss_idt_A = self.loss_dic['mse'](self.idt_A, self.target_real) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.input_real)
            self.loss_idt_B = self.loss_dic['mse'](self.idt_B, self.input_real) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.loss_dic['gan'](self.netD_A(self.fake_clear), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.loss_dic['gan'](self.netD_B(self.fake_hazy), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.loss_dic['l1'](self.rec_hazy, self.input_real) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.loss_dic['l1'](self.rec_clear, self.target_real) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + 10*self.loss_cycle_A + 10*self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
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
        ret_errors['loss_G_A'] = self.loss_G_A.item()
        ret_errors['loss_G_B'] = self.loss_G_B.item()
        ret_errors['loss_D_A'] = self.loss_D_A.item()
        ret_errors['loss_D_B'] = self.loss_D_B.item()
        ret_errors['loss_cycle_A'] = self.loss_cycle_A.item()
        ret_errors['loss_cycle_B'] = self.loss_cycle_B.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals

class Model_cycenhance(Base):
    def name(self):
        return 'FHA_cycenhance'

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
        BaseModel.initialize(self, opt)
        opt.lambda_A = 10
        opt.lambda_B = 10
        opt.lambda_identity = 2
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.netG = 'resnet_9blocks'
        opt.norm = 'instance'
        opt.no_dropout = False
        opt.init_type = 'normal'
        opt.init_gain = 0.02
        opt.ndf = 64
        opt.netD = 'basic'
        opt.n_layers_D = 3

        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
        self.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
        self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])
        self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])

        self.nets = {'netG_A': self.netG_A, 'netG_B': self.netG_B, 'netD_A': self.netD_A, 'netD_B': self.netD_B}
        if self.isTrain:
            for net in self.nets.values():
                nw.init_weights(net, init_type=opt.init_type)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(),
                                                                self.netG_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G, self.optimizer_D])

    def forward(self):
        self.fake_real = self.netG_A(self.input_syn)  #s->r
        self.rec_syn = self.netG_B(self.fake_real)   #s->r->s
        self.fake_syn = self.netG_B(self.input_real)  #r->s
        self.rec_real = self.netG_A(self.fake_syn)   #r->s->r

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_dic['gan'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_dic['gan'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.input_real, self.fake_real)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.input_syn, self.fake_syn)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.input_syn)
            self.loss_idt_A = self.loss_dic['mse'](self.idt_A, self.target_real) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.input_real)
            self.loss_idt_B = self.loss_dic['mse'](self.idt_B, self.input_real) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.loss_dic['gan'](self.netD_A(self.fake_real), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.loss_dic['gan'](self.netD_B(self.fake_syn), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.loss_dic['l1'](self.rec_syn, self.input_syn) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.loss_dic['l1'](self.rec_real, self.input_real) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + 10*self.loss_cycle_A + 10*self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
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
        ret_errors['loss_G_A'] = self.loss_G_A.item()
        ret_errors['loss_G_B'] = self.loss_G_B.item()
        ret_errors['loss_D_A'] = self.loss_D_A.item()
        ret_errors['loss_D_B'] = self.loss_D_B.item()
        ret_errors['loss_cycle_A'] = self.loss_cycle_A.item()
        ret_errors['loss_cycle_B'] = self.loss_cycle_B.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals
class Model_cgan(Base):
    def name(self):
        return 'FHA_cgan'

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
        BaseModel.initialize(self, opt)
        opt.lambda_A = 10
        opt.lambda_B = 10
        opt.lambda_identity = 0.5
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.netG = 'resnet_9blocks'
        opt.norm = 'instance'
        opt.no_dropout = False
        opt.init_type = 'normal'
        opt.init_gain = 0.02
        opt.ndf = 64
        opt.netD = 'basic'
        opt.n_layers_D = 3


        self.A_est = aestimate()
        self.t_est = testimate()
        self.netD_t = define_D(6, opt.ndf, opt.netD,
                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])
        self.netD_At = define_D(6, opt.ndf, opt.netD,
                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0])

        self.nets = {'A_est': self.A_est, 't_est': self.t_est, 'netD_t': self.netD_t, 'netD_A': self.netD_At}
        if self.isTrain:
            for net in self.nets.values():
                nw.init_weights(net, init_type=opt.init_type)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.A_est.parameters(),
                                                                self.t_est.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_t.parameters(), self.netD_At.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G, self.optimizer_D])

    def forward(self):
        self.predt = self.t_est(self.input_syn)
        self.predA = self.A_est(self.input_syn)



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_dic['gan'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_dic['gan'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_t(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_t = self.backward_D_basic(self.netD_t, torch.cat([self.target_t,self.input_syn],dim=1), torch.cat([self.predt,self.input_syn],dim=1))

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_A = self.backward_D_basic(self.netD_At, torch.cat([self.target_A,self.input_syn],dim=1), torch.cat([self.predA,self.input_syn],dim=1))

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss


        # GAN loss D_A(G_A(A))
        self.loss_testimate = self.loss_dic['gan'](self.netD_t(torch.cat([self.predt,self.input_syn])), True)
        # GAN loss D_B(G_B(B))
        self.loss_Aestimate = self.loss_dic['gan'](self.netD_At(torch.cat([self.predA,self.input_syn])), True)
        self.loss_l1 = self.loss_dic['l1'](self.target_t,self.predt)

        self.loss_G = self.loss_testimate + self.loss_Aestimate + self.loss_l1
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_t, self.netD_At], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_t, self.netD_At], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_t()  # calculate gradients for D_A
        self.backward_D_A()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def load(self, resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            try:
                self.nets[key].load_state_dict(state_dict[key])
            except:
                print("there is no " + key)
        self.epoch = state_dict['epoch']
        self.optimizer_G.load_state_dict(state_dict['opt_g'])
        self.optimizer_D.load_state_dict(state_dict['opt_d'])
        return state_dict

    def state_dict(self):
        state_dict = {}
        for key in self.nets:
            state_dict[key] = self.nets[key].state_dict()
        state_dict.update({'opt_g': self.optimizer_G.state_dict(),
                           'opt_d': self.optimizer_D.state_dict(),
                           'epoch': self.epoch})
        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_testimate'] = self.loss_testimate.item()
        ret_errors['loss_Aestimate'] = self.loss_Aestimate.item()
        ret_errors['loss_l1'] = self.loss_l1.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals



class Model_overall(Base):
    def name(self):
        return 'FHA_overall'

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
        BaseModel.initialize(self, opt)
        opt.lambda_A = 10
        opt.lambda_B = 10
        opt.lambda_identity = 0.5
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.netG = 'resnet_9blocks'
        opt.norm = 'instance'
        opt.no_dropout = False
        opt.init_type = 'normal'
        opt.init_gain = 0.02
        opt.ndf = 64
        opt.netD = 'basic'
        opt.n_layers_D = 3


        self.A_est = aestimate().cuda()
        self.t_est = testimate().cuda()
        self.netD_A = define_D(3, opt.ndf, opt.netD,
                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0]).cuda()
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, [0]).cuda()
        self.eq2 = eq2().cuda()
        self.fussion_cnn = fussioncnn().cuda()



        self.nets = {'A_est': self.A_est, 't_est': self.t_est, 'netD_A': self.netD_A,'netG_A':self.netG_A ,'fussion_cnn':self.fussion_cnn}
        if self.isTrain:
            for net in self.nets.values():
                nw.init_weights(net, init_type=opt.init_type)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            self.optimizer_G = torch.optim.Adam(itertools.chain(*[self.nets[key].parameters() for key in filter(lambda x: x!='netD_A',self.nets.keys())]),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)
            self.optimizer_D = torch.optim.Adam(self.nets['netD_A'].parameters(),
                                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G, self.optimizer_D])

    def forward(self):
        # import ipdb;ipdb.set_trace()
        self.predt = self.t_est(self.input_syn)
        self.predA = self.A_est(self.input_syn)
        self.Jcon = self.eq2(self.input_syn, self.predt, self.predA)
        self.Jcyc = self.netG_A(self.input_syn)
        self.Jfuse = self.fussion_cnn(self.Jcon, self.Jcyc)
        return self.Jfuse



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_dic['gan'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_dic['gan'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.target_syn, self.Jfuse)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss


        # GAN loss D_A(G_A(A))
        # import ipdb;ipdb.set_trace()

        self.loss_gan = self.loss_dic['gan'](self.Jfuse, True)
        # GAN loss D_B(G_B(B))
        self.loss_l1 = self.loss_dic['l1'](self.Jfuse, self.Jcyc)+self.loss_dic['l1'](self.Jfuse, self.target_syn)\
                       + self.loss_dic['l1'](self.predA, self.target_A) + self.loss_dic['l1'](self.predt,self.target_t)

        self.loss_G = self.loss_gan + self.loss_l1*100
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD_A, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def load(self, resume_epoch=None):
        ckpt_path = self.opt.ckpt_path
        state_dict = None
        state_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in self.nets:
            try:
                self.nets[key].load_state_dict(state_dict[key])
            except:
                print("there is no " + key)
        self.epoch = state_dict['epoch']
        self.optimizer_G.load_state_dict(state_dict['opt_g'])
        self.optimizer_D.load_state_dict(state_dict['opt_d'])
        return state_dict

    def state_dict(self):
        state_dict = {}
        for key in self.nets:
            state_dict[key] = self.nets[key].state_dict()
        state_dict.update({'opt_g': self.optimizer_G.state_dict(),
                           'opt_d': self.optimizer_D.state_dict(),
                           'epoch': self.epoch})
        return state_dict

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['loss_G'] = self.loss_G.item()
        ret_errors['loss_gan'] = self.loss_gan.item()
        ret_errors['loss_l1'] = self.loss_l1.item()
        ret_errors['loss_D_A'] = self.loss_D_A.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input_syn).astype(np.uint8)
        return ret_visuals


