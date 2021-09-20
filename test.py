
import pdb
import numpy as np
import torch
import models
import time
import os
import torch.backends.cudnn as cudnn
import data.haze_dataset as datasets
from options.prototype.train_options import TrainOptions
from utils import op
from os.path import join
from utils import visualizer
import utils.metrics as metrics
from utils import visualizer
from utils.misc import tensor2im
from PIL import Image

it=0
opt=TrainOptions().parse()

opt.resume= True
opt.name = "eval"

opt.darts = True
print(opt.model)
model = models.__dict__[opt.model]()
print(model.name)
model.initialize(opt=opt)
if opt.resume:
    model.load()
real_dataset = datasets.RealDataset('input')
real_dataloader=datasets.DataLoader(
    real_dataset, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)
with torch.no_grad():

    if 'final' in opt.model:
        for i,data in enumerate(real_dataloader):
            print(i)
            input = data['input'].cuda()
            output = tensor2im(
                model.inference(input))
            Image.fromarray(output.astype(np.uint8)).save(
            'output/' + data['fn'][0])
    else:
        for i,data in enumerate(real_dataloader):
            print(i)
            input = data['input'].cuda()
            fea_syn = model.nets['syn_encoder'](input)
            output = tensor2im(
                model.nets['dehaze_decoder'](fea_syn))
            Image.fromarray(output.astype(np.uint8)).save(
                'output/' + data['fn'][0])
