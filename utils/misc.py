import numpy as np
import os
from PIL import Image
import os
import sys
# import misc_utils as utils


# def init_log(training=True):
#     if training:
#         log_dir = os.path.join(opt.log_dir, opt.tag)
#     else:
#         log_dir = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
#
#     utils.try_make_dir(log_dir)
#     logger = utils.get_logger(f=os.path.join(log_dir, 'log.txt'), level='info')
#
#     logger.info('==================Options==================')
#     for k, v in opt._get_kwargs():
#         logger.info(str(k) + '=' + str(v))
#     logger.info('===========================================')
#     return logger

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy

def parse_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)