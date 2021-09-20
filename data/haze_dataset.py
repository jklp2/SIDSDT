import os.path
from os.path import join
from data.image_folder import make_dataset
from data.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import random
import torch
import torch.utils.data._utils
import math
import pdb
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import glob
import h5py
import data.torchdata as torchdata
import albumentations as A
import cv2
def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(224, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224, 224))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)

    return img_1, img_2


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

class commonDataset(BaseDataset):
    def __init__(self, datadir_x, datadir_gt, fns=None, sec=None,fullsize=False):
        super(commonDataset, self).__init__()
        self.datadir_x = datadir_x
        self.datadir_gt = datadir_gt
        self.fullsize=fullsize

        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_x = sorted(make_dataset(datadir_x, fns), key=lambda x:x.split('/')[-1])
        self.paths_gt = sorted(make_dataset(datadir_gt, fns), key=lambda x: x.split('/')[-1])
        if sec !=None:
            self.paths_x=self.paths_x [sec[0]:sec[1]]
            self.paths_gt=self.paths_gt[sec[0]:sec[1]]

        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_x, self.paths_gt))
            random.shuffle(paths)
            self.paths_x[:], self.paths_gt[:]= zip(*paths)
        # num_paths = len(self.paths) // 2
        self.X_paths = self.paths_x
        self.GT_paths = self.paths_gt


    def __getitem__(self, index):
        index_X = index % len(self.X_paths)
        index_GT = index % len(self.GT_paths)

        X_path = self.X_paths[index_X]
        GT_path = self.GT_paths[index_GT]
        if not self.fullsize:
            xr = random.randint(0,1792)
            yr = random.randint(0,768)
            x_img = Image.open(X_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
            gt_img = Image.open(GT_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
            r = random.random() * 8
            if r > 7:
                x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
                gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif r > 6:
                x_img = x_img.transpose(Image.FLIP_TOP_BOTTOM)
                gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif r > 5:
                x_img = x_img.transpose(Image.ROTATE_90)
                gt_img = gt_img.transpose(Image.ROTATE_90)
            elif r > 4:
                x_img = x_img.transpose(Image.ROTATE_180)
                gt_img = gt_img.transpose(Image.ROTATE_180)
            elif r > 3:
                x_img = x_img.transpose(Image.ROTATE_270)
                gt_img = gt_img.transpose(Image.ROTATE_270)
            elif r > 2:
                x_img = x_img.transpose(Image.TRANSPOSE)
                gt_img = gt_img.transpose(Image.TRANSPOSE)
            elif r > 1:
                x_img = x_img.transpose(Image.TRANSVERSE)
                gt_img = gt_img.transpose(Image.TRANSVERSE)
        else:
            x_img = Image.open(X_path).convert('RGB')
            gt_img = Image.open(GT_path).convert('RGB')
        X = to_tensor(x_img)
        GT = to_tensor(gt_img)
        fn = os.path.basename(X_path)

        return {'input': X, 'target_t': GT, 'target_r': GT, 'fn': fn}

    def __len__(self):

        return max(len(self.X_paths), len(self.GT_paths))
class ITSDataset(BaseDataset):
    def __init__(self, datadir_h, datadir_c, datadir_t, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(ITSDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.datadir_t = datadir_t
        self.enable_transforms = enable_transforms
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x:(x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_t = sorted(make_dataset(datadir_t, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_c = self.paths_c[:size]
            self.paths_t = self.paths_t[:size]

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
        #                                       low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_h, self.paths_c,self.paths_t))
            random.shuffle(paths)
            self.paths_h[:], self.paths_c[:], self.paths_t[:] = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths = self.paths_h
        self.C_paths = self.paths_c
        self.T_paths = self.paths_t

    # def data_synthesis(self, t_img, r_img):
    #     if self.enable_transforms:
    #         t_img, r_img = paired_data_transforms(t_img, r_img)
    #     syn_model = self.syn_model
    #     t_img, r_img, m_img = syn_model(t_img, r_img)
    #
    #     B = to_tensor(t_img)
    #     R = to_tensor(r_img)
    #     M = to_tensor(m_img)
    #
    #     return B, R, M

    def __getitem__(self, index):
        index_H = index % len(self.H_paths)
        index_C = index % len(self.C_paths)
        index_T = index % len(self.T_paths)

        H_path = self.H_paths[index_H]
        C_path = self.C_paths[index_C]
        T_path = self.T_paths[index_T]
        xr = random.randint(0,364)
        yr = random.randint(0,204)
        h_img = Image.open(H_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        c_img = Image.open(C_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        t_img = Image.open(T_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        r = random.random()
        if r>0.5:
            h_img = h_img.transpose(Image.FLIP_LEFT_RIGHT)
            c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
            t_img = t_img.transpose(Image.FLIP_LEFT_RIGHT)
        H = to_tensor(h_img)
        C = to_tensor(c_img)
        T = to_tensor(t_img)

        fn = os.path.basename(H_path)
        return {'input': H, 'target_t': C, 'target_r': T, 'fn': fn, 'j':H_path,'b':C_path,'D':T_path}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.H_paths), len(self.C_paths)), self.size)
        else:
            return max(len(self.H_paths), len(self.C_paths))


class SynandRealDataset(BaseDataset):
    def __init__(self, datadir_sh, datadir_sc, datadir_rh,datadir_rc, fns=None, size= None ):
        super(SynandRealDataset, self).__init__()
        self.size = size
        self.datadir_sh = datadir_sh
        self.datadir_sc = datadir_sc
        self.datadir_rh = datadir_rh
        self.datadir_rc = datadir_rc
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_sh = sorted(make_dataset(datadir_sh, fns), key=lambda x:(x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_sc = sorted(make_dataset(datadir_sc, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_rh = sorted(make_dataset(datadir_rh, fns), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_rc = sorted(make_dataset(datadir_rc, fns), key=lambda x: x.split('/')[-1].split('.')[0])

        if size is not None:
            self.paths_sh = self.paths_sh[:size]
            self.paths_sc = self.paths_sc[:size]
            self.paths_rh = self.paths_rh[:size]
            self.paths_rc = self.paths_rc[:size]


        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_sh, self.paths_sc,self.paths_rh,self.paths_rc))
            random.shuffle(paths)
            self.paths_sh[:], self.paths_sc[:], self.paths_rh[:], self.paths_rc[:]  = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.sH_paths = self.paths_sh
        self.sC_paths = self.paths_sc
        self.rH_paths = self.paths_rh
        self.rC_paths = self.paths_rc



    def __getitem__(self, index):
        index_sH = index % len(self.sH_paths)
        index_sC = index % len(self.sC_paths)
        index_rH = index % len(self.rH_paths)
        index_rC = index % len(self.rC_paths)

        sH_path = self.sH_paths[index_sH]
        sC_path = self.sC_paths[index_sC]
        rH_path = self.rH_paths[index_sH]
        rC_path = self.rC_paths[index_sC]
        xr = random.randint(0,364)
        yr = random.randint(0,204)
        xr2 = random.randint(0, 256)
        yr2 = random.randint(0, 256)

        sh_img = Image.open(sH_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        sc_img = Image.open(sC_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        # sh_img = Image.open(sH_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        # sc_img = Image.open(sC_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        rh_img = Image.open(rH_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        rc_img = Image.open(rC_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        r = random.random()
        if r>0.5:
            sh_img = sh_img.transpose(Image.FLIP_LEFT_RIGHT)
            sc_img = sc_img.transpose(Image.FLIP_LEFT_RIGHT)
            rh_img = rh_img.transpose(Image.FLIP_LEFT_RIGHT)
            rc_img = rc_img.transpose(Image.FLIP_LEFT_RIGHT)
        sH = to_tensor(sh_img)
        sC = to_tensor(sc_img)
        rH = to_tensor(rh_img)
        rC = to_tensor(rc_img)

        # fn = os.path.basename(H_path)
        return {'input_syn': sH, 'target_syn': sC, 'input_real': rH, 'target_real': rC,'fn_input':sH_path,'fn_target':sC_path}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.sH_paths), len(self.sC_paths)), self.size)
        else:
            return max(len(self.sH_paths), len(self.sC_paths))


class SynotsandRealDataset(BaseDataset):
    def __init__(self, datadir_sh, datadir_sc, datadir_rh,datadir_rc,  full_real=False, fns=None, size= None ,isod=True,datadir_t = None, isfullsize=False):
        super(SynotsandRealDataset, self).__init__()
        self.size = size
        self.datadir_sh = datadir_sh
        self.datadir_sc = datadir_sc
        self.datadir_rh = datadir_rh
        self.datadir_rc = datadir_rc
        self.full_real = full_real
        self.isod = isod
        self.isfullsize=isfullsize
        self.has_t = datadir_t is not None
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_sh = sorted(make_dataset(datadir_sh, fns), key=lambda x:(x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        if self.isod:
            self.paths_sc = sorted(make_dataset(datadir_sc, fns, repeat=35), key=lambda x: x.split('/')[-1].split('.')[0])
        else:
            self.paths_sc = sorted(make_dataset(datadir_sc, fns, repeat=10),
                                   key=lambda x: x.split('/')[-1].split('.')[0])
        self.sa = [float(os.path.basename(x).split('_')[1]) for x in self.paths_sh]
        self.paths_rh = sorted(make_dataset(datadir_rh, fns), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_rc = sorted(make_dataset(datadir_rc, fns), key=lambda x: x.split('/')[-1].split('.')[0])
        if self.has_t:
            self.paths_t=[]
            for x in self.paths_sh:
                filename = os.path.basename(x)
                filename = filename.replace('jpg','png')
                self.paths_t.append(os.path.join(datadir_t,filename))



        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        print("reset")
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            if self.has_t:
                paths = list(zip(self.paths_sh, self.paths_sc, self.sa,self.paths_t))
                random.shuffle(paths)
                self.paths_sh[:], self.paths_sc[:], self.sa[:],self.paths_t[:] = zip(*paths)
            else:
                paths = list(zip(self.paths_sh, self.paths_sc, self.sa))
                random.shuffle(paths)
                self.paths_sh[:], self.paths_sc[:], self.sa[:] = zip(*paths)

                random.shuffle(self.rH_paths)
                random.shuffle(self.rC_paths)
        # num_paths = len(self.paths) // 2
        if self.size is not None:
            self.sH_paths = self.paths_sh[:self.size]
            self.sC_paths = self.paths_sc[:self.size]
            self.rH_paths = self.paths_rh[:self.size]
            self.rC_paths = self.paths_rc[:self.size]
            self.sA = self.sa[:self.size]
            if self.has_t:
                self.t_paths = self.paths_t[:self.size]



    def __getitem__(self, index):
        index_sH = index % len(self.sH_paths)
        index_sC = index % len(self.sC_paths)
        index_rH = index % len(self.rH_paths)
        index_rC = index % len(self.rC_paths)

        sH_path = self.sH_paths[index_sH]
        sC_path = self.sC_paths[index_sC]
        rH_path = self.rH_paths[index_sH]
        rC_path = self.rC_paths[index_sC]
        target_A = self.sA[index_sH]
        if self.has_t:
            t_path = self.t_paths[index_sH]
        # xr = random.randint(0,364)
        # yr = random.randint(0,204)
        if self.isod:
            xr2 = random.randint(0, 256)
            yr2 = random.randint(0, 256)
            if self.isfullsize:
                sh_img = Image.open(sH_path).resize((512, 512)).convert('RGB')
                sc_img = Image.open(sC_path).resize((512, 512)).convert('RGB')
                if self.has_t:
                    st_img = Image.open(t_path).resize((512, 512)).convert('RGB')

            else:
                sh_img = Image.open(sH_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
                sc_img = Image.open(sC_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
                if self.has_t:
                    st_img = Image.open(t_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        else:
            xr2 = random.randint(0, 364)
            yr2 = random.randint(0, 204)
            sh_img = Image.open(sH_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            sc_img = Image.open(sC_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            if self.has_t:
                st_img = Image.open(t_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
        if self.isfullsize:
            rh_img = Image.open(rH_path).resize((512, 512)).convert('RGB')
            rc_img = Image.open(rC_path).resize((512, 512)).convert('RGB')
        else:
            if not self.full_real:
                rh_img = Image.open(rH_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            else:
                rh_img = Image.open(rH_path).resize((256, 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((256, 256)).convert('RGB')
        r = random.random()

        if r>0.5:
            sh_img = sh_img.transpose(Image.FLIP_LEFT_RIGHT)
            sc_img = sc_img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.has_t:
                st_img = st_img.transpose(Image.FLIP_LEFT_RIGHT)
            rh_img = rh_img.transpose(Image.FLIP_LEFT_RIGHT)
            rc_img = rc_img.transpose(Image.FLIP_LEFT_RIGHT)
        sH = to_tensor(sh_img)
        sC = to_tensor(sc_img)
        if self. has_t:
            sT = to_tensor(st_img)
        rH = to_tensor(rh_img)
        rC = to_tensor(rc_img)

        # fn = os.path.basename(H_path)
        if self.has_t:
            return {'input_syn': sH, 'target_syn': sC, 'input_real': rH, 'target_real': rC, 'target_A': target_A,
                    'target_t': sT, 'fn_input': sH_path, 'fn_target': sC_path}
        return {'input_syn': sH, 'target_syn': sC, 'input_real': rH, 'target_real': rC, 'target_A': target_A,
                'fn_input': sH_path, 'fn_target': sC_path}


    def __len__(self):
        if self.size is not None:
            return min(max(len(self.sH_paths), len(self.sC_paths)), self.size)
        else:
            return max(len(self.sH_paths), len(self.sC_paths))


class ITSOTSURHI(BaseDataset):
    def __init__(self, datadir_shi, datadir_sci,datadir_sho, datadir_sco,datadir_rh,datadir_rc,full_real=False, fns=None, size= None ,phase=-1):
        super(ITSOTSURHI, self).__init__()
        self.size = size
        self.datadir_shi = datadir_shi
        self.datadir_sci = datadir_sci
        self.datadir_sho = datadir_sho
        self.datadir_sco = datadir_sco
        self.datadir_rh = datadir_rh
        self.datadir_rc = datadir_rc
        self.full_real = full_real
        self.phase=phase
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_shi = sorted(make_dataset(datadir_shi, fns), key=lambda x:(x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_sho = sorted(make_dataset(datadir_sho, fns), key=lambda x:(x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_sco = sorted(make_dataset(datadir_sco, fns, repeat=35), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_sci = sorted(make_dataset(datadir_sci, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])
        # if self.isod:
        #     self.paths_sc = sorted(make_dataset(datadir_sc, fns, repeat=35), key=lambda x: x.split('/')[-1].split('.')[0])
        # else:
        #     self.paths_sc = sorted(make_dataset(datadir_sc, fns, repeat=10),
        #                            key=lambda x: x.split('/')[-1].split('.')[0])

        self.paths_rh = sorted(make_dataset(datadir_rh, fns), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_rc = sorted(make_dataset(datadir_rc, fns), key=lambda x: x.split('/')[-1].split('.')[0])
        # pdb.set_trace()
        if size is not None:
            self.paths_shi = self.paths_shi[:size]
            self.paths_sci = self.paths_sci[:size]
            self.paths_sho = self.paths_sho[:size]
            self.paths_sco = self.paths_sco[:size]
            self.paths_rh = self.paths_rh[:size]
            self.paths_rc = self.paths_rc[:size]


        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_shi, self.paths_sci,self.paths_sho, self.paths_sco,self.paths_rh,self.paths_rc))
            random.shuffle(paths)
            self.paths_shi[:], self.paths_sci[:],self.paths_sho[:], self.paths_sco[:], self.paths_rh[:], self.paths_rc[:]  = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.sHi_paths = self.paths_shi
        self.sCi_paths = self.paths_sci
        self.sHo_paths = self.paths_shi
        self.sCo_paths = self.paths_sci
        self.rH_paths = self.paths_rh
        self.rC_paths = self.paths_rc



    def __getitem__(self, index):
        index_sHi = index % len(self.sHi_paths)

        index_sCi = index % len(self.sCi_paths)
        index_sHo = index % len(self.sHo_paths)
        index_sCo = index % len(self.sCo_paths)
        index_rH = index % len(self.rH_paths)
        index_rC = index % len(self.rC_paths)

        sHi_path = self.sHi_paths[index_sHi]
        sCi_path = self.sCi_paths[index_sCi]
        sHo_path = self.sHo_paths[index_sHo]
        sCo_path = self.sCo_paths[index_sCo]
        rH_path = self.rH_paths[index_rH]
        rC_path = self.rC_paths[index_rC]
        if self.phase==-1:
            xr2 = random.randint(0, 364)
            yr2 = random.randint(0, 204)
            shi_img = Image.open(sHi_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            sci_img = Image.open(sCi_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')


            xr2 = random.randint(0, 256)
            yr2 = random.randint(0, 256)
            sho_img = Image.open(sHo_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            sco_img = Image.open(sCo_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')


            if not self.full_real:
                rh_img = Image.open(rH_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((512,512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            else:
                rh_img = Image.open(rH_path).resize((256, 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((256, 256)).convert('RGB')
            r = random.random()
            if r>0.5:
                shi_img = shi_img.transpose(Image.FLIP_LEFT_RIGHT)
                sci_img = sci_img.transpose(Image.FLIP_LEFT_RIGHT)
                sho_img = sho_img.transpose(Image.FLIP_LEFT_RIGHT)
                sco_img = sco_img.transpose(Image.FLIP_LEFT_RIGHT)
                rh_img = rh_img.transpose(Image.FLIP_LEFT_RIGHT)
                rc_img = rc_img.transpose(Image.FLIP_LEFT_RIGHT)
            sHi = to_tensor(shi_img)
            sCi = to_tensor(sci_img)
            sHo = to_tensor(sho_img)
            sCo = to_tensor(sco_img)
            rH = to_tensor(rh_img)
            rC = to_tensor(rc_img)

            # fn = os.path.basename(H_path)
            return {'input_syni': sHi, 'target_syni': sCi, 'input_syno': sHo, 'target_syno': sCo, 'input_real': rH, 'target_real': rC}
        elif self.phase==0:
            xr2 = random.randint(0, 256)
            yr2 = random.randint(0, 256)
            sho_img = Image.open(sHo_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            sco_img = Image.open(sCo_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')

            r = random.random()
            if r > 0.5:

                sho_img = sho_img.transpose(Image.FLIP_LEFT_RIGHT)
                sco_img = sco_img.transpose(Image.FLIP_LEFT_RIGHT)

            sHo = to_tensor(sho_img)
            sCo = to_tensor(sco_img)

            # fn = os.path.basename(H_path)
            return {'input_syn': sHo, 'target_syn': sCo}
        elif self.phase==1:
            xr2 = random.randint(0, 364)
            yr2 = random.randint(0, 204)
            shi_img = Image.open(sHi_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            sci_img = Image.open(sCi_path).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')


            r = random.random()
            if r > 0.5:
                shi_img = shi_img.transpose(Image.FLIP_LEFT_RIGHT)
                sci_img = sci_img.transpose(Image.FLIP_LEFT_RIGHT)

            sHi = to_tensor(shi_img)
            sCi = to_tensor(sci_img)


            # fn = os.path.basename(H_path)
            return {'input_syn': sHi, 'target_syn': sCi}
        elif self.phase==2:
            xr2 = random.randint(0, 256)
            yr2 = random.randint(0, 256)
            if not self.full_real:
                rh_img = Image.open(rH_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((512, 512)).crop((xr2, yr2, xr2 + 256, yr2 + 256)).convert('RGB')
            else:
                rh_img = Image.open(rH_path).resize((256, 256)).convert('RGB')
                rc_img = Image.open(rC_path).resize((256, 256)).convert('RGB')
            r = random.random()
            if r > 0.5:

                rh_img = rh_img.transpose(Image.FLIP_LEFT_RIGHT)
                rc_img = rc_img.transpose(Image.FLIP_LEFT_RIGHT)
            rH = to_tensor(rh_img)
            rC = to_tensor(rc_img)

            # fn = os.path.basename(H_path)
            return {'input_real': rH,'target_real': rC}



    def __len__(self):
        if self.size is not None:
            return min(max(len(self.sHi_paths), len(self.sCi_paths)), self.size)
        else:
            return max(len(self.sHi_paths), len(self.sCi_paths))

class ITS_gfnDataset(BaseDataset):
    def __init__(self, datadir_h,datadir_h_rgw,datadir_h_cont,datadir_h_gamma,datadir_c, datadir_t, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(ITS_gfnDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_h_rgw = datadir_h_rgw
        self.datadir_h_cont = datadir_h_cont
        self.datadir_h_gamma = datadir_h_gamma
        self.datadir_c = datadir_c
        self.datadir_t = datadir_t
        self.enable_transforms = enable_transforms
        #
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_h_rgw = sorted(make_dataset(datadir_h_rgw, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_h_cont = sorted(make_dataset(datadir_h_cont, fns), key=lambda x:  (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_h_gamma = sorted(make_dataset(datadir_h_gamma, fns), key=lambda x:  (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_t = sorted(make_dataset(datadir_t, fns), key=lambda x:  (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))

        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_h_rgw = self.paths_h_rgw[:size]
            self.paths_h_cont = self.paths_h_cont[:size]
            self.paths_h_gamma = self.paths_h_gamma[:size]
            self.paths_c = self.paths_c[:size]
            self.paths_t = self.paths_t[:size]

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
        #                                       low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_h,self.paths_h_rgw,self.paths_h_cont,self.paths_h_gamma, self.paths_c,self.paths_t))
            random.shuffle(paths)
            self.paths_h[:],self.paths_h_rgw[:],self.paths_h_cont[:],self.paths_h_gamma[:], self.paths_c[:], self.paths_t[:] = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths = self.paths_h
        self.Hrgw_paths = self.paths_h_rgw
        self.Hcont_paths = self.paths_h_cont
        self.Hgamma_paths = self.paths_h_gamma
        self.C_paths = self.paths_c
        self.T_paths = self.paths_t

    # def data_synthesis(self, t_img, r_img):
    #     if self.enable_transforms:
    #         t_img, r_img = paired_data_transforms(t_img, r_img)
    #     syn_model = self.syn_model
    #     t_img, r_img, m_img = syn_model(t_img, r_img)
    #
    #     B = to_tensor(t_img)
    #     R = to_tensor(r_img)
    #     M = to_tensor(m_img)
    #
    #     return B, R, M

    def __getitem__(self, index):
        index_H = index % len(self.H_paths)
        index_C = index % len(self.C_paths)
        index_T = index % len(self.T_paths)

        H_path = self.H_paths[index_H]
        Hrgw_path = self.Hrgw_paths[index_H]
        Hcont_path = self.Hcont_paths[index_H]
        Hgamma_path = self.Hgamma_paths[index_H]
        C_path = self.C_paths[index_C]
        T_path = self.T_paths[index_T]
        xr = random.randint(0,364)
        yr = random.randint(0,204)
        h_img = Image.open(H_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        hrgw_img = Image.open(Hrgw_path).crop((xr, yr, xr + 256, yr + 256)).convert('RGB')
        hcont_img = Image.open(Hcont_path).crop((xr, yr, xr + 256, yr + 256)).convert('RGB')
        hgamma_img = Image.open(Hgamma_path).crop((xr, yr, xr + 256, yr + 256)).convert('RGB')
        c_img = Image.open(C_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        t_img = Image.open(T_path).crop((xr,yr,xr+256,yr+256)).convert('RGB')
        r = random.random()
        if r>0.5:
            h_img = h_img.transpose(Image.FLIP_LEFT_RIGHT)
            hrgw_img = hrgw_img.transpose(Image.FLIP_LEFT_RIGHT)
            hcont_img = hcont_img.transpose(Image.FLIP_LEFT_RIGHT)
            hgamma_img = hgamma_img.transpose(Image.FLIP_LEFT_RIGHT)
            c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
            t_img = t_img.transpose(Image.FLIP_LEFT_RIGHT)
        H = to_tensor(h_img)
        Hrgw = to_tensor(hrgw_img)
        Hcont = to_tensor(hcont_img)
        Hgamma = to_tensor(hgamma_img)
        H=torch.cat([H,Hrgw,Hcont,Hgamma],0)
        C = to_tensor(c_img)
        T = to_tensor(t_img)

        fn = os.path.basename(H_path)
        return {'input': H, 'target_t': C, 'target_r': T, 'fn': fn,'j':H_path,'b':C_path,'D':T_path}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.H_paths), len(self.C_paths)), self.size)
        else:
            return max(len(self.H_paths), len(self.C_paths))
class OTSDataset(BaseDataset):
    def __init__(self, datadir_h, datadir_c, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(OTSDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.enable_transforms = enable_transforms
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=35), key=lambda x: x.split('/')[-1].split('.')[0])

        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_c = self.paths_c[:size]

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
        #                                       low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_h, self.paths_c))
            random.shuffle(paths)
            self.paths_h[:], self.paths_c[:] = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths = self.paths_h
        self.C_paths = self.paths_c

    # def data_synthesis(self, t_img, r_img):
    #     if self.enable_transforms:
    #         t_img, r_img = paired_data_transforms(t_img, r_img)
    #     syn_model = self.syn_model
    #     t_img, r_img, m_img = syn_model(t_img, r_img)
    #
    #     B = to_tensor(t_img)
    #     R = to_tensor(r_img)
    #     M = to_tensor(m_img)
    #
    #     return B, R, M

    def __getitem__(self, index):
        index_H = index % len(self.H_paths)
        index_C = index % len(self.C_paths)

        H_path = self.H_paths[index_H]
        C_path = self.C_paths[index_C]

        h_image=Image.open(H_path)
        c_image=Image.open(C_path)
        ww,hh=h_image.size
        if ww<256:
            h_image=h_image.resize((256,hh))
            c_image = c_image.resize((256,hh))
            ww=256
        if hh < 256:
            h_image = h_image.resize((ww, 256))
            c_image = c_image.resize((ww, 256))
            hh = 256
        xr = random.randint(0, ww-256)
        yr = random.randint(0, hh-256)

        h_img = h_image.crop((xr,yr,xr+256,yr+256)).convert('RGB')
        c_img = c_image.crop((xr,yr,xr+256,yr+256)).convert('RGB')
        r = random.random()
        if r>0.5:
            h_img = h_img.transpose(Image.FLIP_LEFT_RIGHT)
            c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
        H = to_tensor(h_img)
        C = to_tensor(c_img)

        fn = os.path.basename(H_path)
        return {'input': H, 'target_t': C, 'target_r': C, 'fn': fn,'j':H_path,'b':C_path,'D':C_path}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.H_paths), len(self.C_paths)), self.size)
        else:
            return max(len(self.H_paths), len(self.C_paths))
class SOTSODTestDataset(BaseDataset):
    def __init__(self, datadir_h,datadir_c, resize=False,fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None):
        super(SOTSODTestDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.fns = fns or os.listdir(datadir_h)
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.resize=resize
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        # pdb.set_trace()
        self.paths_c=[x.replace(x.split('/')[-1],x.split('/')[-1].split('_')[0]+'.png').replace(datadir_h,datadir_c) for x in self.paths_h]
        # self.paths_c = sorted(make_dataset(datadir_c, fns), key=lambda x: x.split('/')[-1].split('.')[0])

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        h_img = Image.open(self.paths_h[index]).convert('RGB')
        c_img = Image.open(self.paths_c[index]).convert('RGB')
        if self.resize:
            w,h=h_img.size
            h_img=h_img.crop((0,0,w//8*8,h//8*8))
            c_img = c_img.crop((0, 0, w // 8 * 8, h // 8* 8))
        # h_img = h_img.resize((640,640))
        # c_img = c_img.resize((640,640))

        H = to_tensor(h_img)
        C = to_tensor(c_img)
        
        
        dic = {'input': H, 'target_t': C, 'fn': fn, 'real': True, 'target_r': C}  # fake reflection gt
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
class SOTSTestDataset(BaseDataset):
    def __init__(self, datadir_h,datadir_c, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None):
        super(SOTSTestDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.fns = fns or os.listdir(datadir_h)
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1].split('.')[0]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])


        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        h_img = Image.open(self.paths_h[index]).convert('RGB')
        c_img = Image.open(self.paths_c[index]).crop((10, 10, 630, 470)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(h_img, c_img, self.unaligned_transforms)

        H = to_tensor(h_img)
        C = to_tensor(c_img)

        dic = {'input': H, 'target_t': C, 'fn': fn, 'real': True, 'target_r': C}  # fake reflection gt
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


    

class SOTS_gfnTestDataset(BaseDataset):
    def __init__(self, datadir_h,datadir_hrgw,datadir_hcont,datadir_hgamma,datadir_c, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None):
        super(SOTS_gfnTestDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_hrgw = datadir_hrgw
        self.datadir_hcont = datadir_hcont
        self.datadir_hgamma = datadir_hgamma
        self.datadir_c = datadir_c
        self.fns = fns or os.listdir(datadir_h)
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1].split('.')[0]))
        self.paths_hrgw = sorted(make_dataset(datadir_hrgw, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1].split('.')[0]))
        self.paths_hcont = sorted(make_dataset(datadir_hcont, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1].split('.')[0]))
        self.paths_hgamma = sorted(make_dataset(datadir_hgamma, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1].split('.')[0]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=lambda x: x.split('/')[-1].split('.')[0])

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        h_img = Image.open(self.paths_h[index]).convert('RGB')
        hrgw_img = Image.open(self.paths_hrgw[index]).convert('RGB')
        hcont_img = Image.open(self.paths_hcont[index]).convert('RGB')
        hgamma_img = Image.open(self.paths_hgamma[index]).convert('RGB')
        c_img = Image.open(self.paths_c[index]).crop((10, 10, 630, 470)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(h_img, c_img, self.unaligned_transforms)

        H = to_tensor(h_img)
        Hrgw = to_tensor(hrgw_img)
        Hcont = to_tensor(hcont_img)
        Hgamma = to_tensor(hgamma_img)
        H=torch.cat([H,Hrgw,Hcont,Hgamma],0)
        C = to_tensor(c_img)

        dic = {'input': H, 'target_t': C, 'fn': fn, 'real': True, 'target_r': C}  # fake reflection gt
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1

        m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        m_img = m_img.resize((512,512))
        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)

class commontestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(commontestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1

        m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class USDataset(BaseDataset):
    def __init__(self, datadir_h, datadir_c, fns=None, size=2061, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(USDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.enable_transforms = enable_transforms
        #        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: x.split('_')[0]+x.split('_')[1])
        self.paths_h = make_dataset(datadir_h, fns)
        self.paths_c = make_dataset(datadir_c, fns)
        random.shuffle(self.paths_h)
        random.shuffle(self.paths_c)
        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_c = self.paths_c[:size]

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
        #                                       low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_h, self.paths_c))
            random.shuffle(paths)
            self.paths_h[:], self.paths_c[:]= zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths = self.paths_h
        self.C_paths = self.paths_c

    def __getitem__(self, index):
        index_H = index % len(self.H_paths)
        index_C = index % len(self.C_paths)

        H_path = self.H_paths[index_H]
        C_path = self.C_paths[index_C]

        h_img = Image.open(H_path).resize((512,512)).convert('RGB')
        c_img = Image.open(C_path).resize((512,512)).convert('RGB')
        r = random.random()
        if r>0.5:
            h_img = h_img.transpose(Image.FLIP_LEFT_RIGHT)
            c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
        H = to_tensor(h_img)
        C = to_tensor(c_img)

        fn = os.path.basename(H_path)
        return {'input': H, 'target_t': C, 'fn': fn}

    def __len__(self):
        return 2061


class demoDataset(BaseDataset):
    def __init__(self, datadir_h, datadir_c, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(demoDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.enable_transforms = enable_transforms
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: (x.split('/')[-1].split('_')[0],x.split('/')[-1].split('_')[1]))
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=35), key=lambda x: x.split('/')[-1].split('.')[0])
        self.paths_h_part1=self.paths_h[:35000]
        self.paths_h_part2 = self.paths_h[35000:70000]
        self.paths_c_part1 = self.paths_c[:35000]
        self.paths_c_part2 = self.paths_c[35000:70000]


        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_c = self.paths_c[:size]

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
        #                                       low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)
    def reset(self, shuffle=True):
        if shuffle:

            paths = list(zip(self.paths_h_part1, self.paths_c_part1))
            random.shuffle(paths)
            self.paths_h_part1[:], self.paths_c_part1[:] = zip(*paths)
            paths = list(zip(self.paths_h_part2, self.paths_c_part2))
            random.shuffle(paths)
            self.paths_h_part2[:], self.paths_c_part2[:] = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths_part1 = self.paths_h_part1
        self.C_paths_part1 = self.paths_c_part1
        self.H_paths_part2 = self.paths_h_part2
        self.C_paths_part2 = self.paths_c_part2



    def __getitem__(self, index):
        index_H = index % len(self.H_paths_part1)
        index_C = index % len(self.C_paths_part1)

        H_path_part1 = self.H_paths_part1 [index_H]
        C_path_part1 = self.C_paths_part1 [index_C]
        H_path_part2 = self.H_paths_part2[index_H]
        C_path_part2 = self.C_paths_part2[index_C]

        h_image_part1=Image.open(H_path_part1)
        c_image_part1=Image.open(C_path_part1)
        h_image_part2 = Image.open(H_path_part2)
        c_image_part2 = Image.open(C_path_part2)

        h_image_part1=h_image_part1.resize((512,512)).convert('RGB')
        c_image_part1=c_image_part1.resize((512,512)).convert('RGB')

        h_image_part2=h_image_part2.resize((512,512)).convert('RGB')

        c_image_part2=c_image_part2.resize((512,512)).convert('RGB')



        H1 = to_tensor(h_image_part1)
        C1 = to_tensor(c_image_part1)
        H2 = to_tensor(h_image_part2)
        C2 = to_tensor(c_image_part2)

        return {'H1': H1, 'C1': C1, 'H2': H2, 'C2': C2,'H_path_part1':H_path_part1,'C_path_part1':C_path_part1,'H_path_part2':H_path_part2,'C_path_part2':C_path_part2}

    def __len__(self):
        if self.size is not None:
            return self.size
        return 35000

class dadehazeDataset(BaseDataset):
    def __init__(self,synpath, realpath, size=None,full_real=True):
        super(dadehazeDataset).__init__()
        self.paths_syn = sorted(make_dataset(synpath), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.paths_real = sorted(make_dataset(realpath), key=lambda x: os.path.splitext(os.path.basename(x))[0])
        self.paths_syn = self.paths_syn
        self.size = size
        self.full_real = full_real
        # import ipdb;ipdb.set_trace()

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths_syn)
            random.shuffle(self.paths_real)
        if self.size is not None:
            self.s_paths = self.paths_syn[:self.size]
            self.rH_paths = self.paths_real[:self.size]

    def __getitem__(self, index):
        index_s = index % len(self.s_paths)
        index_r = index % len(self.rH_paths)

        s_path = self.s_paths[index_s]
        rH_path = self.rH_paths[index_r]
        s_img = Image.open(s_path).convert('RGB')

        xr = random.randint(0,134)
        yr = random.randint(0,134)
        sh_img = s_img.crop((xr, yr, xr+256, yr+256))
        sc_img = s_img.crop((400+xr, yr, 400+xr+256, yr+256))
        rh_img = Image.open(rH_path).resize((400, 400)).convert('RGB')
        rh_img = rh_img.crop((xr, yr, xr+256, yr+256))
        rc_img = s_img.crop((400, 0, 800, 400)).convert('RGB')
        rc_img = rc_img.crop((xr, yr, xr+256, yr+256))

        r = random.random()
        if r > 0.5:
            sh_img = sh_img.transpose(Image.FLIP_LEFT_RIGHT)
            sc_img = sc_img.transpose(Image.FLIP_LEFT_RIGHT)
            rh_img = rh_img.transpose(Image.FLIP_LEFT_RIGHT)
            rc_img = rc_img.transpose(Image.FLIP_LEFT_RIGHT)
        sH = to_tensor(sh_img)
        sC = to_tensor(sc_img)
        rH = to_tensor(rh_img)
        rC = to_tensor(rc_img)
        #
        # import pdb;pdb.set_trace()
        return {'input_syn': sH, 'target_syn': sC, 'input_real': rH, 'target_real': rC}
    def __len__(self):
        if self.size is not None:
            return self.size
        return 1004

class enhanceDataset():
    def __init__(self,synpath, size=None,full_real=True):
        super(dadehazeDataset).__init__()
        self.paths_syn = sorted(make_dataset(synpath), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.paths_syn = self.paths_syn
        # import ipdb;ipdb.set_trace()

        self.s_paths = self.paths_syn

    def __getitem__(self, index):
        index_s = index % len(self.s_paths)

        s_path = self.s_paths[index_s]
        s_img = Image.open(s_path).convert('RGB')


        sh_img = s_img.crop((0, 0, 400, 400))
        sc_img = s_img.crop((400, 0, 800, 400))




        sH = to_tensor(sh_img)
        sC = to_tensor(sc_img)

        #
        # import pdb;pdb.set_trace()
        return {'input_syn': sH, 'target_syn': sC, 'fn': s_path.split('/')[-1]}
    def __len__(self):
        return len(self.paths_syn)

    def __init__(self, datadir_h,datadir_c):
        super(hazerdDataset, self).__init__()
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        sortkey = lambda key: os.path.split(key)[-1]
        # pdb.set_trace()
        self.paths_h=glob.glob(os.path.join(datadir_h, '*.jpg'))
        self.paths_c=[]
        for path in self.paths_h:
            filename=os.path.splitext(os.path.basename(path))[0]
            idx=-1
            while(filename[idx]!='_'):
                idx-=1
            out=os.path.join(datadir_c,filename[:idx]+'_RGB.jpg')

            self.paths_c.append(out)
        # import ipdb;ipdb.set_tr
        # ace()
    def __getitem__(self, index):
        h_img = Image.open(self.paths_h[index]).resize((512,512)).convert('RGB')
        c_img = Image.open(self.paths_c[index]).resize((512,512)).convert('w')
        # if self.resize:
        #     w,h=h_img.size
        #     h_img=h_img.crop((0,0,w//8*8,h//8*8))
        #     c_img = c_img.crop((0, 0, w // 8 * 8, h // 8* 8))

        H = to_tensor(h_img)
        C = to_tensor(c_img)
        # print(self.paths_h[index]+"    "+self.paths_c[index])
        dic = {'input': H, 'target_t': C}
        return dic

    def __len__(self):

        return len(self.paths_h)

    def __init__(self, datadir_h,datadir_c):
        super(hazerdtrainDataset, self).__init__()
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        sortkey = lambda key: os.path.split(key)[-1]
        # pdb.set_trace()
        self.paths_h=glob.glob(os.path.join(datadir_h, '*.jpg'))
        self.paths_c=[]
        for path in self.paths_h:
            filename=os.path.splitext(os.path.basename(path))[0]
            idx=-1
            while(filename[idx]!='_'):
                idx-=1
            out=os.path.join(datadir_c,filename[:idx]+'_RGB.jpg')

            self.paths_c.append(out)
        # import ipdb;ipdb.set_tr
        # ace()
    def __getitem__(self, index):
        h_img = Image.open(self.paths_h[index]).resize((600,600)).convert('RGB')
        c_img = Image.open(self.paths_c[index]).resize((600,600)).convert('RGB')
        # if self.resize:
        #     w,h=h_img.size
        #     h_img=h_img.crop((0,0,w//8*8,h//8*8))
        #     c_img = c_img.crop((0, 0, w // 8 * 8, h // 8* 8))
        xr2 = random.randint(0, 200)
        yr2 = random.randint(0, 200)
        h_img = h_img.crop((xr2, yr2, xr2 + 400, yr2 + 400))
        c_img = c_img.crop((xr2, yr2, xr2 + 400, yr2 + 400))
        r = random.random()

        if r > 0.5:
            h_img = h_img.transpose(Image.FLIP_LEFT_RIGHT)
            c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
        H = to_tensor(h_img)
        C = to_tensor(c_img)
        # print(self.paths_h[index]+"    "+self.paths_c[index])
        dic = {'input_syn': H, 'target_syn': C,'input_real': H, 'target_real': C}
        return dic

    def __len__(self):

        return len(self.paths_h)

    


def reside_helper(path_h,clear_base):

    fn=path_h.split('/')[-1]
    ext=fn.split('.')[-1]
    fn=fn.split('_')[0]
    return os.path.join(clear_base,fn+'.'+ext)


    def __init__(self,datadir_h,datadir_c,datadir_syn_h,datadir_syn_c):
        self.paths_h=glob.glob(os.path.join(datadir_h,'*'))
        self.paths_c=[path.replace(datadir_h,datadir_c) for path in self.paths_h]
        self.paths_syn_h=glob.glob(os.path.join(datadir_syn_h,'*'))
        self.paths_syn_c=[reside_helper(x,datadir_syn_c) for x in self.paths_syn_h]
        self.paths_h.extend(self.paths_syn_h)
        self.paths_c.extend(self.paths_syn_c)
        import random
        paths = list(zip(self.paths_h, self.paths_c))
        random.shuffle(paths)
        self.paths_h[:], self.paths_c[:] = zip(*paths)



        self.transform = A.Compose(  # FRCNN
        [
            A.SmallestMaxSize(600, p=1.0),  # resize
            # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),

            A.RandomCrop(height=512, width=512, p=1.0),  # 600
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
            #                             val_shift_limit=0.3, p=0.95),
            #     A.RandomBrightnessContrast(brightness_limit=0.3,
            #                                 contrast_limit=0.3, p=0.95),
            # ],p=1.0),
            # A.ToGray(p=0.01),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Normalize(max_pixel_value=1., p=1.0),
            # ToTensorV2(p=1.0),
        ],
        p=1.0,
        additional_targets = {'image_c': 'image'}
        )
        self.transform_input = A.Compose(  # FRCNN
            [
                A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                ToTensorV2(p=1.0),
            ]
        )
        self.transform_target = A.Compose(  # FRCNN
            [
                ToTensorV2(p=1.0),
            ]
        )

    def __getitem__(self,index):
        img_h = cv2.imread(self.paths_h[index])        
        img_h = cv2.cvtColor(img_h,cv2.COLOR_BGR2RGB)
        img_h = img_h.astype('single')/255.0
        img_c = cv2.imread(self.paths_c[index])        
        img_c = cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB)
        img_c = img_c.astype('single')/255.0
        sample = self.transform(**{'image':img_h,'image_c':img_c})
        sample_input = self.transform_input(**{'image': sample['image']})
        sample_target = self.transform_target(**{'image': sample['image_c']})
        return {'input_syn': sample_input['image'], 'target_syn': sample_target['image'], 'C_paths':self.paths_h[index].split('/')[-1]}
    
    def __len__(self):
        return len(self.paths_c)



    def __init__(self,datadir_syn_h,datadir_syn_c):

        self.paths_h=glob.glob(os.path.join(datadir_syn_h,'*'))
        self.paths_c=[reside_helper(x,datadir_syn_c) for x in self.paths_h]

        import random
        paths = list(zip(self.paths_h, self.paths_c))
        random.shuffle(paths)
        self.paths_h[:], self.paths_c[:] = zip(*paths)



        self.transform = A.Compose(  # FRCNN
        [
            A.SmallestMaxSize(600, p=1.0),  # resize
            # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),

            A.RandomCrop(height=512, width=512, p=1.0),  # 600
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
            #                             val_shift_limit=0.3, p=0.95),
            #     A.RandomBrightnessContrast(brightness_limit=0.3,
            #                                 contrast_limit=0.3, p=0.95),
            # ],p=1.0),
            # A.ToGray(p=0.01),
            # A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Normalize(max_pixel_value=1., p=1.0),
            # ToTensorV2(p=1.0),
        ],
        p=1.0,
        additional_targets = {'image_c': 'image'}
        )
        self.transform_input = A.Compose(  # FRCNN
        [
            A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
            ToTensorV2(p=1.0),
        ]
        )
        self.transform_target = A.Compose(  # FRCNN
        [
            ToTensorV2(p=1.0),
        ]
        )


    def __getitem__(self,index):
        img_h = cv2.imread(self.paths_h[index])        
        img_h = cv2.cvtColor(img_h,cv2.COLOR_BGR2RGB)
        img_h = img_h.astype('single')/255.0
        img_c = cv2.imread(self.paths_c[index])        
        img_c = cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB)
        img_c = img_c.astype('single')/255.0
        sample = self.transform(**{'image':img_h,'image_c':img_c})
        sample_input = self.transform_input(**{'image':sample['image']})
        sample_target = self.transform_target(**{'image':sample['image_c']})
        return {'input_syn': sample_input['image'], 'target_syn': sample_target['image'], 'C_paths':self.paths_h[index].split('/')[-1]}
    
    def __len__(self):
        return len(self.paths_c)


    


