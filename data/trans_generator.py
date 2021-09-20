import os
import glob
import h5py
import cv2
import numpy as np
ots_root = '/media/windows/c/datasets/RESIDE/OTS'
hazepaths = glob.glob(os.path.join(ots_root,'hazy/*'))
for path in hazepaths:
    filename = os.path.splitext(os.path.basename(path))[0]
    info = filename.split('_')
    id = info[0]
    beta = float(info[2])
    depth_path = os.path.join(os.path.join(ots_root,'depth'), id+'.mat')
    output_path = os.path.join(os.path.join(ots_root,'trans'), filename+'.png')
    clear_path = os.path.join(os.path.join(ots_root,'clear'), id+'.jpg')
    clear_img = cv2.imread(clear_path)
    depth = h5py.File(depth_path)['depth'].value.transpose()
    depths = np.stack([depth for _ in range(3)], 2)
    trans = np.exp(-beta*depths)
    cv2.imwrite(output_path,trans*255)
    import ipdb;ipdb.set_trace()
    print(filename)
