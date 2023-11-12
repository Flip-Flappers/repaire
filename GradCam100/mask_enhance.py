import argparse
import itertools

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from tqdm import tqdm
from torch import nn as nn

from datasets import ImageDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms as transforms
import glob
import random
import os
from P_models import color_net as p_color_net_maker
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from train_color_dector import models as o_color_net_maker
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='1')

parser.add_argument('--label', type=int)
args = parser.parse_args()
label = args.label
print(label)


root = '../../fin_dataset/cifar10/train'
ori_picture = sorted(glob.glob(os.path.join(root, 'ori_image/' + str(label)) + '/*.*'))
ori_picture_num = len(ori_picture)
mask = []
for i in range(ori_picture_num):
    s = "{:04d}".format(i)
    mask.append(sorted(glob.glob(os.path.join(root, 'color_edge_image/' + str(label) + '/' + s) + '/*.*')))

num = 0
for i in tqdm(range(ori_picture_num)):
    for j in range(len(mask[i])):
        num += 1
image_list = torch.zeros([num, 2, 3, 32, 32])
ans_list = torch.zeros([num, 3])
num = 0
for i in tqdm(range(ori_picture_num)):

    ori_loc = ori_picture[i]

    ori_pic = Image.open(ori_loc)


    ori_tensor = transforms.ToTensor()(ori_pic)
    for j in range(len(mask[i])):
        mask_loc = mask[i][j]
        mask_pic = Image.open(mask_loc)

        mask_tensor = transforms.ToTensor()(mask_pic)
        dist = [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1], [-1, 1], [1, -1]]
        for kk in range(32):
            for jj in range(32):
                if mask_tensor[0][kk][jj] == 1:
                    for zz in range(8):
                        new_x = kk + dist[zz][0]
                        new_y = jj + dist[zz][1]
                        if new_x < 0 or new_y < 0 or new_x >= 32 or new_y >= 32 or mask_tensor[0][new_x][new_y] == 1:
                            continue
                        mask_tensor[0][new_x][new_y] = 0.5

        for ll in range(1):
            visit = torch.zeros(mask_tensor.shape)
            for kk in range(32):
                for jj in range(32):
                    if visit[0][kk][jj] == 0:
                        if mask_tensor[0][kk][jj] == 1 or mask_tensor[0][kk][jj] == 0.5:
                            for zz in range(8):
                                new_x = kk + dist[zz][0]
                                new_y = jj + dist[zz][1]
                                if new_x < 0 or new_y < 0 or new_x >= 32 or new_y >= 32 or mask_tensor[0][new_x][new_y] == 1 or mask_tensor[0][new_x][new_y] == 0.5:
                                    continue
                                mask_tensor[0][new_x][new_y] = 0.5
                                visit[0][new_x][new_y] = 1
        enhance_mask_image = transforms.ToPILImage()(mask_tensor)
        sp = "{:04d}".format(i)
        if not os.path.exists(root + '/enhance_color_edge_image/' + str(label) + '/' + sp):
            os.makedirs(root + '/enhance_color_edge_image/' + str(label) + '/' + sp)

        enhance_mask_image.save(root + '/enhance_color_edge_image/' + str(label) + mask_loc[50:])


