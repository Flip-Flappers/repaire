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


root = '../../fin_dataset/cifar10/test/fgsm'
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
    plt.figure()
    ori_loc = ori_picture[i]

    ori_pic = Image.open(ori_loc)
    plt.subplot(1, 2, 1)
    plt.imshow(ori_pic)

    ori_tensor = transforms.ToTensor()(ori_pic)
    for j in range(len(mask[i])):
        mask_loc = mask[i][j]
        mask_pic = Image.open(mask_loc)

        mask_tensor_pre = transforms.ToTensor()(mask_pic)
        mask_tensor = torch.cat([mask_tensor_pre, mask_tensor_pre], dim = 0)
        mask_tensor = torch.cat([mask_tensor, mask_tensor_pre], dim = 0)
        anti_mask_image = 1 - mask_tensor
        ori_tensor = ori_tensor * anti_mask_image

        tmp = mask_loc[65:][:-4].split('_')
        avg = tmp[0][1:][:-1].split('.')[:-1]
        ans_list[num][0] = (torch.tensor(int(avg[0]))) / 255
        ans_list[num][1] = (torch.tensor(int(avg[1]))) / 255
        ans_list[num][2] = (torch.tensor(int(avg[2]))) / 255

        mask_tensor = torch.cat([mask_tensor_pre * ans_list[num][0], mask_tensor_pre * ans_list[num][1]], dim=0)
        mask_tensor = torch.cat([mask_tensor, mask_tensor_pre * ans_list[num][2]], dim=0)
        ori_tensor = mask_tensor + ori_tensor
    image = transforms.ToPILImage()(ori_tensor)

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()
    num += 1

