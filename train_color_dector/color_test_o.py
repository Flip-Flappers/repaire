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

parser = argparse.ArgumentParser(description='1')

parser.add_argument('--label', type=int)
args = parser.parse_args()
label = args.label
print(label)


root = '../../fin_dataset/cifar10/test'
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
color_net = o_color_net_maker.color_net().cuda()
color_net.load_state_dict(torch.load('./color_net_06160.pth'))
color_net.eval()
haha = 0
rate = 0
for i in tqdm(range(ori_picture_num)):
    error = 0
    color_tmp = 0
    for j in range(len(mask[i])):
        ori_loc = ori_picture[i]
        mask_loc = mask[i][j]
        ori_pic = Image.open(ori_loc)
        mask_pic = Image.open(mask_loc)
        ori_tensor = transforms.ToTensor()(ori_pic)
        mask_tensor_pre = transforms.ToTensor()(mask_pic)
        mask_tensor = torch.cat([mask_tensor_pre, mask_tensor_pre], dim = 0)
        mask_tensor = torch.cat([mask_tensor, mask_tensor_pre], dim = 0)
        anti_mask_image = 1 - mask_tensor
        ori_tensor = ori_tensor * anti_mask_image
        ori_tensor = transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))(ori_tensor)
        mask_tensor = transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))(mask_tensor)
        image_list[num, 0] = ori_tensor
        image_list[num, 1] = mask_tensor

        tmp = mask_loc[60:][:-4].split('_')
        avg = tmp[0][1:][:-1].split('.')[:-1]
        ans_list[num][0] = ((torch.tensor(int(avg[0]))) / 255 - 0.5) * 2
        ans_list[num][1] = ((torch.tensor(int(avg[1]))) / 255 - 0.5) * 2
        ans_list[num][2] = ((torch.tensor(int(avg[2]))) / 255 - 0.5) * 2

        output = color_net(ori_tensor.unsqueeze(0).cuda(), mask_tensor.unsqueeze(0).cuda()).detach()
        loss = abs(output[0] - ans_list[num].cuda()).sum() / 3 * 255 / 2
        color_tmp += loss * (mask_tensor / 2 + 0.5).sum()
        if loss >= 15:
            error += 1
        num += 1
    if error / len(mask[i]) >= 0.4:
        haha += 1
    rate += error / len(mask[i])
    print(i, haha, color_tmp)





