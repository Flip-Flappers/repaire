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

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
root = '../../color_detector_dataset/train/0'
ori_picture = sorted(glob.glob(os.path.join(root, 'ori_picture') + '/*.*'))
mask = []
for i in range(4500):
    s = "{:04d}".format(i)
    mask.append(sorted(glob.glob(os.path.join(root, 'mask/%s' % s) + '/*.*')))

num = 0
for i in tqdm(range(4500)):
    for j in range(len(mask[i])):
        num += 1
image_list = torch.zeros([2, num, 3, 32, 32])
ans_list = torch.zeros([num, 10])
num = 0
for i in tqdm(range(4500)):
    for j in range(len(mask[i])):
        ori_loc = ori_picture[i]
        mask_loc = mask[i][j]
        ori_pic = Image.open(ori_loc)
        mask_pic = Image.open(mask_loc)
        ori_tensor = transforms.ToTensor()(ori_pic)
        mask_tensor = transforms.ToTensor()(mask_pic)
        image_list[0, num] = ori_tensor
        image_list[1, num] = mask_tensor
        tmp = mask_loc[52:][:-4].split('_')
        avg = tmp[0][1:][:-1].split('.')[:-1]
        ans_list[num][0] = torch.tensor(int(avg[0]))
        ans_list[num][1] = torch.tensor(int(avg[1]))
        ans_list[num][2] = torch.tensor(int(avg[2]))
        maxn = tmp[1][1:][:-1].split('.')[:-1]
        ans_list[num][3] = torch.tensor(int(maxn[0]))
        ans_list[num][4] = torch.tensor(int(maxn[1]))
        ans_list[num][5] = torch.tensor(int(maxn[2]))
        minn = tmp[2][1:][:-1].split('.')[:-1]
        ans_list[num][6] = torch.tensor(int(minn[0]))
        ans_list[num][7] = torch.tensor(int(minn[1]))
        ans_list[num][8] = torch.tensor(int(minn[2]))
        num += 1
dataset = torch.utils.data.TensorDataset(image_list, ans_list)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=50, shuffle=True)
for epoch in tqdm(range(10000)):
    for image, ans in dataset_loader:
        image.cuda()
        ans.cuda()
        print(image, ans)