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
"""for i in tqdm(range(ori_picture_num)):
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

        tmp = mask_loc[65:][:-4].split('_')
        avg = tmp[0][1:][:-1].split('.')[:-1]
        ans_list[num][0] = ((torch.tensor(int(avg[0]))) / 255 - 0.5) * 2
        ans_list[num][1] = ((torch.tensor(int(avg[1]))) / 255 - 0.5) * 2
        ans_list[num][2] = ((torch.tensor(int(avg[2]))) / 255 - 0.5) * 2
        num += 1
torch.save(image_list, './test_color_image_list_' + str(label) + '.t')
torch.save(ans_list, './test_color_ans_list_' + str(label) + '.t')"""
image_list = torch.load('./test_color_image_list_' + str(label) + '.t')
ans_list = torch.load('./test_color_ans_list_' + str(label) + '.t')

dataset = torch.utils.data.TensorDataset(image_list, ans_list)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)
color_net = torch.nn.DataParallel(p_color_net_maker())
color_net = torch.nn.DataParallel(o_color_net_maker.color_net2().cuda())
color_net.load_state_dict(torch.load('./color_net_0170.pth'))
color_net.eval()
loss_fn = nn.MSELoss()


tmp = 0
haha = 0
maxn = 0
a = 0
for image, ans in tqdm(dataset_loader):
    shape = image.shape[0]
    image = image.permute([1, 0, 2, 3, 4])
    ori_image = image[0]
    ori_mask_image = image[1].cuda()
    mask_image = 1 - (image[1] / 2 + 0.5)
    ori_image = ori_image.cuda()
    mask_image = mask_image.cuda()
    output = color_net(ori_image, ori_mask_image).detach()
    ans = ans.cuda()
    for kk in range(shape):
        loss = abs(output[kk] - ans[kk]).sum() / 3 * 255 / 2
        tmp += loss
        maxn = max(maxn, loss)
        a += 1
        if loss >= 25:
            haha += 1
    print(tmp / a)

    print(haha)



