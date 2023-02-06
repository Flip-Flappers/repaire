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

from train_color_dector import models

parser = argparse.ArgumentParser(description='1')

parser.add_argument('--label', type=int)
args = parser.parse_args()
label = args.label
print(label)

writer = SummaryWriter('./runs/' + str(label))
root = '../../fin_dataset/cifar10/train/'
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
        tmp = mask_loc[61:][:-4].split('_')
        avg = tmp[0][1:][:-1].split('.')[:-1]
        ans_list[num][0] = ((torch.tensor(int(avg[0]))) / 255 - 0.5) * 2
        ans_list[num][1] = ((torch.tensor(int(avg[1]))) / 255 - 0.5) * 2
        ans_list[num][2] = ((torch.tensor(int(avg[2]))) / 255 - 0.5) * 2
        num += 1
torch.save(image_list, './color_image_list_' + str(label) + '.t')
torch.save(ans_list, './color_ans_list_' + str(label) + '.t')"""

image_list = torch.load('./color_image_list_' + str(label) + '.t')
ans_list = torch.load('./color_ans_list_' + str(label) + '.t')
dataset = torch.utils.data.TensorDataset(image_list, ans_list)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=500, shuffle=True)

color_net = models.color_net().cuda()
#color_net.load_state_dict(torch.load('./color_net_0899.pth'))
optimizer = torch.optim.Adam(color_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()
step = 0
for epoch in tqdm(range(10000)):
    tmp = 0
    for image, ans in tqdm(dataset_loader):
        image = image.permute([1, 0, 2, 3, 4])
        ori_image = image[0]
        mask_image = image[1]
        ori_image = ori_image.cuda()
        mask_image = mask_image.cuda()
        output = color_net(ori_image, mask_image)
        ans = ans.cuda()
        loss = loss_fn(output, ans)
        tmp += loss.detach() * image.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    writer.add_scalar('color_loss', (torch.sqrt(tmp / num))* 255 * 2, step)
    writer.flush()
    step += 1
    if epoch % 10:
        torch.save(color_net.state_dict(), './color_net_0' + str(epoch) + '.pth')

