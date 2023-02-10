import argparse
import os
import shutil
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
flist = []

ori_num_train = [4998, 4998, 4985, 5007, 5001, 4999, 5002, 5004, 5001, 5005]

fo = open("./train.flist", "w")


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
    for j in range(len(mask[i])):
        s = "{:04d}".format(i)
        if not os.path.exists(root + '/diff_color_edge_image/' + str(label) + '/' + s):
            os.makedirs(root + '/diff_color_edge_image/' + str(label) + '/' + s)
        sj = "{:04d}".format(j)
        image = Image.open(mask[i][j])
        image.save(root + '/diff_color_edge_image/' + str(label) + '/' + s + '/' + sj + '.png')

# print(flist)
fo.close()