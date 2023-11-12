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
ori_num_train = [4990, 4996, 4981, 4978, 4991, 4980, 4991, 4998, 4995, 4996]
fo = open("./train_" + ".flist", "w")
for label in range(10):
    flist = []






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

        s = "{:04d}".format(i)
        sj = "{:04d}".format(i)
        path = '../../../fin_dataset/cifar10/train/ori_image/' + str(label) + '/' + s +'.png' + ',' + '../../../fin_dataset/cifar10/train/gradcam_image/' + str(label) + '/' + s  + '/' + sj +'.png' + ',' + sj + '.png'
        fo.write(path + '\n')
        # flist.append(path)

    # print(flist)
fo.close()