import argparse
import itertools

import numpy as np
import torchvision
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
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack
from train_color_dector import Pconv_models

parser = argparse.ArgumentParser(description='1')
parser.add_argument('--label', type=int)
parser.add_argument('--attack', type=str)
parser.add_argument('--target', type=bool)

args = parser.parse_args()
label = args.label
attack = args.attack
target = args.target
num = 0

trainset = torchvision.datasets.CIFAR10(root='../../root_data', train=False, download=True,
                                        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)


root = '../../fin_dataset/cifar10/test/'
ori_picture = sorted(glob.glob(os.path.join(root, 'ori_image/' + str(label)) + '/*.*'))
ori_picture_num = len(ori_picture)


checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")

net_R = torch.nn.DataParallel(checkpoint).cuda()
net_R.eval()



for data in trainloader:
    ori_tensor, labels = data

    true_label = net_R(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(ori_tensor.cuda()))[0].argmax()

    if true_label == labels[0] and labels[0] == 0:
        s = "{:04d}".format(num)
        image = transforms.ToPILImage()(ori_tensor.squeeze(0))

        image.save(root + 'ori_image/' + str(0) + '/' + s + '.png')
        num += 1