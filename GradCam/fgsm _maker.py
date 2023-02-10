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


root = '../../fin_dataset/cifar10/test/'
ori_picture = sorted(glob.glob(os.path.join(root, 'ori_image/' + str(label)) + '/*.*'))
ori_picture_num = len(ori_picture)


checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")

net_R = torch.nn.DataParallel(checkpoint).cuda()
net_R.eval()

if attack == 'BIM':
    adversary = LinfBasicIterativeAttack(
        net_R,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=7 / 255,
        nb_iter=100, eps_iter=255 / 255 / 100, clip_min=0, clip_max=1.0,
        targeted=target)
    # PGD
elif attack == 'PGD':
    if target:
        adversary = PGDAttack(
            net_R,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=7 / 255,
            nb_iter=10, eps_iter=150 / 255 / 5, clip_min=0, clip_max=1.0,
            targeted=target)
    else:
        adversary = PGDAttack(
            net_R,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=7 / 255,
            nb_iter=10, eps_iter=150 / 255 / 5, clip_min=0, clip_max=1.0,
            targeted=target)
    # FGSM
elif attack == 'FGSM':
    adversary = GradientSignAttack(
        net_R,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=4/255,
        targeted=False, clip_min=0, clip_max=1)

elif attack == 'CW':
    adversary = CarliniWagnerL2Attack(
        net_R,
        num_classes=10,
        learning_rate=0.45,
        # loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        binary_search_steps=10,
        max_iterations=12,
        targeted=target, clip_min=-1.0, clip_max=1.0)

loss_fn = nn.CrossEntropyLoss()
step = 0
all_loss = 0

correct = 0
num = 0

for i in tqdm(range(ori_picture_num)):
    s = "{:04d}".format(i)
    ori_loc = ori_picture[i]
    ori_pic = Image.open(ori_loc)
    ori_tensor = transforms.ToTensor()(ori_pic).unsqueeze(0)
    final_image_tmp = adversary.perturb(ori_tensor.cuda(), torch.tensor(label).unsqueeze(0).cuda())
    fgsm_image = transforms.ToPILImage()(final_image_tmp.squeeze(0))
    fgsm_image.save(root + 'fgsm/ori_image/' + str(label) + '/' + s + '.png')

