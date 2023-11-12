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

parser.add_argument('--attack', type=str)


args = parser.parse_args()

attack = args.attack
target = False
num = 0

trainset = torchvision.datasets.CIFAR10(root='../../root_data', train=False, download=True,
                                        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)


root = '../../fin_dataset/cifar10/test/'


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
            eps=4 / 255,
            nb_iter=10, eps_iter=4 / 255 / 5, clip_min=0, clip_max=1.0,
            targeted=False)
    else:
        adversary = PGDAttack(
            net_R,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=8 / 255,
            nb_iter=20, eps_iter=2 / 255, clip_min=0, clip_max=1.0,
            targeted=False)
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

num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for data in tqdm(trainloader):
    ori_tensor, labels = data

    true_label = net_R(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(ori_tensor.cuda()))[0].argmax()
    final_image_tmp = adversary.perturb(ori_tensor.cuda(), torch.tensor(int(labels[0])).unsqueeze(0).cuda())
    fgsm_image = transforms.ToPILImage()(final_image_tmp.squeeze(0))
    tmp_fgsm_image = transforms.ToTensor()(fgsm_image).unsqueeze(0)
    fin_label = net_R(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tmp_fgsm_image))[0].argmax()
    if fin_label != true_label:
        s = "{:04d}".format(num[true_label])
        fgsm_image.save(root + 'pgd/success/' + str(int(true_label)) + '/' + s + '.png')
        num[true_label] += 1
    else:
        s = "{:04d}".format(num2[true_label])
        fgsm_image.save(root + 'pgd/fail/' + str(int(true_label)) + '/' + s + '.png')
        num2[true_label] += 1