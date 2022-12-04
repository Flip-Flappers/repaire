import time
from net_T.mnist import mnist
import PIL

from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

# fgsm
from torch import nn
import argparse
import os
import gc
import sys

import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack

import torch
import torchvision
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
import matplotlib.pyplot as plt

def ans_mnist_fgsm(transform_dict, net_R, net_T, attack, target, numsss):
    fight_correct_ori_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    disssss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for bb in range(numsss):
        # start_time = time.time()
        if attack == 'BIM':
            adversary = LinfBasicIterativeAttack(
                net_R,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=255 / 255,
                nb_iter=100, eps_iter=255 / 255 / 100, clip_min=0, clip_max=1.0,
                targeted=target)
            # PGD
        elif attack == 'PGD':
            if target:
                adversary = PGDAttack(
                    net_R,
                    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=150 / 255,
                    nb_iter=10, eps_iter=150 / 255 / 5, clip_min=0, clip_max=1.0,
                    targeted=target)
            else:
                adversary = PGDAttack(
                    net_R,
                    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=150 / 255,
                    nb_iter=10, eps_iter=150 / 255 / 5, clip_min=0, clip_max=1.0,
                    targeted=target)
            # FGSM
        elif attack == 'FGSM':
            adversary = GradientSignAttack(
                net_R,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=150 / 255,
                targeted=target, clip_min=0, clip_max=1)
        elif attack == 'CW':
            adversary = CarliniWagnerL2Attack(
                net_R,
                num_classes=10,
                learning_rate=0.45,
                # loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                binary_search_steps=10,
                max_iterations=12,
                targeted=target, clip_min=-1.0, clip_max=1.0)

        correct = 0
        mnist_tester = torchvision.datasets.MNIST('../../root_data', transform=transform_dict['tar_s'], train=False)
        mnist_tester_loader = torch.utils.data.DataLoader(dataset=mnist_tester, batch_size=1000, shuffle=False)

        net_R.eval()
        dis = 0
        ori_correct = 0
        fight_correct = 0
        for mnist_images, mnist_labels in mnist_tester_loader:
            mnist_images = mnist_images.cuda()
            mnist_labels = mnist_labels.cuda()
            if target:
                tmp_labels = torch.randint(0, 10, (mnist_images.shape[0],)).cuda()
                for kk in range(tmp_labels.shape[0]):
                    tmp_labels[kk] = 1
                final_image_tmp = adversary.perturb(mnist_images, tmp_labels)
            else:
                final_image_tmp = adversary.perturb(mnist_images, mnist_labels)
            cha = final_image_tmp - mnist_images
            cha = (cha * 255 / 255 / 2 + 0.5) * 255
            for kk in range(mnist_images.shape[0]):
                now_ori = mnist_images[kk]
                now_att = cha[kk]
                image_ori = PIL.Image.fromarray(
                    torch.clamp(now_ori * 255, min=0, max=255).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())
                image_att = PIL.Image.fromarray(
                    torch.clamp(now_att, min=0, max=255).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())
                fin_att =  PIL.Image.fromarray(
                    torch.clamp(final_image_tmp[kk] * 255, min=0, max=255).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())

                plt.figure()

                subplot = plt.subplot(1, 3, 1)
                plt.imshow(image_ori)
                subplot = plt.subplot(1, 3, 2)
                plt.imshow(image_att)
                subplot = plt.subplot(1, 3, 3)
                plt.imshow(fin_att)
                plt.show()
                print(tmp_labels[kk])
            print(cha)


transform_dict = {
    'tar': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'tar_s': transforms.Compose([

        transforms.ToTensor(),

    ]),
    'tar_dis': transforms.Compose([

        transforms.ToTensor(),

    ]),

}

net_T = mnist(pretrained=True).cuda()

net_T.eval()
net_R = mnist(pretrained=True).cuda()
net_R.eval()

mnist_tester = torchvision.datasets.MNIST('../../root_data', transform=transform_dict['tar_s'], train=False)
mnist_tester_loader = torch.utils.data.DataLoader(dataset=mnist_tester, batch_size=1000, shuffle=True)

with torch.no_grad():
    correct = 0
    for mnist_images, mnist_labels in mnist_tester_loader:
        mnist_images = mnist_images.cuda()
        mnist_labels = mnist_labels.cuda()
        outputss = net_R(mnist_images)
        predicteds = outputss.argmax(dim=1)
        correct += torch.eq(predicteds, mnist_labels).sum().float().item()

    print(correct)


print("bim")
print(ans_mnist_fgsm(transform_dict, net_R, net_T, "BIM", True, 10))
print("pgd")
print(ans_mnist_fgsm(transform_dict, net_R, net_T, "PGD", True, 10))
