import time
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

def ans_cifar10_fgsm(transform_dict, net_R, net_T, attack, target, numsss):
    fight_correct_ori_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    disssss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for bb in range(numsss):
        # start_time = time.time()
        if attack == 'BIM':
            adversary = LinfBasicIterativeAttack(
                net_R,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=1 / 255,
                nb_iter=100, eps_iter=1 / 255 / 50 , clip_min=0, clip_max=1.0,
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
        cifar10_tester = torchvision.datasets.CIFAR10('../../root_data', transform=transform_dict['tar'], train=False)
        cifar10_tester_loader = torch.utils.data.DataLoader(dataset=cifar10_tester, batch_size=100, shuffle=False)

        net_R.eval()
        dis = 0
        ori_correct = 0
        fight_correct = 0
        for cifar10_images, cifar10_labels in cifar10_tester_loader:
            cifar10_images = cifar10_images.cuda()
            cifar10_labels = cifar10_labels.cuda()
            if target:
                tmp_labels = torch.randint(0, 10, (cifar10_images.shape[0],)).cuda()
                for kk in range(tmp_labels.shape[0]):
                    tmp_labels[kk] = 0
                final_image_tmp = adversary.perturb(cifar10_images, tmp_labels)
            else:
                final_image_tmp = adversary.perturb(cifar10_images, cifar10_labels)
            cha = transform_dict['unnorm'](final_image_tmp) - transform_dict['unnorm'](cifar10_images)
            for kk in range(cifar10_images.shape[0]):
                now_ori = cifar10_images[kk]
                now_att = cha[kk]
                now_att = now_att / (max(abs(now_att.max()), abs(now_att.min()))) / 2 + 0.5
                image_ori = PIL.Image.fromarray(
                    torch.clamp(transform_dict['unnorm'](now_ori) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
                image_att = PIL.Image.fromarray(
                    torch.clamp((now_att) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
                fin_att =  PIL.Image.fromarray(
                    torch.clamp(transform_dict['unnorm'](final_image_tmp[kk]) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())

                image_att1 = PIL.Image.fromarray(torch.clamp((now_att) * 255, min=0, max=255)[0].unsqueeze(0).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())
                image_att2 = PIL.Image.fromarray(
                    torch.clamp((now_att) * 255, min=0, max=255)[1].unsqueeze(0).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())
                image_att3 = PIL.Image.fromarray(
                    torch.clamp((now_att) * 255, min=0, max=255)[2].unsqueeze(0).byte().permute(1, 2, 0).squeeze(2).cpu().numpy())

                image_ori1 = PIL.Image.fromarray(
                    torch.clamp(transform_dict['unnorm'](now_ori) * 255, min=0, max=255)[0].unsqueeze(0).byte().permute(1, 2,
                                                                                                                  0).squeeze(
                        2).cpu().numpy())
                image_ori2 = PIL.Image.fromarray(
                    torch.clamp(transform_dict['unnorm'](now_ori) * 255, min=0, max=255)[1].unsqueeze(0).byte().permute(1, 2,
                                                                                                                  0).squeeze(
                        2).cpu().numpy())
                image_ori3 = PIL.Image.fromarray(
                    torch.clamp(transform_dict['unnorm'](now_ori) * 255, min=0, max=255)[2].unsqueeze(0).byte().permute(1, 2,
                                                                                                                  0).squeeze(
                        2).cpu().numpy())

                image_ori_one = PIL.Image.fromarray(
                    torch.clamp(transform_dict['tar_one'](transform_dict['unnorm'](now_ori)) * 255, min=0, max=255).byte().permute(
                        1, 2,
                        0).squeeze(
                        2).cpu().numpy())
                image_att_one = PIL.Image.fromarray(
                    torch.clamp(transform_dict['tar_one']((now_att)) * 255, min=0,
                                max=255).byte().permute(
                        1, 2,
                        0).squeeze(
                        2).cpu().numpy())
                fin_att_one = PIL.Image.fromarray(
                    torch.clamp(transform_dict['tar_one'](transform_dict['unnorm'](final_image_tmp[kk])) * 255, min=0,
                                max=255).byte().permute(
                        1, 2,
                        0).squeeze(
                        2).cpu().numpy())

                plt.figure()

                subplot = plt.subplot(1, 12, 1)
                plt.imshow(image_ori)
                subplot = plt.subplot(1, 12, 2)
                plt.imshow(image_ori_one)

                subplot = plt.subplot(1, 12, 3)
                plt.imshow(image_ori1)
                subplot = plt.subplot(1, 12, 4)
                plt.imshow(image_ori2)
                subplot = plt.subplot(1, 12, 5)
                plt.imshow(image_ori3)

                subplot = plt.subplot(1, 12, 6)
                plt.imshow(image_att)
                subplot = plt.subplot(1, 12, 7)
                plt.imshow(image_att_one)
                subplot = plt.subplot(1, 12, 8)
                plt.imshow(image_att1)
                subplot = plt.subplot(1, 12, 9)
                plt.imshow(image_att2)
                subplot = plt.subplot(1, 12, 10)
                plt.imshow(image_att3)
                subplot = plt.subplot(1, 12, 11)
                plt.imshow(fin_att_one)
                subplot = plt.subplot(1, 12, 12)
                plt.imshow(fin_att)
                plt.show()
                print(tmp_labels[kk])
            print(cha)


transform_dict = {
    'tar': transforms.Compose([

        #transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'unnorm': transforms.Compose([
            transforms.Normalize((-0.485 / 0.229, - 0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))
    ]),
    'tar_s':transforms.Compose([
        transforms.ToTensor(),
   ]),
    'tar_one':transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
   ])
}


checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")
net_T = torch.nn.DataParallel(checkpoint).cuda()
net_T.eval()


checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")
net_R = torch.nn.DataParallel(checkpoint).cuda()
net_R.eval()




print("bim")
print(ans_cifar10_fgsm(transform_dict, net_R, net_T, "BIM", True, 10))
print("pgd")
print(ans_cifar10_fgsm(transform_dict, net_R, net_T, "PGD", True, 10))
