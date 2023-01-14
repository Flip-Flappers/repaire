import cv2
import numpy as np
import torchvision
import argparse
import os
import random
import shutil
import numpy as np
import torch
from torch import nn, autograd
from tqdm import tqdm

from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

transform_dict = {
    'tar_s':transforms.Compose([
                    transforms.ToTensor(),

        ])
}


trainset = torchvision.datasets.CIFAR10('./root_data', transform=transform_dict['tar_s'], train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                         shuffle=False, num_workers=0)


testset = torchvision.datasets.CIFAR10('./root_data', transform=transform_dict['tar_s'], train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

def fun1():
    num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, data in enumerate(trainloader):
        image, label = data
        img = torchvision.utils.make_grid(image)
        npimg = img.numpy()
        image = np.transpose(npimg, (1, 2, 0))
        index = int(label[0])
        s = "{:04d}".format(num[index])
        cv2.imwrite('./root_data/pngcifar10/train/'+str(index)+'/'+str(s)+'.png', image*255)
        num[index] += 1
def fun2():
    num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, data in enumerate(testloader):
        image, label = data
        img = torchvision.utils.make_grid(image)
        npimg = img.numpy()
        image = np.transpose(npimg, (1, 2, 0))
        index = int(label[0])
        s = "{:04d}".format(num[index])
        cv2.imwrite('./root_data/pngcifar10/test/'+str(index)+'/'+str(s)+'.png', image*255)
        num[index] += 1
fun1()
fun2()