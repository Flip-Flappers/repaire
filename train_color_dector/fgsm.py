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

from train_color_dector import Pconv_models
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
dataset_loader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=0)

loss_fn = nn.CrossEntropyLoss()
step = 0
all_loss = 0
checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")
net_R = torch.nn.DataParallel(checkpoint).cuda()
net_R.eval()
correct = 0
num = 0
for image, labels in tqdm(dataset_loader):
    net_R.zero_grad()
    image = image.cuda()
    image.requires_grad = True
    labels = labels.cuda()
    output = net_R(image)
    loss = loss_fn(output, labels)

    loss.backward()
    fgsm = torch.sign(image.grad)

    ans = (image + fgsm * 7 / 255).detach()
    ans = torch.clamp(ans, -1, 1)
    output = net_R(ans)
    _, predicted = torch.max(output.data, dim=1)  # 10 to 1
    correct += (predicted == labels).sum().item()
    ans = ans / 2 + 0.5
    for i in range(500):
        if predicted[i] == 0 and labels[i] != 0:
            tmp = transforms.ToPILImage()(ans[i])
            s = "{:04d}".format(num)
            tmp.save('./fgsm/' + s + '.png')
            num += 1

print(correct)