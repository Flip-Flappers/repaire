import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import math
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10):
    f_image = net(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image))[0].cpu().data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(pert_image[None, :], requires_grad=True)

    fs = net.forward(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x[0]))[0]
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):

            # x.zero_grad()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).cpu().data.numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        pert_image = torch.clamp(pert_image, 0, 1)
        x = torch.tensor(pert_image, requires_grad=True)
        fs = net.forward(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x[0]))[0]
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


checkpoint = torch.load("../net_T/pre/resnet20_check_point.pth")
net = torch.nn.DataParallel(checkpoint).cuda()


# Switch to evaluation mode
net.eval()

root = '../../fin_dataset/cifar10/test/'
trainset = torchvision.datasets.CIFAR10(root='../../root_data', train=False, download=True,
                                        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for data in tqdm(trainloader):
    ori_tensor, labels = data
    im_orig = ori_tensor





    r, loop_i, label_orig, label_pert, pert_image = deepfool(im_orig, net,max_iter=50)


    str_label_orig = label_orig
    str_label_pert = label_pert
    fin_label = net(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(pert_image[0]))[0].argmax()

    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)


    image = transforms.ToPILImage()(pert_image[0][0])
    if fin_label != label_orig:
        s = "{:04d}".format(num[label_orig])
        image.save(root + 'deepfool2/success/' + str(int(label_orig)) + '/' + s +  '.png')
        num[label_orig] += 1
    else:
        s = "{:04d}".format(num2[label_orig])
        image.save(root + 'deepfool2/fail/' + str(int(label_orig)) + '/' + s +  '.png')
        num2[label_orig] += 1

