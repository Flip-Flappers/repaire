import os
import random
import shutil

import PIL.Image
import cv2
import torch
import numpy as np
from tqdm import tqdm
from cifar import cifar100
import division.Image_Segmentation as Segmentation

from torchvision.transforms.functional import normalize, resize, to_pil_image

import matplotlib.pyplot as plt
import argparse

fin_ans = 0
zuidacha = 0
pingjuncha = 0
root = '../../fin_dataset/cifar100/train'
#ori_num_train = [4990, 4996, 4981, 4978, 4991, 4980, 4991, 4998, 4995, 4996]
ori_num_test = [1000] * 100
net = cifar100(128, True).cuda()
net.eval()

parser = argparse.ArgumentParser(description='1')

parser.add_argument('--ll', type=int)
args = parser.parse_args()
kk = args.ll
for tmp_ll in range(10):
    ll = tmp_ll + 10 * kk
    k = 0
    for i in os.listdir(root + '/ori_image/' + str(ll)):
        k += 1
    print(k)
    for numss in tqdm(range(k)):
        # copy ori picture
        s = "{:04d}".format(numss)

        # make mask
        image = cv2.imread(root + '/ori_image/' + str(ll) + '/' + s + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_ans = Segmentation.init_R(image, image.shape[0], image.shape[1], 100., 3, 0.8, net, ll)
        ans = seg_ans[0]
        tmp = np.zeros([32, 32, 3])
        tmp2 = np.zeros([3, 32, 32])
        color = np.zeros([ans.max() + 1, 3])
        visit = np.ones((ans.max() + 1)) * -1
        num = 0
        for i in range(32):
            for j in range(32):
                if (visit[ans[i][j]] == -1):
                    visit[ans[i][j]] = num
                    num += 1
        # print(num)
        all = np.zeros([num, 3])
        all_num = np.zeros(num)
        avg = np.zeros([num, 3])
        max_color = np.zeros([num, 3])
        min_color = np.ones([num, 3]) * 255
        for i in range(32):
            for j in range(32):
                for z in range(3):
                    all[int(visit[ans[i][j]])][z] = all[int(visit[ans[i][j]])][z] + image[i][j][z]
                    all_num[int(visit[ans[i][j]])] += 1
                    max_color[int(visit[ans[i][j]])][z] = max(max_color[int(visit[ans[i][j]])][z], image[i][j][z])
                    min_color[int(visit[ans[i][j]])][z] = min(min_color[int(visit[ans[i][j]])][z], image[i][j][z])
        mask = np.zeros([32, 32, 3])
        for i in range(32):
            for j in range(32):
                for z in range(3):
                    mask[i][j][z] = all[int(visit[ans[i][j]])][z]  / all_num[int(visit[ans[i][j]])] * 3
        mask = PIL.Image.fromarray(np.uint8(mask))
        mask.save(root + '/gray_image/' + str(ll) + '/' + s
                  + '.png')
