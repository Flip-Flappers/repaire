import os
import random
import shutil

import PIL.Image
import cv2
import torch
import numpy as np
from tqdm import tqdm

import division.Image_Segmentation as Segmentation


import torch
import torch.nn as nn
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
fin_ans = 0
zuidacha = 0
pingjuncha = 0
for ll in range(10):
    for numss in tqdm(range(4500)):
        # copy ori picture
        s = "{:04d}".format(numss)
        shutil.copy('../../recover_dataset/pngcifar10/train/' + str(ll) + '/' + s + '.png', '../../recover_dataset/train/' + str(ll) + '/ori_picture/' + s + '.png')

        # make mask
        image = cv2.imread('../../recover_dataset/pngcifar10/train/' + str(ll) + '/' + s + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_ans = Segmentation.init_R(image, image.shape[0], image.shape[1], 100., 3, 0.8)
        ans = seg_ans[0]
        tmp = np.zeros([32, 32, 3])
        tmp2 = np.zeros([3, 32, 32])
        tmp3 = np.ones([32 * 3, 32 * 3, 3]) * 255
        color = np.zeros([ans.max() + 1, 3])
        visit = np.ones((ans.max() + 1)) * -1
        num = 0
        for i in range(32):
            for j in range(32):
                if(visit[ans[i][j]] == -1):
                    visit[ans[i][j]] = num
                    num += 1
        #print(num)
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
                    max_color[int(visit[ans[i][j]])][z] = max(max_color[int(visit[ans[i][j]])][z],  image[i][j][z])
                    min_color[int(visit[ans[i][j]])][z] = min(min_color[int(visit[ans[i][j]])][z], image[i][j][z])
        for k in range(num):
            if all_num[k] >= 4:
                mask = np.zeros([32, 32])
                for i in range(32):
                    for j in range(32):
                        if visit[ans[i][j]] == k:
                            mask[i][j] = 255
                mask = PIL.Image.fromarray(np.uint8(mask))
                if not os.path.exists('../../recover_dataset/train/' + str(ll) + '/mask/' + s):
                    os.makedirs('../../recover_dataset/train/' + str(ll) + '/mask/' + s)
                sk = "{:04d}".format(k)
                mask.save('../../recover_dataset/train/' + str(ll) + '/mask/' + s + '/' + sk
                          + '_' + str(np.around(all[k] * 3 / all_num[k]))
                          + '_' + str(max_color[k])
                          + '_' + str(min_color[k])
                          + '.png')
        l_area = 0
        for i in range(num):
            if all_num[i] > 4:
                l_area += 1
        for i in range(num):
            for j in range(3):
                zuidacha = max(zuidacha, max_color[i][j] - min_color[i][j])
                pingjuncha += (max_color[i][j] - min_color[i][j]) / 3 / num
        print(num, l_area, zuidacha, pingjuncha / (ll + 1) / numss)
        for i in range(num):
            for z in range(3):
                avg[i][z] = all[i][z] / all_num[i] * 3
        for i in range(32):
            for j in range(32):
                ans[i][j] = visit[ans[i][j]];
        """for i in range(num):
            color[i][0] = random.randint(0, 255)
            color[i][1] = random.randint(0, 255)
            color[i][2] = random.randint(0, 255)"""
        for i in range(num):
            color[i][0] = int(avg[i][0])
            color[i][1] = int(avg[i][1])
            color[i][2] = int(avg[i][2])
        for i in range(32):
            for j in range(32):
                tmp[i][j][0] = color[ans[i][j]][0]
                tmp[i][j][1] = color[ans[i][j]][1]
                tmp[i][j][2] = color[ans[i][j]][2]
        for i in range(32):
            for j in range(32):
                tmp2[0][i][j] = color[ans[i][j]][0]
                tmp2[1][i][j] = color[ans[i][j]][1]
                tmp2[2][i][j] = color[ans[i][j]][2]

        for i in range(32 * 3):
            for j in range(32 * 3):
                ori_x = int(i / 3)
                ori_y = int(j / 3)
                mode_x = i % 3
                mode_y = j % 3
                new_x = ori_x
                new_y = ori_y
                if mode_x == 0:
                    new_x -= 1
                if mode_y == 0:
                    new_y -= 1
                if mode_x == 2:
                    new_x += 1
                if mode_y == 2:
                    new_y += 1
                if mode_y == 1 and mode_x == 1:
                    continue
                if new_x < 0 or new_y < 0 or new_x >= 32 or new_y >= 32:
                    continue
                same = 1
                for z in range(3):
                    if tmp[ori_x][ori_y][z] != tmp[new_x][new_y][z]:
                        same = 0
                        break
                if same == 0:
                    for z in range(3):
                        tmp3[i][j][z] = 0
        edge = PIL.Image.fromarray(np.uint8(tmp3))
        edge.save('../../recover_dataset/train/' + str(ll) + '/edge/' + s + '.png')
        plt.figure()
        plt.subplot(1, 5, 2)
        plt.imshow(to_pil_image(np.uint8(tmp)))
        plt.subplot(1, 5, 3)
        plt.imshow(to_pil_image(np.uint8(tmp3)))

        #plt.show()