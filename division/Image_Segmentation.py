import time

import PIL
import cv2
import numpy as np
from division import selective_classes
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from tqdm import tqdm
from torch import nn as nn


from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms as transforms
import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class Image_Segment:
    """
    belong 分类后的图片
    int_C 每个块的内部最大忍受点
    area_size 每个块大小
    father 加速用的并查集
    s 边权
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.belong = np.zeros([n, m], np.int)
        self.int_C = np.zeros([n * m], np.float)
        self.area_size = np.ones([n * m], np.float)
        self.father = np.zeros([n * m], np.int)
        self.s = []
        self.visit = np.ones([n * m], np.int)
        self.maxn = np.ones([n * m, 3], np.float)
        self.minn = np.ones([n * m, 3], np.float)
        self.mean = np.ones([n * m, 3], np.float)
        self.visit = self.visit * -1
        self.detail = {}
        self.fathertonum = np.zeros([n * m], np.int)
        self.ori_image = np.zeros([n, m], np.int)
        self.mask = np.zeros([n * m, n, m, 3], np.int)

    def find_father(self, fa):
        if fa == self.father[fa]:
            return fa
        else:
            self.father[fa] = self.find_father(self.father[fa])
            return self.father[fa]


    def make_graph(self, input_img, kernel, sigma):
        direction = [[1, 0], [0, 1]]
        input_img = np.asanyarray(input_img, dtype=np.float)
        self.ori_image = input_img
        input_img = cv2.GaussianBlur(input_img, (kernel, kernel), sigma)
        num = 0
        edge_left = input_img[:self.n - 1] - input_img[1:]
        edge_down = input_img[:, :self.m - 1] - input_img[:, 1:]
        for i in range(self.n):
            for j in range(self.m):

                self.belong[i][j] = num
                self.father[num] = num
                self.area_size[num] = 1
                for zz in range(3):
                    self.mask[num][i][j][zz] = 1
                    self.mean[num][zz] = self.ori_image[i][j][zz]
                    self.maxn[num][zz] = self.ori_image[i][j][zz]
                    self.minn[num][zz] = self.ori_image[i][j][zz]
                num += 1
                for k in range(2):
                    next_x = i + direction[k][0]
                    next_y = j + direction[k][1]
                    if next_x < 0 or next_y < 0 or next_x >= self.n or next_y >= self.m:
                        continue
                    if k == 0:
                        dist = np.linalg.norm(edge_left[i][j])
                    else:
                        dist = np.linalg.norm(edge_down[i][j])
                    self.s.append(selective_classes.Segmentation_node(i, j, next_x, next_y, dist))
        self.s.sort()

    def link_node(self, r_c, net, label):
        for i in range(len(self.s)):
            father1 = self.find_father(
                self.belong[self.s[i].x][self.s[i].y])
            father2 = self.find_father(
                self.belong[self.s[i].next_x][self.s[i].next_y])
            tmp1 = max(abs(self.maxn[father1][0] - self.minn[father2][0]), abs(self.minn[father1][0] - self.maxn[father2][0]))
            tmp2 = max(abs(self.maxn[father1][1] - self.minn[father2][1]), abs(self.minn[father1][1] - self.maxn[father2][1]))
            tmp3 = max(abs(self.maxn[father1][2] - self.minn[father2][2]), abs(self.minn[father1][2] - self.maxn[father2][2]))
            tmp_mean_fa1 = abs(self.mean[father1][0] / self.area_size[father1] - self.mean[father2][0] / self.area_size[father2])
            tmp_mean_fa2 = abs(
                self.mean[father1][1] / self.area_size[father1] - self.mean[father2][1] / self.area_size[
                    father2])
            tmp_mean_fa3 = abs(
                self.mean[father1][2] / self.area_size[father1] - self.mean[father2][2] / self.area_size[
                    father2])
            if father1 != father2 and ((tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3 <= 50 * 50 * 3) or (tmp_mean_fa1 <= 25 and tmp_mean_fa2 <= 25 and tmp_mean_fa3 <= 25)):

                    tmp_area_size = self.area_size[father1] + self.area_size[father2]
                    tmp_mask = self.mask[father1] + self.mask[father2]
                    tmp_v = max(self.int_C[father1], self.int_C[father2])
                    tmp_v = max(self.s[i].v, tmp_v)
                    tmp_c_1 = (self.mean[father1][0] + self.mean[father2][0]) / (self.area_size[father1] + self.area_size[father2])
                    tmp_c_2 = (self.mean[father1][1] + self.mean[father2][1]) / (self.area_size[father1] + self.area_size[father2])
                    tmp_c_3 = (self.mean[father1][2] + self.mean[father2][2]) / (self.area_size[father1] + self.area_size[father2])
                    fin_image = self.ori_image * (1 - tmp_mask) + tmp_mask * [tmp_c_1, tmp_c_2, tmp_c_3]
                    tmp_p = PIL.Image.fromarray(np.uint8(fin_image))

                    fin_tensor = transforms.ToTensor()(tmp_p).cuda()
                    fin_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(fin_tensor).unsqueeze(0)
                    out = net(fin_tensor)[0].argmax()
                    if out == label:
                        tmp = father1
                        father1 = min(father1, father2)
                        father2 = max(tmp, father2)
                        self.area_size[father1] = tmp_area_size
                        self.area_size[father2] = tmp_area_size
                        self.int_C[father1] = tmp_v
                        self.int_C[father2] = tmp_v
                        self.father[father2] = father1

                        self.mask[father1] = tmp_mask
                        self.mask[father2] = tmp_mask



                        for zz in range(3):
                            self.maxn[father1][zz] = max(self.maxn[father1][zz], self.maxn[father2][zz])
                            self.minn[father1][zz] = min(self.minn[father1][zz], self.minn[father2][zz])
                            self.maxn[father2][zz] = max(self.maxn[father1][zz], self.maxn[father2][zz])
                            self.minn[father2][zz] = min(self.minn[father1][zz], self.minn[father2][zz])
                            tmp = self.mean[father1][zz] + self.mean[father2][zz]
                            self.mean[father1][zz] = tmp
                            self.mean[father2][zz] = tmp

    def make_ans(self):
        num = 0
        for i in range(self.n):
            for j in range(self.m):
                father = self.find_father(self.belong[i][j])
                self.belong[i][j] = father
                if self.visit[father] == -1:
                    self.visit[father] = father
                    self.detail[num] = []
                    self.fathertonum[father] = num
                    num += 1
                self.detail[self.fathertonum[father]].append([i, j])
        return self.belong, num, self.detail, self.fathertonum



def init_R(input_img, n, m, r_c, kernel, sigma, net, label):
    Image_Segmenter = Image_Segment(n, m)
    start = time.time()
    Image_Segmenter.make_graph(input_img, kernel, sigma)

    Image_Segmenter.link_node(r_c, net, label)
    ans = Image_Segmenter.make_ans()
    end = time.time()
    #print("分割运行时间:%.2f秒" % (end - start))
    return ans