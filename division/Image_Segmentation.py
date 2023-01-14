import time
import cv2
import numpy as np
from division import selective_classes



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
        self.visit = self.visit * -1
        self.detail = {}
        self.fathertonum = np.zeros([n * m], np.int)
        self.ori_image = np.zeros([n, m], np.int)

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
                for zz in range(3):
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

    def link_node(self, r_c):
        for i in range(len(self.s)):
            father1 = self.find_father(
                self.belong[self.s[i].x][self.s[i].y])
            father2 = self.find_father(
                self.belong[self.s[i].next_x][self.s[i].next_y])
            tmp1 = max(abs(self.maxn[father1][0] - self.minn[father2][0]), abs(self.minn[father1][0] - self.maxn[father2][0]))
            tmp2 = max(abs(self.maxn[father1][1] - self.minn[father2][1]), abs(self.minn[father1][1] - self.maxn[father2][1]))
            tmp3 = max(abs(self.maxn[father1][2] - self.minn[father2][2]), abs(self.minn[father1][2] - self.maxn[father2][2]))
            if father1 != father2 and tmp1 <= 50 and tmp2 <= 50 and tmp3 <= 50:
                if self.s[i].v < min(
                        self.int_C[father1] + r_c / self.area_size[father1],
                        self.int_C[father2] + r_c / self.area_size[father2]):
                    tmp = father1
                    father1 = min(father1, father2)
                    father2 = max(tmp, father2)
                    self.area_size[father1] += self.area_size[father2]
                    self.int_C[father1] = self.s[i].v
                    self.father[father2] = father1
                    for zz in range(3):
                        self.maxn[father1][zz] = max(self.maxn[father1][zz], self.maxn[father2][zz])
                        self.minn[father1][zz] = min(self.minn[father1][zz], self.minn[father2][zz])
                        self.maxn[father2][zz] = max(self.maxn[father1][zz], self.maxn[father2][zz])
                        self.minn[father2][zz] = min(self.minn[father1][zz], self.minn[father2][zz])

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



def init_R(input_img, n, m, r_c, kernel, sigma):
    Image_Segmenter = Image_Segment(n, m)
    start = time.time()
    Image_Segmenter.make_graph(input_img, kernel, sigma)
    Image_Segmenter.link_node(r_c)
    ans = Image_Segmenter.make_ans()
    end = time.time()
    #print("分割运行时间:%.2f秒" % (end - start))
    return ans