import random

import PIL.Image
import cv2
import torch
import numpy as np
from tqdm import tqdm

import Image_Segmentation as Segmentation


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
    for numss in tqdm(range(4000)):
        s = "{:04d}".format(numss + 1)
        image = cv2.imread('./root_data/pngcifar10/train/' + str(ll) + '/' + s + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_ans = Segmentation.init_R(image, image.shape[0], image.shape[1], 50., 3, 0.8)
        ans = seg_ans[0]
        tmp = np.zeros([32, 32, 3])
        tmp2 = np.zeros([3, 32, 32])
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

        plt.figure()

        plt.subplot(1, 5, 3)
        plt.imshow(to_pil_image(np.uint8(tmp)))

        model = torch.load("../net_T/pre/resnet20_check_point.pth")
        model.eval()
        cam_extractor = GradCAM(model)
        # Get your input

        input_tensor = normalize(torch.from_numpy(tmp2) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input_tensor = torch.tensor(input_tensor,dtype=torch.float)
        # Preprocess your data and feed it to the model
        out3 = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        #print(3, out3)

        img = read_image('./root_data/pngcifar10/train/' + str(ll) + '/' + s + '.png')
        # Preprocess it for your chosen model
        tmp = to_pil_image(img)


        plt.subplot(1, 5, 1)
        plt.imshow(tmp)



        input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Preprocess your data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        out0 = out
        # Retrieve the CAM by passing the class index and the model output
        #print(0, out0)

        if out0.argmax(dim=1)[0] == out3.argmax(dim=1)[0]:
            fin_ans += 1
            #print("yes")
        plt.show()
    print(fin_ans)