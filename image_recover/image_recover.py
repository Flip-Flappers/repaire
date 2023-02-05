import tqdm
import cv2
import numpy as np
import os

imgDir = './A'
imgNames = os.listdir(imgDir)  # 图片名列表

# 计算所有图片的指标
for index, name in enumerate(imgNames):
    path = imgDir + '/' + name
    img = cv2.imread(path)
    height, width = img.shape[:2]
    res = cv2.resize(img, (width ** 2, height ** 2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img', res)
    cv2.waitKey(0)
