import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

import GradCam
import PIL.Image
import cv2
import torch
import numpy as np


import torch
import torch.nn as nn
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.utils import save_image
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

dataset = torchvision.datasets.CIFAR10("../../root_data", train=True, transform=transforms.ToTensor())
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
model = torch.load("../net_T/pre/resnet20_check_point.pth")
model.eval()
cam_extractor = GradCAM(model)
num = [0 , 0, 0, 0, 0, 0, 0, 0, 0, 0]
for data, label in tqdm(dataset_loader):

    input_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data)

    # Preprocess your data and feed it to the model
    out = model(input_tensor)
    my_label = int(out[0].argmax())
    activation_map = cam_extractor(int(out[0].argmax()), out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(data.squeeze(0)), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    ori_image = to_pil_image(data.squeeze(0))
    act_image = to_pil_image(activation_map[0].squeeze(0))
    act_image = act_image.resize([32, 32], resample=PIL.Image.BICUBIC)
    s = "{:04d}".format(num[my_label])
    ori_image.save("../../fin_dataset/cifar10/train/ori_image/" + str(my_label) + '/' + s + '.png')
    act_image.save("../../fin_dataset/cifar10/train/gradcam_image/" + str(my_label) + '/' + s + '.png')




    num[my_label] += 1


