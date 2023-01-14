import random

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
for numss in range(50):
    plt.figure()
    # fgsm
    s = "{:04d}".format(numss + 1)
    ori_image = cv2.imread('./fgsm/' + s + '.png')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 8, 5)
    plt.imshow(to_pil_image(np.uint8(ori_image)))

    # gram_ori_B2A
    model = torch.load("../net_T/pre/resnet20_check_point.pth")
    model.eval()
    cam_extractor = GradCAM(model)
    # Get your input
    img = read_image("./fgsm/" + s + ".png")
    input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))

    label = out.argmax(dim=1)
    # Retrieve the CAM by passing the class index and the model output
    print(out)
    activation_map = cam_extractor(0, out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 6)
    plt.imshow(result)

    # fgsm_B2A
    s = "{:04d}".format(numss + 1)
    ori_image = cv2.imread('./fgsm_B2A/' + s + '.png')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 8, 7)
    plt.imshow(to_pil_image(np.uint8(ori_image)))

    # gram_ori_B2A
    model = torch.load("../net_T/pre/resnet20_check_point.pth")
    model.eval()
    cam_extractor = GradCAM(model)
    # Get your input
    img = read_image("./fgsm_B2A/" + s + ".png")
    input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))

    label = out.argmax(dim=1)
    # Retrieve the CAM by passing the class index and the model output
    print(out)
    activation_map = cam_extractor(0, out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 8)
    plt.imshow(result)

    #ori_image
    s = "{:04d}".format(numss)
    ori_image = cv2.imread('./ori/' + s + '.png')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 8, 1)
    plt.imshow(to_pil_image(np.uint8(ori_image)))

    #gram_ori
    model = torch.load("../net_T/pre/resnet20_check_point.pth")
    model.eval()
    cam_extractor = GradCAM(model)
    # Get your input
    img = read_image("./ori/" + s + ".png")
    input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    label = out.argmax(dim = 1)
    # Retrieve the CAM by passing the class index and the model output
    print(out)
    activation_map = cam_extractor(0, out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 2)
    plt.imshow(result)


    #ori_B2A
    s = "{:04d}".format(numss + 1)
    ori_image = cv2.imread('./ori_B2A/' + s + '.png')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)


    plt.subplot(1, 8, 3)
    plt.imshow(to_pil_image(np.uint8(ori_image)))

    #gram_ori_B2A
    model = torch.load("../net_T/pre/resnet20_check_point.pth")
    model.eval()
    cam_extractor = GradCAM(model)
    # Get your input
    img = read_image("./ori_B2A/" + s + ".png")
    input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    label = out.argmax(dim = 1)
    # Retrieve the CAM by passing the class index and the model output
    print(out)
    activation_map = cam_extractor(0, out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 4)
    plt.imshow(result)


    plt.show()