import random

import PIL.Image
import cv2
import torch
import numpy as np


import torch
import torch.nn as nn
from advertorch.attacks import PGDAttack
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.utils import save_image
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
for numss in range(50):
    #ori_image
    s = "{:04d}".format(numss)
    ori_image = cv2.imread('./ori/' + s + '.png')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    plt.figure()

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
    label = out.argmax(dim = 1) * 0
    # Retrieve the CAM by passing the class index and the model output

    activation_map = cam_extractor(int(out[0].argmax()), out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 2)
    plt.imshow(result)




    # fgsm_img
    fgsm_input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    fgsm_input_tensor.requires_grad = True
    fgsm_out = model(fgsm_input_tensor.unsqueeze(0))
    loss = -1 * nn.CrossEntropyLoss()(fgsm_out, label)
    loss.backward()
    fgsm_grad = torch.sign(fgsm_input_tensor.grad)
    tmp_image = torch.clamp(img + fgsm_grad.detach() * 2, 0, 255)
    fgsm_input_tensor2 = normalize(tmp_image / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out2 = model(fgsm_input_tensor2.unsqueeze(0))

    if out2.argmax(dim=1)[0] == 0:
        print(numss)
    plt.subplot(1, 8, 3)
    plt.imshow(to_pil_image(torch.clamp(tmp_image, 0, 255).byte()))
    fgsm_image = to_pil_image(torch.clamp(tmp_image, 0, 255).byte())

    save_image(torch.clamp(tmp_image, 0, 255) / 255, './fgsm/%04d.png' % (numss + 1))

    #gram_fgsm_img
    activation_map = cam_extractor(int(out[0].argmax()), out)
    activation_map = cam_extractor(0, out)
    # Resize the CAM and overlay it

    result = overlay_mask(to_pil_image(torch.clamp(tmp_image, 0, 255).byte()), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(1, 8, 4)
    plt.imshow(result)




