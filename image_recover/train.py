import argparse
import itertools

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

from image_recover import models
writer = SummaryWriter('./runs/0')
root = '../../recover_dataset/train/0'
ori_picture = sorted(glob.glob(os.path.join(root, 'ori_picture') + '/*.*'))
edge_picture = sorted(glob.glob(os.path.join(root, 'edge') + '/*.*'))


"""image_list = torch.zeros([4500, 3, 32 * 3, 32 * 3])
num = 0
for i in tqdm(range(4500)):
    edge_loc = edge_picture[i]
    edge_pic = Image.open(edge_loc)
    edge_tensor = transforms.ToTensor()(edge_pic)
    edge_tensor = transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))(edge_tensor)
    image_list[num] = edge_tensor
    num += 1
torch.save(image_list, './image_list.t')"""

image_list = torch.load('./image_list.t')

dataset = torch.utils.data.TensorDataset(image_list, image_list)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1200, shuffle=True)

recover_net = models.recover_net().cuda()

optimizer = torch.optim.Adam(recover_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()
step = 0
for epoch in tqdm(range(10000)):
    for image, ans in tqdm(dataset_loader):
        image = image.cuda()
        ans = ans.cuda()
        output = recover_net(image)
        ans = ans
        ans = ans.cuda()
        loss = loss_fn(output, ans)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('color_loss', loss, step)
        writer.flush()
        step += 1
    if epoch % 10:
        torch.save(color_net.state_dict(), './color_net_0' + str(epoch) + '.pth')

