'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ori_image_block, mask_block, ori_image_num_blocks, mask_num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.ori_image_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.ori_image_bn1 = nn.BatchNorm2d(16)
        self.ori_image_layer1 = self._make_layer(ori_image_block, 16, ori_image_num_blocks[0], stride=1)
        self.ori_image_layer2 = self._make_layer(ori_image_block, 32, ori_image_num_blocks[1], stride=2)
        self.ori_image_layer3 = self._make_layer(ori_image_block, 64, ori_image_num_blocks[2], stride=2)

        self.in_planes = 16
        self.mask_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask_bn1 = nn.BatchNorm2d(16)
        self.mask_layer1 = self._make_layer(mask_block, 16, mask_num_blocks[0], stride=1)
        self.mask_layer2 = self._make_layer(mask_block, 32, mask_num_blocks[1], stride=2)
        self.mask_layer3 = self._make_layer(mask_block, 64, mask_num_blocks[2], stride=2)

        self.out_linear1 = nn.Linear(64 * 2, 64)
        self.out_linear2 = nn.Linear(64, 32)
        self.out_linear3 = nn.Linear(32, 16)
        self.out_linear4 = nn.Linear(16, 8)
        self.out_linear5 = nn.Linear(8, 4)
        self.out_linear6 = nn.Linear(4, 3)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, ori_image, mask_image):
        ori_image_out = F.relu(self.ori_image_bn1(self.ori_image_conv1(ori_image)))
        ori_image_out = self.ori_image_layer1(ori_image_out)
        ori_image_out = self.ori_image_layer2(ori_image_out)
        ori_image_out = self.ori_image_layer3(ori_image_out)
        ori_image_out = F.avg_pool2d(ori_image_out, ori_image_out.size()[3])
        ori_image_out = ori_image_out.view(ori_image_out.size(0), -1)

        mask_out = F.relu(self.mask_bn1(self.mask_conv1(mask_image)))
        mask_out = self.mask_layer1(mask_out)
        mask_out = self.mask_layer2(mask_out)
        mask_out = self.mask_layer3(mask_out)
        mask_out = F.avg_pool2d(mask_out, mask_out.size()[3])
        mask_out = mask_out.view(mask_out.size(0), -1)

        out = torch.cat([ori_image_out, mask_out], dim=1)
        out = nn.Dropout()(nn.ReLU()(self.out_linear1(out)))
        out = nn.Dropout()(nn.ReLU()(self.out_linear2(out)))
        out = nn.ReLU()(self.out_linear3(out))
        out = nn.ReLU()(self.out_linear4(out))
        out = nn.ReLU()(self.out_linear5(out))
        out = nn.Tanh()(self.out_linear6(out))
        return out


def color_net():
    return ResNet(BasicBlock, BasicBlock, [3, 3, 3], [3, 3, 3])
