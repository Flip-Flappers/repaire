import random

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[32, 32], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.ones = torch.ones([1, 32, 32])
        self.zeros = torch.zeros([1, 32, 32])
        self.fin_mask = []

        for kk in range(3 * 2):
            tmp = torch.zeros([1, 32, 32])
            k = kk + 3
            for i in range(32):
                if i % k < k / 2:
                    for j in range(32):
                        if j % k < k / 2:
                            tmp[0][i][j] = 1
                else:
                    for j in range(32):
                        if j % k > k / 2:
                            tmp[0][i][j] = 1
            self.fin_mask.append(1 - tmp)



        tmp_zero = torch.zeros([2, 2])
        tmp_one = torch.ones([2, 2])
        tmp_zero_one24 = torch.cat([tmp_zero, tmp_one], dim=0)
        tmp_zero_one28 = tmp_zero_one24.clone()
        for i in range(7):
            tmp_zero_one28 = torch.cat([tmp_zero_one28, tmp_zero_one24], dim=0)
        tmp_zero_one32 = tmp_zero_one28.clone()
        for i in range(15):
            if i % 2 == 0:
                tmp_zero_one32 = torch.cat([tmp_zero_one32, 1 - tmp_zero_one28], dim=1)
            else:
                tmp_zero_one32 = torch.cat([tmp_zero_one32, tmp_zero_one28], dim=1)
        self.fin_mask3 = tmp_zero_one32.unsqueeze(0)
        tmp_zero = torch.zeros([4, 4])
        tmp_one = torch.ones([4, 4])
        tmp_zero_one48 = torch.cat([tmp_zero, tmp_one], dim=0)
        tmp_zero_one432 = tmp_zero_one48.clone()
        for i in range(3):
            tmp_zero_one432 = torch.cat([tmp_zero_one432, tmp_zero_one48], dim=0)
        tmp_zero_one3232 = tmp_zero_one432.clone()
        for i in range(7):
            if i % 2 == 0:
                tmp_zero_one3232 = torch.cat([tmp_zero_one3232, 1 - tmp_zero_one432], dim=1)
            else:
                tmp_zero_one3232 = torch.cat([tmp_zero_one3232, tmp_zero_one432], dim=1)
        self.fin_mask4 = tmp_zero_one3232.unsqueeze(0)
        self.fin_mask5 = 1 - self.fin_mask3
        self.fin_mask6 = 1 - self.fin_mask4

        self.fin_mask.append(self.fin_mask3)
        self.fin_mask.append(self.fin_mask4)
        self.fin_mask.append(self.fin_mask5)
        self.fin_mask.append(self.fin_mask6)
        self.window_size = [2, 4, 8]
        self.my_num = 0


    def get_my_mask(self, p, window_size):
        tmp_hang = []
        p = max(p, 1 / (window_size * window_size))

        for i in range(int(32 / window_size)):
            now_window = torch.rand([1, window_size, window_size])
            tmp_num = 0
            for j in range(int(32 / window_size)):
                tmp_window = torch.rand([1, window_size, window_size])
                if tmp_num == 0:
                    now_window = tmp_window
                else:
                    now_window = torch.cat([now_window, tmp_window], dim = 1)
                tmp_num += 1
            tmp_hang.append(now_window)
        fin_window = tmp_hang[0]
        for i in range(len(tmp_hang)):
            if i == 0:
                continue
            fin_window = torch.cat([fin_window, tmp_hang[i]], dim = 2)
        fin_window = torch.where(fin_window <= p, torch.ones([1, 32, 32]), torch.zeros([1, 32, 32]))
        return fin_window


    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index].split(',')
        #原图0.5
        img = self.tfs(self.loader(path[0][3:]))

        #mask 0, 1, 1为mask区域

        """
        mask1 = torch.where(mask <= 0.5, self.zeros, self.ones).byte()
        mask2 = torch.where(mask < 1, self.ones, self.zeros).byte()
        
        mask = mask2 & mask1"""
        """mask = transforms.ToTensor()(Image.open(path[1]))
        mask3 = torch.where(mask <= 0.95, self.zeros, self.ones).byte()
        if self.my_num < 10:
            mask = self.fin_mask[self.my_num].byte() | mask3
        else:"""
        mask = self.get_mask()
        self.my_num += 1
        self.my_num = self.my_num % 20

        #加入噪声后的图片 原图0.5
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)

        cond_image = torch.clamp(cond_image, -1, 1)
        #加入白色后的图片 原图0.5
        mask_img = img*(1. - mask) + mask




        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        if self.mask_mode == 'test':
            ret['path'] = path[0][51:55] + '_' + path[2]
        elif self.mask_mode == 'fgsm':
            ret['path'] = path[0][51:55] + '_' + path[2]
        elif self.mask_mode == 'hybrid2':
            ret['path'] = path[0][47:51] + '_' + path[2]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        self.mask_mode = 'hybrid2'

        regular_mask = bbox2mask(self.image_size, random_bbox())
        irregular_mask = brush_stroke_mask(self.image_size, )
        #mask = regular_mask | irregular_mask | brush_stroke_mask(self.image_size, ) | brush_stroke_mask(self.image_size, ) | brush_stroke_mask(self.image_size, )
        mask = irregular_mask | regular_mask
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[32, 32], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'
        ori_image_file_name = file_name.split(",", 1)
        img = self.tfs(self.loader('{}/{}'.format(self.data_root, ori_image_file_name[0])))
        cond_image = self.tfs(self.loader('{}/{}'.format(self.data_root, ori_image_file_name[0][:35] + "ori_image" + ori_image_file_name[0][44:])))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


