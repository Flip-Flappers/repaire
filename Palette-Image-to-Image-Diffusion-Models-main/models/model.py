import glob
import math
import os

import PIL.Image
import cv2
import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import lpips


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, val_range, window_size=4, window=None, size_average=True, full=False):
    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.net_V = torch.load("../net_T/pre/resnet20_check_point.pth")
        self.net_V = self.net_V.eval()
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.alert_num = 0
        self.fgsm_ok = 0
        self.fgsm_false = 0
        self.all_num = 0
        self.big_task = "fgsm"
        if self.big_task == "fgsm":
            self.writer2 = SummaryWriter("run/fgsm")
            self.root = '../../fin_dataset/cifar10/test/fgsm'
        else:
            self.root = '../../fin_dataset/cifar10/test'

            self.writer2 = SummaryWriter("run/test")

    def set_input(self, data):
        ''' must use set_device in tensor '''
        # 加入噪声后的图片 原图0.5
        self.cond_image = self.set_device(data.get('cond_image'))

        # 原图0.5
        self.gt_image = self.set_device(data.get('gt_image'))

        # mask 0,1
        self.mask = self.set_device(data.get('mask'))

        # 白色噪声
        self.mask_image = data.get('mask_image')

        self.path = data['path']

        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu() + 1) / 2,
            'cond_image': (self.cond_image.detach()[:].float().cpu() + 1) / 2,
        }
        if self.task in ['inpainting', 'uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image + 1) / 2,

            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu() + 1) / 2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx - self.batch_size].detach().float().cpu())

        if self.task in ['inpainting', 'uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                 y_0=self.gt_image, mask=self.mask,
                                                                                 sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                                 sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                          y_0=self.gt_image, mask=self.mask,
                                                                          sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()

        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                 y_0=self.gt_image, mask=self.mask,
                                                                                 sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                                 sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                          y_0=self.gt_image, mask=self.mask,
                                                                          sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                    if key == 'mae':
                        tmp_mae = value
                        self.writer2.add_scalar('mae', value, global_step=self.all_num)
                true_label = []
                true_poss = []
                gt_image = []
                gt_min_max = []
                mask = []
                feature = []
                for key, value in self.get_current_visuals(phase='test').items():

                    self.writer.add_images(key, value)
                    all_num = int(value.shape[0] / 10)
                    if key == 'mask':
                        start = 0
                        end = 10
                        for _ in range(all_num):
                            tmp_value = value[start:end]
                            mask.append(tmp_value)
                            start += 10
                            end += 10
                    if key == 'gt_image':
                        start = 0
                        end = 10

                        for _ in range(all_num):
                            tmp_value = value[start:end]
                            true_out_put_label, features = self.net_V(
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tmp_value))
                            true_out_put_label = true_out_put_label.detach()[0]
                            feature.append(features.detach())
                            true_poss.append(true_out_put_label)
                            true_label.append(true_out_put_label.argmax())
                            gt_image.append(tmp_value)

                            start += 10
                            end += 10

                    if key == 'output':
                        start = 0
                        end = 10
                        for ii in range(all_num):
                            tmp_mae = 0
                            tmp_poss = 0
                            ssims = 0

                            tmp_value = value[start:end]
                            s = "{:04d}".format(self.all_num)
                            color_mask_loc = (sorted(glob.glob(
                                os.path.join(self.root, 'color_edge_image/' + str(0) + '/' + s) + '/*.*')))

                            gradcam_mask = transforms.ToTensor()(
                                PIL.Image.open(self.root + '/gradcam_image/' + '0/' + s + '.png'))
                            gradcam_mask = torch.where(gradcam_mask < 0.5, torch.ones([1, 32, 32]), torch.zeros([1, 32, 32]))
                            for jj in range(10):
                                p1 = gt_image[ii][jj] * mask[ii][jj]

                                p2 = tmp_value[jj] * mask[ii][jj]
                                tmp_mae += nn.L1Loss()(p1, p2)
                                ssims += ssim(gt_image[ii][jj].unsqueeze(0), tmp_value[jj].unsqueeze(0), val_range=255)


                                """for zz in range(len(color_mask_loc)):
                                    color_mask_image = PIL.Image.open(color_mask_loc[zz])
                                    color_mask_tensor = transforms.ToTensor()(color_mask_image).int() & gradcam_mask.int()
                                    if self.big_task == "fgsm":
                                        tmp_color = color_mask_loc[zz][65:][:-4].split('_')
                                    else:
                                        tmp_color = color_mask_loc[zz][60:][:-4].split('_')
                                    avg = tmp_color[0][1:][:-1].split('.')[:-1]
                                    c1 = torch.tensor(int(avg[0])) / 255
                                    c2 = torch.tensor(int(avg[1])) / 255
                                    c3 = torch.tensor(int(avg[2])) / 255
                                    fin_mask_tensor = torch.cat([color_mask_tensor, color_mask_tensor], dim=0)
                                    fin_mask_tensor = torch.cat([fin_mask_tensor, color_mask_tensor], dim=0)
                                    anti_mask_image = 1 - fin_mask_tensor


                                    fin_mask_tensor = torch.cat([color_mask_tensor.float() * c1, color_mask_tensor.float() * c2], dim=0)
                                    fin_mask_tensor = torch.cat([fin_mask_tensor, color_mask_tensor.float() * c3], dim=0)

                                    tmp_value[jj] = tmp_value[jj] * anti_mask_image + fin_mask_tensor"""


                            self.writer2.add_images('fin_img', tmp_value, global_step=self.all_num)
                            out_put_label, features = self.net_V(
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tmp_value))
                            out_put_label = out_put_label.detach()
                            features = features.detach()
                            feature_mse = nn.MSELoss()(features, feature[ii])
                            num = 0
                            for j in range(out_put_label.shape[0]):
                                tmp_poss += nn.KLDivLoss()(nn.Softmax()(true_poss[ii]).log(), nn.Softmax()(out_put_label[j]))
                                if out_put_label[j].argmax() != 0:
                                    num += 1
                            p = num / out_put_label.shape[0]
                            self.writer2.add_scalar('alert_p', p, global_step=self.all_num)
                            self.writer2.add_scalar('tmp_mae', tmp_mae, global_step=self.all_num)
                            self.writer2.add_scalar('IS', tmp_poss, global_step=self.all_num)
                            self.writer2.add_scalar('ssims', ssims, global_step=self.all_num)
                            self.writer2.add_scalar('feature_mse', feature_mse, global_step=self.all_num)
                            if true_label[ii] == 0:

                                if p <= 0.2:
                                    self.fgsm_false += 1
                                    print("false " + str(self.all_num) + ' ' + str(p) + ' ')
                                elif tmp_mae / 10 <= 0.25:
                                    self.fgsm_false += 1
                                    print("false " + str(self.all_num) + ' ' + str(p) + ' ')
                                elif tmp_poss >= 60:
                                    self.fgsm_ok += 1
                                    print("ok " + str(self.all_num) + ' ' + str(p) + ' ')
                                elif (p >= 0.6 or ssims / 10 <= 0.81):
                                    self.fgsm_ok += 1
                                    print("ok " + str(self.all_num) + ' ' + str(p) + ' ')
                                else:
                                    self.fgsm_false += 1
                                    print("false " + str(self.all_num) + ' ' + str(p) + ' ')
                            self.writer2.add_scalar('fgsm_ok', self.fgsm_ok, global_step=self.all_num)
                            self.writer2.add_scalar('fgsm_false', self.fgsm_false, global_step=self.all_num)

                            start += 10
                            end += 10
                            self.all_num += 1
                self.writer.save_images(self.save_current_results())

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label + '_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label + '_ema')
        self.save_training_state()
