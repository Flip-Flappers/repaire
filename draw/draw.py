import PIL.Image as Image
import torchvision.transforms as transforms
import torch


image = Image.open('./image/GT_.png_0163.png')
Mask = Image.open('./image/Mask_.png_0163.png')

mask_tensor = transforms.ToTensor()(Mask)
mask_tensor = torch.where(mask_tensor == 1, torch.ones(mask_tensor.shape), torch.zeros(mask_tensor.shape))
image_tensor = transforms.ToTensor()(image)

cond_image = image_tensor*(1. - mask_tensor) + mask_tensor*torch.randn_like(image_tensor)
cond_image = transforms.ToPILImage()(cond_image)
cond_image.save("./image/cond_image.png")
mask_image = transforms.ToPILImage()(mask_tensor)
mask_image.save("./image/mask_image.png")