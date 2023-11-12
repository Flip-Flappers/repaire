import torch
import torchvision
from torchvision import transforms
import PIL as PIL


trainset = torchvision.datasets.CIFAR10(root='../root_data', train=False, download=True,
                                        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)


root = '../fin_dataset/cifar10/test/'


checkpoint = torch.load("./net_T/pre/resnet20_check_point.pth")

net_R = torch.nn.DataParallel(checkpoint).cuda()
net_R.eval()
ans = 0
num = 0
for data, label in trainloader:
    data, label = data.cuda(), label.cuda()

    true_label = net_R(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data.cuda()))[0].argmax()
    if true_label == label:
        ans += 1
    num += 1
    if (num + 1) % 1000 == 0:
        print(ans)
        num = 0
        ans = 0

test_colorization = torch.load('./test_colorization.pt')
deepfool_success_colorization = torch.load('./pgd4/pgd_success.pt')

tmp_bpg = []
tmp_bpg2 = []
num = 0
num2 = 0
for i in range(test_colorization.shape[0]):
    if test_colorization[i][3] != 0:
        tmp_bpg.append(float(test_colorization[i][3]))
        tmp_bpg2.append(float(test_colorization[i][6]))
        num = num + 1



print(num)
tmp_bpg.sort()
tmp_bpg2.sort()
final_bpd = tmp_bpg[int(num * 0.975)]
final_bpd2 = tmp_bpg2[int(num * 0.975)]
print(final_bpd)

ans_num = 0
all_num = 0
for i in range(deepfool_success_colorization.shape[0]):
    if deepfool_success_colorization[i][6] >= 0.5:
        ans_num += 1
    if deepfool_success_colorization[i][3] != 0:
        all_num += 1

    if all_num % 200 == 0:

        print(all_num )
        print(ans_num )
        ans_num = 0
        all_num = 0
print(all_num)
print(ans_num / all_num)

