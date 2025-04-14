import os
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from pytorch_fid import fid_score
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS
from torchvision import transforms
import warnings
import argparse
import yaml

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config_path",
                    type=str,
                    default="../config/cyclegan.yaml",
                    help="Path to train config file.")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.full_load(f)


def gray_psnr(img1, img2, bit=8):
    max = 2**bit - 1
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(max**2 / mse)
    return psnr


def color_psnr(img1, img2, bit=8):
    psnr = 0
    for i in range(img1.shape[-1]):
        psnr += gray_psnr(img1[:,:,i], img2[:,:,i], bit)
    avg_psnr = psnr / 3.
    return avg_psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=15, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# 计算FID分数
def compute_fid_score(
        real_path: str,
        generated_path: str,
        batch_size: int
) -> np.ndarray:
    """
    :param real_path: 真实图片路径
    :param generated_path: 生成图片路径
    :param batch_size: 批次数
    :return: FID(弗雷歇起始距离)
    """
    return fid_score.calculate_fid_given_paths(paths=[real_path, generated_path],
                                               batch_size=batch_size,
                                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                               dims=2048)

# 图像预处理
def image_preprocess(path: str) -> Tensor:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(path)
    # ndarray->torch->归一化[0-1]
    image = np.array(image)
    image = transform(image)
    image = image.unsqueeze(dim=0)
    return image

# 计算图像感知相似性
def compute_lpips(
        real_path: str,
        fake_path: str,
        net='vgg'
) -> tuple:
    """
    :param real_path: 真实图片路径
    :param fake_path: 生成图片路径，其中真实图片和生成图像须一一对应
    :return: (avg_lpips,lpips_list)
    """
    lpips_vgg = LPIPS(net=net)
    lpips_list = []
    for path in os.listdir(real_path):
        real = image_preprocess(os.path.join(real_path, path))
        fake = image_preprocess(os.path.join(fake_path, path))
        distance = lpips_vgg(real, fake)
        lpips_list.append(distance.item())
    avg_lpips = np.array(lpips_list).mean()
    return avg_lpips, lpips_list

# 计算图像结构相似度
def computer_ssim(
        pred_path: str,
        target_path: str,
        window_size: int=11,
        size_average: bool=True
) -> tuple:
    """
    :param pred_path: 真实图片路径
    :param target_path: 生成图片路径，其中真实图片和生成图像须一一对应
    :return: (avg_ssim, ssim_list)
    """
    ssim = SSIM(window_size=window_size, size_average=size_average)
    ssim_list = []
    for path in os.listdir(pred_path):
        pred = image_preprocess(os.path.join(pred_path, path))
        target = image_preprocess(os.path.join(target_path, path))
        ssim_list.append(ssim(pred, target).item())
    avg_ssim = np.array(ssim_list).mean()
    return avg_ssim, ssim_list

# 计算峰值信噪比
def compute_psnr(
        real_path: str,
        fake_path: str,
        bit: int=8
) -> tuple:
    psnr_list = []
    for path in os.listdir(real_path):
        real = np.array(Image.open(os.path.join(real_path, path)))
        fake = np.array(Image.open(os.path.join(fake_path, path)))
        psnr_list.append(color_psnr(fake, real, bit=bit))
    avg_psnr = np.array(psnr_list).mean()
    return avg_psnr, psnr_list


avg_psnr, psnr_list = compute_psnr(config['EVALUATE']['REAL_PATH'],
                                   config['EVALUATE']['FAKE_PATH'])
print(f'图像峰值信噪比：{avg_psnr}, {psnr_list}')


avg_ssim, ssim_list = computer_ssim(config['EVALUATE']['REAL_PATH'],
                                    config['EVALUATE']['FAKE_PATH'],
                                    window_size=config['EVALUATE']['SSIM']['WINDOW_SIZE'],
                                    size_average=config['EVALUATE']['SSIM']['SIZE_AVE'])
print(f'图像结构相似性：{avg_ssim}, {ssim_list}')

avg_lpips, lpips_list = compute_lpips(config['EVALUATE']['REAL_PATH'],
                                      config['EVALUATE']['FAKE_PATH'],
                                      config['EVALUATE']['LPIPS']['NET'])
print(f'图像感知相似性：{avg_lpips}, {lpips_list}')

fid = compute_fid_score(config['EVALUATE']['REAL_PATH'],
                        config['EVALUATE']['FAKE_PATH'],
                        config['EVALUATE']['FID']['BATCH_SIZE'])

print(f'FID分数：{fid}')
