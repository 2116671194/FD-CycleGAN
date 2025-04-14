import os
import numpy as np
from PIL import Image
from torch import Tensor
import torchvision.transforms as transforms
from torch.autograd import Variable
from typing import Any
from tqdm import tqdm
import argparse
import torch
import yaml

from network.generator import GeneratorResNet


# 判断图像后缀
def is_image(path: str) -> bool:
    return path.split('.')[-1] in ['jpg', 'png', 'bmp']


# 如果输入的数据集是灰度图像，将图片转化为rgb图像
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


# 图像预处理
def image_process(
        path: str,
        config: Any) -> Tensor:
    # 数据预处理
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    image = Image.open(path)
    if image.mode != 'RGB':
        image = to_rgb(image)
    image = image.resize((config['TRAIN']['DATASET']['IMAGE_SIZE'],
                          config['TRAIN']['DATASET']['IMAGE_SIZE']), Image.BICUBIC)
    transformer = transforms.Compose(transforms_)
    image = transformer(image)
    image = image.unsqueeze(dim=0)  # [C, H, W] -> [B, C, H, W]
    return image


def test_model(
        G,
        p: str,
        device: torch.device,
        config: Any
) -> None:

    if not is_image(p):
        raise ValueError('image suffix must is jpg or png or bmp')

    image = image_process(p, config)
    image = Variable(image, requires_grad=False)
    image = image.to(device)
    G.eval()
    with torch.no_grad():
        output = G(image)
    # [-1, 1] -> [0-1]
    output = 0.5 * (output + 1.0)
    if torch.cuda.is_available():
        save_out = np.uint8(255 * output.data.cpu().numpy().squeeze())
    else:
        save_out = np.uint8(255 * output.data.numpy().squeeze())
    save_out = save_out.transpose(1, 2, 0)
    save_out = Image.fromarray(save_out)
    save_out = save_out.resize((200, 250), , Image.BICUBIC)
    if not os.path.exists(f'{config["MODEL"]["G"]["NAME"]}_save_out'):
        os.mkdir(f'{config["MODEL"]["G"]["NAME"]}_save_out')
    filename = p.split('/')[-1]
    save_out.save(os.path.join(f'{config["MODEL"]["G"]["NAME"]}_save_out', filename))


# 测试模型
def test(
        G,
        device: torch.device,
        config: Any
) -> None:
    # ----------
    #  Testing
    # ----------
    # 加载模型
    p = config['TEST']['TEST_PATH']

    if os.path.isfile(p):
        test_model(G, p, device, config)
    elif os.path.isdir(p):
        for pp in tqdm(os.listdir(p)):
            test_model(G, p + pp, device, config)
    print('test finish !!!')


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./config/cyclegan.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    G = GeneratorResNet(config['MODEL']['G']['IN_CHANNELS'], config['MODEL']['G']['ENCODER_LAYERS'], config['MODEL']['G']['NUM_RESIDUAL_BLOCK'])

    G = G.to(device)

    if config['TEST']['IMAGE_KIND'] == 'color':
        checkpoint = torch.load(f'{config["MODEL"]["G"]["NAME"]}_model/G_BA_{config["TRAIN"]["LR"]["EPOCH"]}.pth')
        G.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(f'{config["MODEL"]["G"]["NAME"]}_model/G_AB_{config["TRAIN"]["LR"]["EPOCH"]}.pth')
        G.load_state_dict(checkpoint['model_state_dict'])
    test(G, device, config)
