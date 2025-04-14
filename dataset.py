import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config_path",
                    type=str,
                    default="./config/cyclegan.yaml",
                    help="Path to train config file.")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.full_load(f)


# 如果输入的数据集是灰度图像，将图片转化为rgb图像
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


# 构建数据集
class ImageDataset(Dataset):
    def __init__(self,
                 root: str,
                 transforms_=None,
                 unaligned: bool = False):  # (root = "./CUHK_train", unaligned=True:非对其数据)
        self.transform = transforms.Compose(transforms_)  # transform变为tensor数据
        self.unaligned = unaligned
        
        self.files_A = sorted(glob.glob(os.path.join(root, 'photos') + "/*.*"))  # "./datasets/facades/trainA/*.*"
        self.files_B = sorted(glob.glob(os.path.join(root, 'sketches') + "/*.*"))  # "./datasets/facades/trainB/*.*"
    
    # 获取A,B数据的长度
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])  # 在A中取一张照片
        
        if self.unaligned:  # 如果采用非配对数据，在B中随机取一张
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        
        # 如果是灰度图，把灰度图转换为RGB图
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        image_A = image_A.resize((config['TRAIN']['DATASET']['IMAGE_SIZE'],
                                  config['TRAIN']['DATASET']['IMAGE_SIZE']), Image.BICUBIC)
        image_B = image_B.resize((config['TRAIN']['DATASET']['IMAGE_SIZE'],
                                  config['TRAIN']['DATASET']['IMAGE_SIZE']), Image.BICUBIC)
        # 把RGB图像转换为tensor图, 方便计算，返回字典数据
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

