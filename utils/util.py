import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
import numpy as np
from torchvision.utils import save_image
import logging

__all__ = ['weights_init_normal', 'ReplayBuffer', 'DecayLR', 'logger']


# 配置日志
def logger(filename: str = 'run'):
    # 配置日志记录器
    logging.basicConfig(filename=f'{filename}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 定义控制台处理器的格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # 将控制台处理器添加到日志记录器中
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    return logger


# 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__  # m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv2d") != -1:  # find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  # m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:  # hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("Linear") != -1:  # find():实现查找classname中是否含有Linear字符，没有返回-1；有返回0
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  # m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:  # hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:  # find():实现查找classname中是否含有InstanceNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)  # m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.


# 先前生成的样本的缓冲区
class ReplayBuffer(object):
    def __init__(self,
                 max_size: int = 50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  # 放入一张图像，再从buffer里取一张出来
        to_return = []  # 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  # 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# 设置学习率为初始学习率乘以给定lr_lambda函数的值
class DecayLR(object):
    def __init__(self,
                 n_epochs: int,
                 offset: int,
                 decay_start_epoch: int):  # (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (
                       n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  # 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int):  # return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
