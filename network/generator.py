from torch import Tensor
from utils import *
import torch
from torch import nn
import warnings

warnings.filterwarnings(action='ignore')


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_dim: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, in_dim, 3),
            nn.InstanceNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, in_dim, 3),
            nn.InstanceNorm2d(in_dim),
        )
    
    def forward(self, inp: Tensor) -> Tensor:
        return inp + self.block(inp)


# 生成器
class GeneratorResNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: int = 3,
                 num_residual_block: int = 9) -> None:
        super(GeneratorResNet, self).__init__()
        out_features = 64
        # 边界作为对称轴
        model = [
            nn.ReflectionPad2d(input_dim),
            nn.Conv2d(input_dim, out_features, kernel_size=7),
            nn.ReLU(inplace=True),
        ]
        input_features = out_features
        # 下采样2次
        for _ in range(encoder_layers - 1):
            out_features *= 2
            model += [
                nn.Conv2d(input_features, out_features,
                          kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            input_features = out_features
        # 残差块，循环9次
        for _ in range(num_residual_block):
            model += [ResidualBlock(in_dim=out_features)]
        # 上采样2次
        for _ in range(encoder_layers - 1):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            input_features = out_features
        model += [nn.ReflectionPad2d(input_dim),
                  nn.Conv2d(out_features, input_dim, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, inp: Tensor) -> Tensor:
        return self.model(inp)


def generatorResNet(**kwargs) -> GeneratorResNet:
    model = GeneratorResNet(**kwargs)
    model.apply(weights_init_normal)
    return model
