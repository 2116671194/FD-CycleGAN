import torch
import torch.nn as nn
from torch import Tensor
from utils import *


# 判别器(感受野：1->70)
class PatchDiscriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 spectral_norm: bool = False) -> None:
        """
        :param input_dim: 输入特征维度
        :param spectral_norm: 谱归一化
        """
        super(PatchDiscriminator, self).__init__()
        channels = input_dim  # input_shape:(3， 256， 256)
        self.spectral_norm = spectral_norm

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """
            :param stride: 步长
            :param in_filters: input channels
            :param out_filters: output channels
            :param normalize: 实例归一化
            """
            if self.spectral_norm:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters,
                                                           kernel_size=4, stride=stride, padding=1))]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        :param inp: [N, 3, 256, 256]
        :return: [N, 1, 30, 30]
        """
        return self.model(inp)


def patchDiscriminator(**kwargs) -> PatchDiscriminator:
    model = PatchDiscriminator(**kwargs)
    model.apply(weights_init_normal)
    return model


