from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from new_model import GAU
import torch


def upsample1(in_planes: int, out_planes: int):
    channels_mid = in_planes//4

    return nn.Sequential(
        nn.Upsample(size=10, mode='bilinear'),
        nn.Conv2d(in_channels=in_planes, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels_mid, out_channels=out_planes, kernel_size=3,
                  stride=1, padding=1)
    )
def upsample2(in_planes: int, out_planes: int):
    channels_mid = in_planes//4

    return nn.Sequential(
        nn.Upsample(size=24, mode='bilinear'),
        nn.Conv2d(in_channels=in_planes, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=31, mode='bilinear'),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels_mid, out_channels=out_planes, kernel_size=3,
                  stride=1, padding=1)
    )
def upsample3(in_planes: int, out_planes: int):
    channels_mid = in_planes//4

    return nn.Sequential(
        nn.Upsample(size=41, mode='bilinear'),
        nn.Conv2d(in_channels=in_planes, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=channels_mid, out_channels=channels_mid, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels_mid, out_channels=out_planes, kernel_size=3,
                  stride=1, padding=1)
    )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)

def upsample(in_planes: int, out_planes: int, stride: int = 2):
    return nn.Sequential(deconv2x2(in_planes, out_planes, stride), nn.BatchNorm2d(out_planes))


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        is_upsample: int = 0,
        groups: int = 1,
        base_width: int = 128,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()

        self.is_upsample = is_upsample

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            # if self.is_upsample == 1:
            #     self.conv2 = upsample1(width, width)
            # if self.is_upsample == 2:
            #     self.conv2 = upsample2(width, width)
            # if self.is_upsample == 3:
            #     self.conv2 = upsample3(width, width)
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride



        # self.gau_1 = GAU(2048, 1024)
        # self.gau_2 = GAU(1024, 512)
        # self.gau_3 = GAU(512, 256)

        self.upsample_1 = upsample(2048, 1024)
        self.upsample_2 = upsample(1024, 512)
        self.upsample_3 = upsample(512, 256)
        # self.upsample_1 = upsample1(2048, 1024)
        # self.upsample_2 = upsample2(1024, 512)
        # self.upsample_3 = upsample3(512, 256)

    # def forward(self, x: Tensor, x_low: Tensor = None) -> Tensor:
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_upsample == 1:
            identity = self.upsample_1(x)
        if self.is_upsample == 2:
            identity = self.upsample_2(x)
        if self.is_upsample == 3:
            identity = self.upsample_3(x)

        out += identity
        out = self.relu(out)

        return out


class GAU_Decoder(nn.Module):
    def __init__(self,
                 block: Bottleneck):
        super(GAU_Decoder, self).__init__()
        self.bn_1_0 = block(inplanes=2048, planes=256, stride=2, is_upsample=1)
        self.bn_1_1 = block(inplanes=1024, planes=256)
        self.bn_1_2 = block(inplanes=1024, planes=256)

        self.bn_2_0 = block(inplanes=1024, planes=128, stride=2, is_upsample=2)
        self.bn_2_1 = block(inplanes=512, planes=128)
        self.bn_2_2 = block(inplanes=512, planes=128)

        self.bn_3_0 = block(inplanes=512, planes=64, stride=2, is_upsample=3)
        self.bn_3_1 = block(inplanes=256, planes=64)
        self.bn_3_2 = block(inplanes=256, planes=64)
        self.bn_3_3 = block(inplanes=256, planes=64)
        self.bn_3_4 = block(inplanes=256, planes=64)
        self.bn_3_5 = block(inplanes=256, planes=64)

        self.ca_1 = ChannelAttention(channel=1024)
        self.ca_2 = ChannelAttention(channel=512)
        self.ca_3 = ChannelAttention(channel=256)


        # pass

    def forward(self, x_list: list) -> list:
        x = x_list[0]
        # layer_1
        x = self.bn_1_0(x)
        x = self.bn_1_1(x)
        x = self.bn_1_2(x)
        x = x_list[3] + x


        feather_a = x

        # layer_2
        x = self.bn_2_0(x)
        x = self.bn_2_1(x)
        x = self.bn_2_2(x)
        x = x_list[2] + x
        feather_b = x

        # layer_3
        x = self.bn_3_0(x)
        x = self.bn_3_1(x)
        x = self.bn_3_2(x)
        x = self.bn_3_3(x)
        x = self.bn_3_4(x)
        x = self.bn_3_5(x)
        x = x_list[1] + x
        feather_c = x
        return [feather_c, feather_b, feather_a]




class resnet_50_Decoder(nn.Module):
    def __init__(self,
                 block: Bottleneck):
        super(resnet_50_Decoder, self).__init__()
        self.bn_1_0 = block(inplanes=2048, planes=256, stride=2, is_upsample=1)
        self.bn_1_1 = block(inplanes=1024, planes=256)
        self.bn_1_2 = block(inplanes=1024, planes=256)

        self.bn_2_0 = block(inplanes=1024, planes=128, stride=2, is_upsample=2)
        self.bn_2_1 = block(inplanes=512, planes=128)
        self.bn_2_2 = block(inplanes=512, planes=128)

        self.bn_3_0 = block(inplanes=512, planes=64, stride=2, is_upsample=3)
        self.bn_3_1 = block(inplanes=256, planes=64)
        self.bn_3_2 = block(inplanes=256, planes=64)
        self.bn_3_3 = block(inplanes=256, planes=64)
        self.bn_3_4 = block(inplanes=256, planes=64)
        self.bn_3_5 = block(inplanes=256, planes=64)


    def forward(self, x_list: list) -> list:
        x = x_list[0]
        # layer_1
        x = self.bn_1_0(x)
        x = self.bn_1_1(x)
        x = self.bn_1_2(x)
        feather_a = x

        # layer_2
        x = self.bn_2_0(x)
        x = self.bn_2_1(x)
        x = self.bn_2_2(x)
        feather_b = x

        # layer_3
        x = self.bn_3_0(x)
        x = self.bn_3_1(x)
        x = self.bn_3_2(x)
        x = self.bn_3_3(x)
        x = self.bn_3_4(x)
        x = self.bn_3_5(x)
        feather_c = x
        return [feather_c, feather_b, feather_a]

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output