import torch
import torch.nn as nn
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.ECAAttention import ECAAttention
# from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.ACmixAttention import ACmix
from torch.nn import init

class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        self.sp_attn = ECAAttention(kernel_size=5)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        x_master = self.sp_attn(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        x1_2 = self.sp_attn(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        x2_2 =  self.sp_attn(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        x3_2 = self.sp_attn(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out



class DAM_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.dam_1 = Channel_AE(channel=256)
        self.dam_2 = Channel_AE(channel=512)
        self.dam_3 = Channel_AE(channel=1024)

    def forward(self, x: list):

        x[0] = self.dam_1(x[0])
        x[1] = self.dam_2(x[1])
        x[2] = self.dam_3(x[2])

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Channel_AE(nn.Module):
    def __init__(self, channel=512):
        super(Channel_AE, self).__init__()

        # encoder
        self.conv_1 = nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False)
        self.instance1 = nn.InstanceNorm2d(channel // 2)
        self.conv_2 = nn.Conv2d(channel // 2, channel // 4, kernel_size=3, padding=1, bias=False)
        self.instance2 = nn.InstanceNorm2d(channel // 4)
        self.conv_3 = nn.Conv2d(channel // 4, channel // 8, kernel_size=3, padding=1, bias=False)
        self.instance3 = nn.InstanceNorm2d(channel // 8)
        self.conv_4 = nn.Conv2d(channel // 8, channel // 16, kernel_size=3, padding=1, bias=False)
        self.instance4 = nn.InstanceNorm2d(channel // 16)
        self.conv_5 = nn.Conv2d(channel // 16, 1, kernel_size=3, padding=1, bias=False)
        # self.instance4 = nn.InstanceNorm2d(1)

        # decoder
        self.de_conv_5 = nn.Conv2d(1, channel // 16, kernel_size=3, padding=1, bias=False)
        self.instance5 = nn.InstanceNorm2d(channel // 16)
        self.conv_6 = nn.Conv2d(channel // 16, channel // 8, kernel_size=3, padding=1, bias=False)
        self.instance6 = nn.InstanceNorm2d(channel // 8)
        self.conv_7 = nn.Conv2d(channel // 8, channel // 4, kernel_size=3, padding=1, bias=False)
        self.instance7 = nn.InstanceNorm2d(channel // 4)
        self.conv_8 = nn.Conv2d(channel // 4, channel // 2, kernel_size=3, padding=1, bias=False)
        self.instance8 = nn.InstanceNorm2d(channel // 2)
        self.conv_9 = nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1, bias=False)
        self.instance9 = nn.InstanceNorm2d(channel)
        #
        self.de_conv_10 = nn.Conv2d(channel, channel//32, kernel_size=1, padding=0, bias=False)
        self.de_conv_0 = nn.Conv2d(1, channel//32, kernel_size=1, padding=0, bias=False)



        self.relu = nn.ReLU(inplace=True)
        self.cbam1 = CBAMBlock(channel=channel//2, reduction=16, kernel_size=7)
        self.cbam2 = CBAMBlock(channel=channel//4, reduction=16, kernel_size=7)
        self.cbam3 = CBAMBlock(channel=channel//8, reduction=16, kernel_size=7)


    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        # res = x

        x = self.relu(self.instance1(self.conv_1(x)))
        x = self.cbam1(x)
        x = self.relu(self.instance2(self.conv_2(x)))
        x = self.cbam2(x)
        x = self.relu(self.instance3(self.conv_3(x)))
        x = self.cbam3(x)
        x = self.relu(self.instance4(self.conv_4(x)))
        x = self.cbam4(x)
        x = self.relu(self.conv_5(x))
        x0 = x
        x = self.relu(self.instance5(self.de_conv_5(x)))
        x = self.cbam4(x)
        x1 = x
        x = self.relu(self.instance6(self.conv_6(x)))
        x = self.cbam3(x)
        x2 = x
        x = self.relu(self.instance7(self.conv_7(x)))
        x = self.cbam2(x)
        x3 = x
        x = self.relu(self.instance8(self.conv_8(x)))
        x = self.cbam1(x)
        x4 = x
        x = self.relu(self.instance9(self.conv_9(x)))
        x5 = self.de_conv_10(x)
        x0 = self.de_conv_0(x0)

        out = torch.cat([x5, x0, x1, x2, x3, x4], 1)


        return out

    def __init__(self, channel=512,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        # out=out*self.sa(out)
        return out