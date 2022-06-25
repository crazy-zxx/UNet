import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm3d(num_features=out_channels)
        )

        self.block_dilation = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm3d(num_features=out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm3d(num_features=out_channels)
        )

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.block_dilation(x)
        # x3 = torch.cat([x1, x2], dim=1)
        x3 = x1 + x2
        x4 = F.relu(x3, inplace=True)
        return x + x4


class ZUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ZUNet, self).__init__()

        self.init_conv = nn.Conv3d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.left_conv1 = ResBlock(in_channels=64, out_channels=64)
        self.down1 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.left_conv2 = ResBlock(in_channels=128, out_channels=128)
        self.down2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.left_conv3 = ResBlock(in_channels=256, out_channels=256)
        self.down3 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.left_conv4 = ResBlock(in_channels=512, out_channels=512)
        self.down4 = nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        # self.left_conv5 = ResBlock(in_channels=1024, out_channels=1024)
        # self.down5 = nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.center_conv = ResBlock(in_channels=1024, out_channels=1024)

        # self.up5 = nn.ConvTranspose3d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2, padding=0)
        # self.right_conv5 = ResBlock(in_channels=1024 + 1024, out_channels=1024)
        self.up4 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.right_conv4 = ResBlock(in_channels=512, out_channels=512)
        self.up3 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.right_conv3 = ResBlock(in_channels=256, out_channels=256)
        self.up2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.right_conv2 = ResBlock(in_channels=128, out_channels=128)
        self.up1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.right_conv1 = ResBlock(in_channels=64, out_channels=64)
        self.out_cov = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

        self.out_sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.init_conv(x)

        x1_lc = self.left_conv1(x0)
        x1_down = self.down1(x1_lc)
        x2_lc = self.left_conv2(x1_down)
        x2_down = self.down2(x2_lc)
        x3_lc = self.left_conv3(x2_down)
        x3_down = self.down3(x3_lc)
        x4_lc = self.left_conv4(x3_down)
        x4_down = self.down4(x4_lc)

        xc_c = self.center_conv(x4_down)

        x4_up = self.up4(xc_c)
        x4_rc = self.right_conv4(x4_up + x4_lc)
        x3_up = self.up3(x4_rc)
        x3_rc = self.right_conv3(x3_up + x3_lc)
        x2_up = self.up2(x3_rc)
        x2_rc = self.right_conv2(x2_up + x2_lc)
        x1_up = self.up1(x2_rc)
        x1_rc = self.right_conv1(x1_up + x1_lc)

        xo_c = self.out_cov(x1_rc)
        xo_sig = self.out_sig(xo_c)

        return xo_sig


if __name__ == '__main__':
    net = ZUNet(n_channels=1, n_classes=3)
    print(net)
