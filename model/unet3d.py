import torch
from torch import nn


class DoubleConvLeft(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvLeft, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvRight(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvRight, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.left_conv1 = DoubleConvLeft(in_channels=n_channels, out_channels=64)
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv2 = DoubleConvLeft(in_channels=64, out_channels=128)
        self.left_conv3 = DoubleConvLeft(in_channels=128, out_channels=256)
        self.center_conv = DoubleConvLeft(in_channels=256, out_channels=512)
        self.up1 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.right_conv1 = DoubleConvRight(in_channels=512, out_channels=256)
        self.up2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.right_conv2 = DoubleConvRight(in_channels=256, out_channels=128)
        self.up3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.right_conv3 = DoubleConvRight(in_channels=128, out_channels=64)
        self.out_cov = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

        self.out = nn.Sigmoid()

    def forward(self, x):
        x1 = self.left_conv1(x)
        x1_down = self.down(x1)
        x2 = self.left_conv2(x1_down)
        x2_down = self.down(x2)
        x3 = self.left_conv3(x2_down)
        x3_down = self.down(x3)
        x4 = self.center_conv(x3_down)
        x4_up = self.up1(x4)
        temp = torch.cat([x4_up, x3], dim=1)
        x5 = self.right_conv1(temp)
        x5_up = self.up2(x5)
        temp = torch.cat([x5_up, x2], dim=1)
        x6 = self.right_conv2(temp)
        x6_up = self.up3(x6)
        temp = torch.cat([x6_up, x1], dim=1)
        x7 = self.right_conv3(temp)
        x8 = self.out_cov(x7)
        x9 = self.out(x8)

        return x9


if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=3)
    print(net)
