import torch
from torch import nn


class DoubleConv(nn.Module):
    """double convolution"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # 进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # 进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.left_conv1 = DoubleConv(in_channels=n_channels, out_channels=64)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_conv2 = DoubleConv(in_channels=64, out_channels=128)
        self.left_conv3 = DoubleConv(in_channels=128, out_channels=256)
        self.left_conv4 = DoubleConv(in_channels=256, out_channels=512)
        self.center_conv = DoubleConv(in_channels=512, out_channels=1024)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.right_conv1 = DoubleConv(in_channels=1024, out_channels=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.right_conv2 = DoubleConv(in_channels=512, out_channels=256)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.right_conv3 = DoubleConv(in_channels=256, out_channels=128)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.right_conv4 = DoubleConv(in_channels=128, out_channels=64)
        self.out_cov = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

        self.out = nn.Sigmoid()

    def forward(self, x):
        x1 = self.left_conv1(x)
        x1_down = self.down(x1)
        x2 = self.left_conv2(x1_down)
        x2_down = self.down(x2)
        x3 = self.left_conv3(x2_down)
        x3_down = self.down(x3)
        x4 = self.left_conv4(x3_down)
        x4_down = self.down(x4)
        x5 = self.center_conv(x4_down)
        x6 = self.up1(x5)
        temp = torch.cat([x6, x4], dim=1)
        x6 = self.right_conv1(temp)
        x7 = self.up2(x6)
        temp = torch.cat([x7, x3], dim=1)
        x7 = self.right_conv2(temp)
        x8 = self.up3(x7)
        temp = torch.cat([x8, x2], dim=1)
        x8 = self.right_conv3(temp)
        x9 = self.up4(x8)
        temp = torch.cat([x9, x1], dim=1)
        x9 = self.right_conv4(temp)
        x10 = self.out_cov(x9)

        x10 = self.out(x10)

        return x10


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
