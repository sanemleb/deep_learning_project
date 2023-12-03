import torch
import torch.nn as nn
import torchvision.models as models
from unetPP import Up, Down, OutConv
from settings import num_classes, device

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.resnet_block = models.resnet18(weights=None)
        # Replace the first layer to match the number of input channels
        self.resnet_block.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace the last layer to match the number of output channels
        self.resnet_block.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        return self.resnet_block(x)

class UNetResNet(nn.Module):
    def __init__(self, n_classes=num_classes, deep_supervision=False):
        super(UNetResNet, self).__init__()
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Replace DoubleConv with ResNetBlock
        self.conv0_0 = ResNetBlock(3, 64)
        self.conv1_0 = Down(64, 128)
        self.conv2_0 = Down(128, 256)
        self.conv3_0 = Down(256, 512)
        self.conv4_0 = Down(512, 1024)

        self.conv0_1 = Up(128, 64, 2)
        self.conv1_1 = Up(256, 128, 2)
        self.conv2_1 = Up(512, 256, 2)
        self.conv3_1 = Up(1024, 512, 2)

        self.conv0_2 = Up(128, 64, 3)
        self.conv1_2 = Up(256, 128, 3)
        self.conv2_2 = Up(512, 256, 3)

        self.conv0_3 = Up(128, 64, 4)
        self.conv1_3 = Up(256, 128, 4)

        self.conv0_4 = Up(128, 64, 5)

        if self.deep_supervision:
            self.outc1 = OutConv(64, n_classes)
            self.outc2 = OutConv(64, n_classes)
            self.outc3 = OutConv(64, n_classes)
            self.outc4 = OutConv(64, n_classes)
        else:
            self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(x1_0, x0_0)

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(x2_0, x1_0)
        x0_2 = self.conv0_2(x1_1, x0_0, x0_1)

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(x3_0, x2_0)
        x1_2 = self.conv1_2(x2_1, x1_0, x1_1)
        x0_3 = self.conv0_3(x1_2, x0_0, x0_1, x0_2)

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(x4_0, x3_0)
        x2_2 = self.conv2_2(x3_1, x2_0, x2_1)
        x1_3 = self.conv1_3(x2_2, x1_0, x1_1, x1_2)
        x0_4 = self.conv0_4(x1_3, x0_0, x0_1, x0_2, x0_3)

        if self.deep_supervision:
            out1 = self.outc1(x0_1)
            out2 = self.outc2(x0_2)
            out3 = self.outc3(x0_3)
            out4 = self.outc4(x0_4)
            return out1, out2, out3, out4
        else:
            out = self.outc(x0_4)
            return out

if __name__ == '__main__':
    x = torch.randn((9, 3, 256, 256)).to(device)

    # U-Net with ResNet
    model_resnet = UNetResNet(num_classes).to(device)
    y_resnet = model_resnet(x).cpu()
    print(y_resnet.shape)
    assert y_resnet.size() == (9, 10, 256, 256)
