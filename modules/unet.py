import torch
import torch.nn as nn
import torchvision.models as models
from settings import num_classes, device

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.resnet_block = models.resnet18(weights=None)
        self.resnet_block.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet_block.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        return self.resnet_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip=2):
        super().__init__()
        self.skip = skip
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResNetBlock(out_channels * skip, out_channels)

    def forward(self, x1, *args):
        assert self.skip == len(args) + 1
        x1 = self.up(x1)
        x = torch.cat([*args, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetResNet(nn.Module):
    def __init__(self, n_classes=num_classes):
        super(UNetResNet, self).__init__()
        self.n_classes = n_classes

        self.inc = ResNetBlock(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y = self.outc(x)
        return y

# Example usage
if __name__ == '__main__':
    # Create a random input tensor
    x = torch.randn((9, 3, 256, 256))

    # Create an instance of the UNetResNet model
    model_resnet = UNetResNet(num_classes)

    # Forward pass
    y_resnet = model_resnet(x)

    # Print the output shape
    print(y_resnet.shape)
