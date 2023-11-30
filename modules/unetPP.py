import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
from modules.settings import num_classes, device

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip=2):
        # For UNet   skip = 2, 
        # For UNet++ skip = 2 in UNet(L1), 3 in UNet(L2), 4 in UNet(l3),...
        super().__init__()
        self.skip = skip
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * skip, out_channels)

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
    
class UNetPP(nn.Module):
    def __init__(self, n_classes=num_classes, deep_supervision=False):
        super(UNetPP, self).__init__()
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.conv0_0 = DoubleConv(3, 64)
        self.conv1_0 = Down( 64, 128)
        self.conv2_0 = Down(128, 256)
        self.conv3_0 = Down(256, 512)
        self.conv4_0 = Down(512, 1024)

        self.conv0_1 = Up( 128,  64, 2)
        self.conv1_1 = Up( 256, 128, 2)
        self.conv2_1 = Up( 512, 256, 2)
        self.conv3_1 = Up(1024, 512, 2)

        self.conv0_2 = Up( 128,  64, 3)
        self.conv1_2 = Up( 256, 128, 3)
        self.conv2_2 = Up( 512, 256, 3)

        self.conv0_3 = Up( 128,  64, 4)
        self.conv1_3 = Up( 256, 128, 4)

        self.conv0_4 = Up( 128,  64, 5)

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

        y = self.outc(x0_4)
        return y

if __name__ == '__main__':
    
    x = torch.randn((9, 3, 256, 256)).to(device)

    # UnetPP
    model = UNetPP(num_classes).to(device)
    y = model(x).cpu()
    print(y.shape)
    assert y.size() == (9,10,256,256)

