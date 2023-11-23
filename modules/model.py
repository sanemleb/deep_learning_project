import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from modules.settings import num_classes

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=num_classes, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Convert the final output to the same data type as the input
        x = x.to(dtype=x.dtype)

        x = self.final_conv(x)
        # x = torch.nn.functional.softmax(x, dim=1)  # Apply sigmoid activation

        return x



# import torch 
# import torch.nn as nn
# import torchvision.transforms.functional as TF 
# from modules.settings import num_classes

# class UNET(nn.Module):
    
#     def __init__(self, in_channels=3, classes=num_classes):
#         super(UNET, self).__init__()
#         self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
#         self.double_conv_downs = nn.ModuleList(
#             [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
#         self.up_trans = nn.ModuleList(
#             [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
#              for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
#         self.double_conv_ups = nn.ModuleList(
#         [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
#         self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        
#     def __double_conv(self, in_channels, out_channels):
#         conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         return conv
    
#     def forward(self, x):
#         # down layers
#         concat_layers = []
        
#         for down in self.double_conv_downs:
#             x = down(x)
#             if down != self.double_conv_downs[-1]:
#                 concat_layers.append(x)
#                 x = self.max_pool_2x2(x)
        
#         concat_layers = concat_layers[::-1]
        
#         # up layers
#         for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
#             x = up_trans(x)
#             if x.shape != concat_layer.shape:
#                 x = TF.resize(x, concat_layer.shape[2:])
            
#             concatenated = torch.cat((concat_layer, x), dim=1)
#             x = double_conv_up(concatenated)
            
#         # Convert the final output to the same data type as the input
#         x = x.to(dtype=x.dtype)
#         x = self.final_conv(x)
        
#         return x 