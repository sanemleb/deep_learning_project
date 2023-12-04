import torch
import torch.nn as nn
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

channels = 3
kernel_size = (3, 3)  # <-- Kernel size
conv_stride = 1  # <-- Stride
conv_pad = 1  # <-- Padding
# decoder_conv_val
kernel_size_decoder = (2, 2)
stride_decoder = 2
padding_decoder = 0
# output_conv_val
# in_channels_out = 64
# out_channels_out = 2
kernel_size_out = (1, 1)
stride_out = 1
padding_out = 0
# defining the double convolutional layer for the U-Net
class double_conv(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels):
        super().__init__()

        # first convolution
        self.conv_down_1 = Conv2d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=conv_pad,
        )
        self.b_norm_1 = torch.nn.BatchNorm2d(conv_out_channels)

        # second convolution
        self.conv_down_2 = Conv2d(
            in_channels=conv_out_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=conv_pad,
        )
        self.b_norm_2 = torch.nn.BatchNorm2d(conv_out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = self.conv_down_1(input)
        x = self.b_norm_1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.conv_down_2(x)
        x = self.b_norm_2(x)
        x = relu(x)
        x = self.dropout(x)
        return x


# defining the encoder
class encoder(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels):
        super().__init__()
        self.d_conv = double_conv(conv_in_channels, conv_out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        x = self.d_conv(input)
        p = self.pool(x)
        return x, p


# defining the decoder
class decoder(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            conv_in_channels,
            conv_out_channels,
            kernel_size=kernel_size_decoder,
            stride=stride_decoder,
            padding=padding_decoder,
        )
        self.conv_d = double_conv(
            conv_out_channels + conv_out_channels, conv_out_channels
        )

    def forward(self, input, skip):
        x = self.up(input)
        x = torch.cat([x, skip], axis=1)
        x = self.conv_d(x)
        return x


# defining the las convolutional output
class output_conv(nn.Module):
    def __init__(self, in_channels_out, out_channels_out):
        super().__init__()
        self.output_cv = Conv2d(
            in_channels=in_channels_out,
            out_channels=out_channels_out,
            kernel_size=kernel_size_out,
            stride=stride_out,
            padding=padding_out,
        )
        self.b_norm_out = torch.nn.BatchNorm2d(out_channels_out)

    def forward(self, input):
        x = self.output_cv(input)
        x = self.b_norm_out(x)
        x = relu(x)
        return x


# defining the U-Net
class U_Net_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.dummy_conv = Conv2d(3, 64, 3) for testing purposes
        # four encoder layer
        self.down_1 = encoder(channels, 64)
        self.down_2 = encoder(64, 128)
        self.down_3 = encoder(128, 256)
        self.down_4 = encoder(256, 512)

        self.b_1 = double_conv(512, 1024)

        # four decoder layer
        self.up_1 = decoder(1024, 512)
        self.up_2 = decoder(512, 256)
        self.up_3 = decoder(256, 128)
        self.up_4 = decoder(128, 64)

        # output layer
        self.out = output_conv(
            64, 9
        )  ### has to be changed if more or less classes are availabel. This setting creates a picture with 9 different classes as output
        self.dropout = nn.Dropout(0.1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        s_1, p_1 = self.down_1(input)
        # x = self.dummy_conv(input)
        s_2, p_2 = self.down_2(p_1)
        s_3, p_3 = self.down_3(p_2)
        s_4, p_4 = self.down_4(p_3)

        x_b = self.b_1(p_4)
        x_1 = self.up_1(x_b, s_4)
        x_2 = self.up_2(x_1, s_3)
        x_3 = self.up_3(x_2, s_2)
        x_4 = self.up_4(x_3, s_1)

        out_ = self.out(x_4)
        out_ = self.sig(out_)

        return out_

        # return self.output
