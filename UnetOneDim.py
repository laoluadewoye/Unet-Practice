import torch
import torch.nn as nn


class DoubleConvOne(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            # Transfer to new level of channels + ReLU
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Second Convolution + ReLU
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSampleOne(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Double Convolution Step
        self.conv = DoubleConvOne(in_channels, out_channels)

        # 1D Max Pooling to Shrink image
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # Channel change for skip connection
        down = self.conv(x)

        # Downsample for next step
        p = self.pool(down)

        return down, p


class UpSampleOne(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1D Upscale with channel shrinkage
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

        # Double Convolution Step (assumes skip connection is present to combine long and short paths)
        self.conv = DoubleConvOne(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upscale
        x1 = self.up(x1)

        # Combine then final convolution
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNETOne(nn.Module):
    def __init__(self, in_channels, channel_list, out_layer):
        super().__init__()

        # Create down samplers using nn.ModuleList
        self.down_samplers = nn.ModuleList([
            DownSampleOne(in_channels if i == 0 else channel_list[i - 1], channel_list[i])
            for i in range(len(channel_list) - 1)
        ])

        # Create bottleneck transition
        self.bottle_neck = DoubleConvOne(channel_list[-2], channel_list[-1])

        # Create up samplers using nn.ModuleList
        self.up_samplers = nn.ModuleList([
            UpSampleOne(channel_list[i], channel_list[i - 1])
            for i in range(len(channel_list) - 1, 0, -1)
        ])

        self.out_layer = out_layer

    def forward(self, x):
        skip_connections = []
        cur_pool = x

        # Downsample through list
        for down_sampler in self.down_samplers:
            down_skip, pool = down_sampler(cur_pool)
            skip_connections.append(down_skip)
            cur_pool = pool

        # Bottleneck phase
        bn = self.bottle_neck(cur_pool)
        cur_up = bn

        # Upsample through list
        for up_sampler in self.up_samplers:
            cur_up = up_sampler(cur_up, skip_connections.pop())

        return self.out_layer(cur_up)
