import torch
import math
import torch.nn as nn


# Initially Copied from Denoising Diffusion Tutorial - https://www.youtube.com/watch?v=a4Yfz2FxXiY
class DiffusionSinPosEmbeds(nn.Module):
    def __init__(self, dimensions, theta=10000):
        super().__init__()
        self.dimensions = dimensions
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dimensions = self.dimensions // 2
        embeddings = math.log(self.theta) / (half_dimensions - 1)
        embeddings = torch.exp(torch.arange(half_dimensions, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConvOne(nn.Module):
    def __init__(self, in_channels, out_channels, use_time=False, time_embed_count=0, use_bnorm=False, use_relu=False):
        super().__init__()

        # First Convolution + Batch Norm and ReLU options
        first_conv_list = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_bnorm:
            first_conv_list.append(nn.BatchNorm1d(out_channels))
        if use_relu:
            first_conv_list.append(nn.ReLU(inplace=True))
        self.first_conv = nn.Sequential(*first_conv_list)

        # Optional time modification
        self.need_time = use_time
        if self.need_time:
            self.embed_adjuster = nn.Sequential(
                nn.Linear(time_embed_count, out_channels),
                nn.ReLU()
            )
        else:
            self.embed_adjuster = None

        # Second Convolution + Batch Norm and ReLU options
        sec_conv_list = [nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)]
        if use_bnorm:
            sec_conv_list.append(nn.BatchNorm1d(out_channels))
        if use_relu:
            sec_conv_list.append(nn.ReLU(inplace=True))
        self.sec_conv = nn.Sequential(*sec_conv_list)

    def forward(self, batch, time_embed=None):
        # Do first convolution set
        batch = self.first_conv(batch)

        # Create time embedding if needed
        if self.need_time:
            # Assert that the time_embed is not none
            assert time_embed is not None, (
                "Time embedding is not provided for double convolution step.\n"
                f"\tDouble Conv Layer info: {self}.\n\n"
            )

            # Retrieves the embed given a time step
            adjusted_time_embed = self.embed_adjuster(time_embed)

            # Expands the shape to (batch, out_channels, 1)
            adjusted_time_embed = adjusted_time_embed[(...,) + (None,)]

            # Adds time-sensitive embeddings to batch
            batch = batch + adjusted_time_embed

        # Do second convolution set
        batch = self.sec_conv(batch)
        return batch


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
