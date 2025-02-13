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
    def __init__(self, in_channels, out_channels, dconv_time=False, time_embed_count=0,
                 dconv_bnorm=False, dconv_relu=False):
        super().__init__()

        # Double Convolution Step
        self.conv = DoubleConvOne(
            in_channels, out_channels, use_time=dconv_time, time_embed_count=time_embed_count,
            use_bnorm=dconv_bnorm, use_relu=dconv_relu
        )

        # 1D Max Pooling to Shrink image
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, batch, time_embed=None):
        # Channel change for skip connection
        conv_batch = self.conv(batch, time_embed)

        # Downsample for next step
        encoded_batch = self.pool(conv_batch)

        return conv_batch, encoded_batch


class AttentionOne(nn.Module):
    def __init__(self, dec_skip_channels, inter_channels):
        super().__init__()

        # Perform 1x1 convolution on decoder channels
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(dec_skip_channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
        )

        # Perform 1x1 convolution on skip connections
        self.skip_conv = nn.Sequential(
            nn.Conv1d(dec_skip_channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
        )

        # Create an attention mask
        self.masker = nn.Sequential(
            nn.Conv1d(inter_channels, 1, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        # Activation function for Spatial Attention Block
        self.activation = nn.ReLU(inplace=True)

    def forward(self, decoder, skip):
        # Create an intermediate decoder representation
        decoder_int = self.decoder_conv(decoder)

        # Create an intermediate skip representation
        skip_int = self.skip_conv(skip)

        # Add together then apply activation to keep only noticeable features from both representations
        combo_int = self.activation(decoder_int + skip_int)

        # Create a normalized attention mask to specify where to pay "attention" to
        masked_int = self.masker(combo_int)

        # Adjust the skip connection information using the mask to tailor the information
        return skip * masked_int


# TODO
class UpSampleOne(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, dconv_time=False, time_embed_count=0,
                 dconv_bnorm=False, dconv_relu=False):
        super().__init__()

        # 1D Upscale with channel shrinkage
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

        # Attention block but only if needed
        self.need_attention = use_attention
        if self.need_attention:
            self.attention = AttentionOne(in_channels // 2, out_channels // 2)
        else:
            self.attention = None

        # Double Convolution Step (assumes skip connection is present to combine long and short paths)
        self.conv = DoubleConvOne(
            in_channels, out_channels, use_time=dconv_time, time_embed_count=time_embed_count,
            use_bnorm=dconv_bnorm, use_relu=dconv_relu
        )

    def forward(self, cur, skip, time_embed=None):
        # Upscale from the last encoding
        cur_upscaled = self.upscaler(cur)

        # Apply attention block to skip connection if needed
        if self.need_attention:
            skip = self.attention(cur_upscaled, skip)

        # Combine results then final convolution
        combined = torch.cat([cur_upscaled, skip], 1)
        return self.conv(combined, time_embed)


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
