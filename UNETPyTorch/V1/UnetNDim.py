import math
import copy
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ConvUtils import *


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


class DoubleConvN(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, dconv_act_fn=None, use_time=False, time_embed_count=0,
                 use_res=False):
        super().__init__()

        # Set activation function
        dconv_act_fn_one = copy.deepcopy(dconv_act_fn) if dconv_act_fn is not None else nn.ReLU(inplace=True)
        dconv_act_fn_two = copy.deepcopy(dconv_act_fn) if dconv_act_fn is not None else nn.ReLU(inplace=True)

        # First Convolution + Batch Norm and ReLU options
        self.first_conv = nn.Sequential(
            ConvNd(dimensions, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1),
            BatchNormNd(dimensions, out_channels),
            dconv_act_fn_one
        )

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
        self.sec_conv = nn.Sequential(
            ConvNd(dimensions, out_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1),
            BatchNormNd(dimensions, out_channels),
            dconv_act_fn_two
        )

        # Optional residual modification
        self.need_res = use_res
        if self.need_res:
            self.res_match = ConvNd(
                dimensions, in_channels, out_channels, kernel_size=1, strides=1, padding=0, dilation=1
            )
            self.res_act = nn.ReLU(inplace=True)
        else:
            self.res_match = None
            self.res_act = None

        self.dimensions = dimensions

    def forward(self, batch, time_embed=None):
        # Do first convolution set
        conv_batch = self.first_conv(batch)

        # Create time embedding if needed
        if self.need_time:
            # Assert that the time_embed is not none
            assert time_embed is not None, (
                "Time embedding is not provided for double convolution step.\n"
                f"\tDouble Conv Layer info: {self}.\n\n"
            )

            # Retrieves the embed given a time step
            adjusted_time_embed = self.embed_adjuster(time_embed)

            # Expands the shape to (batch, out_channels, *dimensions)
            adjusted_time_embed = adjusted_time_embed[(...,) + (None,) * self.dimensions]

            # Adds time-sensitive embeddings to batch
            conv_batch = conv_batch + adjusted_time_embed

        # Do second convolution set
        conv_batch = self.sec_conv(conv_batch)

        # Add residuals if needed
        if self.need_res:
            res_batch = self.res_match(batch)
            conv_batch = self.res_act(conv_batch + res_batch)

        return conv_batch


class SpatialAttentionN(nn.Module):
    def __init__(self, dimensions, dec_skip_channels, inter_channels, use_pool=False):
        super().__init__()

        # Perform 1x1 convolution on decoder channels
        self.decoder_conv = nn.Sequential(
            ConvNd(dimensions, dec_skip_channels, inter_channels, kernel_size=1, strides=1, padding=0, dilation=1),
            BatchNormNd(dimensions, inter_channels),
        )

        # Perform 1x1 convolution on skip connections
        self.skip_conv = nn.Sequential(
            ConvNd(dimensions, dec_skip_channels, inter_channels, kernel_size=1, strides=1, padding=0, dilation=1),
            BatchNormNd(dimensions, inter_channels),
        )

        # Create pooling layers for skip connections if needed
        self.need_pool = use_pool
        if self.need_pool:
            self.max_pool = MaxPoolNd(dimensions, kernel_size=2, strides=1, padding=0, dilation=1)
            self.avg_pool = AvgPoolNd(dimensions, kernel_size=2, strides=1, padding=0)
            self.pool_interpolate = InterpolateNd(dimensions)
        else:
            self.max_pool = None
            self.avg_pool = None
            self.pool_interpolate = None

        # Create an attention mask
        self.masker = nn.Sequential(
            ConvNd(dimensions, inter_channels, out_channels=1, kernel_size=1, strides=1, padding=0, dilation=1),
            BatchNormNd(dimensions, out_channels=1),
            nn.Sigmoid()
        )

        # ReLU Activation function for Spatial Attention Block
        self.relu = nn.ReLU(inplace=True)

        self.dimensions = dimensions

    def forward(self, decoder, skip):
        # Create an intermediate decoder representation
        decoder_int = self.decoder_conv(decoder)

        # Create an intermediate skip representation
        skip_int = self.skip_conv(skip)

        # Apply pooling operations if needed
        if self.need_pool:
            skip_max = self.max_pool(skip_int)
            skip_avg = self.avg_pool(skip_int)
            skip_int = self.pool_interpolate(skip_max + skip_avg, size=skip_int.shape[-self.dimensions:])

        # Add together then apply activation to keep only noticeable features from both representations
        combo_int = self.relu(decoder_int + skip_int)

        # Create a normalized attention mask to specify where to pay "attention" to
        masked_int = self.masker(combo_int)

        # Adjust the skip connection information using the mask to tailor the information
        return skip * masked_int


class DownSampleN(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, dconv_act_fn=None, dconv_time=False, time_embed_count=0,
                 dconv_res=False):
        super().__init__()

        # Double Convolution Step
        self.conv = DoubleConvN(
            dimensions, in_channels, out_channels, dconv_act_fn=dconv_act_fn, use_time=dconv_time,
            time_embed_count=time_embed_count, use_res=dconv_res
        )

        # 2x2 Max Pooling to Shrink image
        self.pool = MaxPoolNd(dimensions, kernel_size=2, strides=2, padding=0, dilation=1)

    def forward(self, batch, time_embed=None):
        # Channel change for skip connection
        conv_batch = self.conv(batch, time_embed)

        # Downsample for next step
        encoded_batch = self.pool(conv_batch)

        return conv_batch, encoded_batch


class UpSampleN(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, up_drop_perc=0.3, use_attention=False, attn_pool=False,
                 dconv_act_fn=None, dconv_time=False, time_embed_count=0, dconv_res=False):
        super().__init__()

        # 2x2 Upscale with channel shrinkage
        self.upscaler = [
            ConvTransposeNd(
                dimensions, in_channels, out_channels, kernel_size=2, strides=2,
                padding=0, dilation=1, output_padding=0
            ),
            BatchNormNd(dimensions, out_channels),
            nn.ReLU(inplace=True)
        ]
        if up_drop_perc > 0:
            self.upscaler.append(nn.Dropout(up_drop_perc))
        self.upscaler = nn.Sequential(*self.upscaler)

        # Attention block but only if needed
        self.need_attention = use_attention
        if self.need_attention:
            self.attention = SpatialAttentionN(dimensions, in_channels // 2, out_channels // 2, use_pool=attn_pool)
        else:
            self.attention = None

        # Double Convolution Step (assumes skip connection is present to combine long and short paths)
        self.conv = DoubleConvN(
            dimensions, in_channels, out_channels, dconv_act_fn=dconv_act_fn, use_time=dconv_time,
            time_embed_count=time_embed_count, use_res=dconv_res
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


class UNETNth(nn.Module):
    def __init__(self, dimensions, in_channels, channel_list, out_layer, denoise_diff=False, denoise_embed_count=0,
                 up_attention=False, attn_pool=False, up_drop_perc=0.3, dconv_act_fn=None, dconv_res=False):
        super().__init__()

        # Create Sinusoidal Time Embedding
        self.need_denoise = denoise_diff
        if self.need_denoise:
            self.time_embeds = nn.Sequential(
                DiffusionSinPosEmbeds(denoise_embed_count, theta=10000),
                nn.Linear(denoise_embed_count, denoise_embed_count),
                nn.ReLU()
            )

        # Create down samplers using nn.ModuleList
        down_smap = []
        for i in range(len(channel_list) - 1):
            cur_in_channels = in_channels if i == 0 else channel_list[i - 1]
            down_smap.append(DownSampleN(
                dimensions, cur_in_channels, channel_list[i], dconv_act_fn=dconv_act_fn, dconv_time=self.need_denoise,
                time_embed_count=denoise_embed_count, dconv_res=dconv_res
            ))
        self.down_samplers = nn.ModuleList(down_smap)

        # Create bottleneck transition
        self.bottle_neck = DoubleConvN(
            dimensions, channel_list[-2], channel_list[-1], dconv_act_fn=dconv_act_fn, use_time=self.need_denoise,
            time_embed_count=denoise_embed_count, use_res=dconv_res
        )

        # Create up samplers using nn.ModuleList
        up_samp = []
        for i in range(len(channel_list) - 1, 0, -1):
            cur_attention = up_attention and i > 1
            up_samp.append(UpSampleN(
                dimensions, channel_list[i], channel_list[i - 1], up_drop_perc=up_drop_perc,
                use_attention=cur_attention, attn_pool=attn_pool, dconv_act_fn=dconv_act_fn,
                dconv_time=self.need_denoise, time_embed_count=denoise_embed_count, dconv_res=dconv_res
            ))
        self.up_samplers = nn.ModuleList(up_samp)

        # Set custom output layer
        self.out_layer = out_layer

    def forward(self, batch, time_step=None):
        # Prepare encoder
        skip_connections = []
        cur_down = batch

        # Get time embedding if needed
        if self.need_denoise:
            time_embed = self.time_embeds(time_step)
        else:
            time_embed = None

        # Downsample through list
        for down_sampler in self.down_samplers:
            down_skip, down_encoded = down_sampler(cur_down, time_embed)
            skip_connections.append(down_skip)
            cur_down = down_encoded

        # Bottleneck phase
        cur_up = self.bottle_neck(cur_down, time_embed)

        # Upsample through list
        for up_sampler in self.up_samplers:
            cur_up = up_sampler(cur_up, skip_connections.pop(), time_embed)

        # Apply custom output layer and return
        return self.out_layer(cur_up)
