from dataclasses import dataclass
from enum import auto, StrEnum
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# Altered from Denoising Diffusion Tutorial - https://www.youtube.com/watch?v=a4Yfz2FxXiY
class DiffPosEmbeds(nn.Module):
    def __init__(self, dimensions, theta=10000):
        super().__init__()

        # Assert dimensions are even
        assert dimensions % 2 == 0, "Dimensions must be even for using sin-cosine embeddings."

        half_dimensions = dimensions // 2

        # Altered math for embeds to be precomputed and registered
        embeds = math.log(theta) / (half_dimensions - 1)
        embeds = torch.exp(torch.arange(half_dimensions) * -embeds).unsqueeze(0)
        embeds = torch.stack((embeds.sin(), embeds.cos()), dim=-1)
        embeds = embeds.reshape(embeds.shape[0], -1)

        self.register_buffer('embeds', embeds)

    def forward(self, time_steps):
        device = time_steps.device
        return time_steps.unsqueeze(-1) * self.embeds.to(device)


# Altered from PyTorch Transformer Tutorial - https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
class AttnPosEmbeds(nn.Module):
    def __init__(self, channels, max_seq_length, theta=10000):
        super().__init__()

        # Make channels even
        if channels % 2 != 0:
            print("Channels must be even for using sin-cosine embeddings. Adding 1...")
            channels += 1

        # Create empty embedding
        embeds = torch.zeros(channels, max_seq_length)

        # Create the positions for each channel
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(0)

        # Create the divisor for each channel
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(theta) / channels))
        div_term = div_term.unsqueeze(1)

        # Generate a channels x max_seq_length embedding and apply it every other row
        embeds[0::2, :] = torch.sin(div_term * position)
        embeds[1::2, :] = torch.cos(div_term * position)

        self.register_buffer('embeds', embeds.unsqueeze(0))

    def forward(self, flat_enc):
        return flat_enc + self.embeds[:, :flat_enc.shape[1], :flat_enc.shape[2]]


class ChannelAttention(nn.Module):
    def __init__(self, enc_channels, skip_channels=None, ratio=8):
        super().__init__()

        skip_conv_channels = skip_channels if skip_channels is not None else enc_channels

        # Create an encoder convolution
        self.enc_channel_conv = nn.Sequential(
            nn.Conv1d(enc_channels, skip_conv_channels, kernel_size=1),
            nn.BatchNorm1d(skip_conv_channels),
        )

        # Create pooling layers for skip connections if needed
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=1)

        # Create shared MLP network
        self.mlp = nn.Sequential(
            nn.Linear(skip_conv_channels, skip_conv_channels // ratio),
            nn.Linear(skip_conv_channels // ratio, skip_conv_channels)
        )

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, pe_lin_encoding, pe_lin_skip=None):
        # Assume everything is already flattened and position embedded
        if pe_lin_skip is None:
            pe_lin_skip = pe_lin_encoding

        # Adjust the encoder information
        encoder_int = self.enc_channel_conv(pe_lin_encoding)

        # Aggregate the spatial information
        encoder_avg = F.adaptive_avg_pool1d(encoder_int, 1)
        encoder_max = F.adaptive_max_pool1d(encoder_int, 1)
        skip_avg = F.adaptive_avg_pool1d(pe_lin_skip, 1)
        skip_max = F.adaptive_max_pool1d(pe_lin_skip, 1)

        # Rearrange so channel is last
        encoder_avg = encoder_avg.permute(0, 2, 1)
        encoder_max = encoder_max.permute(0, 2, 1)
        skip_avg = skip_avg.permute(0, 2, 1)
        skip_max = skip_max.permute(0, 2, 1)

        # Pass skip through shared network and add both together
        skip_avg = self.mlp(skip_avg)
        skip_max = self.mlp(skip_max)
        skip_int = skip_avg + skip_max
        encoder_int = encoder_avg + encoder_max

        # Add together then apply activation to keep only noticeable features from both representations
        if pe_lin_skip is None:
            skip_int = self.sigmoid(skip_int)
        else:
            skip_int = self.sigmoid(encoder_int + skip_int)

        return pe_lin_skip * skip_int.permute(0, 2, 1)


class SpatialAttention(nn.Module):
    def __init__(self, enc_channels, skip_channels=None, inter_channels=None):
        super().__init__()

        skip_conv_channels = skip_channels if skip_channels is not None else enc_channels
        int_mask_channels = inter_channels if inter_channels is not None else skip_conv_channels

        # Check if an intermediate encoder convolution is needed
        if inter_channels is not None or skip_channels is not None:
            self.encoder_conv = nn.Sequential(
                nn.Conv1d(enc_channels, int_mask_channels, kernel_size=1),
                nn.BatchNorm1d(int_mask_channels),
            )
        else:
            self.encoder_conv = None

        # Check if an intermediate skip convolution is needed
        if inter_channels is not None:
            self.skip_conv = nn.Sequential(
                nn.Conv1d(skip_conv_channels, inter_channels, kernel_size=1),
                nn.BatchNorm1d(inter_channels),
            )
        else:
            self.skip_conv = None

        # Create pooling layers for skip connections if needed
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=1)

        # Create an attention mask
        self.masker = nn.Sequential(
            nn.Conv1d(int_mask_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(num_features=1),
            nn.Sigmoid()
        )

        # ReLU Activation function for Spatial Attention Block
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pe_lin_encoding, pe_lin_skip=None):
        # Assume everything is already flattened and position embedded
        if pe_lin_skip is None:
            pe_lin_skip = pe_lin_encoding

        # Create an intermediate representations if needed
        encoder_int = self.encoder_conv(pe_lin_encoding) if self.encoder_conv is not None else pe_lin_encoding
        skip_int = self.skip_conv(pe_lin_skip) if self.skip_conv is not None else pe_lin_skip

        # Apply pooling operations if needed
        skip_max = self.max_pool(skip_int)
        skip_avg = self.avg_pool(skip_int)
        skip_int = F.interpolate(skip_max + skip_avg, pe_lin_skip.shape[-1], mode='linear')
        skip_int = F.adaptive_avg_pool1d(skip_int, encoder_int.shape[-1])

        # Add together then apply activation to keep only noticeable features from both representations
        if pe_lin_skip is None:
            combo_int = self.relu(skip_int)
        else:
            combo_int = self.relu(encoder_int + skip_int)

        # Create a normalized attention mask to specify where to pay "attention" to
        masked_int = F.interpolate(self.masker(combo_int), pe_lin_skip.shape[-1], mode='linear')

        return pe_lin_skip * masked_int


class QKVAttention(nn.Module):
    def __init__(self, enc_channels, skip_channels=None, heads=1):
        super().__init__()

        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert enc_channels % heads == 0, (
            "The QKV Attention Block's channel count must be divisible by the number of heads."
        )

        # Initialize dimensions
        self.skip_channels = skip_channels  # Optional skip connection's dimension
        self.heads = heads  # Number of attention heads
        self.channels_per_head = enc_channels // self.heads

        # Weights
        self.query_weights = nn.Linear(enc_channels, enc_channels)
        self.key_weights = nn.Linear(enc_channels, enc_channels)

        if self.skip_channels is not None:
            self.value_weights = nn.Linear(self.skip_channels, self.skip_channels)
            self.out_weights = nn.Linear(self.skip_channels, self.skip_channels)
        else:
            self.value_weights = nn.Linear(enc_channels, enc_channels)
            self.out_weights = nn.Linear(enc_channels, enc_channels)

    def divide_by_heads(self, input_tensor):
        # Pad the last set of data at the end if necessary
        if input_tensor.shape[-1] % self.heads != 0:
            residual = input_tensor.shape[-1] % self.heads
            input_tensor = F.pad(input_tensor, (0, residual), "constant", 0)

        # Divide by heads and move the heads to the second dimension
        input_divided = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], self.heads, -1)
        input_divided = input_divided.permute(0, 2, 1, 3)

        return input_divided

    def forward(self, pe_lin_encoding, pe_lin_skip=None):
        # Assume everything is already flattened and position embedded
        # Create queries, keys, and values
        queries = self.divide_by_heads(self.query_weights(pe_lin_encoding.permute(0, 2, 1)))
        keys = self.divide_by_heads(self.key_weights(pe_lin_encoding.permute(0, 2, 1)))
        if pe_lin_skip is not None and self.skip_channels is not None:
            values = self.divide_by_heads(self.value_weights(pe_lin_skip.permute(0, 2, 1)))
        elif pe_lin_skip is None and self.skip_channels is None:
            values = self.divide_by_heads(self.value_weights(pe_lin_encoding.permute(0, 2, 1)))
        else:
            raise ValueError("Both skip_channels and pe_lin_skip must be provided.")

        # Calculate attention score
        attn_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / math.sqrt(self.channels_per_head)

        # Apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply probability focus to values
        values = F.adaptive_avg_pool2d(values, (pe_lin_encoding.shape[2], values.shape[-1]))
        attn_values = torch.matmul(attn_probs, values)

        # Reshape value output back to the right shape
        attn_values = attn_values.permute(0, 2, 1, 3)
        attn_values = attn_values.reshape(attn_values.shape[0], attn_values.shape[1], -1)
        skip_out_values = self.out_weights(attn_values).permute(0, 2, 1)

        return skip_out_values


@dataclass
class AttentionOptions(StrEnum):
    CHANNEL = auto()
    SPATIAL = auto()
    QKV = auto()


class Attention(nn.Module):
    def __init__(self, attn_order, enc_channels, skip_channels=None, channel_ratio=8, spatial_inter_channels=None,
                 qkv_heads=1, use_pos=False, pos_max_len=0):
        super().__init__()

        skip_conv_channels = skip_channels if skip_channels is not None else enc_channels
        self.uses_qkv = -1
        self.post_qkv = None
        attn_list = []

        # Create positional encoding
        self.need_pos = use_pos
        if self.need_pos:
            self.enc_pos = AttnPosEmbeds(enc_channels, pos_max_len)
            self.skip_pos = AttnPosEmbeds(skip_conv_channels, pos_max_len)
        else:
            self.enc_pos = None
            self.skip_pos = None

        # Create attention list
        for attn in attn_order:
            if attn.lower() == "channel":
                attn_list.append(ChannelAttention(enc_channels, skip_conv_channels, channel_ratio))
            elif attn.lower() == "spatial":
                attn_list.append(SpatialAttention(enc_channels, skip_conv_channels, spatial_inter_channels))
            elif attn.lower() == "qkv":
                self.uses_qkv = len(attn_list)
                attn_list.append(QKVAttention(enc_channels, skip_conv_channels, qkv_heads))

                self.post_qkv = nn.Sequential(
                    nn.Conv1d(skip_conv_channels, skip_conv_channels, kernel_size=1),
                    nn.BatchNorm1d(skip_conv_channels),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unknown attention type: {attn}")

        self.attn_list = nn.ModuleList(attn_list)

    def forward(self, pe_lin_encoding, pe_lin_skip=None):
        # Assume everything is already flattened and position embedded
        skip_out = pe_lin_skip if pe_lin_skip is not None else pe_lin_encoding
        interpolate_shape = pe_lin_skip.shape[-1] if pe_lin_skip is not None else pe_lin_encoding.shape[-1]

        # Apply positional embeddings if needed
        if self.need_pos:
            pe_lin_encoding = self.enc_pos(pe_lin_encoding)
            skip_out = self.skip_pos(skip_out)

        # Apply attention set
        for i in range(len(self.attn_list)):
            skip_out = self.attn_list[i](pe_lin_encoding, skip_out)
            if i == self.uses_qkv:
                skip_out = self.post_qkv(skip_out)
                skip_out = F.interpolate(skip_out, interpolate_shape, mode='linear')

        return skip_out
