import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# Altered from Denoising Diffusion Tutorial - https://www.youtube.com/watch?v=a4Yfz2FxXiY
class DiffPosEmbeds(nn.Module):
    def __init__(self, dimensions, theta=10000):
        super().__init__()

        # Make dimensions even
        if dimensions % 2 != 0:
            print("Warning: Dimensions not even, adding 1...")
            dimensions += 1
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
        return flat_enc + self.embeds[:, :, :flat_enc.size(2)]


class SpatialAttention(nn.Module):
    def __init__(self, dec_skip_channels, inter_channels, use_pool=False):
        super().__init__()

        # Perform 1x1 convolution on decoder channels
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(dec_skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
        )

        # Perform 1x1 convolution on skip connections
        self.skip_conv = nn.Sequential(
            nn.Conv1d(dec_skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
        )

        # Create pooling layers for skip connections if needed
        self.need_pool = use_pool
        if self.need_pool:
            self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
            self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=1)
            self.pool_interpolate = lambda x, size: F.interpolate(x, size=size, mode='linear', align_corners=False)
        else:
            self.max_pool = None
            self.avg_pool = None
            self.pool_interpolate = None

        # Create an attention mask
        self.masker = nn.Sequential(
            nn.Conv1d(inter_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(num_features=1),
            nn.Sigmoid()
        )

        # ReLU Activation function for Spatial Attention Block
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pe_lin_encoding, pe_lin_skip):
        # Assume everything is already flattened and position embedded
        # Create an intermediate decoder representation
        encoder_int = self.encoder_conv(pe_lin_encoding)

        # Create an intermediate skip representation
        skip_int = self.skip_conv(pe_lin_skip)

        # Apply pooling operations if needed
        if self.need_pool:
            skip_max = self.max_pool(skip_int)
            skip_avg = self.avg_pool(skip_int)
            skip_int = self.pool_interpolate(skip_max + skip_avg, size=skip_int.shape[-1:])

        # Add together then apply activation to keep only noticeable features from both representations
        combo_int = self.relu(encoder_int + skip_int)

        # Create a normalized attention mask to specify where to pay "attention" to
        masked_int = self.masker(combo_int)

        # Adjust the skip connection information using the mask to tailor the information
        return pe_lin_skip * masked_int


class QKVAttention(nn.Module):
    def __init__(self, self_channels, heads=1, skip_channels=None):
        super().__init__()

        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert self_channels % heads == 0, (
            "The QKV Attention Block's channel count must be divisible by the number of heads."
        )

        # Initialize dimensions
        self.skip_channels = skip_channels  # Optional skip connection's dimension
        self.heads = heads  # Number of attention heads
        self.self_channels_per_head = self_channels // self.heads

        # Weights
        self.query_weights = nn.Linear(self_channels, self_channels)
        self.key_weights = nn.Linear(self_channels, self_channels)

        if self.skip_channels is not None:
            self.value_weights = nn.Linear(self.skip_channels, self.skip_channels)
            self.out_weights = nn.Linear(self.skip_channels, self.skip_channels)
        else:
            self.value_weights = nn.Linear(self_channels, self_channels)
            self.out_weights = nn.Linear(self_channels, self_channels)

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
        else:
            values = self.divide_by_heads(self.value_weights(pe_lin_encoding.permute(0, 2, 1)))

        # Calculate attention score
        attn_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / math.sqrt(self.self_channels_per_head)

        # Apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply probability focus to values
        if pe_lin_skip is not None and self.skip_channels is not None:
            values = F.adaptive_avg_pool2d(values, (pe_lin_encoding.shape[2], values.shape[-1]))
        attn_values = torch.matmul(attn_probs, values)

        # Reshape value output back to the right shape
        attn_values = attn_values.permute(0, 2, 1, 3)
        attn_values = attn_values.reshape(attn_values.shape[0], attn_values.shape[1], -1)
        skip_out_values = self.out_weights(attn_values).permute(0, 2, 1)

        return skip_out_values


if __name__ == "__main__":
    # Sample time step data
    sample_encoding_one = torch.rand(4, 32, 64, 64)
    sample_encoding_one = sample_encoding_one.reshape(4, 32, -1)

    sample_encoding_two = torch.rand(4, 32, 64, 64)
    sample_encoding_two = sample_encoding_two.reshape(4, 32, -1)

    # Sample embedding
    enc_embeder = AttnPosEmbeds(32, 5000)
    enc_embedding = enc_embeder(sample_encoding_one)

    # Skip embedding
    sample_skip = torch.rand(4, 16, 128, 128)
    sample_skip = sample_skip.reshape(4, 16, -1)
    skip_embeder = AttnPosEmbeds(16, 17000)
    skip_embedding = skip_embeder(sample_skip)

    # Sample QKV attention
    attn = QKVAttention(32, skip_channels=16)
    attn(enc_embedding, skip_embedding)

    attn = QKVAttention(32, skip_channels=32)
    attn(enc_embedding, sample_encoding_two)

    # Sample spatial attention
    attn = SpatialAttention(32, inter_channels=16)
    attn(enc_embedding, sample_encoding_two)
