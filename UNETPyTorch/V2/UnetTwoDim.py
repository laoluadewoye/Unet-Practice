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


class QKVAttentionTwo(nn.Module):
    def __init__(self, channels, heads=1, skip_channels=None):
        super().__init__()

        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert channels % heads == 0, (
            "The QKV Attention Block's channel count must be divisible by the number of heads."
        )

        # Initialize dimensions
        self.channels = channels  # Model's dimension
        self.skip_channels = skip_channels  # Optional skip connection's dimension
        self.heads = heads  # Number of attention heads
        self.channels_per_head = self.channels // self.heads

        # Weights
        self.query_weights = nn.Linear(self.channels, self.channels)
        self.key_weights = nn.Linear(self.channels, self.channels)

        if self.skip_channels is not None:
            self.value_weights = nn.Linear(self.skip_channels, self.skip_channels)
            self.out_weights = nn.Linear(self.skip_channels, self.skip_channels)
        else:
            self.value_weights = nn.Linear(self.channels, self.channels)
            self.out_weights = nn.Linear(self.channels, self.channels)

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
        if pe_lin_skip is not None:
            values = self.divide_by_heads(self.value_weights(pe_lin_skip.permute(0, 2, 1)))
        else:
            values = self.divide_by_heads(self.value_weights(pe_lin_encoding.permute(0, 2, 1)))

        print(values)


if __name__ == "__main__":
    # Sample time step data
    sample_encoding = torch.rand(4, 32, 64, 64)
    sample_encoding = sample_encoding.reshape(4, 32, -1)

    # Sample embedding
    enc_embeder = AttnPosEmbeds(32, 5000)
    enc_embedding = enc_embeder(sample_encoding)

    # Skip embedding
    sample_skip = torch.rand(4, 16, 128, 128)
    sample_skip = sample_skip.reshape(4, 16, -1)
    skip_embeder = AttnPosEmbeds(16, 17000)
    skip_embedding = skip_embeder(sample_skip)

    # Sample attention
    attn = QKVAttentionTwo(32, 8, 16)
    attn(enc_embedding, skip_embedding)
