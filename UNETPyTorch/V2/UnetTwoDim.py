import torch
import math
import torch.nn as nn


# Altered from Denoising Diffusion Tutorial - https://www.youtube.com/watch?v=a4Yfz2FxXiY
class DiffusionEmbeds(nn.Module):
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


# Copied from PyTorch Transformer Tutorial - https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
class QKVPosEmbeds(nn.Module):
    def __init__(self, channels, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, channels)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class QKVAttentionTwo(nn.Module):
    def __init__(self, channels, max_seq_length, heads=1, mask=False):
        super().__init__()

        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert channels % heads == 0, (
            "The QKV Attention Block's channel count must be divisible by the number of heads."
        )

        # Initialize dimensions
        self.channels = channels  # Model's dimension
        self.heads = heads  # Number of attention heads
        self.channels_per_head = self.channels // self.heads

        # Weights
        self.query_weights = nn.Linear(self.channels, self.channels)
        self.key_weights = nn.Linear(self.channels, self.channels)
        self.value_weights = nn.Linear(self.channels, self.channels)
        self.out_weights = nn.Linear(self.channels, self.channels)

        # TODO: Mask

    def forward(self, pe_lin_encoding, pe_lin_skip):
        # Convert encoding to one dimension
        pe_lin_encoding = pe_lin_encoding.reshape(pe_lin_encoding.shape[0], pe_lin_encoding.shape[1], -1)

        # Create queries, keys, and values
        queries = self.query_weights(pe_lin_encoding)
        keys = self.key_weights(pe_lin_encoding)
        values = self.value_weights(pe_lin_encoding)


if __name__ == "__main__":
    # Sample time step data
    ts = torch.randint(0, 300, (4,))

    # Sample embedding
    embeder = DiffusionEmbeds(32, 10000)
    te = embeder(ts)

    print(ts)
    print(te)
