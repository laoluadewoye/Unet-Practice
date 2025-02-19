import torch
import torch.nn as nn


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