import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GenUtils import *


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
    attn = QKVAttention(32, skip_channels=16)
    attn(enc_embedding, skip_embedding)
