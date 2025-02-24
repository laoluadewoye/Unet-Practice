import sys
import os
import copy
from torchinfo import summary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GenUtils import *


class ConvSetTwo(nn.Module):
    def __init__(self, channel_sequence, kernel_sequence, padding_sequence, stride=1, dconv_act_fn=None,
                 use_time=False, time_embed_count=0, use_res=False):
        super().__init__()

        # Assert that the channel sequence is at least 3 items long (for double convolution)
        assert len(channel_sequence) >= 3, "Sequence of channels must be at least three for a proper ConvSet."

        # Assert that the kernel sequence and padding sequence is at least 2 items long (for double convolution)
        assert len(kernel_sequence) >= 2, "Sequence of kernels must be at least two for a proper ConvSet."
        assert len(padding_sequence) >= 2, "Sequence of padding must be at least two for a proper ConvSet."

        # Assert that the kernels, padding, and channel (minus one) are the same length
        assert len(kernel_sequence) == len(padding_sequence) == (len(channel_sequence) - 1), (
            "Kernels and padding must be the same length."
        )

        # List of convolutions to conduct
        conv_list = []
        for i in range(len(channel_sequence) - 1):
            # Set the stride
            if i == 1:
                cur_stride = stride
            else:
                cur_stride = 1

            # Add the convolution module
            conv_list.append(nn.Sequential(
                nn.Conv2d(
                    channel_sequence[i], channel_sequence[i+1], kernel_size=kernel_sequence[i],
                    padding=padding_sequence[i], stride=cur_stride
                ),
                nn.BatchNorm2d(channel_sequence[i+1]),
                copy.deepcopy(dconv_act_fn) if dconv_act_fn is not None else nn.ReLU(inplace=True)
            ))

        # Get it recognized by PyTorch
        self.conv_list = nn.ModuleList(conv_list)

        # Optional time modification after first convolution
        self.need_time = use_time
        if self.need_time:
            self.embed_adjuster = nn.Sequential(
                nn.Linear(time_embed_count, channel_sequence[2]),
                nn.ReLU()
            )
        else:
            self.embed_adjuster = None

        # Optional residual modification
        self.need_res = use_res
        if self.need_res:
            self.res_match = nn.Sequential(
                nn.Conv2d(channel_sequence[0], channel_sequence[-1], kernel_size=1, stride=stride),
                nn.BatchNorm2d(channel_sequence[-1]),
            )
            self.res_act = nn.ReLU(inplace=True)
        else:
            self.res_match = None
            self.res_act = None

    def forward(self, batch, time_embed=None):
        # Do first convolution set
        conv_batch = self.conv_list[0](batch)

        # Create time embedding if needed
        if self.need_time:
            # Assert that the time_embed is not none
            assert time_embed is not None, (
                "Time embedding is not provided for double convolution step.\n"
                f"\tDouble Conv Layer info: {self}.\n\n"
            )

            # Retrieves the embed given a time step
            adjusted_time_embed = self.embed_adjuster(time_embed)

            # Expands the shape to (batch, out_channels, 1, 1)
            adjusted_time_embed = adjusted_time_embed[(...,) + (None,) * 2]

            # Adds time-sensitive embeddings to batch
            conv_batch = conv_batch + adjusted_time_embed

        # Do the rest of the conv sets
        for i in range(1, len(self.conv_list)):
            conv_batch = self.conv_list[i](conv_batch)

        # Add residuals if needed
        if self.need_res:
            res_batch = self.res_match(batch)
            conv_batch = self.res_act(conv_batch + res_batch)

        return conv_batch


def make_res_net_layer(channel_sequence: list, kernel_sequence, padding_sequence, set_count, stride):
    set_list = [ConvSetTwo(channel_sequence, kernel_sequence, padding_sequence, use_res=True, stride=stride)]
    channel_sequence[0] = channel_sequence[-1]

    for i in range(set_count - 1):
        set_list.append(ConvSetTwo(channel_sequence, kernel_sequence, padding_sequence, use_res=True))

    return nn.Sequential(*set_list)


if __name__ == "__main__":
    # Create test conv sets
    test_res_net_module_one = make_res_net_layer(
        [3, 64, 64, 256], (1, 3, 1), (0, 1, 0), 3, 1
    )
    test_res_net_module_two = make_res_net_layer(
        [256, 128, 128, 512], (1, 3, 1), (0, 1, 0), 4, 2
    )
    test_res_net_module_three = make_res_net_layer(
        [512, 256, 256, 1024], (1, 3, 1), (0, 1, 0), 6, 2
    )
    test_res_net_module_four = make_res_net_layer(
        [1024, 512, 512, 2048], (1, 3, 1), (0, 1, 0), 3, 2
    )

    res_net_fifty = nn.Sequential(
        test_res_net_module_one, test_res_net_module_two, test_res_net_module_three, test_res_net_module_four
    )
    summary(res_net_fifty, (4, 3, 64, 64), depth=5)
