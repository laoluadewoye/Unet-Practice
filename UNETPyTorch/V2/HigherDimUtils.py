"""
Module for defining higher dimensional modules.
"""
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add typing
# TODO: Add more documentation
def down_output_size(input_size, input_index, kernel_size, stride, padding, dilation):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]
    d_index = dilation[input_index]
        
    return (input_size + 2 * p_index - d_index * (ks_index - 1) - 1) // s_index + 1


# TODO: Add typing
# TODO: Add more documentation
def avg_output_size(input_size, input_index, kernel_size, stride, padding):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]

    return (input_size + 2 * p_index - ks_index) // s_index + 1


# TODO: Add typing
# TODO: Add more documentation
def up_output_size(input_size, input_index, kernel_size, stride, padding, dilation, output_padding):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]
    d_index = dilation[input_index]
    op_index = output_padding[input_index]

    return (input_size - 1) * s_index - 2 * p_index + d_index * (ks_index - 1) + op_index + 1


# Simplified version of ConvNd using this as a guide: https://github.com/pvjosue/pytorch_convNd/blob/master/convNd.py
# TODO: Add typing
# TODO: Add more documentation
class ConvNd(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, **kwargs):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Things to feed into the Conv block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int inputs to tuples
        self.kernel_size = kwargs.get('kernel_size', 1)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * dimensions
        else:
            assert len(self.kernel_size) == dimensions, "Passed tuple is not the same length as dimensions"

        self.stride = kwargs.get('stride', 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * dimensions
        else:
            assert len(self.stride) == dimensions, "Passed tuple is not the same length as dimensions"

        self.padding = kwargs.get('padding', 0)
        if isinstance(self.padding, int):
            self.padding = (self.padding,) * dimensions
        else:
            assert len(self.padding) == dimensions, "Passed tuple is not the same length as dimensions"

        self.dilation = kwargs.get('dilation', 1)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation,) * dimensions
        else:
            assert len(self.dilation) == dimensions, "Passed tuple is not the same length as dimensions"

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, ConvNd(
                self.dimensions - 1, self.in_channels, self.out_channels, kernel_size=self.kernel_size[1:],
                stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:]
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.Conv3d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size[1:],
                stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:]
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.Conv1d(
            in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size[0],
            stride=self.stride[0], padding=self.padding[0], dilation=self.dilation[0]
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.reshape(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [down_output_size(
            lower_dim_shape[i], i+1, self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.reshape(shape[0], shape[2], -1, *lower_dim_shape)

        # Create a list from 0 to n dimensions
        # The list should go batch, highest dimension, channels, lower dimensions
        order = [i for i in range(self.dimensions + 2)]

        # Move the channels and highest dimension behind the lower dimensions, then reduce the batch to 1 dimension
        # First though, count how many scalars should fit into the lower-view batch dimension using the actual lower
        #   dimensional shape
        one_dim_count = shape[0]
        for dim_shape in lower_dim_shape:
            one_dim_count *= dim_shape

        one_dim_tensor = lower_conv_tensor.permute(0, *order[-self.dimensions + 1:], 2, 1)
        one_dim_tensor = one_dim_tensor.reshape(one_dim_count, -1, shape[2])

        # Work the last dimension to obtain connections between N-1D dimensions
        one_dim_conv_tensor = self.last_dim(one_dim_tensor)

        # The shape is currently batch, lower dimensions, channel, then highest dimension
        # Reshape everything back to normal
        high_dim_size = down_output_size(
            shape[2], 0, self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        final_tensor = one_dim_conv_tensor.reshape(shape[0], *lower_dim_shape, self.out_channels, high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


# TODO: Add typing
# TODO: Add more documentation
class ConvTransposeNd(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, **kwargs):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Things to feed into the Conv block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int inputs to tuples
        self.kernel_size = kwargs.get('kernel_size', 1)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * dimensions
        else:
            assert len(self.kernel_size) == dimensions, "Passed tuple is not the same length as dimensions"

        self.stride = kwargs.get('stride', 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * dimensions
        else:
            assert len(self.stride) == dimensions, "Passed tuple is not the same length as dimensions"

        self.padding = kwargs.get('padding', 0)
        if isinstance(self.padding, int):
            self.padding = (self.padding,) * dimensions
        else:
            assert len(self.padding) == dimensions, "Passed tuple is not the same length as dimensions"

        self.dilation = kwargs.get('dilation', 1)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation,) * dimensions
        else:
            assert len(self.dilation) == dimensions, "Passed tuple is not the same length as dimensions"

        self.output_padding = kwargs.get('output_padding', 0)
        if isinstance(self.output_padding, int):
            self.output_padding = (self.output_padding,) * dimensions
        else:
            assert len(self.output_padding) == dimensions, "Passed tuple is not the same length as dimensions"

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, ConvTransposeNd(
                self.dimensions - 1, self.in_channels, self.out_channels, kernel_size=self.kernel_size[1:],
                stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:],
                output_padding=self.output_padding[1:]
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.ConvTranspose3d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size[1:],
                stride=self.stride[1:], padding=self.padding[1:], dilation=self.dilation[1:],
                output_padding=self.output_padding[1:]
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.ConvTranspose1d(
            in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size[0],
            stride=self.stride[0], padding=self.padding[0], dilation=self.dilation[0],
            output_padding=self.output_padding[0]
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.reshape(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [up_output_size(
            lower_dim_shape[i], i+1, self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, output_padding=self.output_padding) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.reshape(shape[0], shape[2], -1, *lower_dim_shape)

        # Create a list from 0 to n dimensions
        # The list should go batch, highest dimension, channels, lower dimensions
        order = [i for i in range(self.dimensions + 2)]

        # Move the channels and highest dimension behind the lower dimensions, then reduce the batch to 1 dimension
        # First though, count how many scalars should fit into the lower-view batch dimension using the actual lower
        #   dimensional shape
        one_dim_count = shape[0]
        for dim_shape in lower_dim_shape:
            one_dim_count *= dim_shape

        one_dim_tensor = lower_conv_tensor.permute(0, *order[-self.dimensions + 1:], 2, 1)
        one_dim_tensor = one_dim_tensor.reshape(one_dim_count, -1, shape[2])

        # Work the last dimension to obtain connections between N-1D dimensions
        one_dim_conv_tensor = self.last_dim(one_dim_tensor)

        # The shape is currently batch, lower dimensions, channel, then highest dimension
        # Reshape everything back to normal
        high_dim_size = up_output_size(
            shape[2], 0, self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, output_padding=self.output_padding
        )
        final_tensor = one_dim_conv_tensor.reshape(shape[0], *lower_dim_shape, self.out_channels, high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


# TODO: Add typing
# TODO: Add more documentation
class BatchNormNd(nn.Module):
    def __init__(self, dimensions, out_channels):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        self.out_channels = out_channels
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)

        # Reduce shape to one dimension
        nd_tensor = nd_tensor.reshape(shape[0], shape[1], -1)

        # Conduct 1D Batch Normalization
        norm_tensor = self.norm(nd_tensor)

        # Change everything back
        norm_tensor = norm_tensor.reshape(*shape)
        return norm_tensor


# TODO: Add typing
# TODO: Add more documentation
class MaxPoolNd(nn.Module):
    def __init__(self, dimensions, **kwargs):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Convert int inputs to tuples
        self.kernel_size = kwargs.get('kernel_size', 1)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * dimensions
        else:
            assert len(self.kernel_size) == dimensions, "Passed tuple is not the same length as dimensions"

        self.stride = kwargs.get('stride', 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * dimensions
        else:
            assert len(self.stride) == dimensions, "Passed tuple is not the same length as dimensions"

        self.padding = kwargs.get('padding', 0)
        if isinstance(self.padding, int):
            self.padding = (self.padding,) * dimensions
        else:
            assert len(self.padding) == dimensions, "Passed tuple is not the same length as dimensions"

        self.dilation = kwargs.get('dilation', 1)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation,) * dimensions
        else:
            assert len(self.dilation) == dimensions, "Passed tuple is not the same length as dimensions"

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, MaxPoolNd(
                self.dimensions - 1, kernel_size=self.kernel_size[1:], stride=self.stride[1:],
                padding=self.padding[1:], dilation=self.dilation[1:]
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.MaxPool3d(
                kernel_size=self.kernel_size[1:], stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:]
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.MaxPool1d(
            kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[0],
            dilation=self.dilation[0]
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.reshape(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [down_output_size(
            lower_dim_shape[i], i + 1, self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.reshape(shape[0], shape[2], -1, *lower_dim_shape)

        # Create a list from 0 to n dimensions
        # The list should go batch, highest dimension, channels, lower dimensions
        order = [i for i in range(self.dimensions + 2)]

        # Move the channels and highest dimension behind the lower dimensions, then reduce the batch to 1 dimension
        # First though, count how many scalars should fit into the lower-view batch dimension using the actual lower
        #   dimensional shape
        one_dim_count = shape[0]
        for dim_shape in lower_dim_shape:
            one_dim_count *= dim_shape

        one_dim_tensor = lower_conv_tensor.permute(0, *order[-self.dimensions + 1:], 2, 1)
        one_dim_tensor = one_dim_tensor.reshape(one_dim_count, -1, shape[2])

        # Work the last dimension to obtain connections between N-1D dimensions
        one_dim_conv_tensor = self.last_dim(one_dim_tensor)

        # The shape is currently batch, lower dimensions, channel, then highest dimension
        # Reshape everything back to normal
        high_dim_size = down_output_size(
            shape[2], 0, self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        final_tensor = one_dim_conv_tensor.reshape(shape[0], *lower_dim_shape, shape[1], high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


# TODO: Add typing
# TODO: Add more documentation
class AvgPoolNd(nn.Module):
    def __init__(self, dimensions, **kwargs):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Convert int inputs to tuples
        self.kernel_size = kwargs.get('kernel_size', 1)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * dimensions
        else:
            assert len(self.kernel_size) == dimensions, "Passed tuple is not the same length as dimensions"

        self.stride = kwargs.get('stride', 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * dimensions
        else:
            assert len(self.stride) == dimensions, "Passed tuple is not the same length as dimensions"

        self.padding = kwargs.get('padding', 0)
        if isinstance(self.padding, int):
            self.padding = (self.padding,) * dimensions
        else:
            assert len(self.padding) == dimensions, "Passed tuple is not the same length as dimensions"

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, AvgPoolNd(
                self.dimensions - 1, kernel_size=self.kernel_size[1:], stride=self.stride[1:], padding=self.padding[1:]
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.AvgPool3d(
                kernel_size=self.kernel_size[1:], stride=self.stride[1:], padding=self.padding[1:]
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.AvgPool1d(
            kernel_size=self.kernel_size[0], stride=self.stride[0], padding=self.padding[0]
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.reshape(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [avg_output_size(
            lower_dim_shape[i], i + 1, self.kernel_size, stride=self.stride, padding=self.padding)
            for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.reshape(shape[0], shape[2], -1, *lower_dim_shape)

        # Create a list from 0 to n dimensions
        # The list should go batch, highest dimension, channels, lower dimensions
        order = [i for i in range(self.dimensions + 2)]

        # Move the channels and highest dimension behind the lower dimensions, then reduce the batch to 1 dimension
        # First though, count how many scalars should fit into the lower-view batch dimension using the actual lower
        #   dimensional shape
        one_dim_count = shape[0]
        for dim_shape in lower_dim_shape:
            one_dim_count *= dim_shape

        one_dim_tensor = lower_conv_tensor.permute(0, *order[-self.dimensions + 1:], 2, 1)
        one_dim_tensor = one_dim_tensor.reshape(one_dim_count, -1, shape[2])

        # Work the last dimension to obtain connections between N-1D dimensions
        one_dim_conv_tensor = self.last_dim(one_dim_tensor)

        # The shape is currently batch, lower dimensions, channel, then highest dimension
        # Reshape everything back to normal
        high_dim_size = avg_output_size(
            shape[2], 0, self.kernel_size, stride=self.stride, padding=self.padding
        )
        final_tensor = one_dim_conv_tensor.reshape(shape[0], *lower_dim_shape, shape[1], high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


# TODO: Add typing
# TODO: Add more documentation
class InterpolateNd(nn.Module):
    def __init__(self, dimensions):
        super().__init__()

        # Assert dimensions is higher than three
        self.dimensions = dimensions
        assert self.dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, InterpolateNd(dimensions - 1))
        else:
            self.lower_name = 'lower'
            setattr(
                self, self.lower_name,
                lambda x, size: F.interpolate(x, size=size, mode='trilinear', align_corners=False)
            )

        # Capture the last dimension left out by the lower representation
        self.last_dim = lambda x, size: F.interpolate(x, size=size, mode='linear', align_corners=False)

    def forward(self, nd_tensor, size):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.reshape(shape[0] * shape[2], shape[1], *shape[-self.dimensions + 1:])

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor, size[1:])

        # Change everything back
        lower_conv_tensor = lower_tensor.reshape(shape[0], shape[2], -1, *size[1:])

        # Create a list from 0 to n dimensions
        # The list should go batch, highest dimension, channels, lower dimensions
        order = [i for i in range(self.dimensions + 2)]

        # Move the channels and highest dimension behind the lower dimensions, then reduce the batch to 1 dimension
        # First though, count how many scalars should fit into the lower-view batch dimension using the actual lower
        #   dimensional shape
        one_dim_count = shape[0]
        for dim_shape in size[1:]:
            one_dim_count *= dim_shape

        one_dim_tensor = lower_conv_tensor.permute(0, *order[-self.dimensions + 1:], 2, 1)
        one_dim_tensor = one_dim_tensor.reshape(one_dim_count, -1, shape[2])

        # Work the last dimension to obtain connections between N-1D dimensions
        one_dim_conv_tensor = self.last_dim(one_dim_tensor, size[0])

        # The shape is currently batch, lower dimensions, channel, then highest dimension
        # Reshape everything back to normal
        final_tensor = one_dim_conv_tensor.reshape(shape[0], *size[1:], -1, size[0])
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor
