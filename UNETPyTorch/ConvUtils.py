import torch
import torch.nn as nn


def down_output_size(input_size, input_index, kernel_size, stride, padding, dilation):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]
    d_index = dilation[input_index]
        
    return (input_size + 2 * p_index - d_index * (ks_index - 1) - 1) // s_index + 1


def avg_output_size(input_size, input_index, kernel_size, stride, padding):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]

    return (input_size + 2 * p_index - ks_index) // s_index + 1


def up_output_size(input_size, input_index, kernel_size, stride, padding, dilation, output_padding):
    ks_index = kernel_size[input_index]
    s_index = stride[input_index]
    p_index = padding[input_index]
    d_index = dilation[input_index]
    op_index = output_padding[input_index]

    return (input_size - 1) * s_index - 2 * p_index + d_index * (ks_index - 1) + op_index + 1


# Simplified version of ConvNd using this as a guide: https://github.com/pvjosue/pytorch_convNd/blob/master/convNd.py
class ConvNd(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, strides, padding, dilation):
        super().__init__()

        # Assert dimensions is higher than three
        assert dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Dimension count
        self.dimensions = dimensions

        # Things to feed into the Conv block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int inputs to tuples
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * dimensions
        else:
            assert len(kernel_size) == dimensions
            self.kernel_size = kernel_size
            
        if isinstance(strides, int):
            self.strides = (strides,) * dimensions
        else:
            assert len(strides) == dimensions
            self.strides = strides
            
        if isinstance(padding, int):
            self.padding = (padding,) * dimensions
        else:
            assert len(padding) == dimensions
            self.padding = padding
            
        if isinstance(dilation, int):
            self.dilation = (dilation,) * dimensions
        else:
            assert len(dilation) == dimensions
            self.dilation = dilation

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, ConvNd(
                self.dimensions - 1, self.in_channels, self.out_channels, kernel_size,
                strides=strides, padding=padding, dilation=dilation
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.Conv3d(
                in_channels=self.in_channels, out_channels=self.out_channels,
                kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.Conv1d(
            in_channels=self.out_channels, out_channels=self.out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.view(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [down_output_size(
            lower_dim_shape[i], i+1, self.kernel_size, stride=self.strides, padding=self.padding,
            dilation=self.dilation) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.view(shape[0], shape[2], -1, *lower_dim_shape)

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
            shape[2], 0, self.kernel_size, stride=self.strides, padding=self.padding, dilation=self.dilation
        )
        final_tensor = one_dim_conv_tensor.view(shape[0], *lower_dim_shape, self.out_channels, high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


class ConvTransposeNd(nn.Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, strides, padding, dilation, output_padding):
        super().__init__()

        # Assert dimensions is higher than three
        assert dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Dimension count
        self.dimensions = dimensions

        # Things to feed into the Conv Transpose block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int inputs to tuples
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * dimensions
        else:
            assert len(kernel_size) == dimensions
            self.kernel_size = kernel_size

        if isinstance(strides, int):
            self.strides = (strides,) * dimensions
        else:
            assert len(strides) == dimensions
            self.strides = strides

        if isinstance(padding, int):
            self.padding = (padding,) * dimensions
        else:
            assert len(padding) == dimensions
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation,) * dimensions
        else:
            assert len(dilation) == dimensions
            self.dilation = dilation

        if isinstance(output_padding, int):
            self.output_padding = (output_padding,) * dimensions
        else:
            assert len(output_padding) == dimensions
            self.output_padding = output_padding

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, ConvTransposeNd(
                self.dimensions - 1, self.in_channels, self.out_channels, kernel_size,
                strides=strides, padding=padding, dilation=dilation, output_padding=output_padding
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.ConvTranspose3d(
                in_channels=self.in_channels, out_channels=self.out_channels,
                kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation,
                output_padding=output_padding
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.ConvTranspose1d(
            in_channels=self.out_channels, out_channels=self.out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation,
            output_padding=output_padding
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.view(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [up_output_size(
            lower_dim_shape[i], i+1, self.kernel_size, stride=self.strides, padding=self.padding,
            dilation=self.dilation, output_padding=self.output_padding) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.view(shape[0], shape[2], -1, *lower_dim_shape)

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
            shape[2], 0, self.kernel_size, stride=self.strides, padding=self.padding,
            dilation=self.dilation, output_padding=self.output_padding
        )
        final_tensor = one_dim_conv_tensor.view(shape[0], *lower_dim_shape, self.out_channels, high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


class BatchNormNd(nn.Module):
    def __init__(self, dimensions, out_channels):
        super().__init__()

        # Assert dimensions is higher than three
        assert dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        self.dimensions = dimensions
        self.out_channels = out_channels
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)

        # Reduce shape to one dimension
        nd_tensor = nd_tensor.view(shape[0], shape[1], -1)

        # Conduct 1D Batch Normalization
        norm_tensor = self.norm(nd_tensor)

        # Change everything back
        norm_tensor = norm_tensor.view(*shape)
        return norm_tensor


# TODO: Add tuple support for parameters
class MaxPoolNd(nn.Module):
    def __init__(self, dimensions, kernel_size, strides, padding, dilation):
        super().__init__()

        # Assert dimensions is higher than three
        assert dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Dimension count
        self.dimensions = dimensions

        # Convert int inputs to tuples
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * dimensions
        else:
            assert len(kernel_size) == dimensions
            self.kernel_size = kernel_size

        if isinstance(strides, int):
            self.strides = (strides,) * dimensions
        else:
            assert len(strides) == dimensions
            self.strides = strides

        if isinstance(padding, int):
            self.padding = (padding,) * dimensions
        else:
            assert len(padding) == dimensions
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation,) * dimensions
        else:
            assert len(dilation) == dimensions
            self.dilation = dilation

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, MaxPoolNd(
                self.dimensions - 1, kernel_size, strides, padding, dilation
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.MaxPool3d(
                kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.MaxPool1d(
            kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.view(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [down_output_size(
            lower_dim_shape[i], i + 1, self.kernel_size, stride=self.strides, padding=self.padding,
            dilation=self.dilation) for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.view(shape[0], shape[2], -1, *lower_dim_shape)

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
            shape[2], 0, self.kernel_size, stride=self.strides, padding=self.padding, dilation=self.dilation
        )
        final_tensor = one_dim_conv_tensor.view(shape[0], *lower_dim_shape, shape[1], high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


# TODO: Add tuple support for parameters
class AvgPoolNd(nn.Module):
    def __init__(self, dimensions, kernel_size, strides, padding):
        super().__init__()

        # Assert dimensions is higher than three
        assert dimensions > 3, "This block is only for cases where the dimensions are higher than 3."

        # Dimension count
        self.dimensions = dimensions

        # Convert int inputs to tuples
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * dimensions
        else:
            assert len(kernel_size) == dimensions
            self.kernel_size = kernel_size

        if isinstance(strides, int):
            self.strides = (strides,) * dimensions
        else:
            assert len(strides) == dimensions
            self.strides = strides

        if isinstance(padding, int):
            self.padding = (padding,) * dimensions
        else:
            assert len(padding) == dimensions
            self.padding = padding

        # Lower dimension representation through recursion until hitting 4D, then just 3D + 1D.
        if self.dimensions > 4:
            self.lower_name = f'lower_{self.dimensions - 1}'
            setattr(self, self.lower_name, AvgPoolNd(
                self.dimensions - 1, kernel_size, strides, padding
            ))
        else:
            self.lower_name = 'lower'
            setattr(self, self.lower_name, nn.AvgPool3d(
                kernel_size=kernel_size, stride=strides, padding=padding
            ))

        # Capture the last dimension left out by the lower representation
        self.last_dim = nn.AvgPool1d(
            kernel_size=kernel_size, stride=strides, padding=padding
        )

    def forward(self, nd_tensor):
        # Get shape of tensor and shape of tensor's lower dimensions
        shape = list(nd_tensor.shape)
        lower_dim_shape = shape[-self.dimensions + 1:]

        # Adjust the view to merge the batch and highest dimension, then the channels, then the lower dimensions
        lower_tensor = nd_tensor.view(shape[0] * shape[2], shape[1], *lower_dim_shape)

        # Work the lower dimension
        lower_tensor = getattr(self, self.lower_name)(lower_tensor)

        # Recalculate the output of lower dimension shapes to account for convolution
        lower_dim_shape = [avg_output_size(
            lower_dim_shape[i], i + 1, self.kernel_size, stride=self.strides, padding=self.padding)
            for i in range(len(lower_dim_shape))
        ]

        # Change everything back
        lower_conv_tensor = lower_tensor.view(shape[0], shape[2], -1, *lower_dim_shape)

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
            shape[2], 0, self.kernel_size, stride=self.strides, padding=self.padding
        )
        final_tensor = one_dim_conv_tensor.view(shape[0], *lower_dim_shape, shape[1], high_dim_size)
        final_tensor = final_tensor.permute(0, order[-2], order[-1], *order[1:-2])
        return final_tensor


if __name__ == '__main__':
    tensor = torch.randn(2, 1, 4, 4, 4, 4, 4)

    conv = ConvNd(5, 1, 2, 3, 1, 0, 1)
    conv_tensor = conv(tensor)

    print("Convolution")
    print(tensor.shape)
    print(conv_tensor.shape)

    conv = ConvTransposeNd(5, 1, 2, 3, 1, 0, 1, 0)
    conv_tensor = conv(tensor)

    print("Convolution Transpose")
    print(tensor.shape)
    print(conv_tensor.shape)

    norm = BatchNormNd(5, 1)
    n_tensor = norm(tensor)

    print("Batch Norm")
    print(tensor.shape)
    print(n_tensor.shape)

    pool = MaxPoolNd(5, 2, 2, 0, 1)
    pool_tensor = pool(tensor)

    print("Max Pool")
    print(tensor.shape)
    print(pool_tensor.shape)

    pool = AvgPoolNd(5, 2, 2, 0)
    pool_tensor = pool(tensor)

    print("Avg Pool")
    print(tensor.shape)
    print(pool_tensor.shape)
