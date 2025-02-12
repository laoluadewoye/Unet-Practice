import tensorflow as tf
from tensorflow.keras import layers, Model


class DoubleConv(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_op = tf.keras.Sequential([
            # Transfer to new level of channels + ReLU
            layers.Conv2D(out_channels, 3, padding="same"),
            layers.ReLU(),

            # Second Convolution + ReLU
            layers.Conv2D(out_channels, 3, padding="same"),
            layers.ReLU(),
        ])

    def call(self, x):
        return self.conv_op(x)


class DownSample(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        # Double Convolution Step
        self.conv = DoubleConv(out_channels)

        # 2x2 Max Pooling to Shrink image
        self.pool = layers.MaxPool2D()

    def call(self, x):
        # Channel change for skip connection
        down = self.conv(x)

        # Downsample for next step
        p = self.pool(down)

        return down, p


class UpSample(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        # 2x2 Upscale with channel shrinkage
        self.up = layers.Conv2DTranspose(out_channels, 2, 2)

        # Double Convolution Step (assumes skip connection is present to combine long and short paths)
        self.conv = DoubleConv(out_channels)

    def call(self, inputs):
        x1, x2 = inputs  # Unpack inputs

        # Upscale
        x1 = self.up(x1)

        # Ensure x1 and x2 have the same height & width before concatenation
        x1 = tf.image.resize(x1, tf.shape(x2)[1:3])

        x = tf.concat([x2, x1], axis=-1)  # Channel concatenation
        return self.conv(x)


def get_UNET(in_channels, channel_list, out_layer):
    inputs = tf.keras.Input(shape=(None, None, in_channels))

    # Downsampling path
    skips = []
    x = inputs
    for out_channels in channel_list[:-1]:  # Skip the last bottleneck layer
        down, x = DownSample(out_channels)(x)
        skips.append(down)

    # Bottleneck
    x = DoubleConv(channel_list[-1])(x)

    # Upsampling path
    for out_channels in reversed(channel_list[:-1]):
        skip = skips.pop()
        x = UpSample(out_channels)([x, skip])

    # Output layer
    outputs = out_layer(x)

    return Model(inputs, outputs)


class UNETModel:
    def __init__(self, in_channels, conv_channels, out_layer):
        assert len(conv_channels) > 1, (
            "channel_list must have at least two elements for downsampling and bottleneck."
        )

        # Model
        self.model = get_UNET(in_channels, conv_channels, out_layer)


if __name__ == "__main__":
    # Channels (i.e. RGB, Grayscale)
    channels = 3

    # Filter list
    conv_filters = [64, 128, 256, 512, 1024]

    # The output classes
    num_classes = 10
    my_out_layer = layers.Conv2D(num_classes, 1)

    model = UNETModel(channels, conv_filters, my_out_layer)
    model.model.summary()
