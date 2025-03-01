from UNETPyTorch.V1 import GeneralUNETModel as GenUnetOne
from UNETPyTorch.V2 import GeneralUNETModel as GenUnetTwo
from UNETPyTorch.V2 import AttentionOptions

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Create the output layer
class PoolSoftmaxOutput(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        # 1x1 Convolution to go from UNET channels to MNIST classes
        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_classes, kernel_size=1)

    def forward(self, unet_output):
        conv_output = self.final_conv(unet_output)

        # Global average pooling
        pooled_output = F.adaptive_avg_pool2d(conv_output, (1, 1))

        # Flatten down to 1 dimension
        class_output = pooled_output.view(pooled_output.size(0), -1)

        # Return the raw values of each class
        return class_output


# Specify normalization transformation for data
IMG_SIZE = 64
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resizes to a power of two for simple handling
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images (mean, std)
])

# Load MNIST dataset (training and test sets)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader to load data in batches
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Visualize a sample image to ensure it's loaded correctly
first_image, first_label = train_dataset[0]
plt.imshow(first_image.squeeze(), cmap='gray')  # MNIST is single-channel, using cmap='gray'
plt.title(f"Label: {first_label}")
plt.show()

# Create different model versions
dimensions = 2
channels = 1
conv_filters = [64, 128, 256, 512]
mnist_classes = 10
my_out_layer = PoolSoftmaxOutput(in_channels=conv_filters[0], out_classes=mnist_classes)
optim_loss_rate = 0.002

# Version 1
mnist_model_v1 = GenUnetOne(
    name='version_one_unet', in_dimensions=dimensions, in_channels=channels, conv_channels=conv_filters,
    out_layer=my_out_layer, use_up_atten=True, use_attn_pool=True, loss_rate=optim_loss_rate
)

# Version 2
spatial_attn_args = {
    'attn_order': [AttentionOptions.SPATIAL]
}
mnist_model_v2 = GenUnetTwo(
    name='version_two_unet', in_channels=channels, channel_list=conv_filters, out_layer=my_out_layer,
    data_dims=dimensions, up_attn_args=spatial_attn_args, loss_rate=optim_loss_rate
)

# Train the models
epoch_count = 1
print_count = 100
print_interval = max(1, len(train_loader) // print_count)
loss_module = nn.CrossEntropyLoss()
model_train_stats_one = mnist_model_v1.train_model(
    train_loader=train_loader, epochs=epoch_count, loss_func=loss_module, print_interval=print_interval
)
print(model_train_stats_one)
model_train_stats_two = mnist_model_v2.train_model(
    train_loader=train_loader, epochs=epoch_count, loss_func=loss_module, print_interval=print_interval
)
print(model_train_stats_two)

# Test the models
loss_module = nn.CrossEntropyLoss()
mnist_model_v1.test_model(test_loader=test_loader, loss_func=loss_module)
mnist_model_v2.test_model(test_loader=test_loader, loss_func=loss_module)
