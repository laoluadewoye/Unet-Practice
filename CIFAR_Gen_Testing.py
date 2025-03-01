from UNETPyTorch.V1 import DiffusionUNETModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def load_transformed_dataset(img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)

    return torch.utils.data.ConcatDataset([train, test])


if __name__ == '__main__':
    # Get data into dataloader
    IMG_SIZE = 64
    BATCH_SIZE = 100
    data = load_transformed_dataset(IMG_SIZE)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Create model
    dimensions = 2
    channels = 3
    conv_filters = [64, 128, 256, 512]
    mnist_classes = 10
    my_out_layer = nn.Conv2d(conv_filters[0], channels, 1)
    optim_loss_rate = 0.002
    time_steps = 300

    model = DiffusionUNETModel(
        name='unet_diffusion_cifar10_model', in_dimensions=dimensions, in_channels=channels, conv_channels=conv_filters,
        out_layer=my_out_layer, use_up_atten=True, use_attn_pool=True, use_dconv_res=True, loss_rate=optim_loss_rate,
        time_steps=time_steps, time_embed_count=32
    )

    print(model)

    # Simulate forward diffusion
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(13, 2))
    plt.axis('off')
    num_images = 10
    step_size = int(time_steps/num_images)

    for idx in range(0, time_steps, step_size):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/step_size) + 1)
        img, noise = model.forward_diffusion_sample(image, t, torch.device('cpu'))
        model.show_tensor_image(img)

    plt.tight_layout()
    plt.show()

    # Train the model
    epoch_count = 1
    print_count = 100
    print_interval = max(1, len(dataloader) // print_count)
    model_train_stats = model.train_model(
        train_loader=dataloader, epochs=epoch_count, print_interval=print_interval,
        batch_size=BATCH_SIZE, sample_img_size=IMG_SIZE
    )

    print(model_train_stats)
