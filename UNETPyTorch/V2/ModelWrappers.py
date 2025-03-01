import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torchvision import transforms
from torchinfo import summary
from typing import Union, Iterable
from enum import auto, StrEnum
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HigherDimUtils import *
from EmbedAttnUtils import AttentionOptions, AttentionArgs
from ModelModules import UNET, ResNet


def assert_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


class DiffusionUNETModel:
    def __init__(self, name, in_channels, channel_list, in_layer=None, out_layer=None, data_dims=2, up_drop_perc=0.3,
                 up_attn_args=None, conv_act_fn=None, conv_attn_args=None, conv_residual=False, loss_rate=0.002,
                 time_steps=300, time_embed_count=32):

        # Assertion list
        assert data_dims > 0, "in_dimensions must be greater than 0."
        assert len(channel_list) > 2, (
            "channel_list must have at least three elements for down sampling and bottleneck."
        )
        assert assert_ascending(channel_list), (
            "channel_list must be in ascending order, from least amount of encodings to most."
        )
        assert isinstance(in_layer, Union[nn.Module, None]), "in_layer must be an instance of nn.Module."
        assert isinstance(out_layer, Union[nn.Module, None]), "out_layer must be an instance of nn.Module."

        # Basic model information
        self.model_name = name
        self.dimension_count = data_dims
        if self.dimension_count == 1:
            conv_function = nn.Conv1d
            bn_function = nn.BatchNorm1d
            mp_function = nn.MaxPool1d
            conv_trans_func = nn.ConvTranspose1d
        elif self.dimension_count == 2:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
            mp_function = nn.MaxPool2d
            conv_trans_func = nn.ConvTranspose2d
        elif self.dimension_count == 3:
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
            mp_function = nn.MaxPool3d
            conv_trans_func = nn.ConvTranspose3d
        else:
            conv_function = ConvNd
            bn_function = BatchNormNd
            mp_function = MaxPoolNd
            conv_trans_func = ConvTransposeNd

        self.model = UNET(
            in_channels, channel_list, in_layer=in_layer, out_layer=out_layer, data_dims=self.dimension_count,
            conv_function=conv_function, bn_function=bn_function, mp_function=mp_function,
            conv_trans_func=conv_trans_func, denoise_diff=True, denoise_embed_count=time_embed_count,
            up_drop_perc=up_drop_perc, up_attn_args=up_attn_args, conv_act_fn=conv_act_fn,
            conv_attn_args=conv_attn_args, conv_residual=conv_residual
        )
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.optimizer = Adam(self.model.parameters(), lr=loss_rate)

        # Pre-calculations for noise scheduling
        # Calculate beta and alpha scheduling
        self.time_steps = time_steps
        self.beta_schedule = self.linear_beta_schedule(self.time_steps)
        alpha_schedule = 1.0 - self.beta_schedule

        # Get the alpha cumulative product and shift it to the start
        a_s_cum_product = torch.cumprod(alpha_schedule, dim=0)
        a_s_cum_product_shift = F.pad(a_s_cum_product[:-1], (1, 0), value=1.0)  # Previous time step?

        # Record the squared reciprocal and squared cumulative product of the alpha schedule
        self.sqrt_recip_a_s = torch.sqrt(1.0 / alpha_schedule)
        self.sqrt_a_s_cum_product = torch.sqrt(a_s_cum_product)

        # Record the square root of 1 - alpha cumulative product
        self.sqrt_minus_a_s_cum_product = torch.sqrt(1. - a_s_cum_product)

        # Record the posterior variance
        self.posterior_variance = self.beta_schedule * (1.0 - a_s_cum_product_shift) / (1.0 - a_s_cum_product)

    def __str__(self):
        title = f"{self.model_name}\n{self.dimension_count}-Dimensional UNET\n------------------------\n\n"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_summary = f"{summary(self.model, depth=10, device=device)}\n"
        return title + model_summary

    @staticmethod
    def linear_beta_schedule(time_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, time_steps)

    @staticmethod
    def get_index_from_list(vals, time_step, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = time_step.shape[0]
        out = vals.gather(-1, time_step.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)

    def forward_diffusion_sample(self, batch_imgs, batch_time_steps, device):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(batch_imgs)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_a_s_cum_product, batch_time_steps, batch_imgs.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_minus_a_s_cum_product, batch_time_steps, batch_imgs.shape
        )

        mean = sqrt_alphas_cumprod_t.to(device) * batch_imgs.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return mean + variance, noise.to(device)

    @staticmethod
    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))

    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.beta_schedule, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_minus_a_s_cum_product, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_a_s, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self, img_size, device, file_path):
        print("Creating Sample Image...")

        # Sample noise
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(13, 2))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.time_steps / num_images)

        for i in range(0, self.time_steps)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i / stepsize) + 1)
                self.show_tensor_image(img.detach().cpu())
        plt.savefig(file_path, dpi=300)
        plt.close()

    def train_model(self, train_loader, epochs, print_interval, batch_size, sample_img_size) -> pd.DataFrame:
        # Create training output folder if needed
        result_folder = f"training_output/{self.model_name}"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        # Create dictionary to save results
        results = {"epoch": [], "batch": [], "loss": []}

        # Create a running loss metric to save the best model
        running_loss = -1.0

        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to training mode
        self.model.train()

        # Loop through epoch
        for epoch in range(epochs):
            for i, (batch_data, _) in enumerate(train_loader, 1):
                # Reset optimizer
                self.optimizer.zero_grad()

                # Generate random time step for testing
                rand_batch_time_steps = torch.randint(0, self.time_steps, (batch_size,), device=device).long()

                # Get a pre-compiled forward diffusion step and random noise to try to predict
                noisy_batch_imgs, rand_batch_noise = self.forward_diffusion_sample(
                    batch_data, rand_batch_time_steps, device
                )
                pred_batch_noise = self.model(noisy_batch_imgs, rand_batch_time_steps)

                # Calculate loss
                noise_abs_loss = F.l1_loss(rand_batch_noise, pred_batch_noise)
                noise_abs_loss.backward()
                self.optimizer.step()

                # Save findings
                results["epoch"].append(epoch + 1)
                results["batch"].append(i)
                results["loss"].append(noise_abs_loss.cpu().item())

                # Print findings and generate progress sample image
                if i % print_interval == 0:
                    print(f"Epoch {epoch+1} | step {i:03d} Loss: {noise_abs_loss.cpu().item()} ")
                    fp = f"{result_folder}/sample_img_{epoch+1:03d}_{i:03d}.png"
                    self.sample_plot_image(sample_img_size, device, fp)

                # Save the model if it beats the current lowest loss
                if noise_abs_loss.cpu().item() < running_loss or running_loss == -1.0:
                    print(f"New best loss score: {noise_abs_loss.cpu().item()}")
                    print("Saving epoch model...")
                    running_loss = noise_abs_loss.cpu().item()
                    torch.save(self.model.state_dict(), f"{result_folder}/{self.model_name}_epoch_{epoch + 1}.pt")

        # Move everything back to cpu
        self.model.to("cpu")

        # Create dataframe then csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv", index=False)

        return results_df


class GeneralUNETModel:
    def __init__(self, name, in_channels, channel_list, in_layer=None, out_layer=None, data_dims=2, up_drop_perc=0.3,
                 up_attn_args=None, conv_act_fn=None, conv_attn_args=None, conv_residual=False, loss_rate=0.002):

        # Assertion list
        assert data_dims > 0, "in_dimensions must be greater than 0."
        assert len(channel_list) > 2, (
            "channel_list must have at least three elements for down sampling and bottleneck."
        )
        assert assert_ascending(channel_list), (
            "channel_list must be in ascending order, from least amount of encodings to most."
        )
        assert isinstance(in_layer, Union[nn.Module, None]), "in_layer must be an instance of nn.Module."
        assert isinstance(out_layer, Union[nn.Module, None]), "out_layer must be an instance of nn.Module."

        # Set the model characteristics and create the model
        self.model_name = name
        self.dimension_count = data_dims

        if self.dimension_count == 1:
            conv_function = nn.Conv1d
            bn_function = nn.BatchNorm1d
            mp_function = nn.MaxPool1d
            conv_trans_func = nn.ConvTranspose1d
        elif self.dimension_count == 2:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
            mp_function = nn.MaxPool2d
            conv_trans_func = nn.ConvTranspose2d
        elif self.dimension_count == 3:
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
            mp_function = nn.MaxPool3d
            conv_trans_func = nn.ConvTranspose3d
        else:
            conv_function = ConvNd
            bn_function = BatchNormNd
            mp_function = MaxPoolNd
            conv_trans_func = ConvTransposeNd

        self.model = UNET(
            in_channels, channel_list, in_layer=in_layer, out_layer=out_layer, data_dims=self.dimension_count,
            conv_function=conv_function, bn_function=bn_function, mp_function=mp_function,
            conv_trans_func=conv_trans_func, up_drop_perc=up_drop_perc, up_attn_args=up_attn_args,
            conv_act_fn=conv_act_fn, conv_attn_args=conv_attn_args, conv_residual=conv_residual
        )

        self.param_count = sum(p.numel() for p in self.model.parameters())

        # Training parameters
        self.optimizer = Adam(self.model.parameters(), lr=loss_rate)

    def __str__(self):
        title = f"{self.model_name}\n{self.dimension_count}-Dimensional UNET\n------------------------\n\n"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_summary = f"{summary(self.model, depth=10, device=device)}\n"
        return title + model_summary

    def train_model(self, train_loader, epochs, loss_func, print_interval) -> pd.DataFrame:
        # Create training output folder if needed
        result_folder = f"training_output/{self.model_name}"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        # Create dictionary to save results
        results = {"epoch": [], "batch": [], "loss": []}

        # Create a running loss metric to save the best model
        running_loss = -1.0

        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to training mode
        self.model.train()

        # Loop through epoch
        for epoch in range(epochs):
            for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
                # Reset optimizer
                self.optimizer.zero_grad()

                # Move data to device
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)

                # Retrieve the batch of class predictions
                train_preds = self.model(train_inputs)

                # Calculate loss using cross entropy
                batch_loss = loss_func(train_preds, train_labels)

                # Conduct backpropagation
                batch_loss.backward()
                self.optimizer.step()

                # Save findings
                results["epoch"].append(epoch+1)
                results["batch"].append(i)
                results["loss"].append(batch_loss.cpu().item())

                # Print findings
                if i % print_interval == 0:
                    print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {batch_loss.cpu().item():.4f}")

                # Save the model if it beats the current lowest loss
                if batch_loss.cpu().item() < running_loss or running_loss == -1.0:
                    print(f"New best loss score: {batch_loss.cpu().item()}")
                    print("Saving epoch model...")
                    running_loss = batch_loss.cpu().item()
                    torch.save(self.model.state_dict(), f"{result_folder}/{self.model_name}_epoch_{epoch+1}.pt")

        # Move everything back to cpu
        self.model.to("cpu")

        # Create dataframe then csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv", index=False)

        return results_df

    def test_model(self, test_loader, loss_func):
        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to eval mode
        self.model.eval()

        # Create total loss and count metric
        total_loss = 0
        total_count = 0

        # Run the tests
        for i, (test_inputs, test_labels) in enumerate(test_loader, 1):
            # Move data to device
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            # Retrieve the batch of class predictions
            test_preds = self.model(test_inputs)

            # Get the loss
            test_loss = loss_func(test_preds, test_labels)

            # Add to loss and count
            total_loss += test_loss.cpu().item()
            total_count += 1

            # Retrieve the highest index from each prediction
            test_preds = torch.argmax(test_preds, dim=1)

            # Print findings
            print(f"Batch: {i}")
            print(f"Predictions:    {test_preds}")
            print(f"Labels:         {test_labels}")
            print(f"Loss:           {test_loss.cpu().item():.4f}\n")

        # Move everything back to cpu
        self.model.to("cpu")

        # Return the average loss
        return total_loss / total_count


class ResNetPresets(StrEnum):
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET50 = auto()
    RESNET101 = auto()
    RESNET152 = auto()


@dataclass
class ResNetArgs:
    channel_list: Iterable[Iterable[int]]
    kernel_list: Iterable[Iterable[int]]
    padding_list: Iterable[Iterable[int]]
    set_list: Iterable[int]


class DiffusionResNetModel:
    def __init__(self, name, in_channels, preset=None, preset_out_classes=0, custom_resnet_args=None, in_layer=None,
                 out_layer=None, data_dims=2, conv_act_fn=None, conv_attn_args=None, conv_residual=False,
                 loss_rate=0.002, time_steps=300, time_embed_count=32):

        # Assertion list
        assert data_dims > 0, "in_dimensions must be greater than 0."
        assert preset is not None or custom_resnet_args is not None, (
            "preset or custom_resnet_args must be specified."
        )
        assert not (preset is not None and custom_resnet_args is not None), (
            "preset and custom_resnet_args cannot both be specified."
        )
        assert preset is not None and preset_out_classes > 1, (
            "preset_out_classes must be greater than 1 when preset is used"
        )
        assert isinstance(custom_resnet_args, Union[ResNetArgs, None]), (
            "custom_resnet_args must be an instance of ResNetArgs for safety."
        )
        assert isinstance(in_layer, Union[nn.Module, None]), "in_layer must be an instance of nn.Module."
        assert isinstance(out_layer, Union[nn.Module, None]), "out_layer must be an instance of nn.Module."

        # Basic model information
        self.model_name = name
        self.dimension_count = data_dims

        if self.dimension_count == 1:
            conv_function = nn.Conv1d
            bn_function = nn.BatchNorm1d
            mp_function = nn.MaxPool1d
        elif self.dimension_count == 2:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
            mp_function = nn.MaxPool2d
        elif self.dimension_count == 3:
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
            mp_function = nn.MaxPool3d
        else:
            conv_function = ConvNd
            bn_function = BatchNormNd
            mp_function = MaxPoolNd

        # Model component tree
        if preset is not None:
            if in_layer is None:
                # Create dimension-specific input layer features
                if self.dimension_count <= 3:
                    input_conv = conv_function(in_channels, 64, kernel_size=7, stride=2, padding=3)
                    input_batch = bn_function(64)
                    input_max_pool = mp_function(kernel_size=3, stride=2, padding=1)
                else:
                    input_conv = conv_function(
                        self.dimension_count, in_channels, 64, kernel_size=7, stride=2, padding=3
                    )
                    input_batch = bn_function(self.dimension_count, 64)
                    input_max_pool = mp_function(self.dimension_count, kernel_size=3, stride=2, padding=1)

                # Create input layer
                in_layer = nn.Sequential(input_conv, input_batch, nn.ReLU(inplace=True), input_max_pool)

            # Set up the package
            if preset.lower() == 'resnet18':
                list_of_channel_lists = [
                    [64, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512]
                ]
                list_of_kernel_lists = [(3, 3)] * 4
                list_of_padding_lists = [(1, 1)] * 4
                list_of_set_counts = [2, 2, 2, 2]
            elif preset.lower() == 'resnet34':
                list_of_channel_lists = [
                    [64, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512]
                ]
                list_of_kernel_lists = [(3, 3)] * 4
                list_of_padding_lists = [(1, 1)] * 4
                list_of_set_counts = [3, 4, 6, 3]
            elif preset.lower() == 'resnet50':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 4, 6, 3]
            elif preset.lower() == 'resnet101':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 4, 23, 3]
            elif preset.lower() == 'resnet152':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 8, 36, 3]
            else:
                raise ValueError(
                    "Preset must be one of 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'"
                )

            # Create output layer
            if out_layer is None:
                out_layer = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.AdaptiveAvgPool1d(list_of_channel_lists[-1][-1]),
                    nn.Linear(list_of_channel_lists[-1][-1], preset_out_classes),
                    nn.Softmax(dim=1)
                )
        elif custom_resnet_args is not None:
            list_of_channel_lists = custom_resnet_args.channel_list
            list_of_kernel_lists = custom_resnet_args.kernel_list
            list_of_padding_lists = custom_resnet_args.padding_list
            list_of_set_counts = custom_resnet_args.set_list
        else:
            list_of_channel_lists = None
            list_of_kernel_lists = None
            list_of_padding_lists = None
            list_of_set_counts = None

        self.model = ResNet(
            list_of_channel_lists, list_of_kernel_lists, list_of_padding_lists, list_of_set_counts, in_layer=in_layer,
            out_layer=out_layer, data_dims=data_dims, conv_function=conv_function, bn_function=bn_function,
            denoise_diff=True, denoise_embed_count=time_embed_count, conv_act_fn=conv_act_fn,
            conv_attn_args=conv_attn_args, conv_residual=conv_residual
        )
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.optimizer = Adam(self.model.parameters(), lr=loss_rate)

        # Pre-calculations for noise scheduling
        # Calculate beta and alpha scheduling
        self.time_steps = time_steps
        self.beta_schedule = self.linear_beta_schedule(self.time_steps)
        alpha_schedule = 1.0 - self.beta_schedule

        # Get the alpha cumulative product and shift it to the start
        a_s_cum_product = torch.cumprod(alpha_schedule, dim=0)
        a_s_cum_product_shift = F.pad(a_s_cum_product[:-1], (1, 0), value=1.0)  # Previous time step?

        # Record the squared reciprocal and squared cumulative product of the alpha schedule
        self.sqrt_recip_a_s = torch.sqrt(1.0 / alpha_schedule)
        self.sqrt_a_s_cum_product = torch.sqrt(a_s_cum_product)

        # Record the square root of 1 - alpha cumulative product
        self.sqrt_minus_a_s_cum_product = torch.sqrt(1. - a_s_cum_product)

        # Record the posterior variance
        self.posterior_variance = self.beta_schedule * (1.0 - a_s_cum_product_shift) / (1.0 - a_s_cum_product)

    def __str__(self):
        title = f"{self.model_name}\n{self.dimension_count}-Dimensional UNET\n------------------------\n\n"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_summary = f"{summary(self.model, depth=10, device=device)}\n"
        return title + model_summary

    @staticmethod
    def linear_beta_schedule(time_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, time_steps)

    @staticmethod
    def get_index_from_list(vals, time_step, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = time_step.shape[0]
        out = vals.gather(-1, time_step.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)

    def forward_diffusion_sample(self, batch_imgs, batch_time_steps, device):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(batch_imgs)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_a_s_cum_product, batch_time_steps, batch_imgs.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_minus_a_s_cum_product, batch_time_steps, batch_imgs.shape
        )

        mean = sqrt_alphas_cumprod_t.to(device) * batch_imgs.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return mean + variance, noise.to(device)

    @staticmethod
    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))

    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.beta_schedule, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_minus_a_s_cum_product, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_a_s, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self, img_size, device, file_path):
        print("Creating Sample Image...")

        # Sample noise
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(13, 2))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.time_steps / num_images)

        for i in range(0, self.time_steps)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i / stepsize) + 1)
                self.show_tensor_image(img.detach().cpu())
        plt.savefig(file_path, dpi=300)
        plt.close()

    def train_model(self, train_loader, epochs, print_interval, batch_size, sample_img_size) -> pd.DataFrame:
        # Create training output folder if needed
        result_folder = f"training_output/{self.model_name}"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        # Create dictionary to save results
        results = {"epoch": [], "batch": [], "loss": []}

        # Create a running loss metric to save the best model
        running_loss = -1.0

        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to training mode
        self.model.train()

        # Loop through epoch
        for epoch in range(epochs):
            for i, (batch_data, _) in enumerate(train_loader, 1):
                # Reset optimizer
                self.optimizer.zero_grad()

                # Generate random time step for testing
                rand_batch_time_steps = torch.randint(0, self.time_steps, (batch_size,), device=device).long()

                # Get a pre-compiled forward diffusion step and random noise to try to predict
                noisy_batch_imgs, rand_batch_noise = self.forward_diffusion_sample(
                    batch_data, rand_batch_time_steps, device
                )
                pred_batch_noise = self.model(noisy_batch_imgs, rand_batch_time_steps)

                # Calculate loss
                noise_abs_loss = F.l1_loss(rand_batch_noise, pred_batch_noise)
                noise_abs_loss.backward()
                self.optimizer.step()

                # Save findings
                results["epoch"].append(epoch + 1)
                results["batch"].append(i)
                results["loss"].append(noise_abs_loss.cpu().item())

                # Print findings and generate progress sample image
                if i % print_interval == 0:
                    print(f"Epoch {epoch + 1} | step {i:03d} Loss: {noise_abs_loss.cpu().item()} ")
                    fp = f"{result_folder}/sample_img_{epoch + 1:03d}_{i:03d}.png"
                    self.sample_plot_image(sample_img_size, device, fp)

                # Save the model if it beats the current lowest loss
                if noise_abs_loss.cpu().item() < running_loss or running_loss == -1.0:
                    print(f"New best loss score: {noise_abs_loss.cpu().item()}")
                    print("Saving epoch model...")
                    running_loss = noise_abs_loss.cpu().item()
                    torch.save(self.model.state_dict(), f"{result_folder}/{self.model_name}_epoch_{epoch + 1}.pt")

        # Move everything back to cpu
        self.model.to("cpu")

        # Create dataframe then csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv", index=False)

        return results_df


class GeneralResNetModel:
    def __init__(self, name, in_channels, preset=None, preset_out_classes=0, custom_resnet_args=None, in_layer=None,
                 out_layer=None, data_dims=2, conv_act_fn=None, conv_attn_args=None, conv_residual=False,
                 loss_rate=0.002):

        # Assertion list
        assert data_dims > 0, "in_dimensions must be greater than 0."
        assert preset is not None or custom_resnet_args is not None, (
            "preset or custom_resnet_args must be specified."
        )
        assert not (preset is not None and custom_resnet_args is not None), (
            "preset and custom_resnet_args cannot both be specified."
        )
        assert preset is not None and preset_out_classes > 1, (
            "preset_out_classes must be greater than 1 when preset is used"
        )
        assert isinstance(custom_resnet_args, Union[ResNetArgs, None]), (
            "custom_resnet_args must be an instance of ResNetArgs for safety."
        )
        assert isinstance(in_layer, Union[nn.Module, None]), "in_layer must be an instance of nn.Module."
        assert isinstance(out_layer, Union[nn.Module, None]), "out_layer must be an instance of nn.Module."

        # Set the model characteristics and create the model
        self.model_name = name
        self.dimension_count = data_dims

        if self.dimension_count == 1:
            conv_function = nn.Conv1d
            bn_function = nn.BatchNorm1d
            mp_function = nn.MaxPool1d
        elif self.dimension_count == 2:
            conv_function = nn.Conv2d
            bn_function = nn.BatchNorm2d
            mp_function = nn.MaxPool2d
        elif self.dimension_count == 3:
            conv_function = nn.Conv3d
            bn_function = nn.BatchNorm3d
            mp_function = nn.MaxPool3d
        else:
            conv_function = ConvNd
            bn_function = BatchNormNd
            mp_function = MaxPoolNd

        # Model component tree
        if preset is not None:
            if in_layer is None:
                # Create dimension-specific input layer features
                if self.dimension_count <= 3:
                    input_conv = conv_function(in_channels, 64, kernel_size=7, stride=2, padding=3)
                    input_batch = bn_function(64)
                    input_max_pool = mp_function(kernel_size=3, stride=2, padding=1)
                else:
                    input_conv = conv_function(
                        self.dimension_count, in_channels, 64, kernel_size=7, stride=2, padding=3,
                    )
                    input_batch = bn_function(self.dimension_count, 64)
                    input_max_pool = mp_function(self.dimension_count, kernel_size=3, stride=2, padding=1)

                # Create input layer
                in_layer = nn.Sequential(input_conv, input_batch, nn.ReLU(inplace=True), input_max_pool)

            # Set up the package
            if preset.lower() == 'resnet18':
                list_of_channel_lists = [
                    [64, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512]
                ]
                list_of_kernel_lists = [(3, 3)] * 4
                list_of_padding_lists = [(1, 1)] * 4
                list_of_set_counts = [2, 2, 2, 2]
            elif preset.lower() == 'resnet34':
                list_of_channel_lists = [
                    [64, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512]
                ]
                list_of_kernel_lists = [(3, 3)] * 4
                list_of_padding_lists = [(1, 1)] * 4
                list_of_set_counts = [3, 4, 6, 3]
            elif preset.lower() == 'resnet50':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 4, 6, 3]
            elif preset.lower() == 'resnet101':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 4, 23, 3]
            elif preset.lower() == 'resnet152':
                list_of_channel_lists = [
                    [64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]
                ]
                list_of_kernel_lists = [(1, 3, 1)] * 4
                list_of_padding_lists = [(0, 1, 0)] * 4
                list_of_set_counts = [3, 8, 36, 3]
            else:
                raise ValueError(
                    "Preset must be one of 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'"
                )

            # Create output layer
            if out_layer is None:
                out_layer = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.AdaptiveAvgPool1d(list_of_channel_lists[-1][-1]),
                    nn.Linear(list_of_channel_lists[-1][-1], preset_out_classes),
                    nn.Softmax(dim=1)
                )
        elif custom_resnet_args is not None:
            list_of_channel_lists = custom_resnet_args.channel_list
            list_of_kernel_lists = custom_resnet_args.kernel_list
            list_of_padding_lists = custom_resnet_args.padding_list
            list_of_set_counts = custom_resnet_args.set_list
        else:
            list_of_channel_lists = None
            list_of_kernel_lists = None
            list_of_padding_lists = None
            list_of_set_counts = None

        self.model = ResNet(
            list_of_channel_lists, list_of_kernel_lists, list_of_padding_lists, list_of_set_counts, in_layer=in_layer,
            out_layer=out_layer, data_dims=data_dims, conv_function=conv_function, bn_function=bn_function,
            conv_act_fn=conv_act_fn, conv_attn_args=conv_attn_args, conv_residual=conv_residual
        )

        self.param_count = sum(p.numel() for p in self.model.parameters())

        # Training parameters
        self.optimizer = Adam(self.model.parameters(), lr=loss_rate)

    def __str__(self):
        title = f"{self.model_name}\n{self.dimension_count}-Dimensional UNET\n------------------------\n\n"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_summary = f"{summary(self.model, depth=10, device=device)}\n"
        return title + model_summary

    def train_model(self, train_loader, epochs, loss_func, print_interval) -> pd.DataFrame:
        # Create training output folder if needed
        result_folder = f"training_output/{self.model_name}"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        # Create dictionary to save results
        results = {"epoch": [], "batch": [], "loss": []}

        # Create a running loss metric to save the best model
        running_loss = -1.0

        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to training mode
        self.model.train()

        # Loop through epoch
        for epoch in range(epochs):
            for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
                # Reset optimizer
                self.optimizer.zero_grad()

                # Move data to device
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)

                # Retrieve the batch of class predictions
                train_preds = self.model(train_inputs)

                # Calculate loss using cross entropy
                batch_loss = loss_func(train_preds, train_labels)

                # Conduct backpropagation
                batch_loss.backward()
                self.optimizer.step()

                # Save findings
                results["epoch"].append(epoch + 1)
                results["batch"].append(i)
                results["loss"].append(batch_loss.cpu().item())

                # Print findings
                if i % print_interval == 0:
                    print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {batch_loss.cpu().item():.4f}")

                # Save the model if it beats the current lowest loss
                if batch_loss.cpu().item() < running_loss or running_loss == -1.0:
                    print(f"New best loss score: {batch_loss.cpu().item()}")
                    print("Saving epoch model...")
                    running_loss = batch_loss.cpu().item()
                    torch.save(self.model.state_dict(), f"{result_folder}/{self.model_name}_epoch_{epoch + 1}.pt")

        # Move everything back to cpu
        self.model.to("cpu")

        # Create dataframe then csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv", index=False)

        return results_df

    def test_model(self, test_loader, loss_func):
        # Check if model can be used on gpu and move it there if available
        print(f"Cuda availability status: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set the model to eval mode
        self.model.eval()

        # Create total loss and count metric
        total_loss = 0
        total_count = 0

        # Run the tests
        for i, (test_inputs, test_labels) in enumerate(test_loader, 1):
            # Move data to device
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            # Retrieve the batch of class predictions
            test_preds = self.model(test_inputs)

            # Get the loss
            test_loss = loss_func(test_preds, test_labels)

            # Add to loss and count
            total_loss += test_loss.cpu().item()
            total_count += 1

            # Retrieve the highest index from each prediction
            test_preds = torch.argmax(test_preds, dim=1)

            # Print findings
            print(f"Batch: {i}")
            print(f"Predictions:    {test_preds}")
            print(f"Labels:         {test_labels}")
            print(f"Loss:           {test_loss.cpu().item():.4f}\n")

        # Move everything back to cpu
        self.model.to("cpu")

        # Return the average loss
        return total_loss / total_count


def transformer_unet(in_channels, spatial_dims, attn_pos_max_len):
    cbam_args = AttentionArgs(
        attn_order=[AttentionOptions.CHANNEL, AttentionOptions.SPATIAL],
        use_pos=True,
        pos_max_len=attn_pos_max_len
    )
    transformer_args = {
        'attn_order': [AttentionOptions.QKV], 'qkv_heads': 2, 'use_pos': True,
        'pos_max_len': attn_pos_max_len
    }
    return GeneralUNETModel(
        name='sample_model', in_channels=in_channels, channel_list=[64, 128, 256], out_layer=nn.Sigmoid(),
        data_dims=spatial_dims, up_drop_perc=0.5, up_attn_args=transformer_args,
        conv_act_fn=nn.LeakyReLU(0.2, inplace=True), conv_attn_args=cbam_args, conv_residual=True
    )


def res_net_fifty(in_channels, spatial_dims, out_classes, use_cbam=False):
    # Create CBAM Attention, this time using a attnargs class
    if use_cbam:
        cbam_args = AttentionArgs(attn_order=[AttentionOptions.CHANNEL, AttentionOptions.SPATIAL])
    else:
        cbam_args = None

    # Create ResNet with default 2D data handling
    return GeneralResNetModel('sample_model', in_channels, preset=ResNetPresets.RESNET18,
                              preset_out_classes=out_classes, data_dims=spatial_dims, conv_attn_args=cbam_args,
                              conv_residual=True)


if __name__ == "__main__":
    # Set the aspect size and channels
    test_data_size = 16
    test_data_dim = 4
    test_channels = 1
    test_batch_size = 1

    test_data = torch.randn(test_batch_size, test_channels, *([test_data_size] * test_data_dim))
    test_time_steps = torch.randint(0, 300, (test_batch_size,))

    # Create a test UNET that uses CBAM Residual Convolution Blocks and Up-scaling Transformer Blocks
    var_dim_model = transformer_unet(test_channels, test_data_dim, test_data_size**test_data_dim)

    # View UNET summary
    print(var_dim_model)

    # Create a test ResNet that uses CBAM Residual Convolution Blocks
    var_dim_model = res_net_fifty(test_channels, test_data_dim, 10, use_cbam=True)

    # View UNET summary
    print(var_dim_model)
