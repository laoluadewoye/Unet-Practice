import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
import matplotlib.pyplot as plt

from UnetOneDim import UNETOne
from UnetTwoDim import UNETTwo


# Model based on DeepFind's "Diffusion models from scratch in PyTorch" video using my dynamic UNET as a base.
class DiffusionUNETModel:
    def __init__(self, name, in_dimensions, in_channels, conv_channels, out_layer, use_up_atten=False,
                 use_dconv_bn=False, use_dconv_relu=False, loss_rate=0.002, time_steps=300,
                 time_embed_count=32):
        assert in_dimensions in [1, 2], "in_dimensions must be 1 or 2."

        assert len(conv_channels) > 1, (
            "channel_list must have at least two elements for down sampling and bottleneck."
        )

        # Basic model information
        self.model_name = name
        self.dim_count = in_dimensions
        if in_dimensions == 1:
            self.model = UNETOne(in_channels, conv_channels, out_layer)
        elif in_dimensions == 2:
            self.model = UNETTwo(
                in_channels, conv_channels, out_layer, up_attention=use_up_atten, denoise_diff=True,
                denoise_embed_count=time_embed_count, dconv_bnorm=use_dconv_bn, dconv_relu=use_dconv_relu
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
        return f"{self.model_name}\n{self.dim_count}-Dimension UNET\n------------------------\n\n{self.model}\n"

    @staticmethod
    def linear_beta_schedule(time_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, time_steps)

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, batch_img_0, time_steps, device):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(batch_img_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_a_s_cum_product, time_steps, batch_img_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_minus_a_s_cum_product, time_steps, batch_img_0.shape
        )

        mean = sqrt_alphas_cumprod_t.to(device) * batch_img_0.to(device)
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
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_minus_a_s_cum_product, t, x.shape
        )
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
        if not os.path.exists(result_folder):
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
            for i, train_batch in enumerate(train_loader, 1):
                # Reset optimizer
                self.optimizer.zero_grad()

                # Generate random time steps
                rand_time_step = torch.randint(0, self.time_steps, (batch_size,), device=device).long()

                # Use only the first element for some reason???
                # Get a pre-compiled forward diffusion step and random noise to try to predict
                noisy_batch_img_0, rand_img_noise = self.forward_diffusion_sample(
                    train_batch[0], rand_time_step, device
                )
                pred_img_noise = self.model(noisy_batch_img_0, rand_time_step)

                # Calculate loss
                noise_abs_loss = F.l1_loss(rand_img_noise, pred_img_noise)
                noise_abs_loss.backward()
                self.optimizer.step()

                # Save findings
                results["epoch"].append(epoch + 1)
                results["batch"].append(i)
                results["loss"].append(noise_abs_loss.cpu().item())

                # Print findings and generate progress sample image
                if i % print_interval == 0:
                    print(f"Epoch {epoch} | step {i:03d} Loss: {noise_abs_loss.cpu().item()} ")
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
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv")

        return results_df


class GeneralUNETModel:
    def __init__(self, name, in_dimensions, in_channels, conv_channels, out_layer, use_up_atten=False,
                 use_dconv_bn=False, use_dconv_relu=False, loss_rate=0.002):
        assert in_dimensions in [1, 2], (
            "in_dimensions must be 1 or 2."
        )

        assert len(conv_channels) > 1, (
            "channel_list must have at least two elements for down sampling and bottleneck."
        )

        self.model_name = name
        self.dim_count = in_dimensions

        # Model
        if in_dimensions == 1:
            self.model = UNETOne(in_channels, conv_channels, out_layer)
        elif in_dimensions == 2:
            self.model = UNETTwo(
                in_channels, conv_channels, out_layer, up_attention=use_up_atten,
                dconv_bnorm=use_dconv_bn, dconv_relu=use_dconv_relu
            )

        self.param_count = sum(p.numel() for p in self.model.parameters())

        # Training parameters
        self.optimizer = Adam(self.model.parameters(), lr=loss_rate)

    def __str__(self):
        return f"{self.model_name}\n{self.dim_count}-Dimension UNET\n------------------------\n\n{self.model}\n"

    def train_model(self, train_loader, epochs, loss_func, print_interval) -> pd.DataFrame:
        # Create training output folder if needed
        result_folder = f"training_output/{self.model_name}"
        if not os.path.exists(result_folder):
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
        results_df.to_csv(f"{result_folder}/{self.model_name}_training_results.csv")

        return results_df


def two_dim_unet():
    # Dimensions
    dimensions = 2

    # Channels (i.e. RGB, Grayscale)
    channels = 3

    # Filter list
    conv_filters = [64, 128, 256, 512, 1024]

    # The output classes
    num_classes = 10
    my_out_layer = nn.Conv2d(in_channels=conv_filters[0], out_channels=num_classes, kernel_size=1)

    # Create sample model
    default_model = GeneralUNETModel(
        'default', dimensions, channels, conv_filters, my_out_layer,
        True, True, True, 0.001
    )
    print(default_model)
    print(f"Parameter Count: {default_model.param_count:,}")


if __name__ == "__main__":
    two_dim_unet()
