import shutil
import pandas as pd
from torchinfo import summary
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HigherDimUtils import *
from EmbedAttnUtils import AttentionOptions
from ModelModules import UNET, ResNet
from torch.optim import Adam


def assert_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


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
        assert isinstance(out_layer, nn.Module), "out_layer must be an instance of nn.Module."

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
        return f"{self.model_name}\n{self.dimension_count}-Dimension UNET\n------------------------\n\n{self.model}\n"

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


def transformer_unet(in_channels, spatial_dims, attn_pos_max_len):
    cbam_args = {
        'attn_order': [AttentionOptions.CHANNEL, AttentionOptions.SPATIAL], 'use_pos': True,
        'pos_max_len': attn_pos_max_len
    }
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
    # Create custom input layer
    in_layer = nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # Create layer list inputs
    lcl = [[64, 64, 64, 256], [256, 128, 128, 512], [512, 256, 256, 1024], [1024, 512, 512, 2048]]
    lkl = [(1, 3, 1)] * 4
    lpl = [(0, 1, 0)] * 4
    lsl = [3, 4, 6, 3]

    # Create custom output layer
    out_layer = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, out_classes),
        nn.Softmax(dim=1)
    )

    # Create CBAM Attention
    if use_cbam:
        cbam_args = {'attn_order': [AttentionOptions.CHANNEL, AttentionOptions.SPATIAL]}
    else:
        cbam_args = None

    # Create ResNet with default 2D data handling
    return ResNet(
        lcl, lkl, lpl, lsl, in_layer=in_layer, out_layer=out_layer, denoise_diff=True, denoise_embed_count=32,
        conv_attn_args=cbam_args, conv_residual=True
    )


if __name__ == "__main__":
    # Set the aspect size and channels
    test_data_size = 16
    test_data_dim = 4
    test_channels = 1
    test_batch_size = 1

    data = torch.randn(test_batch_size, test_channels, *([test_data_size] * test_data_dim))
    time_steps = torch.randint(0, 300, (test_batch_size,))

    # Create a test UNET that uses CBAM Residual Convolution Blocks and Up-scaling Transformer Blocks
    four_dim_model = transformer_unet(test_channels, 4, test_data_size**test_data_dim)

    # View UNET summary
    summary(four_dim_model.model, input_data=data, depth=10)
