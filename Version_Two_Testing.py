from torch import nn
from UNETPyTorch.V2 import GeneralUNETModel, GeneralResNetModel, ResNetPresets, AttentionArgs, AttentionOptions


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

    # Create a test UNET that uses CBAM Residual Convolution Blocks and Up-scaling Transformer Blocks
    var_dim_model = transformer_unet(test_channels, test_data_dim, test_data_size**test_data_dim)

    # View UNET summary
    print(var_dim_model)

    # Create a test ResNet that uses CBAM Residual Convolution Blocks
    var_dim_model = res_net_fifty(test_channels, test_data_dim, 10, use_cbam=True)

    # View ResNet summary
    print(var_dim_model)
