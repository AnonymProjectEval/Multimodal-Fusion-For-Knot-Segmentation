import os
import torch
from monai.utils import set_determinism

from prepare import prepare
from train import train

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

set_determinism(seed=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/CombinedNNUnet256WetOnly/Model"
os.makedirs(model_dir, exist_ok=True)
data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"

def build_nnunet_direct(
    in_channels=1,
    out_channels=2,
    img_size=(256, 256, 256),
    base_num_features=32,
    max_num_features=320,
    spatial_dims=3,
    n_conv_per_stage_encoder=None,
    n_conv_per_stage_decoder=None,
    num_pool_per_axis=None,
    pool_op_kernel_sizes=None,
    conv_kernel_sizes=None,
    dropout_p=0.0,
    nonlin_first=True,
):
    if num_pool_per_axis is None:
        min_dim = min(img_size)
        min_dim = max(min_dim, 8)
        max_pools = int(torch.log2(torch.tensor(min_dim // 8)).item())
        max_pools = max(1, min(max_pools, 5))
        num_pool_per_axis = [max_pools] * spatial_dims
    n_stages = max(num_pool_per_axis) + 1
    if n_conv_per_stage_encoder is None:
        n_conv_per_stage_encoder = [2] * n_stages
    if n_conv_per_stage_decoder is None:
        n_conv_per_stage_decoder = [2] * (n_stages - 1)
    if pool_op_kernel_sizes is None:
        pool_op_kernel_sizes = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    if conv_kernel_sizes is None:
        conv_kernel_sizes = [[3, 3, 3]] * n_stages
    conv_op = convert_dim_to_conv_op(spatial_dims)
    norm_op = get_matching_instancenorm(conv_op)
    features_per_stage = [min(base_num_features * (2 ** i), max_num_features) for i in range(n_stages)]
    network = PlainConvUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        n_conv_per_stage=n_conv_per_stage_encoder,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        num_classes=out_channels,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=torch.nn.Dropout3d if spatial_dims == 3 else torch.nn.Dropout2d,
        dropout_op_kwargs={"p": float(dropout_p), "inplace": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        nonlin_first=nonlin_first,
    )
    return network

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    train_loader, test_loader, val_loader = prepare(
        data_dir,
        cache=True,
        spatial_size=(256, 256, 256),
        batch_size=1,
        shuffle=True,
    )

    model = build_nnunet_direct(
        in_channels=1,
        out_channels=2,
        img_size=(256, 256, 256),
        base_num_features=32,
        max_num_features=320,
        spatial_dims=3,
        dropout_p=0.0,
        nonlin_first=True,
    ).to(device)

    total, trainable = count_params(model)
    print(f"[nnUNet-Direct] Total params: {total:,} (~{total/1e6:.2f}M) | Trainable: {trainable:,} (~{trainable/1e6:.2f}M)")

    num_epochs = 300
    print(f"Starting nnUNet-Direct training for {num_epochs} epochs")

    train(
        model,
        (train_loader, test_loader, val_loader),
        num_epochs,
        device,
        model_dir,
        val_infer_roi=(256, 256, 256),
        val_infer_overlap=0.25,
        val_sw_batch_size=1,
        lr=1e-4,
        weight_decay=1e-5,
        step_size=10,
        gamma=0.8,
        accum_steps=2,
        early_stop_patience=30,
        use_amp=True,
    )

    print("nnUNet-Direct training completed successfully")
