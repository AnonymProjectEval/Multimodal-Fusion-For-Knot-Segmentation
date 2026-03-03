from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR

from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Make Fusion/common importable when running from Fusion/EarlyFusion
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

#from prepare import prepare
#from train import train

from EarlyFusion.prepare import prepare #for other fusion to import from EarlyFusion.train_unet import build_nnunet_direct, count_params
from EarlyFusion.train import train

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)


def build_nnunet_direct(
    in_channels: int = 2,
    out_channels: int = 2,
    img_size: tuple[int, int, int] = (256, 256, 256),
    base_num_features: int = 32,
    max_num_features: int = 320,
    spatial_dims: int = 3,
    n_conv_per_stage_encoder=None,
    n_conv_per_stage_decoder=None,
    num_pool_per_axis=None,
    pool_op_kernel_sizes=None,
    conv_kernel_sizes=None,
    dropout_p: float = 0.0,
    nonlin_first: bool = True,
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

    features_per_stage = [
        min(base_num_features * (2 ** i), max_num_features) for i in range(n_stages)
    ]

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


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Update these paths for your environment
    data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"
    model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/EarlyFusion/Model5"

    os.makedirs(model_dir, exist_ok=True)

    # Data
    train_loader, test_loader, val_loader = prepare(
        in_dir=data_dir,
        spatial_size=(256, 256, 256),
        cache=True,
        batch_size=1,
        shuffle=True,
        mask_prefix="SEG_",
        seed=42,
    )

    # Model for early fusion, 2 channel input [wet, dry]
    model = build_nnunet_direct(
        in_channels=2,
        out_channels=2,
        img_size=(256, 256, 256),
        base_num_features=32,
        max_num_features=320,
        spatial_dims=3,
        dropout_p=0.0,
        nonlin_first=True,
    ).to(device)

    total, trainable = count_params(model)
    print(
        f"[EarlyFusion nnUNet] Total params: {total:,} "
        f"(~{total / 1e6:.2f}M) | Trainable: {trainable:,} "
        f"(~{trainable / 1e6:.2f}M)"
    )

    # Loss and metric
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    metric_fn = DiceMetric(include_background=False, reduction="mean")

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        exp_dir=model_dir,
        num_epochs=300,
        scheduler=scheduler,
        accum_steps=2,
        use_amp=True,
        roi_size=(256, 256, 256),
        sw_batch_size=1,
        overlap=0.25,
        post_pred=post_pred,
        post_label=post_label,
        validate_every=1,
        early_stopping_patience=30,
        min_delta=0.0,
        max_grad_norm=None,
        save_every_epoch=False,
        seed=42,
    )

    print("Early fusion training finished.")
    print(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model: {results['best_model_path']}")