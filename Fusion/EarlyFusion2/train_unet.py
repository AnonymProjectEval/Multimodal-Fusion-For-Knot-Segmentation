from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Make Fusion root importable when running from Fusion/EarlyFusion2
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

# Reuse prepare and train from EarlyFusion to avoid redundant code
from EarlyFusion.prepare import prepare
from EarlyFusion.train import train

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)


def build_nnunet_direct(
    in_channels: int = 1,
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


class EarlyFusion2LinearMixUNet(nn.Module):
    """
    Early fusion with an explicit learnable linear mixing layer.

    Input:
      x shape = B x 2 x D x H x W   where channel 0 is wet and channel 1 is dry

    Fusion:
      1x1x1 Conv3d acts as a learnable linear layer across the two modalities
      and produces a single fused channel.

    Backbone:
      nnUNet style PlainConvUNet receives the fused single channel volume.
    """

    def __init__(
        self,
        out_channels: int = 2,
        img_size: tuple[int, int, int] = (256, 256, 256),
        base_num_features: int = 32,
        max_num_features: int = 320,
        spatial_dims: int = 3,
        dropout_p: float = 0.0,
        nonlin_first: bool = True,
        init_as_average: bool = True,
    ):
        super().__init__()

        # Linear fusion layer, matches the "right side" idea in your figure
        self.fuse_linear = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if init_as_average:
            with torch.no_grad():
                self.fuse_linear.weight.zero_()
                self.fuse_linear.weight[0, 0, 0, 0, 0] = 0.5  # wet
                self.fuse_linear.weight[0, 1, 0, 0, 0] = 0.5  # dry
                self.fuse_linear.bias.zero_()

        self.backbone = build_nnunet_direct(
            in_channels=1,
            out_channels=out_channels,
            img_size=img_size,
            base_num_features=base_num_features,
            max_num_features=max_num_features,
            spatial_dims=spatial_dims,
            dropout_p=dropout_p,
            nonlin_first=nonlin_first,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fused = self.fuse_linear(x)
        return self.backbone(x_fused)

    def get_fusion_weights(self):
        with torch.no_grad():
            w = self.fuse_linear.weight.detach().cpu().view(1, 2).squeeze(0)
            b = self.fuse_linear.bias.detach().cpu().view(-1)
        return {
            "wet_weight": float(w[0].item()),
            "dry_weight": float(w[1].item()),
            "bias": float(b[0].item()),
        }


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Update these paths for your environment
    data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"
    model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/EarlyFusion2/Model"

    os.makedirs(model_dir, exist_ok=True)

    # Reused dataloader from EarlyFusion, output keys are still:
    #   vol -> B x 2 x D x H x W
    #   seg -> B x 1 x D x H x W
    train_loader, test_loader, val_loader = prepare(
        in_dir=data_dir,
        spatial_size=(256, 256, 256),
        cache=True,
        batch_size=1,
        shuffle=True,
        mask_prefix="SEG_",
        seed=42,
    )

    # EarlyFusion2 model with explicit learnable linear mixing at input
    model = EarlyFusion2LinearMixUNet(
        out_channels=2,
        img_size=(256, 256, 256),
        base_num_features=32,
        max_num_features=320,
        spatial_dims=3,
        dropout_p=0.0,
        nonlin_first=True,
        init_as_average=True,
    ).to(device)

    total, trainable = count_params(model)
    print(
        f"[EarlyFusion2 LinearMix nnUNet] Total params: {total:,} "
        f"(~{total / 1e6:.2f}M) | Trainable: {trainable:,} "
        f"(~{trainable / 1e6:.2f}M)"
    )
    print("Initial fusion weights:", model.get_fusion_weights())

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

    print("EarlyFusion2 training finished.")
    print(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model: {results['best_model_path']}")
    print("Learned fusion weights at end:", model.get_fusion_weights())