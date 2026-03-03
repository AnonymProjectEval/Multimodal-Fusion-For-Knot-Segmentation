from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Make Fusion root importable when running from Fusion/IntermediateFusion
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

# Reuse data pipeline and shared trainer
from EarlyFusion.prepare import prepare
from EarlyFusion.train import train


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv = ConvBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ModalityEncoder3D(nn.Module):
    """
    Returns modality specific skip features and bottleneck feature.
    """

    def __init__(self, in_channels: int, feature_channels: Sequence[int]):
        super().__init__()
        if len(feature_channels) < 3:
            raise ValueError("feature_channels must contain at least 3 levels.")

        self.feature_channels = list(feature_channels)

        self.stem = ConvBlock3D(in_channels, self.feature_channels[0])

        self.downs = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1):
            self.downs.append(DownBlock3D(self.feature_channels[i], self.feature_channels[i + 1]))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)

        bottleneck = x
        return skips, bottleneck


class BottleneckConcatLinearFusion3D(nn.Module):
    """
    Intermediate fusion at bottleneck using concat plus 1x1x1 Conv3d.

    Wet is the main stream. Dry contributes through a learned linear fusion term.

    fused = wet_bottleneck + gamma * Conv1x1(concat(wet_bottleneck, dry_bottleneck))
    """

    def __init__(
        self,
        channels: int,
        init_gamma: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels

        self.fuse_conv = nn.Conv3d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Start from wet only path, then learn how much fused correction to add
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    def forward(self, wet_bottleneck: torch.Tensor, dry_bottleneck: torch.Tensor) -> torch.Tensor:
        x = torch.cat([wet_bottleneck, dry_bottleneck], dim=1)
        delta = self.fuse_conv(x)
        delta = self.drop(delta)
        fused = wet_bottleneck + self.gamma * delta
        return fused

    def get_fusion_info(self) -> Dict[str, object]:
        with torch.no_grad():
            w = self.fuse_conv.weight.detach().cpu()
            b = self.fuse_conv.bias.detach().cpu() if self.fuse_conv.bias is not None else None
        return {
            "gamma": float(self.gamma.detach().cpu().item()),
            "fusion_weight_shape": tuple(w.shape),
            "fusion_bias_shape": None if b is None else tuple(b.shape),
        }


class SharedDecoder3D(nn.Module):
    """
    Shared decoder that uses wet skip connections.
    Wet is the primary modality, dry contributes at bottleneck fusion.
    """

    def __init__(self, feature_channels: Sequence[int], num_classes: int):
        super().__init__()
        ch = list(feature_channels)
        if len(ch) < 3:
            raise ValueError("feature_channels must contain at least 3 levels.")

        self.up_blocks = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock3D(
                    in_channels=ch[i],
                    skip_channels=ch[i - 1],
                    out_channels=ch[i - 1],
                )
            )

        self.out_head = nn.Conv3d(ch[0], num_classes, kernel_size=1, bias=True)

    def forward(self, bottleneck: torch.Tensor, wet_skips: Sequence[torch.Tensor]) -> torch.Tensor:
        expected_skips = len(self.up_blocks)
        if len(wet_skips) != expected_skips:
            raise ValueError(f"Expected {expected_skips} wet skip tensors, got {len(wet_skips)}")

        x = bottleneck
        for up_block, skip in zip(self.up_blocks, reversed(wet_skips)):
            x = up_block(x, skip)

        logits = self.out_head(x)
        return logits


class IntermediateConcatFusionUNet(nn.Module):
    """
    Intermediate fusion segmentation model with bottleneck concat plus 1x1x1 fusion.

    Input:
      x shape = B x 2 x D x H x W
      channel 0 is wet
      channel 1 is dry

    Design:
      1. Wet encoder and Dry encoder are separate
      2. Bottleneck features are concatenated and linearly fused with 1x1x1 Conv3d
      3. Fusion is added back to wet bottleneck with a residual connection
      4. Shared decoder predicts segmentation from fused bottleneck and wet skips
    """

    def __init__(
        self,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (16, 32, 64, 128, 256, 320),
        fusion_dropout: float = 0.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()

        self.feature_channels = tuple(feature_channels)

        self.wet_encoder = ModalityEncoder3D(in_channels=1, feature_channels=self.feature_channels)
        self.dry_encoder = ModalityEncoder3D(in_channels=1, feature_channels=self.feature_channels)

        self.bottleneck_fusion = BottleneckConcatLinearFusion3D(
            channels=self.feature_channels[-1],
            init_gamma=init_gamma,
            dropout=fusion_dropout,
        )

        self.decoder = SharedDecoder3D(
            feature_channels=self.feature_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape B x 2 x D x H x W, got {tuple(x.shape)}")

        wet = x[:, 0:1]
        dry = x[:, 1:2]

        wet_skips, wet_bottleneck = self.wet_encoder(wet)
        _, dry_bottleneck = self.dry_encoder(dry)

        fused_bottleneck = self.bottleneck_fusion(wet_bottleneck, dry_bottleneck)
        logits = self.decoder(fused_bottleneck, wet_skips)
        return logits

    def get_fusion_info(self) -> Dict[str, object]:
        return self.bottleneck_fusion.get_fusion_info()


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Update these paths for your environment
    data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"
    model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/IntermediateFusion2/Model"

    os.makedirs(model_dir, exist_ok=True)

    # Reuse EarlyFusion dataloader
    # vol shape = B x 2 x D x H x W
    # seg shape = B x 1 x D x H x W
    train_loader, test_loader, val_loader = prepare(
        in_dir=data_dir,
        spatial_size=(256, 256, 256),
        cache=True,
        batch_size=1,
        shuffle=True,
        mask_prefix="SEG_",
        seed=42,
    )

    # Two encoders plus decoder is heavier than single backbone
    # Keep channels a bit smaller than the plain single model
    model = IntermediateConcatFusionUNet(
        num_classes=2,
        feature_channels=(16, 32, 64, 128, 256, 320),
        fusion_dropout=0.0,
        init_gamma=0.0,
    ).to(device)

    total, trainable = count_params(model)
    print(
        f"[Intermediate Concat1x1 Fusion] Total params: {total:,} "
        f"(~{total / 1e6:.2f}M) | Trainable: {trainable:,} "
        f"(~{trainable / 1e6:.2f}M)"
    )
    print("Initial fusion info:", model.get_fusion_info())

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

    print("Intermediate concat plus 1x1x1 fusion training finished.")
    print(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model: {results['best_model_path']}")
    print("Final fusion info:", model.get_fusion_info())