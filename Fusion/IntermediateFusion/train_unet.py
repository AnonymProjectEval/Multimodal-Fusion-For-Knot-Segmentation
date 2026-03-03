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
            # Last stage output is bottleneck, do not use it as a skip
            if i < len(self.downs) - 1:
                skips.append(x)

        bottleneck = x
        return skips, bottleneck


class CrossAttentionBottleneck3D(nn.Module):
    """
    Cross attention at bottleneck.
    Query comes from wet features.
    Key and Value come from dry features.

    Fused output is added back to the wet bottleneck through a residual path:
      fused = wet + gamma * cross_attn(wet, dry, dry)
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads.")

        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        self.q_proj = nn.Conv3d(in_channels, attn_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv3d(in_channels, attn_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv3d(in_channels, attn_dim, kernel_size=1, bias=True)

        self.q_norm = nn.LayerNorm(attn_dim)
        self.k_norm = nn.LayerNorm(attn_dim)
        self.v_norm = nn.LayerNorm(attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Conv3d(attn_dim, in_channels, kernel_size=1, bias=True)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B x N x C
        return tokens, (d, h, w)

    @staticmethod
    def _to_map(tokens: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        d, h, w = spatial_shape
        x = tokens.transpose(1, 2).reshape(b, c, d, h, w)
        return x

    def forward(self, wet_feat: torch.Tensor, dry_feat: torch.Tensor) -> torch.Tensor:
        q_map = self.q_proj(wet_feat)
        k_map = self.k_proj(dry_feat)
        v_map = self.v_proj(dry_feat)

        q_tokens, spatial_shape = self._to_tokens(q_map)
        k_tokens, _ = self._to_tokens(k_map)
        v_tokens, _ = self._to_tokens(v_map)

        q_tokens = self.q_norm(q_tokens)
        k_tokens = self.k_norm(k_tokens)
        v_tokens = self.v_norm(v_tokens)

        attn_tokens, _ = self.attn(
            query=q_tokens,
            key=k_tokens,
            value=v_tokens,
            need_weights=False,
        )
        attn_tokens = self.out_drop(attn_tokens)

        attn_map = self._to_map(attn_tokens, spatial_shape)
        attn_map = self.out_proj(attn_map)

        fused = wet_feat + self.gamma * attn_map
        return fused


class SharedDecoder3D(nn.Module):
    """
    Shared decoder that uses wet skip connections.
    Wet is the primary modality, dry contributes through cross attention.
    """

    def __init__(self, feature_channels: Sequence[int], num_classes: int):
        super().__init__()
        ch = list(feature_channels)

        # ch example: [16, 32, 64, 128, 256, 320]
        # skips are ch[0]..ch[-2], bottleneck is ch[-1]
        self.up_blocks = nn.ModuleList([
            UpBlock3D(ch[-1], ch[-2], ch[-2]),
            UpBlock3D(ch[-2], ch[-3], ch[-3]),
            UpBlock3D(ch[-3], ch[-4], ch[-4]),
            UpBlock3D(ch[-4], ch[-5], ch[-5]),
            UpBlock3D(ch[-5], ch[-6], ch[-6]),
        ])

        self.out_head = nn.Conv3d(ch[0], num_classes, kernel_size=1, bias=True)

    def forward(self, bottleneck: torch.Tensor, wet_skips: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(wet_skips) != 5:
            raise ValueError(f"Expected 5 wet skip tensors, got {len(wet_skips)}")

        x = bottleneck
        x = self.up_blocks[0](x, wet_skips[4])
        x = self.up_blocks[1](x, wet_skips[3])
        x = self.up_blocks[2](x, wet_skips[2])
        x = self.up_blocks[3](x, wet_skips[1])
        x = self.up_blocks[4](x, wet_skips[0])

        logits = self.out_head(x)
        return logits


class IntermediateCrossAttentionUNet(nn.Module):
    """
    Intermediate fusion segmentation model with bottleneck cross attention.

    Input:
      x shape = B x 2 x D x H x W
      channel 0 is wet
      channel 1 is dry

    Design:
      1. Wet encoder and Dry encoder are separate
      2. Cross attention at bottleneck uses Wet as Query, Dry as Key and Value
      3. Shared decoder predicts segmentation from fused bottleneck and wet skips
    """

    def __init__(
        self,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (16, 32, 64, 128, 256, 320),
        attn_dim: int = 128,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()

        self.feature_channels = tuple(feature_channels)

        self.wet_encoder = ModalityEncoder3D(in_channels=1, feature_channels=self.feature_channels)
        self.dry_encoder = ModalityEncoder3D(in_channels=1, feature_channels=self.feature_channels)

        self.cross_attn = CrossAttentionBottleneck3D(
            in_channels=self.feature_channels[-1],
            attn_dim=attn_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            init_gamma=init_gamma,
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

        fused_bottleneck = self.cross_attn(wet_bottleneck, dry_bottleneck)
        logits = self.decoder(fused_bottleneck, wet_skips)
        return logits

    def get_fusion_info(self) -> Dict[str, float]:
        return {
            "gamma": float(self.cross_attn.gamma.detach().cpu().item()),
            "attn_dim": float(self.cross_attn.attn_dim),
            "attn_heads": float(self.cross_attn.num_heads),
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
    model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/IntermediateFusion/Model"

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

    # Intermediate fusion with cross attention at bottleneck
    # Base channels are set a bit lower because this model has two encoders
    model = IntermediateCrossAttentionUNet(
        num_classes=2,
        feature_channels=(16, 32, 64, 128, 256, 320),
        attn_dim=128,
        attn_heads=4,
        attn_dropout=0.0,
        init_gamma=0.0,
    ).to(device)

    total, trainable = count_params(model)
    print(
        f"[Intermediate CrossAttention] Total params: {total:,} "
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

    print("Intermediate cross attention training finished.")
    print(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model: {results['best_model_path']}")
    print("Final fusion info:", model.get_fusion_info())