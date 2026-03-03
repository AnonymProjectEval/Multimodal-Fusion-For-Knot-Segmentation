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

# Make Fusion root importable when running from Fusion/LateFusion
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

# Reuse prepare and train wrapper from EarlyFusion
from EarlyFusion.prepare import prepare
from EarlyFusion.train import train

# Reuse nnUNet builder helpers to avoid redundant code
from EarlyFusion.train_unet import build_nnunet_direct, count_params


class LateFusionUNet(nn.Module):
    """
    Late fusion for two image modalities using two separate segmentation branches.

    Input:
      x shape = B x 2 x D x H x W
      x[:, 0:1] is wet
      x[:, 1:2] is dry

    Branch outputs:
      wet_logits shape = B x C x D x H x W
      dry_logits shape = B x C x D x H x W

    Fusion options:
      avg            simple average of logits
      global_weight  learn two scalar weights and fuse logits
      conv1x1        concatenate logits and fuse with 1x1x1 Conv3d
    """

    def __init__(
        self,
        num_classes: int = 2,
        img_size: tuple[int, int, int] = (256, 256, 256),
        base_num_features: int = 32,
        max_num_features: int = 320,
        spatial_dims: int = 3,
        dropout_p: float = 0.0,
        nonlin_first: bool = True,
        fusion_mode: str = "conv1x1",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        self.wet_branch = build_nnunet_direct(
            in_channels=1,
            out_channels=num_classes,
            img_size=img_size,
            base_num_features=base_num_features,
            max_num_features=max_num_features,
            spatial_dims=spatial_dims,
            dropout_p=dropout_p,
            nonlin_first=nonlin_first,
        )

        self.dry_branch = build_nnunet_direct(
            in_channels=1,
            out_channels=num_classes,
            img_size=img_size,
            base_num_features=base_num_features,
            max_num_features=max_num_features,
            spatial_dims=spatial_dims,
            dropout_p=dropout_p,
            nonlin_first=nonlin_first,
        )

        if fusion_mode == "avg":
            self.fusion_head = None

        elif fusion_mode == "global_weight":
            # Learnable scalar weights for wet and dry logits
            # We normalize them with softmax in forward for stability
            self.logit_fusion_alpha = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
            self.fusion_head = None

        elif fusion_mode == "conv1x1":
            # Voxelwise linear fusion in class logit space
            # Input channels are wet_logits and dry_logits concatenated, so 2C
            # Output channels are final fused logits, so C
            self.fusion_head = nn.Conv3d(
                in_channels=2 * num_classes,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(
                f"Unsupported fusion_mode '{fusion_mode}'. "
                f"Choose from 'avg', 'global_weight', or 'conv1x1'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape B x 2 x D x H x W, got {tuple(x.shape)}")

        wet = x[:, 0:1]
        dry = x[:, 1:2]

        wet_logits = self.wet_branch(wet)
        dry_logits = self.dry_branch(dry)

        if self.fusion_mode == "avg":
            fused_logits = 0.5 * wet_logits + 0.5 * dry_logits

        elif self.fusion_mode == "global_weight":
            weights = torch.softmax(self.logit_fusion_alpha, dim=0)
            fused_logits = weights[0] * wet_logits + weights[1] * dry_logits

        elif self.fusion_mode == "conv1x1":
            both_logits = torch.cat([wet_logits, dry_logits], dim=1)
            fused_logits = self.fusion_head(both_logits)

        else:
            raise RuntimeError("Invalid fusion mode state.")

        return fused_logits

    def get_fusion_info(self) -> dict:
        info = {"fusion_mode": self.fusion_mode}

        if self.fusion_mode == "avg":
            info["wet_weight"] = 0.5
            info["dry_weight"] = 0.5

        elif self.fusion_mode == "global_weight":
            with torch.no_grad():
                weights = torch.softmax(self.logit_fusion_alpha.detach().cpu(), dim=0)
            info["wet_weight"] = float(weights[0].item())
            info["dry_weight"] = float(weights[1].item())

        elif self.fusion_mode == "conv1x1":
            with torch.no_grad():
                w = self.fusion_head.weight.detach().cpu()
                b = self.fusion_head.bias.detach().cpu() if self.fusion_head.bias is not None else None
            info["fusion_conv_weight_shape"] = tuple(w.shape)
            info["fusion_conv_bias_shape"] = None if b is None else tuple(b.shape)

        return info


if __name__ == "__main__":
    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Update these paths for your environment
    data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"
    model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/LateFusion/Model"

    os.makedirs(model_dir, exist_ok=True)

    # Reuse EarlyFusion dataloader, returns:
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

    # Choose late fusion variant
    # Options: "avg", "global_weight", "conv1x1"
    fusion_mode = "conv1x1"

    model = LateFusionUNet(
        num_classes=2,
        img_size=(256, 256, 256),
        base_num_features=32,
        max_num_features=320,
        spatial_dims=3,
        dropout_p=0.0,
        nonlin_first=True,
        fusion_mode=fusion_mode,
    ).to(device)

    total, trainable = count_params(model)
    print(
        f"[LateFusion {fusion_mode}] Total params: {total:,} "
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

    print("Late fusion training finished.")
    print(f"Best val metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model: {results['best_model_path']}")
    print("Final fusion info:", model.get_fusion_info())