from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

# Make Fusion/common importable when running from Fusion/EarlyFusion
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

from common.train_shared import train_model


def train(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    metric_fn,
    device: Union[str, torch.device],
    exp_dir: Union[str, Path],
    num_epochs: int = 300,
    scheduler: Any = None,
    accum_steps: int = 1,
    use_amp: bool = True,
    roi_size: Tuple[int, int, int] = (256, 256, 256),
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    post_pred=None,
    post_label=None,
    validate_every: int = 1,
    early_stopping_patience: int = 50,
    min_delta: float = 0.0,
    max_grad_norm: Optional[float] = None,
    save_every_epoch: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Early fusion training wrapper.

    Expects the dataloader batch to contain:
      - 'vol' : stacked 2 channel tensor [wet, dry]
      - 'seg' : segmentation mask
    """

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        exp_dir=exp_dir,
        num_epochs=num_epochs,
        scheduler=scheduler,
        accum_steps=accum_steps,
        use_amp=use_amp,
        input_keys="vol",
        target_key="seg",
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        post_pred=post_pred,
        post_label=post_label,
        validate_every=validate_every,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
        max_grad_norm=max_grad_norm,
        save_every_epoch=save_every_epoch,
        seed=seed,
    )