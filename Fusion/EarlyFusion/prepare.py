from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

# Make Fusion/common importable when running from Fusion/EarlyFusion
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

from common.prepare_shared import (
    collect_paired_samples,
    build_transforms,
    build_dataloaders,
)


def prepare(
    in_dir: str,
    spatial_size: Tuple[int, int, int] = (256, 256, 256),
    cache: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
    mask_prefix: str = "SEG_",
    seed: int | None = None,
):
    """
    Early fusion dataloader builder.

    Output batch keys
      vol  -> stacked tensor with 2 channels where channel 0 is wet and channel 1 is dry
      seg  -> segmentation mask

    Folder expectation under in_dir

      train/WetImage
      train/DryImage
      train/DryMask2Class

      val/WetImage
      val/DryImage
      val/WetMask2Class

      test/WetImage
      test/DryImage
      test/WetMask2Class
    """

    split_samples = {
        "train": collect_paired_samples(
            wet_img_dir=os.path.join(in_dir, "train", "WetImage"),
            dry_img_dir=os.path.join(in_dir, "train", "DryImage"),
            mask_dir=os.path.join(in_dir, "train", "DryMask2Class"),
            mask_prefix=mask_prefix,
        ),
        "val": collect_paired_samples(
            wet_img_dir=os.path.join(in_dir, "val", "WetImage"),
            dry_img_dir=os.path.join(in_dir, "val", "DryImage"),
            mask_dir=os.path.join(in_dir, "val", "WetMask2Class"),
            mask_prefix=mask_prefix,
        ),
        "test": collect_paired_samples(
            wet_img_dir=os.path.join(in_dir, "test", "WetImage"),
            dry_img_dir=os.path.join(in_dir, "test", "DryImage"),
            mask_dir=os.path.join(in_dir, "test", "WetMask2Class"),
            mask_prefix=mask_prefix,
        ),
    }

    train_tf = build_transforms(
        spatial_size=spatial_size,
        image_keys=("wet", "dry"),
        seg_key="seg",
        concat_to="vol",      # Early fusion stacks wet and dry into one tensor
        crop_image_key="dry", # Crop guidance uses dry, since annotation is mainly on dry
        train=True,
    )

    eval_tf = build_transforms(
        spatial_size=spatial_size,
        image_keys=("wet", "dry"),
        seg_key="seg",
        concat_to="vol",
        crop_image_key="dry",
        train=False,
    )

    return build_dataloaders(
        split_samples=split_samples,
        train_transform=train_tf,
        eval_transform=eval_tf,
        cache=cache,
        batch_size=batch_size,
        shuffle_train=shuffle,
        seed=seed,
    )