from __future__ import annotations

import os
import re
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

from natsort import natsorted
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
    RandFlipd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    ConcatItemsd,
    DeleteItemsd,
    MapTransform,
)


def _strip_prefix(text: str, prefix: str) -> str:
    if prefix and text.startswith(prefix):
        return text[len(prefix):]
    return text


def _canonical_pair_key(stem: str) -> str:
    """
    Build a shared pairing key from wet or dry stems.

    Examples
      Drydisk01.1        -> 01.1
      Green Disk_01.1    -> 01.1
      SEG_Drydisk01.1    -> handled before this function
    """
    s = stem.strip()

    # Primary rule for your naming style, use the numeric token at the end
    # such as 01.1 or 12.3
    m = re.search(r"(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1)

    # Fallback normalization for unexpected names
    s = s.lower()
    s = s.replace("_", "")
    s = s.replace(" ", "")
    s = s.replace("greendisk", "")
    s = s.replace("drydisk", "")
    return s


def _build_keyed_map(
    paths: List[str],
    *,
    is_mask: bool,
    mask_prefix: str,
) -> Dict[str, str]:
    """
    Build canonical_key -> file_path map and detect duplicate canonical keys.
    """
    out: Dict[str, str] = {}
    duplicates: List[str] = []

    for p in paths:
        stem = Path(p).stem
        if is_mask:
            stem = _strip_prefix(stem, mask_prefix)

        key = _canonical_pair_key(stem)

        if key in out:
            duplicates.append(f"{key}: {out[key]}  |  {p}")
        else:
            out[key] = p

    if duplicates:
        preview = "; ".join(duplicates[:5])
        raise ValueError(
            f"Duplicate canonical keys found while pairing files. Examples: {preview}"
        )

    return out


def collect_paired_samples(
    wet_img_dir: str,
    dry_img_dir: str,
    mask_dir: str,
    pattern: str = "*.nrrd",
    mask_prefix: str = "SEG_",
    fail_on_missing: bool = True,
) -> List[Dict[str, str]]:
    """
    Returns a list of dicts with keys: wet, dry, seg.
    Matches by canonical key extracted from the filename stem.
    """

    wet_imgs = natsorted(glob(os.path.join(wet_img_dir, pattern)))
    dry_imgs = natsorted(glob(os.path.join(dry_img_dir, pattern)))
    masks = natsorted(glob(os.path.join(mask_dir, pattern)))

    wet_map = _build_keyed_map(wet_imgs, is_mask=False, mask_prefix=mask_prefix)
    dry_map = _build_keyed_map(dry_imgs, is_mask=False, mask_prefix=mask_prefix)
    mask_map = _build_keyed_map(masks, is_mask=True, mask_prefix=mask_prefix)

    all_keys = natsorted(list(set(wet_map.keys()) | set(dry_map.keys()) | set(mask_map.keys())))

    samples: List[Dict[str, str]] = []
    missing: List[str] = []

    for key in all_keys:
        wet_path = wet_map.get(key)
        dry_path = dry_map.get(key)
        seg_path = mask_map.get(key)

        if wet_path and dry_path and seg_path:
            samples.append(
                {
                    "wet": wet_path,
                    "dry": dry_path,
                    "seg": seg_path,
                }
            )
        else:
            parts = []
            if wet_path is None:
                parts.append("wet")
            if dry_path is None:
                parts.append("dry")
            if seg_path is None:
                parts.append("mask")
            missing.append(f"{key}: missing {', '.join(parts)}")

    if missing and fail_on_missing:
        examples = "; ".join(missing[:5])
        raise ValueError(
            f"Found {len(missing)} incomplete paired samples. Examples: {examples}"
        )

    return samples


class MatchSpatialShapeByCropd(MapTransform):
    """
    Crop all listed keys to the minimum common spatial shape.

    This fixes small off by one mismatches after resampling or registration.
    Cropping is done from the end, so index zero alignment is preserved.
    Expects channel first tensors with shape (C, D, H, W).
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        spatial_shapes = []
        for key in self.keys:
            if key not in d:
                continue
            x = d[key]
            if x is None:
                continue
            if len(x.shape) < 4:
                raise ValueError(
                    f"Expected channel first 3D tensor for key '{key}', got shape {x.shape}"
                )
            spatial_shapes.append(tuple(int(s) for s in x.shape[1:]))

        if not spatial_shapes:
            return d

        min_d = min(s[0] for s in spatial_shapes)
        min_h = min(s[1] for s in spatial_shapes)
        min_w = min(s[2] for s in spatial_shapes)
        target_spatial = (min_d, min_h, min_w)

        for key in self.keys:
            if key not in d:
                continue
            x = d[key]
            if x is None:
                continue

            current_spatial = tuple(int(s) for s in x.shape[1:])
            if current_spatial != target_spatial:
                d[key] = x[:, :min_d, :min_h, :min_w]

        return d


def build_transforms(
    spatial_size: Tuple[int, int, int] = (256, 256, 256),
    image_keys: Sequence[str] = ("wet", "dry"),
    seg_key: str = "seg",
    concat_to: Optional[str] = None,
    crop_image_key: str = "dry",
    train: bool = True,
):
    """
    Shared transform builder.

    If concat_to is provided, image_keys are concatenated into a single tensor key.
    This is used by early fusion and can also be reused later if needed.
    """

    keys_all = list(image_keys) + [seg_key]

    xforms = [
        LoadImaged(keys=keys_all),
        EnsureChannelFirstd(keys=keys_all),
        MatchSpatialShapeByCropd(keys=keys_all),
        NormalizeIntensityd(keys=list(image_keys), nonzero=True, channel_wise=True),
    ]

    if train:
        xforms.extend(
            [
                SpatialPadd(keys=keys_all, spatial_size=spatial_size, method="end"),
                RandCropByPosNegLabeld(
                    keys=keys_all,
                    label_key=seg_key,
                    spatial_size=spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key=crop_image_key,
                    image_threshold=0.0,
                ),
                RandFlipd(keys=keys_all, spatial_axis=[0], prob=0.5),
                RandFlipd(keys=keys_all, spatial_axis=[1], prob=0.5),
            ]
        )

    if concat_to is not None:
        xforms.extend(
            [
                ConcatItemsd(keys=list(image_keys), name=concat_to, dim=0),
                DeleteItemsd(keys=list(image_keys)),
            ]
        )

    final_keys = [concat_to, seg_key] if concat_to is not None else keys_all
    xforms.append(EnsureTyped(keys=final_keys))

    return Compose(xforms)


def build_dataloaders(
    split_samples: Dict[str, List[Dict[str, str]]],
    train_transform,
    eval_transform,
    cache: bool = False,
    batch_size: int = 1,
    shuffle_train: bool = True,
    seed: Optional[int] = None,
):
    """
    Builds train, val, and test dataloaders.
    Returns train_loader, test_loader, val_loader to stay compatible with your current code.
    """

    if seed is not None:
        random.seed(seed)

    train_samples = list(split_samples["train"])
    if shuffle_train:
        random.shuffle(train_samples)

    Ds = CacheDataset if cache else Dataset
    ds_kwargs = {"cache_rate": 1.0} if cache else {}

    train_ds = Ds(data=train_samples, transform=train_transform, **ds_kwargs)
    val_ds = Ds(data=split_samples["val"], transform=eval_transform, **ds_kwargs)
    test_ds = Ds(data=split_samples["test"], transform=eval_transform, **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    return train_loader, test_loader, val_loader