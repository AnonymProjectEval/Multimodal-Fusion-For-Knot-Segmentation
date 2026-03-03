from __future__ import annotations
import os
import random
from glob import glob
from pathlib import Path
from typing import List, Dict, Tuple

from natsort import natsorted
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
    RandFlipd,
    SpatialPadd,
    RandCropByPosNegLabeld,
)
from monai.data import DataLoader, Dataset, CacheDataset


def _collect_pairs(
    img_dir: str,
    mask_dir: str,
    pattern: str = "*.nrrd",
    mask_prefix: str = "SEG_",
    fail_on_missing: bool = True,
) -> List[Dict[str, str]]:
    imgs = natsorted(glob(os.path.join(img_dir, pattern)))
    masks = natsorted(glob(os.path.join(mask_dir, pattern)))

    def strip_prefix(text: str) -> str:
        return text[len(mask_prefix):] if text.startswith(mask_prefix) else text

    img_map = {Path(p).stem: p for p in imgs}
    mask_map = {strip_prefix(Path(p).stem): p for p in masks}

    pairs, missing = [], []
    for stem, img_path in img_map.items():
        if stem in mask_map:
            pairs.append({"vol": img_path, "seg": mask_map[stem]})
        else:
            missing.append(img_path)

    if missing and fail_on_missing:
        raise ValueError(
            f"{len(missing)} image(s) in {img_dir} have no matching mask in {mask_dir} "
            f"(first: {Path(missing[0]).name})"
        )

    return pairs


def prepare(
    in_dir: str,
    pixdim: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    a_min: int = 0,
    a_max: int = 4506,
    spatial_size: Tuple[int, int, int] = (256, 256, 256),
    cache: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
):
    splits = ["train", "val", "test"]
    data_files: Dict[str, List[Dict[str, str]]] = {sp: [] for sp in splits}

    # Wet only for training
    train_img_dir = os.path.join(in_dir, "train/DryImage")
    train_mask_dir = os.path.join(in_dir, "train/DryMask2Class")
    data_files["train"].extend(_collect_pairs(train_img_dir, train_mask_dir))

    # Wet only for validation
    val_img_dir = os.path.join(in_dir, "val/WetImage")
    val_mask_dir = os.path.join(in_dir, "val/WetMask2Class")
    data_files["val"].extend(_collect_pairs(val_img_dir, val_mask_dir))

    # Wet only for testing
    test_img_dir = os.path.join(in_dir, "test/WetImage")
    test_mask_dir = os.path.join(in_dir, "test/WetMask2Class")
    data_files["test"].extend(_collect_pairs(test_img_dir, test_mask_dir))

    if shuffle:
        random.shuffle(data_files["train"])

    train_tf = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["vol", "seg"], spatial_size=spatial_size, method="end"),
        RandCropByPosNegLabeld(
            keys=["vol", "seg"],
            label_key="seg",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="vol",
            image_threshold=0.0,
        ),
        RandFlipd(keys=["vol", "seg"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["vol", "seg"], spatial_axis=[1], prob=0.5),
        EnsureTyped(keys=["vol", "seg"]),
    ])

    val_test_tf = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["vol", "seg"]),
    ])

    Ds = CacheDataset if cache else Dataset
    ds_kwargs = {"cache_rate": 1.0} if cache else {}

    train_ds = Ds(data=data_files["train"], transform=train_tf, **ds_kwargs)
    val_ds = Ds(data=data_files["val"], transform=val_test_tf, **ds_kwargs)
    test_ds = Ds(data=data_files["test"], transform=val_test_tf, **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
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

    return train_loader, test_loader, val_loader
