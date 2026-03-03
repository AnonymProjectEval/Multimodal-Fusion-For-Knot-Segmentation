from __future__ import annotations

import sys
from pathlib import Path

import torch
from monai.utils import set_determinism

# Make Fusion root importable when running from Fusion/EarlyFusion2
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

# Reuse data preparation from EarlyFusion
from EarlyFusion.prepare import prepare

# Reuse model class from EarlyFusion2 train_unet
from train_unet import EarlyFusion2LinearMixUNet, count_params


def tensor_info(name: str, x: torch.Tensor) -> None:
    x_cpu = x.detach().cpu()
    print(f"{name}")
    print(f"  shape   : {tuple(x_cpu.shape)}")
    print(f"  dtype   : {x_cpu.dtype}")
    print(f"  min max : {float(x_cpu.min()):.4f} , {float(x_cpu.max()):.4f}")
    print(f"  mean std: {float(x_cpu.mean()):.4f} , {float(x_cpu.std()):.4f}")


def unique_values_info(name: str, x: torch.Tensor, max_show: int = 20) -> None:
    vals = torch.unique(x.detach().cpu())
    vals_list = vals.tolist()
    if len(vals_list) > max_show:
        preview = vals_list[:max_show]
        print(f"{name} unique values first {max_show}: {preview} ... total {len(vals_list)}")
    else:
        print(f"{name} unique values: {vals_list}")


def small_crop_3d(x: torch.Tensor, crop_size=(64, 64, 64)) -> torch.Tensor:
    _, _, d, h, w = x.shape
    cd, ch, cw = crop_size

    if d < cd or h < ch or w < cw:
        raise ValueError(
            f"Input is smaller than crop size. Input spatial {(d, h, w)}, crop {crop_size}"
        )

    d0 = (d - cd) // 2
    h0 = (h - ch) // 2
    w0 = (w - cw) // 2
    return x[:, :, d0:d0 + cd, h0:h0 + ch, w0:w0 + cw]


def check_batch(batch: dict, split_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"{split_name} batch keys: {list(batch.keys())}")
    print("=" * 80)

    if "vol" not in batch or "seg" not in batch:
        raise KeyError(f"{split_name} batch must contain 'vol' and 'seg' keys.")

    vol = batch["vol"]
    seg = batch["seg"]

    if not torch.is_tensor(vol) or not torch.is_tensor(seg):
        raise TypeError(f"{split_name} batch values 'vol' and 'seg' must be tensors.")

    tensor_info(f"{split_name} vol", vol)
    tensor_info(f"{split_name} seg", seg)
    unique_values_info(f"{split_name} seg", seg)

    if vol.ndim != 5:
        raise ValueError(f"{split_name} vol must have shape B,C,D,H,W. Got {tuple(vol.shape)}")
    if seg.ndim != 5:
        raise ValueError(f"{split_name} seg must have shape B,C,D,H,W. Got {tuple(seg.shape)}")
    if vol.shape[1] != 2:
        raise ValueError(f"{split_name} vol must have 2 channels. Got {vol.shape[1]}")
    if seg.shape[1] != 1:
        print(f"Warning: {split_name} seg channel is {seg.shape[1]}, expected 1")

    wet = vol[:, 0:1]
    dry = vol[:, 1:2]
    tensor_info(f"{split_name} wet channel", wet)
    tensor_info(f"{split_name} dry channel", dry)


def main():
    set_determinism(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Update this path to your dataset root
    data_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"

    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")

    train_loader, test_loader, val_loader = prepare(
        in_dir=data_dir,
        spatial_size=(256, 256, 256),
        cache=False,
        batch_size=1,
        shuffle=True,
        mask_prefix="SEG_",
        seed=42,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    check_batch(train_batch, "TRAIN")
    check_batch(val_batch, "VAL")

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
    print("\n" + "=" * 80)
    print(f"Model params total: {total:,} which is {total / 1e6:.2f}M")
    print(f"Model params trainable: {trainable:,} which is {trainable / 1e6:.2f}M")
    print("Initial linear fusion weights:", model.get_fusion_weights())
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        x = train_batch["vol"].to(device, non_blocking=True)
        x_small = small_crop_3d(x, crop_size=(64, 64, 64))
        print(f"\nForward sanity input shape: {tuple(x_small.shape)}")

        # Check explicit fusion layer output shape
        x_fused = model.fuse_linear(x_small)
        print(f"Linear fusion output shape: {tuple(x_fused.shape)}")

        y_hat = model(x_small)
        print(f"Model output shape: {tuple(y_hat.shape)}")

        if x_fused.shape[1] != 1:
            raise RuntimeError("Fusion layer output channel count mismatch. Expected 1.")
        if y_hat.shape[0] != x_small.shape[0]:
            raise RuntimeError("Batch size mismatch between input and output.")
        if y_hat.shape[1] != 2:
            raise RuntimeError("Output channel count mismatch. Expected 2 classes.")

        print("\nSanity check passed.")
        print("Data loading, linear fusion layer, and model forward pass are working.")


if __name__ == "__main__":
    main()