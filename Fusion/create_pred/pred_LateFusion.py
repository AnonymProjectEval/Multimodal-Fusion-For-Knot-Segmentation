from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import nrrd
import nibabel as nib

from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm


# =========================
# Paths and settings
# =========================
DATA_ROOT = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"

MODEL_PATH = (
    "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/"
    "Multimodal/Fusion/LateFusion/Model/checkpoints/best_model.pt"
)

WET_DIR = os.path.join(DATA_ROOT, "test", "WetImage")
DRY_DIR = os.path.join(DATA_ROOT, "test", "DryImage")

FUSION_MODE = "conv1x1"  # must match training: "avg", "global_weight", or "conv1x1"

SAVE_DIR_WET = os.path.join(WET_DIR, f"prediction_LateFusion_{FUSION_MODE}")
SAVE_DIR_DRY = os.path.join(DRY_DIR, f"prediction_LateFusion_{FUSION_MODE}")

ROI_SIZE = (256, 256, 256)
SW_BATCH_SIZE = 1
OVERLAP = 0.25
SW_MODE = "gaussian"

MIN_COMPONENT_SIZE = 800
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
MODEL_BUILD_SIZE = (256, 256, 256)


# =========================
# Pairing helpers
# =========================
def canonical_key_from_stem(stem: str) -> str:
    s = stem.strip()
    m = re.search(r"(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1)
    s = s.lower().replace(" ", "").replace("_", "")
    s = s.replace("greendisk", "").replace("drydisk", "")
    return s


def list_images(folder: str) -> List[str]:
    exts = ("*.nrrd", "*.nii.gz", "*.nii")
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)


def pair_wet_dry(wet_dir: str, dry_dir: str) -> List[Dict[str, str]]:
    wet_files = list_images(wet_dir)
    dry_files = list_images(dry_dir)

    wet_map: Dict[str, str] = {}
    for p in wet_files:
        wet_map[canonical_key_from_stem(Path(p).stem)] = p

    dry_map: Dict[str, str] = {}
    for p in dry_files:
        dry_map[canonical_key_from_stem(Path(p).stem)] = p

    keys = sorted(set(wet_map.keys()) & set(dry_map.keys()))
    pairs: List[Dict[str, str]] = []
    for k in keys:
        pairs.append({"key": k, "wet": wet_map[k], "dry": dry_map[k]})

    if not pairs:
        raise RuntimeError("No paired wet and dry files found in test folders.")
    return pairs


# =========================
# IO helpers
# =========================
def read_volume_with_meta(path: str) -> Tuple[np.ndarray, str, object, Optional[np.ndarray], Tuple[int, int, int], str]:
    """
    Returns:
      vol: float32 ndarray with shape D,H,W
      ext: ".nrrd" or ".nii" or ".nii.gz"
      header: nrrd header dict or nib header
      affine: nib affine or None
      orig_shape: (D,H,W)
      filename: original filename
    """
    p = Path(path)
    name = p.name
    ext = p.suffix.lower()
    if ext == ".gz":
        ext = ".nii.gz"

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        nii = nib.load(str(p))
        vol = nii.get_fdata(dtype=np.float32)
        header = nii.header
        affine = nii.affine
        orig_shape = tuple(int(x) for x in vol.shape)
        return vol, ext, header, affine, orig_shape, name

    vol, hdr = nrrd.read(str(p))
    vol = vol.astype(np.float32, copy=False)
    header = hdr
    affine = None
    orig_shape = tuple(int(x) for x in vol.shape)
    return vol, ".nrrd", header, affine, orig_shape, name


def save_mask(mask: np.ndarray, out_path: str, ext: str, header: object, affine: Optional[np.ndarray]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_u8 = mask.astype(np.uint8, copy=False)

    if ext in [".nii", ".nii.gz"]:
        hdr = header.copy() if header is not None else None
        img = nib.Nifti1Image(mask_u8, affine if affine is not None else np.eye(4), hdr)
        if hdr is not None:
            hdr.set_data_dtype(np.uint8)
        nib.save(img, out_path)
        return

    nrrd_header = header if header is not None else {}
    nrrd.write(out_path, mask_u8, header=nrrd_header)


# =========================
# Preprocessing and postprocessing
# =========================
def normalize_intensity_nonzero_channelwise(vol_ch: np.ndarray) -> np.ndarray:
    """
    vol_ch: float32 ndarray with shape C,D,H,W
    """
    out = vol_ch.copy()
    for c in range(out.shape[0]):
        x = out[c]
        m = x != 0
        if m.any():
            mean = float(x[m].mean())
            std = float(x[m].std())
            if std > 1e-8:
                out[c] = (x - mean) / std
    return out


def match_spatial_by_crop_end(wet: np.ndarray, dry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_d = min(wet.shape[0], dry.shape[0])
    min_h = min(wet.shape[1], dry.shape[1])
    min_w = min(wet.shape[2], dry.shape[2])
    return wet[:min_d, :min_h, :min_w], dry[:min_d, :min_h, :min_w]


def pad_or_crop_to_shape(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = mask.shape
    td, th, tw = target_shape

    if (d, h, w) == (td, th, tw):
        return mask

    m2 = mask[: min(d, td), : min(h, th), : min(w, tw)]

    pd = td - m2.shape[0]
    ph = th - m2.shape[1]
    pw = tw - m2.shape[2]
    if pd >= 0 and ph >= 0 and pw >= 0:
        return np.pad(m2, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=0)

    factors = (td / mask.shape[0], th / mask.shape[1], tw / mask.shape[2])
    return zoom(mask.astype(np.uint8), zoom=factors, order=0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, class_index: int = 1, min_size: int = 800) -> np.ndarray:
    out = mask.copy()
    binary = (mask == class_index).astype(np.uint8)
    lab, n = cc_label(binary)
    if n == 0:
        return out
    counts = np.bincount(lab.ravel())
    for comp_id in range(1, n + 1):
        if counts[comp_id] < min_size:
            out[lab == comp_id] = 0
    return out


# =========================
# Model definition: LateFusion
# =========================
def build_nnunet_direct(
    in_channels: int = 1,
    out_channels: int = 2,
    img_size: Tuple[int, int, int] = (256, 256, 256),
    base_num_features: int = 32,
    max_num_features: int = 320,
    spatial_dims: int = 3,
    dropout_p: float = 0.0,
    nonlin_first: bool = True,
) -> PlainConvUNet:
    min_dim = max(min(img_size), 8)
    max_pools = int(torch.log2(torch.tensor(min_dim // 8)).item())
    max_pools = max(1, min(max_pools, 5))
    num_pool_per_axis = [max_pools] * spatial_dims
    n_stages = max(num_pool_per_axis) + 1

    n_conv_per_stage_encoder = [2] * n_stages
    n_conv_per_stage_decoder = [2] * (n_stages - 1)
    pool_op_kernel_sizes = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    conv_kernel_sizes = [[3, 3, 3]] * n_stages

    conv_op = convert_dim_to_conv_op(spatial_dims)
    norm_op = get_matching_instancenorm(conv_op)
    features_per_stage = [min(base_num_features * (2 ** i), max_num_features) for i in range(n_stages)]

    return PlainConvUNet(
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
        dropout_op=torch.nn.Dropout3d,
        dropout_op_kwargs={"p": float(dropout_p), "inplace": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        nonlin_first=nonlin_first,
    )


class LateFusionUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        img_size: Tuple[int, int, int] = (256, 256, 256),
        base_num_features: int = 32,
        max_num_features: int = 320,
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
            spatial_dims=3,
            dropout_p=dropout_p,
            nonlin_first=nonlin_first,
        )
        self.dry_branch = build_nnunet_direct(
            in_channels=1,
            out_channels=num_classes,
            img_size=img_size,
            base_num_features=base_num_features,
            max_num_features=max_num_features,
            spatial_dims=3,
            dropout_p=dropout_p,
            nonlin_first=nonlin_first,
        )

        if fusion_mode == "avg":
            self.fusion_head = None
        elif fusion_mode == "global_weight":
            self.logit_fusion_alpha = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
            self.fusion_head = None
        elif fusion_mode == "conv1x1":
            self.fusion_head = nn.Conv3d(2 * num_classes, num_classes, kernel_size=1, bias=True)
        else:
            raise ValueError("fusion_mode must be one of: avg, global_weight, conv1x1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wet = x[:, 0:1]
        dry = x[:, 1:2]

        wet_logits = self.wet_branch(wet)
        dry_logits = self.dry_branch(dry)

        if self.fusion_mode == "avg":
            fused = 0.5 * wet_logits + 0.5 * dry_logits
        elif self.fusion_mode == "global_weight":
            w = torch.softmax(self.logit_fusion_alpha, dim=0)
            fused = w[0] * wet_logits + w[1] * dry_logits
        else:
            both = torch.cat([wet_logits, dry_logits], dim=1)
            fused = self.fusion_head(both)
        return fused


def torch_load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: str, device: torch.device, fusion_mode: str) -> nn.Module:
    model = LateFusionUNet(
        num_classes=NUM_CLASSES,
        img_size=MODEL_BUILD_SIZE,
        base_num_features=32,
        max_num_features=320,
        dropout_p=0.0,
        nonlin_first=True,
        fusion_mode=fusion_mode,
    ).to(device)

    ckpt = torch_load_checkpoint(model_path, torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict) and len(state) and next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: nn.Module, vol_2ch: torch.Tensor) -> np.ndarray:
    logits = sliding_window_inference(
        inputs=vol_2ch,
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=OVERLAP,
        mode=SW_MODE,
    )
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=False)
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def main() -> None:
    set_determinism(SEED)

    os.makedirs(SAVE_DIR_WET, exist_ok=True)
    os.makedirs(SAVE_DIR_DRY, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Late fusion mode: {FUSION_MODE}")
    print(f"Wet dir: {WET_DIR}")
    print(f"Dry dir: {DRY_DIR}")
    print(f"Save wet: {SAVE_DIR_WET}")
    print(f"Save dry: {SAVE_DIR_DRY}")

    pairs = pair_wet_dry(WET_DIR, DRY_DIR)
    print(f"Found paired test samples: {len(pairs)}")

    model = load_model(MODEL_PATH, DEVICE, fusion_mode=FUSION_MODE)
    print("Model loaded.")

    for i, item in enumerate(pairs, 1):
        wet_path = item["wet"]
        dry_path = item["dry"]
        key = item["key"]

        wet_np, wet_ext, wet_hdr, wet_aff, wet_shape, wet_name = read_volume_with_meta(wet_path)
        dry_np, dry_ext, dry_hdr, dry_aff, dry_shape, dry_name = read_volume_with_meta(dry_path)

        wet_np, dry_np = match_spatial_by_crop_end(wet_np, dry_np)

        vol_ch = np.stack([wet_np, dry_np], axis=0).astype(np.float32)
        vol_ch = normalize_intensity_nonzero_channelwise(vol_ch)

        vol_t = torch.from_numpy(vol_ch).unsqueeze(0).to(DEVICE)

        pred = predict_mask(model, vol_t)
        pred = remove_small_components(pred, class_index=1, min_size=MIN_COMPONENT_SIZE)

        pred_wet = pad_or_crop_to_shape(pred, wet_shape)
        out_wet = os.path.join(SAVE_DIR_WET, f"pred_{wet_name}")
        save_mask(pred_wet, out_wet, wet_ext, wet_hdr, wet_aff)

        pred_dry = pad_or_crop_to_shape(pred, dry_shape)
        out_dry = os.path.join(SAVE_DIR_DRY, f"pred_{dry_name}")
        save_mask(pred_dry, out_dry, dry_ext, dry_hdr, dry_aff)

        print(f"{i:03d}/{len(pairs)} key {key} saved:")
        print(f"  {out_wet}")
        print(f"  {out_dry}")


if __name__ == "__main__":
    main()