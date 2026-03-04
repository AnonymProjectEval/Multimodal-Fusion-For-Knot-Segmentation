from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nrrd
import nibabel as nib

from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# =========================
# Paths and settings
# =========================
DATA_ROOT = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"

MODEL_PATH = (
    "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/"
    "Multimodal/Fusion/IntermediateFusion/Model/checkpoints/best_model.pt"
)

WET_DIR = os.path.join(DATA_ROOT, "test", "WetImage")
DRY_DIR = os.path.join(DATA_ROOT, "test", "DryImage")

SAVE_DIR_WET = os.path.join(WET_DIR, "prediction_IntermediateFusionV1")
SAVE_DIR_DRY = os.path.join(DRY_DIR, "prediction_IntermediateFusionV1")

ROI_SIZE = (256, 256, 256)
SW_BATCH_SIZE = 1
OVERLAP = 0.25
SW_MODE = "gaussian"

MIN_COMPONENT_SIZE = 800
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2

# These must match training for IntermediateFusionV1
FEATURE_CHANNELS = (16, 32, 64, 128, 256, 320)
ATTN_DIM = 128
ATTN_HEADS = 4
ATTN_DROPOUT = 0.0


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
def match_spatial_by_crop_end(wet: np.ndarray, dry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops both to min D,H,W by slicing from the end, preserving index 0 alignment.
    """
    min_d = min(wet.shape[0], dry.shape[0])
    min_h = min(wet.shape[1], dry.shape[1])
    min_w = min(wet.shape[2], dry.shape[2])
    wet_c = wet[:min_d, :min_h, :min_w]
    dry_c = dry[:min_d, :min_h, :min_w]
    return wet_c, dry_c


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


def pad_or_crop_to_shape(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    mask: D,H,W
    Pads with zeros at the end if needed, or crops at the end if mask is larger.
    Falls back to nearest neighbor zoom if needed.
    """
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
# Model definition: IntermediateFusionV1 Cross Attention
# Matches your training file
# =========================
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
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ModalityEncoder3D(nn.Module):
    def __init__(self, in_channels: int, feature_channels: Sequence[int]):
        super().__init__()
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

        return skips, x


class CrossAttentionBottleneck3D(nn.Module):
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
            raise ValueError("attn_dim must be divisible by num_heads")

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
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (d, h, w)

    @staticmethod
    def _to_map(tokens: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        d, h, w = spatial_shape
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)

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

        attn_tokens, _ = self.attn(query=q_tokens, key=k_tokens, value=v_tokens, need_weights=False)
        attn_tokens = self.out_drop(attn_tokens)

        attn_map = self._to_map(attn_tokens, spatial_shape)
        attn_map = self.out_proj(attn_map)

        return wet_feat + self.gamma * attn_map


class SharedDecoder3D(nn.Module):
    def __init__(self, feature_channels: Sequence[int], num_classes: int):
        super().__init__()
        ch = list(feature_channels)

        self.up_blocks = nn.ModuleList(
            [
                UpBlock3D(ch[-1], ch[-2], ch[-2]),
                UpBlock3D(ch[-2], ch[-3], ch[-3]),
                UpBlock3D(ch[-3], ch[-4], ch[-4]),
                UpBlock3D(ch[-4], ch[-5], ch[-5]),
                UpBlock3D(ch[-5], ch[-6], ch[-6]),
            ]
        )

        self.out_head = nn.Conv3d(ch[0], num_classes, kernel_size=1, bias=True)

    def forward(self, bottleneck: torch.Tensor, wet_skips: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(wet_skips) != 5:
            raise ValueError("Expected 5 wet skip tensors")
        x = bottleneck
        x = self.up_blocks[0](x, wet_skips[4])
        x = self.up_blocks[1](x, wet_skips[3])
        x = self.up_blocks[2](x, wet_skips[2])
        x = self.up_blocks[3](x, wet_skips[1])
        x = self.up_blocks[4](x, wet_skips[0])
        return self.out_head(x)


class IntermediateCrossAttentionUNet(nn.Module):
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

        self.decoder = SharedDecoder3D(feature_channels=self.feature_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wet = x[:, 0:1]
        dry = x[:, 1:2]

        wet_skips, wet_bottleneck = self.wet_encoder(wet)
        _, dry_bottleneck = self.dry_encoder(dry)

        fused_bottleneck = self.cross_attn(wet_bottleneck, dry_bottleneck)
        return self.decoder(fused_bottleneck, wet_skips)


# =========================
# Checkpoint loading and inference
# =========================
def torch_load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = IntermediateCrossAttentionUNet(
        num_classes=NUM_CLASSES,
        feature_channels=FEATURE_CHANNELS,
        attn_dim=ATTN_DIM,
        attn_heads=ATTN_HEADS,
        attn_dropout=ATTN_DROPOUT,
        init_gamma=0.0,
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
    print(f"Wet dir: {WET_DIR}")
    print(f"Dry dir: {DRY_DIR}")
    print(f"Save wet: {SAVE_DIR_WET}")
    print(f"Save dry: {SAVE_DIR_DRY}")

    pairs = pair_wet_dry(WET_DIR, DRY_DIR)
    print(f"Found paired test samples: {len(pairs)}")

    model = load_model(MODEL_PATH, DEVICE)
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