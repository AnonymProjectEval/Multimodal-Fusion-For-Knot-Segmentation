from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise ImportError(
        "This script needs SciPy for connected component blob removal. "
        "Install it with pip install scipy."
    ) from e


# -----------------------
# Paths
# -----------------------
DATA_ROOT = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"

# Update this to your IntermediateFusion Approach1 checkpoint
MODEL_PATH = (
    "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/"
    "Multimodal/Fusion/IntermediateFusion/Model/checkpoints/best_model.pt"
)

CSV_PATH = str(Path(MODEL_PATH).resolve().parent / "test_metrics_dice_hd95_blob800_intermediateA1.csv")

# -----------------------
# Model config
# Must match training
# -----------------------
FEATURE_CHANNELS = (16, 32, 64, 128, 256, 320)
ATTN_DIM = 128
ATTN_HEADS = 4
ATTN_DROPOUT = 0.0
INIT_GAMMA = 0.0

# -----------------------
# Data and inference config
# -----------------------
SPATIAL_SIZE = (256, 256, 256)
MASK_PREFIX = "SEG_"

ROI_SIZE = (256, 256, 256)
SW_BATCH_SIZE = 1
OVERLAP = 0.25

CACHE_TEST = False

# Blob removal
MIN_BLOB_SIZE = 800
CC_CONNECTIVITY = 2  # 2 means 26 connected in 3D, 1 means 6 connected


# -----------------------
# Imports from your repo
# Put this script in Fusion or Fusion/eval so parents[1] is Fusion
# -----------------------
FUSION_ROOT = Path(__file__).resolve().parents[1]
if str(FUSION_ROOT) not in sys.path:
    sys.path.append(str(FUSION_ROOT))

from common.prepare_shared import collect_paired_samples, build_transforms
from IntermediateFusion.train_unet import IntermediateCrossAttentionUNet


def _strip_prefix(text: str, prefix: str) -> str:
    if prefix and text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _canonical_pair_key(stem: str) -> str:
    s = stem.strip()
    m = re.search(r"(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1)
    s = s.lower().replace("_", "").replace(" ", "")
    s = s.replace("greendisk", "").replace("drydisk", "")
    return s


def _ensure_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return str(x[0])
    return str(x)


def safe_mean(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return ckpt

    model.load_state_dict(ckpt, strict=True)
    return {}


def remove_small_blobs_3d(binary_mask: torch.Tensor, min_size: int = 800, connectivity: int = 2) -> torch.Tensor:
    if binary_mask.dim() == 3:
        mask_np = binary_mask.detach().cpu().numpy().astype(np.uint8)
        structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
        labeled, num = ndi.label(mask_np, structure=structure)

        if num == 0:
            cleaned = mask_np
        else:
            counts = np.bincount(labeled.ravel())
            keep = counts >= int(min_size)
            keep[0] = False
            cleaned = keep[labeled].astype(np.uint8)

        return torch.from_numpy(cleaned).to(device=binary_mask.device, dtype=binary_mask.dtype)

    if binary_mask.dim() == 4:
        out = []
        for b in range(binary_mask.shape[0]):
            out.append(remove_small_blobs_3d(binary_mask[b], min_size=min_size, connectivity=connectivity))
        return torch.stack(out, dim=0)

    raise ValueError(f"Expected 3D or 4D mask, got shape {tuple(binary_mask.shape)}")


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_first_item(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _spacing_from_affine(affine_any: Any) -> Optional[List[float]]:
    aff = _to_numpy(affine_any)
    if aff is None:
        return None

    aff = _maybe_first_item(aff)

    if aff.ndim != 2 or aff.shape[0] < 3 or aff.shape[1] < 3:
        return None

    col0 = aff[:3, 0]
    col1 = aff[:3, 1]
    col2 = aff[:3, 2]

    sp_xyz = [
        float(np.sqrt(np.sum(col0 * col0))),
        float(np.sqrt(np.sum(col1 * col1))),
        float(np.sqrt(np.sum(col2 * col2))),
    ]
    return sp_xyz


def _extract_spacing_xyz(meta: Optional[dict]) -> Optional[List[float]]:
    if not isinstance(meta, dict):
        return None

    if "spacing" in meta and meta["spacing"] is not None:
        sp = _to_numpy(meta["spacing"])
        if sp is not None:
            sp = _maybe_first_item(sp)
            sp = np.asarray(sp).reshape(-1)
            if sp.size >= 3:
                return [float(sp[0]), float(sp[1]), float(sp[2])]

    if "pixdim" in meta and meta["pixdim"] is not None:
        sp = _to_numpy(meta["pixdim"])
        if sp is not None:
            sp = _maybe_first_item(sp)
            sp = np.asarray(sp).reshape(-1)
            if sp.size >= 4:
                return [float(sp[1]), float(sp[2]), float(sp[3])]

    aff = meta.get("affine", None)
    if aff is None:
        aff = meta.get("original_affine", None)

    if aff is not None:
        sp_xyz = _spacing_from_affine(aff)
        if sp_xyz is not None:
            return sp_xyz

    return None


def get_spacing_dhw(batch: Dict[str, Any]) -> Optional[List[float]]:
    for md_key in ("seg_meta_dict", "vol_meta_dict", "wet_meta_dict", "dry_meta_dict"):
        md = batch.get(md_key)
        sp_xyz = _extract_spacing_xyz(md if isinstance(md, dict) else None)
        if sp_xyz is not None:
            return [sp_xyz[2], sp_xyz[1], sp_xyz[0]]

    for tensor_key in ("seg", "vol"):
        x = batch.get(tensor_key, None)
        meta = getattr(x, "meta", None)
        sp_xyz = _extract_spacing_xyz(meta if isinstance(meta, dict) else None)
        if sp_xyz is not None:
            return [sp_xyz[2], sp_xyz[1], sp_xyz[0]]

    return None


def build_test_loader_only(
    data_root: str,
    spatial_size: Tuple[int, int, int],
    mask_prefix: str,
    cache: bool,
) -> DataLoader:
    test_samples = collect_paired_samples(
        wet_img_dir=os.path.join(data_root, "test", "WetImage"),
        dry_img_dir=os.path.join(data_root, "test", "DryImage"),
        mask_dir=os.path.join(data_root, "test", "WetMask2Class"),
        mask_prefix=mask_prefix,
    )

    for s in test_samples:
        s["wet_path"] = s["wet"]
        s["dry_path"] = s["dry"]
        s["seg_path"] = s["seg"]
        seg_stem = Path(s["seg"]).stem
        seg_stem = _strip_prefix(seg_stem, mask_prefix)
        s["case_id"] = _canonical_pair_key(seg_stem)

    eval_tf = build_transforms(
        spatial_size=spatial_size,
        image_keys=("wet", "dry"),
        seg_key="seg",
        concat_to="vol",
        crop_image_key="dry",
        train=False,
    )

    Ds = CacheDataset if cache else Dataset
    ds_kwargs = {"cache_rate": 1.0} if cache else {}
    test_ds = Ds(data=test_samples, transform=eval_tf, **ds_kwargs)

    print(f"Test samples: {len(test_ds)}")

    return DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )


def main() -> None:
    set_determinism(seed=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader = build_test_loader_only(
        data_root=DATA_ROOT,
        spatial_size=SPATIAL_SIZE,
        mask_prefix=MASK_PREFIX,
        cache=CACHE_TEST,
    )

    model = IntermediateCrossAttentionUNet(
        num_classes=2,
        feature_channels=FEATURE_CHANNELS,
        attn_dim=ATTN_DIM,
        attn_heads=ATTN_HEADS,
        attn_dropout=ATTN_DROPOUT,
        init_gamma=INIT_GAMMA,
    ).to(device)
    model.eval()

    ckpt_info = load_checkpoint_into_model(model, MODEL_PATH, device)
    if "epoch" in ckpt_info:
        print(f"Loaded checkpoint from epoch: {ckpt_info['epoch']}")
    if "best_metric" in ckpt_info:
        print(f"Checkpoint best val metric stored: {ckpt_info['best_metric']}")

    if hasattr(model, "get_fusion_info"):
        print("Fusion info:", model.get_fusion_info())

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    rows: List[Dict[str, Any]] = []
    dice_all: List[float] = []
    hd95_all: List[float] = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader, start=1):
            vol = batch["vol"].to(device)
            seg = batch["seg"].to(device)

            case_id = _ensure_str(batch.get("case_id"))
            wet_path = _ensure_str(batch.get("wet_path"))
            dry_path = _ensure_str(batch.get("dry_path"))
            seg_path = _ensure_str(batch.get("seg_path"))

            spacing_dhw = get_spacing_dhw(batch)

            logits = sliding_window_inference(
                inputs=vol,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=OVERLAP,
            )

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            label_list = [post_label(y) for y in decollate_batch(seg)]

            pred_b = torch.stack(pred_list, dim=0)
            lab_b = torch.stack(label_list, dim=0)

            pred_fg_vox_before = int(pred_b[:, 1].sum().item())
            label_fg_vox = int(lab_b[:, 1].sum().item())

            fg = pred_b[:, 1]
            fg_clean = remove_small_blobs_3d(fg, min_size=MIN_BLOB_SIZE, connectivity=CC_CONNECTIVITY)

            pred_b_clean = pred_b.clone()
            pred_b_clean[:, 1] = fg_clean
            pred_b_clean[:, 0] = 1.0 - fg_clean

            pred_fg_vox_after = int(pred_b_clean[:, 1].sum().item())

            dice_metric.reset()
            dice_vals = dice_metric(y_pred=pred_b_clean, y=lab_b)
            dice_case = float(np.nanmean(dice_vals.detach().cpu().numpy()))
            dice_all.append(dice_case)

            hd95_metric.reset()
            if spacing_dhw is None:
                hd_vals = hd95_metric(y_pred=pred_b_clean, y=lab_b)
                hd_unit = "voxel"
            else:
                hd_vals = hd95_metric(y_pred=pred_b_clean, y=lab_b, spacing=spacing_dhw)
                hd_unit = "physical"

            hd95_case = float(np.nanmean(hd_vals.detach().cpu().numpy()))
            hd95_all.append(hd95_case)

            rows.append(
                {
                    "case_id": case_id,
                    "wet_path": wet_path,
                    "dry_path": dry_path,
                    "seg_path": seg_path,
                    "dice_fg": dice_case,
                    "hd95_fg": hd95_case,
                    "hd95_unit": hd_unit,
                    "spacing_d": spacing_dhw[0] if spacing_dhw else "",
                    "spacing_h": spacing_dhw[1] if spacing_dhw else "",
                    "spacing_w": spacing_dhw[2] if spacing_dhw else "",
                    "pred_fg_voxels_before": pred_fg_vox_before,
                    "pred_fg_voxels_after": pred_fg_vox_after,
                    "label_fg_voxels": label_fg_vox,
                    "min_blob_size": MIN_BLOB_SIZE,
                    "attn_dim": ATTN_DIM,
                    "attn_heads": ATTN_HEADS,
                    "init_gamma": INIT_GAMMA,
                }
            )

            spacing_str = "None" if spacing_dhw is None else f"{spacing_dhw}"
            print(
                f"[{idx:03d}] case_id={case_id}  Dice={dice_case:.4f}  "
                f"HD95={hd95_case:.4f}  unit={hd_unit}  spacing_dhw={spacing_str}  "
                f"pred_fg_before={pred_fg_vox_before}  pred_fg_after={pred_fg_vox_after}  "
                f"label_fg={label_fg_vox}"
            )

    out_path = Path(CSV_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "wet_path",
        "dry_path",
        "seg_path",
        "dice_fg",
        "hd95_fg",
        "hd95_unit",
        "spacing_d",
        "spacing_h",
        "spacing_w",
        "pred_fg_voxels_before",
        "pred_fg_voxels_after",
        "label_fg_voxels",
        "min_blob_size",
        "attn_dim",
        "attn_heads",
        "init_gamma",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dice_mean = safe_mean(dice_all)
    dice_sd = safe_std(dice_all)
    hd_mean = safe_mean(hd95_all)
    hd_sd = safe_std(hd95_all)

    print("\n===== Test summary =====")
    print(f"Cases: {len(rows)}")
    print(f"Dice mean: {dice_mean:.4f}  Dice std: {dice_sd:.4f}")
    print(f"HD95 mean: {hd_mean:.4f}  HD95 std: {hd_sd:.4f}")
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()