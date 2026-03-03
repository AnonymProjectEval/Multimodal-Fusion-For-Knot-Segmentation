import os
from glob import glob
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label, sum as ndi_sum
from medpy.metric.binary import hd95
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, EnsureTyped, ToTensord
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

model_dir = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Code/CombinedNNUnet/Model"
model_path = os.path.join(model_dir, "best_metric_model.pth")
data_root = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData"
seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
roi_size = (256, 256, 256)
sw_batch_size = 1
overlap = 0.25
voxel_spacing = (0.5, 0.5, 0.5)
min_comp_size = 400

def remove_small_components(pred_mask, class_index=1, min_size=200, replace_with=0):
    cleaned = pred_mask.copy()
    binary = (pred_mask == class_index).astype(np.uint8)
    labeled, num = label(binary)
    if num == 0:
        return cleaned
    sizes = ndi_sum(binary, labeled, index=range(1, num + 1))
    for i, s in enumerate(sizes, start=1):
        if s < min_size:
            cleaned[(labeled == i) & (pred_mask == class_index)] = replace_with
    return cleaned

def _first_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None

def pair_images_and_masks(root):
    img_dir = _first_existing([
        os.path.join(root, "test", "WetImage"),
        os.path.join(root, "test", "Images"),
        os.path.join(root, "test", "WetImageResized"),
    ])
    mask_dir = _first_existing([
        os.path.join(root, "test", "WetMask2Class"),
        os.path.join(root, "test", "BinaryMasks"),
        os.path.join(root, "test", "WetMask2ClassResized"),
    ])
    if img_dir is None or mask_dir is None:
        raise FileNotFoundError("Missing test image/mask folders")
    exts = ["*.nrrd", "*.nii.gz", "*.nii"]
    imgs = []
    for e in exts:
        imgs.extend(glob(os.path.join(img_dir, e)))
    imgs = sorted(imgs)
    masks = []
    for e in exts:
        masks.extend(glob(os.path.join(mask_dir, e)))
    masks = sorted(masks)
    by_name = {}
    for m in masks:
        b = Path(m).name
        by_name[b] = m
        if b.startswith("SEG_"):
            by_name[b[4:]] = m
    pairs = []
    for v in imgs:
        b = Path(v).name
        m = by_name.get(b) or by_name.get("SEG_" + b)
        if m is not None:
            pairs.append({"vol": v, "seg": m})
    if not pairs:
        raise RuntimeError("No image mask pairs found")
    return pairs

def build_loader(root):
    data = pair_images_and_masks(root)
    tf = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["vol", "seg"]),
        ToTensord(keys=["vol", "seg"]),
    ])
    ds = Dataset(data=data, transform=tf)
    return DataLoader(ds, batch_size=1, pin_memory=True), len(ds)

def build_nnunet_direct(in_channels=1, out_channels=2, img_size=(256, 256, 256), base_num_features=32, max_num_features=320, spatial_dims=3, n_conv_per_stage_encoder=None, n_conv_per_stage_decoder=None, num_pool_per_axis=None, pool_op_kernel_sizes=None, conv_kernel_sizes=None, dropout_p=0.0, nonlin_first=True):
    if num_pool_per_axis is None:
        min_dim = min(img_size)
        min_dim = max(min_dim, 8)
        max_pools = int(torch.log2(torch.tensor(min_dim // 8)).item())
        max_pools = max(1, min(max_pools, 5))
        num_pool_per_axis = [max_pools] * spatial_dims
    n_stages = max(num_pool_per_axis) + 1
    if n_conv_per_stage_encoder is None:
        n_conv_per_stage_encoder = [2] * n_stages
    if n_conv_per_stage_decoder is None:
        n_conv_per_stage_decoder = [2] * (n_stages - 1)
    if pool_op_kernel_sizes is None:
        pool_op_kernel_sizes = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    if conv_kernel_sizes is None:
        conv_kernel_sizes = [[3, 3, 3]] * n_stages
    conv_op = convert_dim_to_conv_op(spatial_dims)
    norm_op = get_matching_instancenorm(conv_op)
    features_per_stage = [min(base_num_features * (2 ** i), max_num_features) for i in range(n_stages)]
    net = PlainConvUNet(
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
        dropout_op=torch.nn.Dropout3d if spatial_dims == 3 else torch.nn.Dropout2d,
        dropout_op_kwargs={"p": float(dropout_p), "inplace": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        nonlin_first=nonlin_first,
    )
    return net

def load_model(path, dev):
    model = build_nnunet_direct(in_channels=1, out_channels=2, img_size=(256, 256, 256), base_num_features=32, max_num_features=320, spatial_dims=3, dropout_p=0.0, nonlin_first=True).to(dev)
    state = torch.load(path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if len(state) and next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model

def dice_foreground(pred_np, gt_np):
    p = pred_np.astype(bool)
    g = gt_np.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    denom = p.sum(dtype=np.float64) + g.sum(dtype=np.float64)
    if denom == 0:
        return 0.0
    return float(2.0 * inter / denom)

@torch.no_grad()
def infer_logits(model, vol_t):
    return sliding_window_inference(inputs=vol_t, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap, mode="gaussian")

if __name__ == "__main__":
    set_determinism(seed)
    test_loader, n_cases = build_loader(data_root)
    model = load_model(model_path, device)
    dice_list = []
    hd95_list = []
    names = []
    for i, batch in enumerate(test_loader, 1):
        vol = batch["vol"].to(device)
        seg = batch["seg"].to(device).long()
        logits = infer_logits(model, vol)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
        p_np = pred[0, 0].cpu().numpy()
        g_np = seg[0, 0].cpu().numpy()
        p_np = remove_small_components(p_np, class_index=1, min_size=min_comp_size, replace_with=0)
        d = dice_foreground(p_np == 1, g_np == 1)
        dice_list.append(d)
        if (p_np == 1).any() and (g_np == 1).any():
            try:
                h = float(hd95(p_np == 1, g_np == 1, voxelspacing=voxel_spacing, connectivity=1))
            except Exception:
                h = float("nan")
        else:
            h = float("nan")
        hd95_list.append(h)
        try:
            name = Path(batch["vol_meta_dict"]["filename_or_obj"][0]).name
        except Exception:
            name = f"case_{i:03d}"
        names.append(name)
        if np.isfinite(h):
            print(f"{i:03d}/{n_cases} {name} | Dice: {d:.4f} | HD95(mm): {h:.2f}")
        else:
            print(f"{i:03d}/{n_cases} {name} | Dice: {d:.4f} | HD95(mm): NaN")
    dice_arr = np.asarray(dice_list, dtype=np.float64)
    hd95_arr = np.asarray(hd95_list, dtype=np.float64)
    dice_mean = float(np.nanmean(dice_arr)) if dice_arr.size else float("nan")
    dice_std = float(np.nanstd(dice_arr)) if dice_arr.size else float("nan")
    hd95_mean = float(np.nanmean(hd95_arr)) if np.isfinite(hd95_arr).any() else float("nan")
    hd95_std = float(np.nanstd(hd95_arr)) if np.isfinite(hd95_arr).any() else float("nan")
    print("===== Final =====")
    print(f"Avg Dice: {dice_mean:.4f} + {dice_std:.4f}")
    print(f"Avg HD95 (mm): {hd95_mean:.2f} + {hd95_std:.2f}")
