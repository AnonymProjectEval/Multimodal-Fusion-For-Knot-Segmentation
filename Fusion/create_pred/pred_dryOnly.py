import os
from glob import glob
from pathlib import Path
import numpy as np
import torch
import nrrd
import nibabel as nib
from scipy.ndimage import label, sum as ndi_sum, zoom
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, EnsureTyped, ToTensord
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

MODEL_PATH = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/CombinedNNUnet256DryOnly/Model/best_metric_model.pth"
IMG_DIR = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/test/DryImage"
SAVE_DIR = "/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/test/DryImage/prediction_dryOnly"
ROI_SIZE = (256, 256, 256)
SW_BATCH_SIZE = 1
OVERLAP = 0.25
MIN_COMPONENT_SIZE = 800
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_nnunet_direct(in_channels=1, out_channels=2, img_size=(256, 256, 256), base_num_features=32, max_num_features=320, spatial_dims=3, n_conv_per_stage_encoder=None, n_conv_per_stage_decoder=None, num_pool_per_axis=None, pool_op_kernel_sizes=None, conv_kernel_sizes=None, dropout_p=0.0, nonlin_first=True):
    if num_pool_per_axis is None:
        min_dim = max(min(img_size), 8)
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

def load_model(path, device):
    model = build_nnunet_direct(in_channels=1, out_channels=2, img_size=ROI_SIZE).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    if len(state) and next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model

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

def list_images(folder):
    exts = ("*.nrrd", "*.nii.gz", "*.nii")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)

def build_loader(img_paths):
    data = [{"vol": p} for p in img_paths]
    tf = Compose([LoadImaged(keys=["vol"]), EnsureChannelFirstd(keys=["vol"]), NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True), EnsureTyped(keys=["vol"]), ToTensord(keys=["vol"])])
    ds = Dataset(data=data, transform=tf)
    return DataLoader(ds, batch_size=1, pin_memory=True), ds

def get_name_and_header(img_path):
    p = Path(img_path)
    stem = p.name
    ext = p.suffix.lower()
    if ext == ".gz":
        ext = ".nii.gz"
    elif ext in [".nii", ".nrrd"]:
        ext = ext
    else:
        ext = p.suffixes[-1] if p.suffixes else p.suffix
    header = None
    affine = None
    orig_shape = None
    if stem.endswith(".nii") or stem.endswith(".nii.gz"):
        nii = nib.load(str(p))
        affine = nii.affine
        header = nii.header
        orig_shape = nii.get_fdata(dtype=np.float32).shape
    else:
        vol, hdr = nrrd.read(str(p))
        header = hdr
        orig_shape = vol.shape
    return stem, ext, header, affine, orig_shape

@torch.no_grad()
def predict_volume(model, vol_t):
    logits = sliding_window_inference(inputs=vol_t, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP, mode="gaussian")
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1, keepdim=False)
    return pred[0].cpu().numpy().astype(np.uint8)

def save_pred(pred_np, original_name, ext, header, affine, orig_shape, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if pred_np.shape != orig_shape:
        factors = [o / p for o, p in zip(orig_shape, pred_np.shape)]
        pred_np = zoom(pred_np, zoom=factors, order=0).astype(np.uint8)
    out_name = f"pred_{original_name}"
    out_path = os.path.join(save_dir, out_name)
    if ext in [".nii", ".nii.gz"]:
        img = nib.Nifti1Image(pred_np, affine if affine is not None else np.eye(4), header)
        if header is not None:
            header.set_data_dtype(np.uint8)
        nib.save(img, out_path)
    else:
        nrrd.write(out_path, pred_np, header=header if header is not None else {})
    print(out_path)

if __name__ == "__main__":
    set_determinism(SEED)
    model = load_model(MODEL_PATH, DEVICE)
    img_paths = list_images(IMG_DIR)
    loader, ds = build_loader(img_paths)
    for i, batch in enumerate(loader, 1):
        img_path = ds.data[i - 1]["vol"]
        name, ext, hdr, aff, orig_shape = get_name_and_header(img_path)
        vol = batch["vol"].to(DEVICE)
        pred = predict_volume(model, vol)
        pred = remove_small_components(pred, class_index=1, min_size=MIN_COMPONENT_SIZE, replace_with=0)
        save_pred(pred, name, ext, hdr, aff, orig_shape, SAVE_DIR)
