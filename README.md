# Multimodal Fusion for Knot Segmentation

This repository contains code for supervised 3D knot segmentation from wood CT using paired wet and dry scans. The main goal is to compare multimodal fusion strategies under the same training protocol and backbone settings, and to evaluate how fusion helps in challenging wet sapwood regions where knot boundaries are harder to separate.

The dataset used in this project is private and is not included in this repository. The wet and dry volumes are resampled and registered so that voxel locations match across stages.

## Models 

Single modality baselines
- Wet only nnUNet type model
- Dry only nnUNet type model

Multimodal fusion models
- Early fusion by input channel stacking
- Early fusion with learnable linear mixing
- Intermediate fusion at the bottleneck with cross attention
- Intermediate fusion at the bottleneck with concatenation and 1 x 1 x 1 mixing
- Late fusion at the logit level

## Repository structure

Top level
- `CombinedNNUnet256WetOnly/`  
  Training code for the wet only baseline.
- `CombinedNNUnet256DryOnly/`  
  Training code for the dry only baseline.
- `Fusion/`  
  All multimodal fusion experiments, shared utilities, evaluation, and prediction creation.


Inside `Fusion/`
- `common/`  
  Shared code used across fusion experiments, such as dataset pairing, transforms, and training utilities.
- `EarlyFusion/`  
  Early fusion with input channel stacking.
- `EarlyFusion2/`  
  Early fusion with learnable linear mixing before the backbone.
- `IntermediateFusion/`  
  Bottleneck cross attention intermediate fusion.
- `IntermediateFusion2/`  
  Bottleneck concat plus 1 x 1 x 1 mixing intermediate fusion.
- `LateFusion/`  
  Logit level fusion with separate wet and dry branches.
- `eval/`  
  Evaluation scripts.
- `create_pred/`  
  Scripts that generate prediction masks and save them to disk.
- `check_img_size.py`  
  Utility script for sanity checking input shapes.

## Requirements

This project uses PyTorch and MONAI for training and inference, and dynamic network architectures for the nnUNet style backbone.

Dependencies
- Python 3.9
- PyTorch
- MONAI
- dynamic network architectures
- numpy, scipy
- pynrrd and nibabel for file IO
- medpy for some evaluation scripts
- VTK for qualitative grid visualization scripts

A CUDA GPU is strongly recommended for training and for fast sliding window inference.

