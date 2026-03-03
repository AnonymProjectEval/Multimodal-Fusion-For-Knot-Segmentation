#!/bin/bash
#SBATCH --job-name=IntermFusion  
#SBATCH --output=unet_train_%j.log  
#SBATCH --error=unet_train_%j.err  
#SBATCH --partition=gpu             
#SBATCH --gres=gpu:1                
#SBATCH --mem=250G                  
#SBATCH --time=288:00:00             
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=8           

# Proper Conda activation
eval "$(/mnt/users/mohahoss/miniconda3/bin/conda shell.bash hook)"
conda activate fva2

# Navigate to your code directory
cd /net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/Multimodal/Fusion/IntermediateFusion
# Run training script
python train_unet.py
