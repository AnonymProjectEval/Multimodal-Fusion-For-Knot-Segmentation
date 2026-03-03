import nrrd

# Path to your NRRD file
file_path = r"/net/fs-2/scale/OrionStore/Projects/WaiKnotCT/FullData/train/WetImage/Green Disk_01.2.nrrd"

# Read NRRD file
data, header = nrrd.read(file_path)

# Image dimensions (number of voxels)
print("Image shape (voxels):", data.shape)

# Voxel spacing
spacing = header.get("space directions")
print("Voxel spacing (mm):", spacing)

# Calculate physical size if spacing exists
if spacing is not None:
    voxel_size = [abs(spacing[i][i]) for i in range(len(spacing))]
    physical_size = [data.shape[i] * voxel_size[i] for i in range(len(data.shape))]
    print("Physical size (mm):", physical_size)