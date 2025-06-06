import nibabel as nib
import numpy as np

# Load the NIfTI file
nii_img = nib.load(" example_data/01_MRA1_seg.nii.gz")
image_data = nii_img.get_fdata()

print("Image data shape:", image_data.shape)
print("Image data type:", image_data.dtype)

# Check unique values
unique_values = np.unique(image_data)
print("Unique values:", unique_values)
print("Unique value types:", [type(val) for val in unique_values])

# Check exact values
for val in unique_values:
    count = np.sum(image_data == val)
    print(f"Value {val} (type: {type(val)}): {count} voxels")

# Check the non-zero values more carefully
non_zero_mask = image_data != 0
print(f"Non-zero voxels: {np.sum(non_zero_mask)}")

if np.sum(non_zero_mask) > 0:
    non_zero_values = image_data[non_zero_mask]
    print(f"Non-zero value range: {np.min(non_zero_values)} to {np.max(non_zero_values)}")
    print(f"Non-zero unique values: {np.unique(non_zero_values)}")

# Test different masking approaches
label_1_exact = np.sum(image_data == 1.0)
label_1_close = np.sum(np.abs(image_data - 1.0) < 1e-6)
label_1_nonzero = np.sum(image_data > 0.5)

print(f"\nMasking test:")
print(f"Exact == 1.0: {label_1_exact} voxels")
print(f"Close to 1.0 (1e-6): {label_1_close} voxels") 
print(f"Greater than 0.5: {label_1_nonzero} voxels") 