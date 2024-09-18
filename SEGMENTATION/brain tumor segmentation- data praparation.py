# -*- coding: utf-8 -*-
"""
#Combine 
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""
import numpy as np
import nibabel as nib
import glob
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tifffile import imsave
import splitfolders

scaler = MinMaxScaler()

# Define base paths using raw strings
BASE_TRAIN_PATH = r'C:/survival prediction/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
BASE_VALIDATION_PATH = r'C:/survival prediction/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
OUTPUT_PATH = r'C:/survival prediction/BraTS2020_TrainingData/input_data'

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_PATH, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'masks'), exist_ok=True)

# Load sample images and visualize
example_t1_path = os.path.join(BASE_TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1.nii')
example_flair_path = os.path.join(BASE_TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_flair.nii')
example_t1ce_path = os.path.join(BASE_TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t1ce.nii')
example_t2_path = os.path.join(BASE_TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_t2.nii')
example_mask_path = os.path.join(BASE_TRAIN_PATH, 'BraTS20_Training_355', 'BraTS20_Training_355_seg.nii')

test_image_flair = nib.load(example_flair_path).get_fdata()
print(test_image_flair.max())

# Scalers are applied to 1D so let us reshape and then reshape back to original shape.
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

test_image_t1 = nib.load(example_t1_path).get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce = nib.load(example_t1ce_path).get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2 = nib.load(example_t2_path).get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask = nib.load(example_mask_path).get_fdata()
test_mask = test_mask.astype(np.uint8)

print(np.unique(test_mask))  # 0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3)
test_mask[test_mask == 4] = 3  # Reassign mask values 4 to 3
print(np.unique(test_mask))

n_slice = random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

# Combine t1ce, t2, and flair into single multichannel image
combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)

# Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
combined_x = combined_x[56:184, 56:184, 13:141]  # Crop to 128x128x128x4

# Do the same for mask
test_mask = test_mask[56:184, 56:184, 13:141]

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

# Save the combined image and mask
imsave(r'C:/survival prediction/BraTS2020_TrainingData/combined255.tiff', combined_x)
np.save(r'C:/survival prediction/BraTS2020_TrainingData/combined255.npy', combined_x)

# Verify image is being read properly
my_img = np.load(r'C:/survival prediction/BraTS2020_TrainingData/combined255.npy')

# Check if the arrays are equal
print(np.array_equal(combined_x, my_img))

# Process the mask
test_mask = to_categorical(test_mask, num_classes=4)

# Use os.path.join to create full paths and glob to get the sorted file lists
t2_list = sorted(glob.glob(os.path.join(BASE_TRAIN_PATH, '*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(BASE_TRAIN_PATH, '*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(BASE_TRAIN_PATH, '*/*flair.nii')))
mask_list = sorted(glob.glob(os.path.join(BASE_TRAIN_PATH, '*/*seg.nii')))

# Print lists to verify
print("T2 List:", t2_list)
print("T1CE List:", t1ce_list)
print("FLAIR List:", flair_list)
print("Mask List:", mask_list)

# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes
# Check and process each patient
for img in range(len(t2_list)):  # Assuming all lists are of the same size
    # Extract patient ID from the filename
    patient_id = os.path.basename(t2_list[img]).split('_')[2]
    print("Now preparing image and masks for patient ID:", patient_id)
    
    # Check if corresponding files exist for the patient ID
    t2_path = t2_list[img]
    t1ce_path = t1ce_list[img]
    flair_path = flair_list[img]
    mask_path = mask_list[img]

    # Load images and masks
    temp_image_t2 = nib.load(t2_path).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce = nib.load(t1ce_path).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair = nib.load(flair_path).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask = nib.load(mask_path).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3

    # Combine images into a single multichannel image
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(os.path.join(OUTPUT_PATH, 'images', f'image_{patient_id}.npy'), temp_combined_images)
        np.save(os.path.join(OUTPUT_PATH, 'masks', f'mask_{patient_id}.npy'), temp_mask)
    else:
        print("I am useless")
     
################################################################
#Repeat the same from above for validation data folder OR

import numpy as np
import nibabel as nib
import glob
from sklearn.preprocessing import MinMaxScaler
import os

scaler = MinMaxScaler()

# Define base paths using raw strings
BASE_VALIDATION_PATH = r'C:/survival prediction/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
OUTPUT_VAL_PATH = r'C:/survival prediction/BraTS2020_ValidationData/input_valdata'

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_VAL_PATH, 'images'), exist_ok=True)

# Use os.path.join to create full paths and glob to get the sorted file lists
t2_list = sorted(glob.glob(os.path.join(BASE_VALIDATION_PATH, '*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(BASE_VALIDATION_PATH, '*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(BASE_VALIDATION_PATH, '*/*flair.nii')))

# Print lists to verify
print(f"Total T2 validation images: {len(t2_list)}")
print(f"Total T1ce validation images: {len(t1ce_list)}")
print(f"Total Flair validation images: {len(flair_list)}")

# Process and save validation images
for img in range(len(t2_list)):  # Assuming all lists are of the same size
    try:
        # Extract patient ID from the filename
        patient_id = os.path.basename(t2_list[img]).split('_')[2]
        print(f"Now preparing images for patient ID: {patient_id}")

        temp_image_t2 = nib.load(t2_list[img]).get_fdata()
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

        temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

        temp_image_flair = nib.load(flair_list[img]).get_fdata()
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

        # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

        # Save the processed images
        np.save(os.path.join(OUTPUT_VAL_PATH, 'images', f'image_{patient_id}.npy'), temp_combined_images)
        print(f"Saved {os.path.join(OUTPUT_VAL_PATH, 'images', f'image_{patient_id}.npy')}")

    except Exception as e:
        print(f"Error processing patient ID: {patient_id}, Error: {e}")
        
import glob
import os

# Define paths using raw strings
IMAGE_PATH = r'C:/survival prediction/BraTS2020_TrainingData/input_data/images'
MASK_PATH = r'C:/survival prediction/BraTS2020_TrainingData/input_data/masks'

# Use os.path.join to create full paths and glob to get the sorted file lists
image_files = sorted(glob.glob(os.path.join(IMAGE_PATH, '*.npy')))
mask_files = sorted(glob.glob(os.path.join(MASK_PATH, '*.npy')))

print(f"Total saved images: {len(image_files)}")
print(f"Total saved masks: {len(mask_files)}")

# Check if all images have corresponding masks
image_ids = [os.path.basename(f).split('_')[1].split('.')[0] for f in image_files]
mask_ids = [os.path.basename(f).split('_')[1].split('.')[0] for f in mask_files]

print("Matching IDs:", image_ids == mask_ids)

#Split training data into train and validation

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""


input_folder = r'C:/survival prediction/BraTS2020_TrainingData/input_data'
output_folder = r'C:/survival prediction/BraTS2020_TrainingData/train_val'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e., `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.80, .20), group_prefix=None)  # default values
########################################

import os

# Directories for training and validation sets
train_images_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/images/'
train_masks_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/masks/'

val_images_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/images/'
val_masks_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/masks/'

# Function to extract patient IDs from filenames
def extract_patient_id(filename):
    return filename.split('_')[1].split('.')[0]

# List and sort files
train_images = sorted(os.listdir(train_images_dir))
train_masks = sorted(os.listdir(train_masks_dir))
val_images = sorted(os.listdir(val_images_dir))
val_masks = sorted(os.listdir(val_masks_dir))

# Extract patient IDs
train_image_ids = [extract_patient_id(f) for f in train_images]
train_mask_ids = [extract_patient_id(f) for f in train_masks]
val_image_ids = [extract_patient_id(f) for f in val_images]
val_mask_ids = [extract_patient_id(f) for f in val_masks]

# Compare IDs and print mismatches if any
def compare_ids(image_ids, mask_ids, set_name):
    mismatches = [(img_id, mask_id) for img_id, mask_id in zip(image_ids, mask_ids) if img_id != mask_id]
    if mismatches:
        print(f"Mismatches in {set_name} set:")
        for img_id, mask_id in mismatches:
            print(f"Image ID: {img_id}, Mask ID: {mask_id}")
    else:
        print(f"All IDs match in the {set_name} set.")

# Check training and validation sets
compare_ids(train_image_ids, train_mask_ids, 'training')
compare_ids(val_image_ids, val_mask_ids, 'validation')

# Print counts to verify
print(f"Training images: {len(train_images)}, Training masks: {len(train_masks)}")
print(f"Validation images: {len(val_images)}, Validation masks: {len(val_masks)}")
