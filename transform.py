"""Data transformation from hdf5 to nifti image (For monai)
"""
    
import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

root_dir = "/cluster/home/kimsa/data/BraTs2020/BraTS2020_training_data/content/data"
save_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)
brats_files = glob(os.path.join(root_dir, "*.h5"))

shape_outlier = list()
status = dict()
pbar = tqdm(total=len(brats_files))


def save_nii(nii_arr, path):
    img = nib.Nifti1Image(nii_arr, np.eye(4))
    nib.save(img, path)
    return os.path.exists(path)
    
    
if __name__ == "__main__":

    # Iterate through brats files
    for file_idx, file in enumerate(brats_files):
        h5_obj = h5py.File(file)
        image_arr = h5_obj["image"][()]
        label_arr = h5_obj["mask"][()]
        
        # Transpose original image
        image_arr = np.transpose(image_arr, (2, 0, 1))
        label_arr = np.transpose(label_arr, (2, 0, 1))
        
        file_name = os.path.basename(file).replace(".h5", ".nii.gz")
        img_save_path = os.path.join(save_dir, "image", file_name)
        lbl_save_path = os.path.join(save_dir, "label", file_name)
        
        img_saved = save_nii(image_arr, img_save_path)
        lbl_saved = save_nii(label_arr, lbl_save_path)

        status[f'{file_name}'] = [img_saved, lbl_saved]
            
        if (image_arr.shape != (4, 240, 240)) or (label_arr.shape != (3, 240, 240)):
            shape_outlier.append(dict(
                name=os.path.basename(file),
                image_shape=image_arr.shape,
                label_shape=label_arr.shape
            ))
        pbar.update(1)

    print("Shape Outlier")
    print(shape_outlier)

    status_df = pd.DataFrame.from_dict(status).T
    status_df.columns = ["Image Saved", "Label Saved"]
    status_df.to_csv("./tmp/transform.csv")
