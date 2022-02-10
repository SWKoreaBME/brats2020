import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"

import torch
import torch.nn as nn
# import numpy as np
# import pandas as pd
# import nibabel as nib
import matplotlib.pyplot as plt

from glob import glob
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from glob import glob
# from copy import copy
from shutil import rmtree
from torch.utils import data

import monai.transforms as tf
# from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, UNETR, DynUNet
from monai.networks.layers import Norm
# from monai.networks import one_hot
from utils.Utils import concat_bg
from utils.Metric import compute_meandice

from utils.Device import *
from utils.Data import load_dataloader
from utils.Loss import *


# train transforms
train_transforms = tf.Compose([
    tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
    tf.AsDiscreted(keys=['label'], threshold_values=True),
    # tf.ToNumpyd(keys=['image', 'label']),
    # tf.NormalizeIntensityd(keys=['image'], channel_wise=True, nonzero=True),
    tf.ToTensord(keys=['image', 'label']),
#         tf.DeleteItemsd(keys=['image_transforms', 'label_transforms'])
])


# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
    tf.AsDiscreted(keys=['label'], threshold_values=True),
    # tf.ToNumpyd(keys=['image', 'label']),
    # tf.NormalizeIntensityd(keys=['image'], channel_wise=True, nonzero=True),
    tf.ToTensord(keys=['image', 'label'])
])

# root_dir = "/cluster/home/kimsa/data/BraTs2020/BraTS2020_training_data/content/data_monai"
root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"

loader_params = dict(
    batch_size=64,
    shuffle=True
)

test_loader_params = dict(
    batch_size=64,
    shuffle=False
)

train_dataloader = load_dataloader(root_dir, "train", train_transforms, loader_params)
valid_dataloader = load_dataloader(root_dir, "valid", val_transforms, test_loader_params)
test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)

# Logging
ckpt_save_dir = "./result/exps/unetr-noaug"
if os.path.exists(ckpt_save_dir):
    rmtree(ckpt_save_dir)
os.makedirs(ckpt_save_dir, exist_ok=True)

img_save_dir = os.path.join(ckpt_save_dir, "figures")
os.makedirs(img_save_dir, exist_ok=True)

images_seqs = [f"{idx+1}" for idx in range(4)]

# Model
amp = True
device, multi_gpu = gpu_setting()

model = UNETR(in_channels=4,
              out_channels=3+1,
              img_size=240,
              feature_size=8,
              dropout_rate=0.3,
              norm_name='batch',
              spatial_dims=2).to(device)
model = model_dataparallel(model, multi_gpu)


# Loss function
lambda_ce, lambda_dice = 0.7, 0.3

# Change loss function
loss_function = seg_loss_fn_3d

softmax = nn.Softmax(dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler() if amp else None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=10)
num_epochs = 300
val_freq = 2

train_losses = list()
val_losses = list()

best_loss = 1e+2
print(f"Train Start")
for epoch in range(num_epochs):
    
    current_lr = optimizer.param_groups[0]['lr']
    
    batch_loss = dict(
        train=0,
        val=0
    )
    batch_dice = dict(
        train=0,
        val=0
    )
    total_num_imgs = dict(
        train=0,
        val=0
    )
    
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, labels = batch["image"], concat_bg(batch["label"])
        inputs, labels = inputs.to(device), labels.to(device)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels, lambda_ce, lambda_dice, False, False)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if torch.isnan(loss).any():
                print(f"NAN in outputs", torch.isnan(outputs).any())
                print(f"NAN in inputs", torch.isnan(inputs).any())
                print(f"NAN in labels", torch.isnan(labels).any())
                pass
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
        batch_dice['train'] += compute_meandice(outputs, labels, include_background=False) * inputs.size(0)
        batch_loss['train'] += float(loss.data) * inputs.size(0)
        total_num_imgs['train'] += inputs.size(0)

        
    # validation
    model.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            inputs, labels = batch["image"], concat_bg(batch["label"])
            inputs, labels = inputs.to(device), labels.to(device)

            if amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels, lambda_ce, lambda_dice, False, False)
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            batch_dice['val'] += compute_meandice(outputs, labels, include_background=False) * inputs.size(0)
            if torch.isnan(batch_dice['val']).any():
                print(batch_dice['val'])
                pass
            batch_loss['val'] += float(loss.data) * inputs.size(0)
            total_num_imgs['val'] += inputs.size(0)
            
    
    batch_loss['train'] = batch_loss['train'] / total_num_imgs['train']
    batch_dice['train'] = batch_dice['train'] / total_num_imgs['train']
    
    batch_loss['val'] = batch_loss['val'] / total_num_imgs['val']
    batch_dice['val'] = batch_dice['val'] / total_num_imgs['val']
    
    train_losses.append(batch_loss['train'])
    val_losses.append(batch_loss['val'])
    scheduler.step(batch_loss['val'])
    
    # print if scheduler changes current lr
    if optimizer.param_groups[0]['lr'] != current_lr:
        print(f"Learning Rate Changed {current_lr:.4f} --> {optimizer.param_groups[0]['lr']:.4f}")
        
    torch.save(model.state_dict(), os.path.join(ckpt_save_dir, "checkpoint.pth"))
    if best_loss >= batch_loss['val']:
        best_loss = batch_loss['val']
        torch.save(model.state_dict(), os.path.join(ckpt_save_dir, "best.pth"))

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train loss: {batch_loss['train']:.4f} Val loss: {batch_loss['val']:.4f} Val Dice: {batch_dice['val']:.4f} (best loss: {best_loss:.4f}) lr: {optimizer.param_groups[0]['lr']:.4f}")

    # Visualization
    output_image = softmax(outputs).detach().cpu().numpy()
    output_image_tr = output_image.argmax(1)

    loaded_image = inputs.detach().cpu().numpy()
    loaded_label = labels.detach().cpu().numpy().argmax(1)
    
    for img_idx in range(inputs.size(0)):
        if img_idx > 4:
            break
        fig, axes = plt.subplots(2, inputs.size(1), figsize=(20, 10))
        plt.suptitle(f"Epoch {epoch+1}", y=0.95)
        for img_seq_idx in range(inputs.size(1)):
            loaded_seq_image = loaded_image[img_idx][img_seq_idx].squeeze()
            axes[0, img_seq_idx].imshow(loaded_seq_image, cmap='gray',
                                        # vmin=loaded_seq_image.mean() - loaded_seq_image.std() * 1.96,
                                        vmax=loaded_seq_image.mean() + loaded_seq_image.std() * 1.96)
            axes[0, img_seq_idx].set_title(f"Input ({images_seqs[img_seq_idx]})")
        axes[1, 0].imshow(loaded_label[img_idx].squeeze(), cmap='gray')
        axes[1, 0].set_title("Label")
        axes[1, 1].imshow(output_image_tr[img_idx].squeeze(), cmap='gray')
        axes[1, 1].set_title("Output")
        fig.delaxes(axes[1, 2])
        fig.delaxes(axes[1, 3])
        plt.savefig(os.path.join(img_save_dir, f"epoch-{epoch+1}_loss_{batch_loss['val']:.4f}.jpg"), bbox_inches='tight', dpi=50)
        plt.show()
        plt.close()
    
    # break
    pass

    
# Test
test_dice = 0
total_num_imgs = 0

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch["image"], concat_bg(batch["label"])
        inputs, labels = inputs.to(device), labels.to(device)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        test_dice += compute_meandice(outputs, labels, include_background=False) * inputs.size(0)
        total_num_imgs += inputs.size(0)
        
        # Visualization
        output_image = softmax(outputs).detach().cpu().numpy()
        output_image_tr = output_image.argmax(1)

        loaded_image = inputs.detach().cpu().numpy()
        loaded_label = labels.detach().cpu().numpy().argmax(1)
        
    for img_idx in range(inputs.size(0)):
        if img_idx > 16:
            break
        fig, axes = plt.subplots(2, inputs.size(1), figsize=(20, 10))
        plt.suptitle(f"Test {batch_idx+1}", y=0.95)
        for img_seq_idx in range(inputs.size(1)):
            loaded_seq_image = loaded_image[img_idx][img_seq_idx].squeeze()
            axes[0, img_seq_idx].imshow(loaded_seq_image, cmap='gray',
                                        # vmin=loaded_seq_image.mean() - loaded_seq_image.std() * 1.96,
                                        vmax=loaded_seq_image.mean() + loaded_seq_image.std() * 1.96)
            axes[0, img_seq_idx].set_title(f"Input ({images_seqs[img_seq_idx]})")
        axes[1, 0].imshow(loaded_label[img_idx].squeeze(), cmap='gray')
        axes[1, 0].set_title("Label")
        axes[1, 1].imshow(output_image_tr[img_idx].squeeze(), cmap='gray')
        axes[1, 1].set_title("Output")
        fig.delaxes(axes[1, 2])
        fig.delaxes(axes[1, 3])
        plt.savefig(os.path.join(img_save_dir, f"test-{batch_idx+1}.jpg"), bbox_inches='tight', dpi=50)
        plt.show()
        plt.close()

        
test_dice /= total_num_imgs
print(f"Test Dice: {test_dice:.4f}")

# rmtree(ckpt_save_dir)
