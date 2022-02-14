import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import pandas as pd
# import nibabel as nib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from copy import copy
from shutil import rmtree
from tqdm import tqdm
from torch.utils import data

import monai.transforms as tf
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, UNETR, DynUNet
from monai.networks.layers import Norm
# from monai.networks import one_hot
from utils.Utils import concat_bg
from utils.Metric import compute_meandice, compute_meandice_multilabel

from utils.Device import *
from utils.Data import load_dataloader, convert_label_to_brats
from utils.Loss import *
from utils.Gradient import plot_grad_flow
from utils.Visualize import plot_whole_imgs

from datetime import datetime

debug = False
device, multi_gpu = gpu_setting()


# Get Environment variables
batch_size_train = int(os.environ["BATCH_SIZE_TRAIN"]) if not debug else 64
batch_size_test = int(os.environ["BATCH_SIZE_TEST"]) if not debug else 64
root_dir = os.environ["ROOT_DIR"] if not debug else "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
ckpt_save_dir = os.environ["CKPT_SAVE_DIR"] if not debug else "./result/exps/unet-merge-noaug-gradient-check"
num_epochs = int(os.environ["NUM_EPOCHS"]) if not debug else 5

# Model configuration
in_channels = int(os.environ["IN_CHANNELS"]) if not debug else 4
out_channels = int(os.environ["OUT_CHANNELS"]) if not debug else 4
dropout_rate = float(os.environ["DROPOUT_RATE"]) if not debug else 0.3
norm_name = os.environ["NORM_NAME"] if not debug else "instance"
spatial_dims = int(os.environ["SPATIAL_DIMS"]) if not debug else 2

lambda_ce = float(os.environ["LAMBDA_CE"]) if not debug else 0.5
lambda_dice = float(os.environ["LAMBDA_DICE"]) if not debug else 0.5
lr = float(os.environ["LR"]) if not debug else 0.01


# Tensorboard logging
if not debug:
    datetime_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/{os.path.basename(ckpt_save_dir)}/{datetime_now}")


# train transforms
train_transforms = tf.Compose([
    tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
    # tf.EnsureChannelFirstd(keys="image"),
    # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    tf.EnsureTyped(keys=["image", "label"]),
    tf.ToTensord(keys=['image', 'label']),
    tf.ToDeviced(keys=["image", "label"], device=device)
])


# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
    # tf.EnsureChannelFirstd(keys="image"),
    # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    tf.EnsureTyped(keys=["image", "label"]),
    tf.ToTensord(keys=['image', 'label']),
    tf.ToDeviced(keys=["image", "label"], device=device)
])


loader_params = dict(
    batch_size=batch_size_train,
    shuffle=True
)

test_loader_params = dict(
    batch_size=batch_size_test,
    shuffle=False
)

train_dataloader = load_dataloader(root_dir, "train", train_transforms, loader_params)
valid_dataloader = load_dataloader(root_dir, "valid", val_transforms, test_loader_params)
test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)


# Logging
if os.path.exists(ckpt_save_dir):
    rmtree(ckpt_save_dir)
os.makedirs(ckpt_save_dir, exist_ok=True)

img_save_dir = os.path.join(ckpt_save_dir, "figures")
os.makedirs(img_save_dir, exist_ok=True)

grad_save_dir = os.path.join(ckpt_save_dir, "gradients")
os.makedirs(grad_save_dir, exist_ok=True)

images_seqs = ["T1", "T1Gd", "T2", "FLAIR"]


# Model
amp = True
model = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    act="RELU",
    norm=norm_name,
    dropout=dropout_rate,
    bias=True
).to(device)
# num_params = count_parameters(model)  # count number of parameters
model = model_dataparallel(model, multi_gpu)

# Loss function
# Change loss function
loss_function = DiceCELoss(include_background=False,
                           smooth_nr=1e-5, 
                           smooth_dr=1e-5,
                           squared_pred=True,
                           to_onehot_y=False,
                           lambda_ce=lambda_ce,
                           lambda_dice=lambda_dice,
                           sigmoid=True)

# softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler() if amp else None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=30)
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
        # inputs, labels = batch["image"], batch["label"]
        # labels = one_hot(labels.argmax(1).unsqueeze(1), num_classes=4)  # make one-hot
        inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            # plot_grad_flow(model.named_parameters(),
            #                os.path.join(grad_save_dir, f"gradient-e{str(epoch+1).zfill(4)}-b{str(batch_idx+1).zfill(4)}.jpg"))  # plot gradient flow
            scaler.step(optimizer)
            scaler.update()
            if torch.isnan(loss).any():
                print(f"NAN in outputs", torch.isnan(outputs).any())
                print(f"NAN in inputs", torch.isnan(inputs).any())
                print(f"NAN in labels", torch.isnan(labels).any())
                break
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
        batch_dice['train'] += compute_meandice_multilabel(outputs, labels, include_background=False) * inputs.size(0)
        batch_loss['train'] += float(loss.data) * inputs.size(0)
        total_num_imgs['train'] += inputs.size(0)
        
        # break
        if debug and (batch_idx == 10):
            break

        
    # validation
    model.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            # inputs, labels = batch["image"], batch["label"]
            # labels = one_hot(labels.argmax(1).unsqueeze(1), num_classes=4)  # make one-hot
            inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods

            if amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            batch_dice['val'] += compute_meandice_multilabel(outputs, labels, include_background=False) * inputs.size(0)
            if torch.isnan(batch_dice['val']).any():
                print(batch_dice['val'])
                pass
            batch_loss['val'] += float(loss.data) * inputs.size(0)
            total_num_imgs['val'] += inputs.size(0)
            
            if debug:
                break
            
    batch_loss['train'] = batch_loss['train'] / total_num_imgs['train']
    batch_dice['train'] = batch_dice['train'] / total_num_imgs['train']
    
    batch_loss['val'] = batch_loss['val'] / total_num_imgs['val']
    batch_dice['val'] = batch_dice['val'] / total_num_imgs['val']
    
    train_losses.append(batch_loss['train'])
    val_losses.append(batch_loss['val'])
    scheduler.step(batch_loss['val'])
    
    if not debug:
        # Tensorboard logging
        for loss_name, loss_value in batch_loss.items():
            writer.add_scalar(f'loss/{loss_name}', loss_value, global_step=epoch+1)
        for score_name, score_value in batch_dice.items():
            writer.add_scalar(f'dice/{score_name}', score_value, global_step=epoch+1)
        
    
    # print if scheduler changes current lr
    if optimizer.param_groups[0]['lr'] != current_lr:
        print(f"Learning Rate Changed {current_lr:.4f} --> {optimizer.param_groups[0]['lr']:.4f}")
        
        
    # Model save
    torch.save(model.state_dict(), os.path.join(ckpt_save_dir, "checkpoint.pth"))
    if best_loss >= batch_loss['val']:
        best_loss = batch_loss['val']
        torch.save(model.state_dict(), os.path.join(ckpt_save_dir, "best.pth"))

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train loss: {batch_loss['train']:.4f} Val loss: {batch_loss['val']:.4f} Val Dice: {batch_dice['val']:.4f} (best loss: {best_loss:.4f}) lr: {optimizer.param_groups[0]['lr']:.4f}")


    # Visualization
    output_image = torch.where(sigmoid(outputs).detach().cpu() > 0.5, 1, 0).numpy()
    output_image_tr = output_image[:, 1:].sum(1)

    loaded_image = inputs.detach().cpu().numpy()
    loaded_label = labels.detach().cpu().numpy()[:, 1:].sum(1)
    
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
    
    if debug:
        break

    
# Test
test_dice = 0
total_num_imgs = 0

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        test_dice += compute_meandice_multilabel(outputs, labels, include_background=False) * inputs.size(0)
        total_num_imgs += inputs.size(0)
        
        # Visualization
        output_image = torch.where(sigmoid(outputs).detach().cpu() > 0.5, 1, 0).numpy()
        output_image_tr = output_image[:, 1:].sum(1)

        loaded_image = inputs.detach().cpu().numpy()
        loaded_label = labels.detach().cpu().numpy()[:, 1:].sum(1)
        
        if debug:
            break
        
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
        
        if debug:
            break


test_dice /= total_num_imgs
print(f"Test Dice: {test_dice:.4f}")

if debug:
    rmtree(ckpt_save_dir)
