import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import monai.transforms as tf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import pandas as pd
# import nibabel as nib
import matplotlib.pyplot as plt
from monai.optimizers import Novograd

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from copy import copy
from shutil import rmtree
from tqdm import tqdm
from torch.utils import data
import numpy as np
import random

import monai.transforms as tf
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, UNETR, DynUNet, SegResNet
from monai.networks.layers import Norm
# from monai.networks import one_hot
from utils.Utils import concat_bg
from utils.Metric import compute_meandice, compute_meandice_multilabel

from utils.Device import *
from utils.Data import load_dataloader, convert_label_to_brats
from utils.Loss import *
from utils.Gradient import plot_grad_flow
from utils.Visualize import plot_whole_imgs
from utils.Model import load_model_weights
from utils.Metric import Score

from datetime import datetime
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ToDeviced
)
from monai.utils import set_determinism

import torch

print_config()

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
# import numpy as np
# import pandas as pd
# import nibabel as nib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from copy import copy
from shutil import rmtree
from tqdm import tqdm
from torch.utils import data
import numpy as np
import random

import monai.transforms as tf
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, UNETR, DynUNet,SegResNet
from monai.networks.layers import Norm
# from monai.networks import one_hot
from utils.Utils import concat_bg
from utils.Metric import compute_meandice, compute_meandice_multilabel

from utils.Device import *
from utils.Data import load_dataloader, convert_label_to_brats
from utils.Loss import *
from utils.Gradient import plot_grad_flow
from utils.Visualize import plot_whole_imgs
from utils.Model import load_model_weights
from utils.Metric import Score

from datetime import datetime

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

debug = False
#test_only = bool(int(os.environ.get("TEST_ONLY"))) if not debug else False
test_only = False

#device, multi_gpu = gpu_setting()
device="cuda:0"


# Get Environment variables
batch_size_train = 4
batch_size_test =  4
root_dir =  "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
ckpt_save_dir =  "result/"
num_epochs = 10
augmentation =  False


img_save_dir ="result/figures"

grad_save_dir = "result/gradients"


images_seqs = ["T1", "T1Gd", "T2", "FLAIR"]


# Model
amp = True
device = torch.device("cuda:0")
model = SegResNet(
    spatial_dims=2,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=4,
    dropout_prob=0.2,
).to(device)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# Tensorboard logging
if not debug:
    datetime_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/{os.path.basename(ckpt_save_dir)}/{datetime_now}")


# train transforms
# Add augmentation
if augmentation:
    train_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        tf.OneOf([
            tf.RandGaussianNoised(keys=["image"], prob=0.4, mean=0, std=1.0),  # 0.8 x 0.5
            tf.RandRotated(keys=["image", "label"], prob=0.4, range_x=(-90, 90)),  # 0.8 x 0.5
            tf.RandFlipd(keys=["image", "label"], prob=0.4, spatial_axis=0),  # 0.8 x 0.5
        ]),
        tf.EnsureTyped(keys=["image", "label"]),
        tf.ToTensord(keys=['image', 'label']),
        tf.ToDeviced(keys=["image", "label"], device=device),
        tf.DeleteItemsd(keys=["image_transforms", "label_transforms"])
    ])
    
else:
    train_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        tf.EnsureTyped(keys=["image", "label"]),
        tf.ToTensord(keys=['image', 'label']),
        tf.ToDeviced(keys=["image", "label"], device=device),
    ])

# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
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
if test_only:
    print("Test only")
    model_weights = os.path.join(ckpt_save_dir, "best_metric_model_with_augmentation.pth")
    assert os.path.exists(model_weights), f"Model weight does not exist"
    #print("Load model ",model_weights)
    model.load_state_dict(
        torch.load(os.path.join(ckpt_save_dir, "best_metric_model_with_augmentation.pth"))
    )
    #model = load_model_weights(model, model_weights, dp=False)
    print("model loaded!")
    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    print(len(test_dataloader))



max_epochs = 10
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
#optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
optimizer = Novograd(model.parameters(), learning_rate * 10)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.95, patience=3)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = tf.Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()


if not test_only:
    train_dataloader = load_dataloader(root_dir, "train", train_transforms, loader_params)
    print(len(train_dataloader))
    valid_dataloader = load_dataloader(root_dir, "valid", val_transforms, test_loader_params)
    print(len(valid_dataloader))
    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    print(len(test_dataloader))
    print(f"Train Start")
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    max_epochs=10

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"],
                convert_label_to_brats(concat_bg(batch_data["label"]))
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            #print(
            #    f"{step}/{len(train_dataloader)*batch_size_train}"
            #    f", train_loss: {loss.item():.4f}"
            #    f", step time: {(time.time() - step_start):.4f}"
            #)
        #lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                for val_data in valid_dataloader:
                    val_inputs, val_labels = (
                        val_data["image"],
                        convert_label_to_brats(concat_bg(val_data["label"]))
                    )
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i)
                                for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                #scheduler.step(batch_loss['val'])

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[1].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[2].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[3].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(
                        time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join("result/best_metric_model_with_no_augmentation.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(
            f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    
# Test

sigmoid = nn.Sigmoid()
pbar = tqdm(total=len(test_dataloader))
scorer = Score(
    include_background=False,
    sigmoid=True
)


model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)
            
        scorer.evaluate(outputs, labels)
        pbar.update(1)
        
        if debug:
            break
        
    # Visualization
    output_image = torch.where(sigmoid(outputs).detach().cpu() > 0.5, 1, 0).numpy()
    output_image_tr = output_image[:, 1:].sum(1)

    loaded_image = inputs.detach().cpu().numpy()
    loaded_label = labels.detach().cpu().numpy()[:, 1:].sum(1)
        
    for img_idx in range(inputs.size(0)):
        if img_idx > 4:
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
        
        # if debug:
        #     break

scorer.save(ckpt_save_dir)

post_transforms = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
model.eval()
with torch.no_grad():
    for batch_idx, val_data in enumerate(test_dataloader):
        val_inputs, val_labels = val_data["image"], convert_label_to_brats(concat_bg(val_data["label"]))  # use all four mods
        val_outputs = inference(val_inputs)
        val_outputs = [post_trans(i)
                       for i in decollate_batch(val_outputs)]
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)

    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc, metric_wt, metric_et = metric_batch[1].item(), metric_batch[2].item(), metric_batch[3].item()

print("Metric on original image spacing: ", metric)
print(f"metric_tc: {metric_tc:.4f}")
print(f"metric_wt: {metric_wt:.4f}")
print(f"metric_et: {metric_et:.4f}")
# if debug:
#     rmtree(ckpt_save_dir)
