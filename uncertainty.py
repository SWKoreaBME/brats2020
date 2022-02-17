import torch
import random
import numpy as np
import os

from utils.Device import gpu_setting, model_dataparallel
from utils.Model import load_model_weights
from utils.Data import load_dataloader
from utils.Visualize import plot_with_hist
from utils.Utils import concat_bg
from utils.Data import convert_label_to_brats
from utils.Uncertainty import get_dropout_uncertainty

from monai.networks.nets import UNETR, UNet

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


if __name__ == "__main__":
    import monai.transforms as tf
    
    # Model setting
    amp = True
    device, multi_gpu = gpu_setting()
    model = UNETR(in_channels=4,
                  out_channels=4,
                  img_size=240,
                  feature_size=8,
                  dropout_rate=0.3,
                  hidden_size=64,
                  num_heads=4,
                  num_layers=4,
                  mlp_dim=128,
                  norm_name='instance',
                  spatial_dims=2).to(device)
    # model = UNet(
    #     spatial_dims=2,
    #     in_channels=4,
    #     out_channels=3+1,
    #     channels=(8, 16, 32, 64),
    #     strides=(2, 2, 2, 2),
    #     act="RELU",
    #     norm='instance',
    #     dropout=0.3,
    #     bias=True
    # ).to(device)
    ckpt_dir = "./result/exps/unetr-merge-4layer-withaug"
    model_weights = os.path.join(ckpt_dir, "best.pth")
    model = load_model_weights(model, model_weights, dp=False)
    model = model_dataparallel(model, multi_gpu)
    
    test_loader_params = dict(
        batch_size=16,
        shuffle=True
    )
    
    # validation and test transforms
    val_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        # tf.AsDiscreted(keys=['label'], threshold_values=True),
        tf.EnsureTyped(keys=["image", "label"]),
        tf.ToTensord(keys=['image', 'label']),
        tf.ToDeviced(keys=["image", "label"], device=device)
    ])
    
    
    # Logging
    img_save_dir = os.path.join(ckpt_dir, "uncertainty", "dropout")
    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
    os.makedirs(img_save_dir, exist_ok=True)

    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods
        labels_np = labels.detach().cpu().numpy()
        inputs = inputs.to(device)
        
        # Predict Uncertainty
        get_dropout_uncertainty(model, 
                                inputs,
                                labels, 
                                num_iters=10, 
                                vis=True, 
                                img_dir=os.path.join(img_save_dir, f"b{batch_idx}"))
        
        if batch_idx > 8:
            break
        # break