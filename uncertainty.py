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

from model.unetr import UNETR
from monai.networks.nets import SegResNet

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
                  mlp_dim=128,
                  pos_embed='conv',
                  norm_name='instance',
                  spatial_dims=2).to(device)
    ckpt_dir = "./result/exps/unetr-merge-4layer-withaug"
    model_weights = os.path.join(ckpt_dir, "best.pth")
    model = load_model_weights(model, model_weights, dp=False)
    model = model_dataparallel(model, multi_gpu)
    
    test_loader_params = dict(
        batch_size=1,
        shuffle=False
    )
    
    # validation and test transforms
    val_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        tf.EnsureTyped(keys=["image", "label"]),
        tf.ToTensord(keys=['image', 'label']),
        tf.ToDeviced(keys=["image", "label"], device=device)
    ])
    
    
    # Logging
    # img_save_dir = os.path.join(ckpt_dir, "uncertainty", "dropout")
    img_save_dir = os.path.join("asset", "dropout")
    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
    os.makedirs(img_save_dir, exist_ok=True)

    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    target_files = ["volume_310_slice_85.nii.gz", "volume_316_slice_89.nii.gz"]
    
    for batch_idx, batch in enumerate(test_dataloader):
        
        file_names = [x["image"] for x in test_dataloader.dataset.data[batch_idx * test_loader_params["batch_size"]: (batch_idx + 1) * test_loader_params["batch_size"]]]
        if os.path.basename(file_names[0]) not in target_files:
            continue

        inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods
        labels_np = labels.detach().cpu().numpy()
        inputs = inputs.to(device)
        
        # Predict Uncertainty
        get_dropout_uncertainty(model,
                                inputs,
                                labels,
                                num_iters=30,
                                vis=True,
                                file_names=file_names,
                                img_dir=img_save_dir)