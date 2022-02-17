import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from utils.Visualize import plot_whole_imgs
from utils.Utils import concat_bg
from utils.Data import convert_label_to_brats


random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def remove_input(tensor, axis=0):
    """Remove a single input modality
        
        input tensor: B x C x H x W
        ouptut tensor: B x C x H x W
    """
    output_tensor = torch.zeros_like(tensor)
    single_size = (tensor.size(0), *tensor.size()[2:])
    for i in range(output_tensor.size(1)):
        if i == axis:
            output_tensor[:, i] = torch.zeros(single_size)
        else:
            output_tensor[:, i] = tensor[:, i]
    return output_tensor


if __name__ == "__main__":
    
    from utils.Device import gpu_setting, model_dataparallel
    from utils.Model import load_model_weights
    from utils.Data import load_dataloader
    from utils.Uncertainty import get_dropout_uncertainty
    from monai.networks.nets import UNETR, UNet
    
    import monai.transforms as tf
    
    sigmoid = nn.Sigmoid()
    
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
        batch_size=56,
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
    
    # Logging directories
    img_save_dir = os.path.join(ckpt_dir, "input_removed")
    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
    os.makedirs(img_save_dir, exist_ok=True)
    
    
    # Test Dataloder
    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    
    
    # Image seqs
    images_seqs = ["T1", "T1Gd", "T2", "FLAIR"]
    tumors_names = ["TC", "WT", "ET"]
    
    
    # Iterate through mini-batches
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods
            labels_np = labels.detach().cpu().numpy()
            inputs = inputs.to(device)
            
            # remove
            with torch.cuda.amp.autocast():
                outputs = torch.where(sigmoid(model(inputs)) > 0.5, 1, 0)
                
                plot_whole_imgs(inputs[:, 0].detach().cpu().numpy(), os.path.join(img_save_dir, f"inputs.jpg"))
                plot_whole_imgs(labels[:, 1:].sum(1).detach().cpu().numpy(), os.path.join(img_save_dir, f"labels.jpg"))
                plot_whole_imgs(outputs[:, 1:].sum(1).detach().cpu().numpy(), os.path.join(img_save_dir, f"outputs.jpg"))
                
                for i in range(4):  # Iterate over image sequences
                    removed_input = remove_input(inputs, i)
                    removed_outputs = torch.where(sigmoid(model(removed_input)) > 0.5, 1, 0)
                    removed_mean, removed_std = get_dropout_uncertainty(model, removed_input, labels, num_iters=10, vis=False)
                    removed_uncertainty = removed_mean + removed_std
                    
                    plot_whole_imgs(removed_outputs[:, 1:].sum(1).detach().cpu().numpy(), os.path.join(img_save_dir, f"{images_seqs[i]}_removed.jpg"))
                    for k in range(labels.size(1)):
                        if k == 0:
                            continue
                        plot_whole_imgs(removed_uncertainty[:, k],
                                        os.path.join(img_save_dir, f"{images_seqs[i]}_removed_uncertainty_{tumors_names[k-1]}.jpg"))
            break
