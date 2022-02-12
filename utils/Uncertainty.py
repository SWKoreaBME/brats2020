import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"

from time import time
from Visualize import plot_with_hist
from Utils import concat_bg


def get_dropout_uncertainty(model, 
                            x,
                            labels = None,
                            num_iters:int = 30, 
                            vis:bool = False, 
                            img_dir="../asset/dropout"):
    """Uncertainty prediction using MCdropout

        ----
        args
        ----
        
        - model: PyTorch model object, Model (should contain dropout)
        - x: Input x images, B X C X ...
        - num_iters: Integer, Number of iterations for the inference (default: 30)
        
    """
    
    # AMP
    amp = True
    
    # Set model to train
    model.train()
    num_batches, num_labels = x.size(0), labels.size(1)
    softmax = nn.Softmax(1)
    
    # check model device and input device

    try:
        model_device = next(model.module.parameters()).device
    except:
        model_device = next(model.parameters()).device
    input_device = x.device
    assert model_device == input_device, f"Devices should be same, but got {model_device} and {input_device}"
    
    total_outs = torch.zeros((num_iters, *labels.size()))
    # total_outs = torch.zeros((num_iters, num_batches, num_labels, 240, 240))
    with torch.cuda.amp.autocast():
        for iter_idx in range(num_iters):
            iter_pred = softmax(model(x))
            total_outs[iter_idx] = iter_pred.unsqueeze(0)
    
        model.eval()
        with torch.no_grad():
            static_outs = softmax(model(x)).argmax(1)
            static_outs_np = static_outs.detach().cpu().numpy()
    
    # Calculate model outputs, calculate average and std over axis 0 (axis of iteration)
    total_outs = total_outs.detach().cpu().numpy()
    total_mean, total_std = total_outs.mean(0), total_outs.std(0)
    
    if vis:
        # Logging
        if img_dir is not None:
            os.makedirs(img_dir, exist_ok=True)
        
        x = x.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        columns = ["Input", "Label", "Prediction", "Uncertainty (NCR/NET)", "Uncertainty (ED)", "Uncertainty (ET)"]
        for b_idx in range(num_batches):
            x_np = x[b_idx]
            static_np = static_outs_np[b_idx]
            mean_np = total_mean[b_idx]  # num_labels X 240 X 240
            std_np = total_std[b_idx]  # num_labels X 240 X 240
            
            imgs_to_show = [x_np, labels[b_idx].argmax(0), static_np]
            for label_idx in range(num_labels):
                if label_idx == 0:  # Ignore background
                    continue
                else:
                    uncertainty_map_np = mean_np[label_idx] + std_np[label_idx]
                    imgs_to_show.append(uncertainty_map_np)
                
            img_save_path = os.path.join(img_dir, f"dropout_{np.random.randint(1000)}.jpg") if img_dir is not None else None
            plot_with_hist(imgs_to_show, columns, img_save_path=img_save_path, show=True)
        
    return None


if __name__ == "__main__":
    
    from Device import gpu_setting, model_dataparallel
    from Model import load_model_weights
    from Data import load_dataloader
    from monai.networks.nets import UNETR, UNet
    
    import monai.transforms as tf
    
    # Model setting
    amp = True
    device, multi_gpu = gpu_setting()
    # model = UNETR(in_channels=4,
    #             out_channels=3+1,
    #             img_size=240,
    #             feature_size=8,
    #             dropout_rate=0.3,
    #             norm_name='batch',
    #             spatial_dims=2).to(device)
    model = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=3+1,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        act="RELU",
        norm='batch',
        dropout=0.3,
        bias=True
    ).to(device)
    model_weights = os.path.join("./result/exps/unet-noaug/best.pth")
    model = load_model_weights(model, model_weights, dp=False)
    model = model_dataparallel(model, multi_gpu)
    
    test_loader_params = dict(
        batch_size=4,
        shuffle=True
    )
    
    # validation and test transforms
    val_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        # tf.AsDiscreted(keys=['label'], threshold_values=True),
        tf.ToTensord(keys=['image', 'label'])
    ])
    
    img_save_dir = "./result/exps/unet-noaug/uncertainty/dropout"
    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"

    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch["image"], concat_bg(batch["label"])
        labels_np = labels.detach().cpu().numpy()
        inputs = inputs.to(device)
        
        # Predict Uncertainty
        get_dropout_uncertainty(model, inputs, labels, num_iters=10, vis=True, img_dir=img_save_dir)
        
        if batch_idx > 4:
            break
        # break
