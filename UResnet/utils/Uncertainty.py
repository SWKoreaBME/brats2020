import torch
import torch.nn as nn
import os

from utils.Visualize import plot_with_hist


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
    sigmoid = nn.Sigmoid()
    
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
            iter_pred = sigmoid(model(x))
            total_outs[iter_idx] = iter_pred.unsqueeze(0)
    
        model.eval()
        with torch.no_grad():
            static_outs = torch.where(sigmoid(model(x)) > 0.5, 1, 0)[:, 1:].sum(1)
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
        columns = ["Input", "Label", "Prediction", "Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
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
                
            img_save_path = os.path.join(img_dir, f"dropout_{b_idx}.jpg") if img_dir is not None else None
            plot_with_hist(imgs_to_show, columns, img_save_path=img_save_path, show=True)
        
    return total_mean, total_std


if __name__ == "__main__":
    pass
