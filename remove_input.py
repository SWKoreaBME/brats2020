from statistics import mode
from sklearn.feature_extraction import image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from utils.Metric import compute_meandice_multilabel

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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def remove_input(tensor, axis=0, mode='zeros'):
    """Remove a single input modality
        
        input tensor: B x C x H x W
        ouptut tensor: B x C x H x W
    """
    output_tensor = torch.zeros_like(tensor)
    single_size = (tensor.size(0), *tensor.size()[2:])
    for i in range(output_tensor.size(1)):
        if i == axis:
            if mode == 'zeros':
                output_tensor[:, i] = torch.zeros(single_size)
            elif mode == 'random':
                output_tensor[:, i] = torch.randn(single_size)
        else:
            output_tensor[:, i] = tensor[:, i]
    return output_tensor


if __name__ == "__main__":
    
    from utils.Device import gpu_setting, model_dataparallel
    from utils.Model import load_model_weights
    from utils.Data import load_dataloader
    from utils.Uncertainty import get_dropout_uncertainty

    from model.attnunet import AttU_Net
    from model.unetr import UNETR
    from tqdm import tqdm
    
    import monai.transforms as tf
    
    sigmoid = nn.Sigmoid()
    
    # Model setting
    amp = True
    device, multi_gpu = gpu_setting()
    unetr = UNETR(in_channels=4,
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
    model_weights = os.path.join("./result/exps/unetr-merge-4layer-withaug", "best.pth")
    unetr = load_model_weights(unetr, model_weights, dp=False)
    unetr = model_dataparallel(unetr, multi_gpu)
    
    attn_unet = AttU_Net(img_ch=4, output_ch=4)
    model_weights = os.path.join("./asset/WithAug_Attnet.pt")
    attn_unet = load_model_weights(attn_unet, model_weights, dp=False, device=torch.device('cpu'))
    attn_unet = attn_unet.to(device)
    attn_unet = model_dataparallel(attn_unet, multi_gpu)
    
    test_loader_params = dict(
        batch_size=8,
        shuffle=False
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
    img_save_dir = os.path.join("./asset", "input_removed")
    os.makedirs(img_save_dir, exist_ok=True)
    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
    
    
    # Test Dataloder
    test_dataloader = load_dataloader(root_dir, "test", val_transforms, test_loader_params)
    
    
    # Image seqs
    images_seqs = ["T1", "T1Gd", "T2", "FLAIR"]
    tumors_names = ["TC", "WT", "ET"]
    model_dice = list()
    # replace_modes = ["random", "zeros"]
    replace_modes = ["zeros"]

    # models = [unetr, attn_unet]
    models = [unetr]
    for replace_mode in replace_modes:
        for model_idx, model in enumerate(models):
        
            dice_dict = {
                "original": 0.,
                "T1": 0.,
                "T1Gd": 0.,
                "T2": 0.,
                "FLAIR": 0.
            }
            total_num_imgs = 0
            
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(test_dataloader)):
                    inputs, labels = batch["image"], convert_label_to_brats(concat_bg(batch["label"]))  # use all four mods
                    labels_np = labels.detach().cpu().numpy()
                    inputs = inputs.to(device)
                    total_num_imgs += inputs.size(0)
                    
                    # remove
                    outputs = torch.where(sigmoid(model(inputs)) > 0.5, 1, 0)
                    original_dice = compute_meandice_multilabel(outputs, labels, include_background=False) * inputs.size(0)
                    dice_dict["original"] += float(original_dice.data)
                    
                    # plot_whole_imgs(inputs[:, 0].detach().cpu().numpy(), 
                    #                 os.path.join(img_save_dir, f"inputs.jpg"), 
                    #                 num_cols=int(np.sqrt(inputs.size(0))))
                    
                    # plot_whole_imgs(outputs[:, 1:].sum(1).detach().cpu().numpy(),
                    #                 os.path.join(img_save_dir, f"outputs.jpg"),
                    #                 num_cols=int(np.sqrt(inputs.size(0))))
                    # plot_whole_imgs(labels[:, 1:].sum(1).detach().cpu().numpy(), os.path.join(img_save_dir, f"labels.jpg"))
                    
                    if float(original_dice / inputs.size(0)) > 0.75 and (labels_np[:, 1:].sum(1).sum() > 10000):
                        for i in range(4):  # Iterate over image sequences
                            removed_input = remove_input(inputs, i, mode=replace_mode)
                            removed_outputs = torch.where(sigmoid(model(removed_input)) > 0.5, 1, 0)
                            # removed_dice = compute_meandice_multilabel(removed_outputs, labels, include_background=False) * inputs.size(0)
                            # dice_dict[images_seqs[i]] += float(removed_dice.data)
                            # removed_mean, removed_std = get_dropout_uncertainty(model, removed_input, labels, num_iters=100, vis=False)
                            # removed_uncertainty = removed_mean + removed_std
                            
                            input_np = inputs[:, i].detach().cpu().numpy()
                            out_np = outputs[:, 1:].sum(1).detach().cpu().numpy()
                            label_np = labels_np[:, 1:].sum(1)
                            removed_out_np = removed_outputs[:, 1:].sum(1).detach().cpu().numpy()
                            
                            plot_whole_imgs(input_np,
                                            os.path.join(img_save_dir, f"inputs-{images_seqs[i]}-{replace_mode}-{batch_idx}.jpg"),
                                            num_cols=2)
                            
                            img_to_vis = np.concatenate([label_np, out_np, removed_out_np], -1)
                            plot_whole_imgs(img_to_vis,
                                            os.path.join(img_save_dir, f"outputs-{images_seqs[i]}_removed-{replace_mode}-{batch_idx}.jpg"),
                                            num_cols=2)
                            
                        # break
                            
                        # Visualize by each tumor
                        # for k in range(labels.size(1)):
                        #     if k == 0:
                        #         continue
                            # removed_uncertainty_np = removed_uncertainty[:, k]
                            # img_to_vis = np.concatenate([img_to_vis, removed_uncertainty_np], -1)
                            # plot_whole_imgs(img_to_vis,
                            #                 os.path.join(img_save_dir, f"{images_seqs[i]}_removed_uncertainty_{tumors_names[k-1]}.jpg"), 
                            #                 num_cols=int(np.sqrt(inputs.size(0))))
                if batch_idx == 192:
                    break
                
            for key, val in dice_dict.items():
                dice_dict[key] = val / total_num_imgs
            model_dice.append(list(dice_dict.values()))

    # df = pd.DataFrame(data=model_dice)
    # df.index = ["UNETR (random)", "Attn-UNet (random)"] + ["UNETR (zeros)", "Attn-UNet (zeros)"]
    # df.columns = ["Original"] + images_seqs
    # df.to_csv("./asset/input_removed.csv")
    