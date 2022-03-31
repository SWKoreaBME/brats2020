from sklearn.feature_extraction import image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from utils.Metric import compute_meandice_multilabel
import albumentations
from albumentations.pytorch import ToTensorV2
from utils.Visualize import plot_whole_imgs
from utils.Model import AttU_Net,load_model_weights
from utils.dataloader import BRATS2020_2D
from tqdm import tqdm
    

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


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
    
    # Model setting
    amp = True
    model_weights = os.path.join("./Results/Attn_Unet_multi_exp1_rotaug/saved_models/Checkpoint_23Feb12_12_04.pt")
    root_dir = "/home/ramanav/projects/rrg-amartel/ramanav/Downloads/BraTS2020_training_data"

    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn_unet = AttU_Net(img_ch=4, output_ch=4)
    attn_unet = load_model_weights(attn_unet, model_weights, device)
    attn_unet = attn_unet.to(device)
    
    # validation and test transforms
    val_transforms = albumentations.Compose([
        ToTensorV2()])
    # Logging directories
    img_save_dir = os.path.join("./Results/Attn_Unet_multi_exp1_rotaug", "input_removed")
    os.makedirs(img_save_dir, exist_ok=True)
    testset = BRATS2020_2D(path=root_dir, mode="testing",transform=val_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True)
    
    # Image seqs
    images_seqs = ["T1", "T1Gd", "T2", "FLAIR"]
    tumors_names = ["TC", "WT", "ET"]
    model_dice = list()
    # replace_modes = ["random", "zeros"]
    replace_modes = ["random","zeros"]

    models = [attn_unet]
    for replace_mode in replace_modes:
        print(f"Experiment Mode: {replace_mode}")
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
                for batch_idx, batch in tqdm(enumerate(test_loader)):
                    inputs, labels = batch  # use all four mods
                    labels_np = labels.detach().cpu().numpy()
                    inputs = inputs.to(device)
                    total_num_imgs += inputs.size(0)
                    
                    # remove
                    outputs = torch.where(sigmoid(model(inputs)) > 0.5, 1, 0)
                    original_dice,_ = compute_meandice_multilabel(outputs, labels, include_background=False)
                    dice_dict["original"] += float(original_dice.data * inputs.size(0))
                                   
                    # if float(original_dice) > 0.55 and (labels_np[:, 1:].sum(1).sum() > 10000):
                    for i in range(4):  # Iterate over image sequences
                        removed_input = remove_input(inputs, i, mode=replace_mode)
                        removed_outputs = torch.where(sigmoid(model(removed_input)) > 0.5, 1, 0)

                        input_np = inputs[:, i].detach().cpu().numpy()
                        out_np = outputs[:, 1:].sum(1).detach().cpu().numpy()
                        label_np = labels_np[:, 1:].sum(1)
                        removed_out_np = removed_outputs[:, 1:].sum(1).detach().cpu().numpy()
                        
                        # plot_whole_imgs(input_np,
                        #                 os.path.join(img_save_dir, f"inputs-{images_seqs[i]}-{replace_mode}-{batch_idx}.jpg"),
                        #                 num_cols=2)
                        
                        # img_to_vis = np.concatenate([label_np, out_np, removed_out_np], -1)
                        # plot_whole_imgs(img_to_vis,
                        #                 os.path.join(img_save_dir, f"outputs-{images_seqs[i]}_removed-{replace_mode}-{batch_idx}.jpg"),
                        #                 num_cols=2)

                if batch_idx == 192:
                    break
                
            for key, val in dice_dict.items():
                dice_dict[key] = val / total_num_imgs
            model_dice.append(list(dice_dict.values()))
            print(model_dice)
