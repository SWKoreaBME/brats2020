#This script gets uncertainty maps/ dice score/ Specificity / Sensitivity for each of the class separately and average

import torch
import albumentations
from utils.Model import AttU_Net,load_model_weights
from albumentations.pytorch import ToTensorV2
from utils.dataloader import BRATS2020_2D
from utils.Uncertainty import get_dropout_uncertainty
from tqdm import tqdm
from utils.parallel import DataParallelModel, DataParallelCriterion, gather
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils.Metric import Score


if __name__ == "__main__":
    
    NUM_CLASSES = 4
    MULTI_GPU = False
    DEVICE_LIST = [0,1]
    NUM_GPUS = len(DEVICE_LIST)
    CLASSES = {0:'background',1:'tumour core',2:'whole tumour',3:'enhancing tumour'}
    MODEL_PATH = "./Results/Attn_Unet_multi_exp1_rotaug/saved_models/Checkpoint_23Feb12_12_04.pt"
    DATASET_PATH = "/home/ramanav/projects/rrg-amartel/ramanav/Downloads/BraTS2020_training_data"
    SAVE_DIR = str(Path(MODEL_PATH).parent.parent)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttU_Net(img_ch=4,output_ch=NUM_CLASSES)
    model = load_model_weights(model, MODEL_PATH, torch.device("cpu"))

    #Multiple GPUs
    if torch.cuda.device_count() > 1  and MULTI_GPU:
        print("Using {} GPUs".format(NUM_GPUS))
        model = DataParallelModel(model, device_ids=DEVICE_LIST)
    
    model = model.to(device)

    ###################################
    #        Data Augmentations       #
    ###################################   
    transform_test =  albumentations.Compose([
        ToTensorV2()])
    ###################################
    #          Data Loaders           #
    ###################################  

    #Loading the images for test set
    testset = BRATS2020_2D(path=DATASET_PATH, mode="testing",transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True)
    Metric_Score = Score(include_background=False,sigmoid=True)

    for batch_idx, data in tqdm(enumerate(test_loader)):
        inputs, labels = data
        labels_np = labels.detach().cpu().numpy()
        inputs = inputs.to(device)
        
        # Predict Uncertainty and save results for some outputs
        if batch_idx<8:
            get_dropout_uncertainty(model, inputs, labels, num_iters=10, vis=True, img_dir=SAVE_DIR)
        
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
            Metric_Score.evaluate(outputs, labels)

    Metric_Score.save(str(Path(MODEL_PATH).parent.parent))
