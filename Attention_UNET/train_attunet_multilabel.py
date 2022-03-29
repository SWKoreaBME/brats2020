import torch, os
import torch._utils
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
from pathlib import Path
import time
# import torchvision.transforms as transforms
# import torchvision
import torch.nn as nn
import wandb
import albumentations
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
# from utils.Metric import compute_meandice
# from utils.Loss import seg_loss_fn_3d
from utils.lossfunction import DiceScore, MultiFocalLoss
from utils.Model import AttU_Net
from utils.dataloader import BRATS2020_2D
from utils.parallel import DataParallelModel, DataParallelCriterion, gather
import numpy as np
# import torchmetrics
from monai.losses import DiceCELoss, DiceFocalLoss
from utils.Metric import compute_meandice_multilabel
import random

################################################################ Load arguments #################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="hyperparameter location",required=True)
parser.add_argument("-n", help="project name",required=True)
#For compute canada
parser.add_argument("-l",help="Data location modfier",default=None)
parser.add_argument("-m",help="(True/False) To use dataparallel or not",default="True")
parser.add_argument("-r", help="resume location",default=None)
parser.add_argument("-s",help="(True/False)save checkpoints?",default="False")
parser.add_argument("-w",help="(True/False)save wandb?",default="True")


args = parser.parse_args()

hyp = args.p
loc_mod = args.l
name = args.n
multi_gpu = eval(args.m)
resume_location = args.r
is_save = eval(args.s)
is_wandb = eval(args.w)

with open(hyp,"r") as f: 
	data_hyp=json.load(f) 
print(f"Experiment Name: {name}\nHyperparameters:{data_hyp}")
print("CUDA is available:{}".format(torch.cuda.is_available()))

if is_wandb:
    #For visualizations
    wandb.init(project="BRATS",config=data_hyp)
    wandb.run.name = name
    wandb.run.save()

if loc_mod is None:
    dataset_path = Path(data_hyp["DATASET_PATH"])
else:
    dataset_path = str(Path(loc_mod)/Path(data_hyp["DATASET_PATH"]))

if is_save:
    PARENT_NAME = Path(data_hyp["SAVE_DIR"])
    FOLDER_NAME = PARENT_NAME / Path("Results")
    MODEL_SAVE = FOLDER_NAME / Path(name) / Path("saved_models")

    if not FOLDER_NAME.is_dir():
        os.mkdir(FOLDER_NAME)
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    elif not MODEL_SAVE.parent.is_dir():
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    else:
        pass

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
#########################################################################################################################################################

def save_checkpoint(epoch,model,optimizer,scheduler,metric):
    """
    Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
    """
    print("Saving Checkpoint ...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric': metric
        }, MODEL_SAVE/Path("Checkpoint_{}.pt".format(time.strftime("%d%b%H_%M_%S",time.gmtime())))
    )
    torch.cuda.empty_cache()

def load_checkpoint(path,model,optimizer,scheduler):
    """
    Loads the saved checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metric = checkpoint['metric']
    del checkpoint
    torch.cuda.empty_cache()

    return model,optimizer,scheduler,epoch,metric

def train(model,trainloader, optimizer, LossFun,MetricFun):
    running_loss = 0.0
    total_dice = 0
    dice_1 = 0
    dice_2 = 0
    dice_3 = 0
    model.train()
    for data in tqdm(trainloader):
        inputs, labels = data
        #Convert to GPU
        inputs,labels = inputs.to(device), labels.to(device)

        # initialize gradients to zero
        optimizer.zero_grad() 
        # forward pass
        outputs = model(inputs)
        if torch.cuda.device_count() > 1 and multi_gpu:
            outputs = gather(outputs,device)
        #compute Loss with respect to target
        loss = LossFun(outputs, labels)
        #Metric Calculation
        dice_1+=MetricFun[0](outputs, labels)
        dice_2+=MetricFun[1](outputs, labels)
        dice_3+=MetricFun[2](outputs, labels)
        total_dice+=MetricFun[3](outputs, labels)
        # metrics_calc = MetricFun(outputs,labels)
        # back propagate
        loss.backward()
        # do SGD step i.e., update parameters
        optimizer.step()
        # by default loss is averaged over all elements of batch
        running_loss += loss.data

        if is_wandb:
            wandb.log({
                "Epoch Train Loss":loss.data,
            })        

    running_loss = running_loss.cpu().numpy()
    # metrics_calc = MetricFun.compute()
    # print(metrics_calc)
    # MetricFun.reset()
    if is_wandb:
        wandb.log({
            f"Epoch Train {CLASSES[1]} Dice Score":dice_1.cpu().numpy() / len(trainloader),
            f"Epoch Train {CLASSES[2]}  Dice Score":dice_2.cpu().numpy() / len(trainloader),
            f"Epoch Train {CLASSES[3]}  Dice Score":dice_3.cpu().numpy() / len(trainloader),
            "Epoch Train Total Dice Score":total_dice.cpu().numpy() / len(trainloader)
        })

    return running_loss

def test(model,testloader,LossFun,epoch,MetricFun):
    running_loss = 0.0
    total_dice = 0
    dice_1 = 0
    dice_2 = 0
    dice_3 = 0
    # evaluation mode which takes care of architectural disablings
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            #Convert to GPU
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            if torch.cuda.device_count() > 1:
                outputs = gather(outputs,device)
            loss = LossFun(outputs, labels)
            #Metric Calculation
            dice_1+=MetricFun[0](outputs, labels)
            dice_2+=MetricFun[1](outputs, labels)
            dice_3+=MetricFun[2](outputs, labels)
            total_dice+=MetricFun[3](outputs, labels)
            # metrics_calc = MetricFun(outputs,labels)
            running_loss += loss.data

    if is_wandb and epoch%5==0:
        log_wandb_table_stats(inputs,outputs,labels,epoch,MetricFun)
    
    # metrics_calc = MetricFun.compute()
    # print(metrics_calc)
    running_loss = running_loss.cpu().numpy()
    print(f"Total Dice Score on test data : {total_dice.cpu().numpy() / len(testloader)}")
    # MetricFun.reset()
    if is_wandb:
        wandb.log({
            "Test Loss":running_loss /  len(testloader),
            f"Epoch Test {CLASSES[1]} Dice Score":dice_1.cpu().numpy() / len(testloader),
            f"Epoch Test {CLASSES[2]}  Dice Score":dice_2.cpu().numpy() / len(testloader),
            f"Epoch Test {CLASSES[3]}  Dice Score":dice_3.cpu().numpy() / len(testloader),
            "Test Total Dice Score":total_dice.cpu().numpy() / len(testloader)
        })

    return total_dice.cpu().numpy() / len(testloader) , running_loss /  len(testloader)

def log_wandb_table_stats(Input,Output,Truth,epoch,MetricFun):
    # W&B: Create a Table to store predictions for each test step
    sigmoid = nn.Sigmoid()
    columns=["id", "image_T1","image_T1Gd","image_T2","image_T2-FLAIR",f"Masks_{CLASSES[0]}",f"Masks_{CLASSES[1]}",f"Masks_{CLASSES[2]}",f"Masks_{CLASSES[3]}",f"Dice_{CLASSES[1]}",f"Dice_{CLASSES[2]}",f"Dice_{CLASSES[3]}","Total Dice Score"]
    test_table = wandb.Table(columns=columns)
    # Y_pred_target=torch.argmax(Output,dim=1)
    # Y_true_target=torch.argmax(Truth,dim=1)
    output_image = torch.where(sigmoid(Output).detach().cpu() > 0.5, 1, 0).numpy()
    for i in range(Input.shape[0]):
        idx = f"{epoch}_{i}"
        image1 = wandb.Image(Input[i,0,:,:])
        image2 = wandb.Image(Input[i,1,:,:])
        image3 = wandb.Image(Input[i,2,:,:])
        image4 = wandb.Image(Input[i,3,:,:])
        mask1 = wandb.Image(Input[i,0,:,:].cpu().numpy(), masks={
            "prediction": {"mask_data" : output_image[i,0,:,:], "class_labels" : {0:"none",1:CLASSES[0]}},
            "ground_truth": {"mask_data" : Truth[i,0,:,:].cpu().numpy(), "class_labels" : {0:"none",1:CLASSES[0]}}
        })
        mask2 = wandb.Image(Input[i,0,:,:].cpu().numpy(), masks={
            "prediction": {"mask_data" : output_image[i,1,:,:], "class_labels" : {0:"none",1:CLASSES[1]}},
            "ground_truth": {"mask_data" : Truth[i,1,:,:].cpu().numpy(), "class_labels" :{0:"none",1:CLASSES[1]}}
        })
        mask3 = wandb.Image(Input[i,0,:,:].cpu().numpy(), masks={
            "prediction": {"mask_data" : output_image[i,2,:,:], "class_labels" : {0:"none",1:CLASSES[2]}},
            "ground_truth": {"mask_data" : Truth[i,2,:,:].cpu().numpy(), "class_labels" : {0:"none",1:CLASSES[2]}}
        })
        mask4 = wandb.Image(Input[i,0,:,:].cpu().numpy(), masks={
            "prediction": {"mask_data" : output_image[i,3,:,:], "class_labels" : {0:"none",1:CLASSES[3]}},
            "ground_truth": {"mask_data" : Truth[i,3,:,:].cpu().numpy(), "class_labels" : {0:"none",1:CLASSES[3]}}
        })
        
        # dicescore = MetricFun(Output[[i],:,:,:], Truth[[i],:,:,:], include_background=False) * Input.size(0)
        dicescore = MetricFun[3](Output[[i],:,:,:], Truth[[i],:,:,:])
        dice1 = MetricFun[0](Output[[i],:,:,:], Truth[[i],:,:,:])
        dice2 = MetricFun[1](Output[[i],:,:,:], Truth[[i],:,:,:])
        dice3 = MetricFun[2](Output[[i],:,:,:], Truth[[i],:,:,:])

        test_table.add_data(idx, image1,image2,image3,image4,mask1,mask2,mask3,mask4,dice1,dice2,dice3,dicescore)
    wandb.log({"table_key": test_table})

if __name__=="__main__":
    ###################################
    #          Model Setup            #
    ###################################
    NUM_CLASSES = 4
    DEVICE_LIST = data_hyp["DEVICE_LIST"]
    NUM_GPUS = len(DEVICE_LIST)
    # CLASSES = {0:'non-enhancing tumor core',1:'peritumoral edema',2:'GD-enhancing tumor',3:'background'}
    CLASSES = {0:'background',1:'tumour core',2:'whole tumour',3:'enhancing tumour'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AttU_Net(img_ch=4,output_ch=NUM_CLASSES)
    # model = smp.Unet(encoder_name="resnet18",
    #             encoder_depth=5,
    #             decoder_channels=[64, 32, 16, 8, 4],
    #             classes=NUM_CLASSES,
    #             in_channels=4)

    #Multiple GPUs
    if torch.cuda.device_count() > 1  and multi_gpu:
        print("Using {} GPUs".format(NUM_GPUS))
        model = DataParallelModel(model, device_ids=DEVICE_LIST)
    
    model = model.to(device)


    ###################################
    #        Data Augmentations       #
    ###################################   
    
    transform_train =  albumentations.Compose([    
        albumentations.OneOf([
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),              
            albumentations.Rotate(p=0.5)
            ],p=0.8),
            # albumentations.GaussianBlur(p=0.5),
            # albumentations.RandomBrightnessContrast(p=0.4),
        ToTensorV2()])
    # transform_train =  albumentations.Compose([
    #     ToTensorV2()]) 
    transform_test =  albumentations.Compose([
        ToTensorV2()])
    ###################################
    #          Data Loaders           #
    ###################################  
    #Loading the images for train set
    trainset = BRATS2020_2D(path=dataset_path,mode="training",transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=data_hyp["TRAIN_BATCH_SIZE"],shuffle=True, num_workers=4)
    #Loading the images for test set
    testset = BRATS2020_2D(path=dataset_path, mode="validation",transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=data_hyp["TEST_BATCH_SIZE"],shuffle=True, num_workers=4)

    ###########################################
    #  Loss function, optimizers and metrics  #
    ########################################### 
    
    #Loss function
    # LossFun = MultiFocalLoss(alpha=data_hyp['FOCAL_IMP'],gamma=data_hyp['FOCAL_GAMMA'])
    LossFun = DiceCELoss(include_background=False,
                           smooth_nr=1e-5, 
                           smooth_dr=1e-5, 
                           squared_pred=True,
                           to_onehot_y=False,
                           lambda_ce=data_hyp["LAMBDA_CE"],
                           lambda_dice=data_hyp["LAMBDA_DICE"],
                           sigmoid=True,
                           ce_weight=torch.Tensor([0.4,0.2,0.4]))
    # LossFun = DiceFocalLoss(include_background=False,
    #                        smooth_nr=1e-5, 
    #                        smooth_dr=1e-5, 
    #                        squared_pred=True,
    #                        to_onehot_y=False,
    #                        lambda_focal=data_hyp["LAMBDA_CE"],
    #                        lambda_dice=data_hyp["LAMBDA_DICE"],
    #                        sigmoid=True,
    #                        focal_weight=[0.38,0.24,0.38])
    # LossFun = nn.CrossEntropyLoss().cuda(device)
    # LossFun = LossFun.to(device)
    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=data_hyp['LEARNINGRATE'], weight_decay = data_hyp['LAMBDA'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=data_hyp["PATIENCE"])
    #Metrics
    # metrics = torchmetrics.MetricCollection([DiceScore(labels=[0,1,2],dist_sync_on_step=multi_gpu),
    #                                         torchmetrics.F1Score(num_classes=NUM_CLASSES,average=None,dist_sync_on_step=multi_gpu)])
    # metrics = torchmetrics.MetricCollection([DiceScore(labels=[0,1,2],dist_sync_on_step=multi_gpu)])
    
    # TrainMetricDict = metrics.clone(prefix='train_').to(device)
    # TestMetricDict = metrics.clone(prefix='test_').to(device)
    # DiceScore = torchmetrics.functional.dice_score
    # dicescore = DiceScore(labels=[0,1,2])
    dicescore_1 = DiceScore(labels=[1])
    dicescore_2 = DiceScore(labels=[2])
    dicescore_3 = DiceScore(labels=[3])
    dice_mean = compute_meandice_multilabel
    MetricFun = [dicescore_1,dicescore_2,dicescore_3,dice_mean]

    best_metric = -10000
    ###################################
    #           Training              #
    ###################################    
    if is_wandb:
        wandb.watch(model, log='all') 

    if resume_location is not None:
        model,optimizer,scheduler,epoch_start,best_metric = load_checkpoint(resume_location,model,optimizer,scheduler)
        best_metric = best_metric.cpu().numpy()
        print("Resuming from saved checkpoint...")
    else:
        epoch_start = 0
    
    for epoch in range(epoch_start,data_hyp['EPOCHS']):
        print("EPOCH: {}".format(epoch))
        train(model, train_loader ,optimizer, LossFun,MetricFun)
        metric,test_loss = test(model,test_loader,LossFun,epoch,MetricFun)
        ### Saving model checkpoints
        if is_save and (epoch+1) % 15 == 0 and metric>best_metric:
            save_checkpoint(epoch,model,optimizer,scheduler,metric)
            best_metric = metric
        scheduler.step(test_loss)
        wandb.log({"Learning rate":optimizer.param_groups[0]["lr"]})
    print("Training done...")
