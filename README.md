# BraTs Segmentation 2020: Course Project
This repository implements three different models and analyses the results on the BRATS 2020 dataset downloaded from the [link](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). This work is done as part of the course project **MBP 1413H: Biomedical Applications of AI**.  

## Project description

This project aims at two goals

1. Brain tumour segmentation
   - UResNet
   - Attention U-Net
   - UNet Transformer(UNETR)
      
2. Uncertainty prediction
   - Dropout

## Results
BRATS dataset contains multimodal MRI scans with extensive annotation, comprising GD-enhancing Tumor, peritumoral edema, and necrotic and non-enhancing tumor core. For our experiments, we group labels to make new set of labels as follows:-
- Tumor core label was created by joining non enhancing tumor core and GD enhancing tumor
- Whole tumor label was created by joining non enhancing tumor core, peritumoral edema, and GD enhancing tumor
- Enhancing tumor label was created by just considering class GD enhancing tumor
- Background label was created by considering all the pixels not included in the Whole tumor label

Using these labels, we performed multi label segmentation using three models. We report the dice score for our three experiments.

|        **Dice**       | **TC** | **ET** | **WT** | **AVG** |
|:---------------------:|:-------:|:------:|:------:|:-------:|
|  Resunet (WithNoAug)  |   0.4   |  0.79  |  0.68  |   0.62  |
|   Resunet (WithAug)   |   0.70  |  0.74  |  0.68  |   0.69  |
|   Attention(WithAug)  |   0.72  |  0.73  |  0.71  |   0.72  |
| Attention (WithNoAug) |   0.65  |  0.69  |  0.66  |   0.67  |
|     UnetR(WithAug)    |   0.76  |  0.84  |  0.72  |   0.77  |
|    UnetR(WithNoAug)   |   0.50  |  0.65  |  0.69  |   0.61  |

Please refer to the report for more details.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Note that your local environment should be cuda enabled.  

### Prerequisites
Libraries needed

```
monai
torchmetrics
pytorch
torchvision
segmentation_models_pytorch
pathlib
```
### Dataset
The dataset can be downloaded from the [link](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). For running Attention U-Net experiments, one can directly use the downloaded dataset stored in `h5` format. For running UResnet and UNETR, the dataset should be in `nifty` format.  


## File Descriptions
### UNETR
* `1 - remove_input.py` - This script runs ablation study by removing channels and replacing them with random noise or 0
* `2 - train_unetr.py` - This script trains transformer unet
* `3 - transform.py` - This script converts `hdf5` to `nifti` format. Run this script before training the network 
* `4 - uncertainity` - This script gets the uncertaininty maps for a given model using Monte Carlo Dropout
* `5 - utils\` - This directory contains helper functions for dataloading, metric calculations, visualizations, uncertaininty calculation, etc.
### Attention UNet
* `1 - Hyperparameters\` - This directory is used by `train_attunet_multilabel.py`. It contains `json` files with hyperparameter settings for experiment
* `2 - evaluate.py` - This script gets the metric scores and uncertainty map for a given model and dataset
* `3 - remove_input_attn.py` - This script runs ablation study by removing channels and replacing them with random noise or 0
* `4 - utils\` - This directory contains helper functions for dataloading, metric calculations, visualizations, uncertaininty calculation, etc.

## Train Examples

### Training UNETR
```bash
# To train UNETR
# With Slurm Scripts
>>> sbatch -p gpu /path/to/script

# Directly run python
>>> python ./UNet_transformer/train_unetr.py
```
### Training Attention Unet
```bash
# Directly run python
>>> python ./Attention_UNET/train_attunet_multilabel.py\
   -p [str: HYPERPARAMETER LOCATION]\
   -n [str: NAME OF EXPERIMENT]\
   -l [str: DATASET LOCATION MODIFIER (compute canada)]\
   -m [bool: MULTI GPU (True/False)]\
   -r [str: RESUME FROM CHECKPOINT(default: None)]\
   -s [bool: SAVE CHECKPOINTS (True/False)]\
   -w [bool: USE WANDB (True/False)]\
```

## Contributor
The following people contributed to this project.
* [Sangwook](https://github.com/SWKoreaBME)
* [Siham]()
* [Vishwesh](https://github.com/Vishwesh4)
