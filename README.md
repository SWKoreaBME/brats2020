# BraTs Segmentation 2020



### Project description

This project aims at two goals.


1. Brain tumour segmentation
   1. UResNet
   2. Attention U-Net
   3. UNETR
      
2. Uncertainty prediction
   1. Dropout


### Train Examples

```bash
# To train UNETR

# With Slurm Scripts

>>> sbatch -p gpu /path/to/script

# Directly run python
>>> python train_unetr.py
```

### Contributor

Sangwook, Vishwesh, and Siham contribute to this project.