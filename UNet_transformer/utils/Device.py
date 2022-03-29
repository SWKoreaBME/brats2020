import torch
from torch.nn import DataParallel
import os


def gpu_setting():
    """Return GPU settings
    """
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    elif torch.cuda.device_count() == 1:
        multi_gpu = False
    else:
        multi_gpu = False
        
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Number of Cuda device: {torch.cuda.device_count()}")
    elif device == "cpu":
        print("Number of Cuda device: None, CPU instead")
        
    return device, multi_gpu
    

def model_dataparallel(model, multi_gpu):
    """Get model and return with the multi-gpu computational device
    """
    if multi_gpu:
        model = DataParallel(model)
    print(f"Model Loaded, Multi GPU: {multi_gpu}")
    return model


def get_device_info():
    num_device = torch.cuda.device_count()
    name_device = torch.cuda.get_device_name()
    cuda = torch.cuda.is_available()

    print(f"\nGPU DEVICE INFO")
    print(f"GPU STATUS: {cuda}")
    print(f"NAME OF DEVICE:\t {name_device}")
    print(f"NUMBER OF DEVICE:\t {num_device}")

    return cuda


if __name__ == '__main__':
    pass
