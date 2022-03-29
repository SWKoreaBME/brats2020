import torch.utils.data as data
import torch
from tqdm import tqdm
import cv2
import h5py
from pathlib import Path
import pandas as pd
import numpy as np

class BRATS2020_2D(data.Dataset):
    ''' Gets the 2D slices of the BRATS dataset'''
    def __init__(self,
                path="/home/ramanav/projects/rrg-amartel/ramanav/Downloads/BraTS2020_training_data",
                mode="training",
                normalization="mean",
                transform=None):
        '''
        path: (str) Path to BRATS2020 dataset
        mode: (str) Select one of training/validation/test
        normalization: (str) minmax for min-max stratergy , mean for z-normalization
        transform: (albumtentations.core.transforms) Transform object
        '''
        super(BRATS2020_2D,self).__init__()
        self.mode = mode
        self.path = Path(path)
        self.transform = transform
        self.normalization = normalization
        
        #Get the path list
        self._get_all_paths()

    def __len__(self):
        return len(self.master)
    
    def __getitem__(self,index):
        filename = self.master.iloc[index]
        path = Path(filename.slice_path[1:])

        with h5py.File(str(self.path/path), "r") as f:
            # List all groups
            a_group_key = list(f.keys())
            # Get the data
            image = np.array(f.get(a_group_key[0]))
            mask = np.array(f.get(a_group_key[1]))
            #Since there are 4 classes constructing 4 masks
            h,w,_ = mask.shape
            mask = mask.astype(np.int32)
            mask = np.dstack((mask,np.zeros((h,w,1))))
            #If the first three classes are not present i.e 0,1,2 then put it into rest class which is at the end
            mask[:,:,-1] = mask[:,:,-1] + np.abs(np.sum(mask,axis=-1)-1)
            #Process the mask further to convert the label into different label space
            mask = self._convert_label_to_brats(mask)

        if self.transform!=None:
            image = image.astype(np.float32)
            #Converts into 4 binary masks for augmentation
            mask_all = [mask[:,:,j] for j in range(mask.shape[-1])]
            out = self.transform(image= image,masks=mask_all) 
            return out["image"], torch.Tensor(np.stack(out["masks"]))
        else:
            return image,mask
    
    def _get_all_paths(self):
        '''
        Gets paths for training/validation/testing
        '''
        all_paths = pd.read_csv(str(self.path / Path("content/data/meta_data.csv")))
        if self.mode=="training":
            self.master = all_paths.loc[all_paths.volume<=260]
        elif self.mode=="validation":
            self.master = all_paths.loc[(all_paths.volume>260) & (all_paths.volume<=300)]
        elif self.mode=="testing":
            self.master = all_paths.loc[all_paths.volume>300]
        else:
            raise ValueError(f"Mode {self.mode} is not a valid input")
        self.master.reset_index(drop=True, inplace=True)
    def _get_normalize(self,x,a,b):
        '''
        Normalizes the slices based on given methodology
        '''
        if self.normalization=="minmax":
            """
            b:np.array = max(volume)
            a:np.array = min(volume)
            """
            return (x-a)/(b-a)

        elif self.normalization=="mean":
            raise NotImplementedError
        else:
            raise ValueError(f"Normalization {self.normalization} is not a valid input")

    def _convert_label_to_brats(self,arr):
        """
        arr: HxWxC
        Converts the label space [0:'non-enhancing tumor core',1:'peritumoral edema',2:'GD-enhancing tumor',3:'background'] -> 
        The possible classes are
        TC (Tumor core) --> 0, 2,
        WT (Whole tumor) --> 0, 1, 2
        ET (Enhancing tumor) --> 2
        """
        result = []
        arr_b = np.argmax(arr,axis=-1)
        result.append((arr_b==3)*1.0)
        result.append((np.logical_or(arr_b==0,arr_b==2))*1.0)
        result.append((
            np.logical_or(
                np.logical_or(arr_b == 2, arr_b == 0), arr_b == 1
            )
        )*1.0)
        result.append((arr_b==2)*1.0)
        return np.stack(result,axis=-1)

    
    
    def read_data(self):
        print("Loading the data...")
        DATA = []
        for i in tqdm(range(len(self.master_path))):
            path = self.master_path[i][:-1]
            DATA.append(cv2.imread(path,cv2.IMREAD_GRAYSCALE))
        return DATA
