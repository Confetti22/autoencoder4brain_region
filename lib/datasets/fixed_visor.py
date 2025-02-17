import sys
sys.path.append("/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder")

import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os

class FixedVisor(Dataset):
    def __init__(self, cfg,valid=False):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        self.e5 = cfg.SYSTEM.e5
        self.input_shape = cfg.DATASET.input_size

        # Set data paths based on configuration and mode
        self.data_path = cfg.DATASET.e5_data_path_dir if self.e5 else cfg.DATASET.data_path_dir
        self.valid_data_path = cfg.DATASET.e5_valid_data_path_dir if self.e5 else cfg.DATASET.valid_data_path_dir

        self.g_clip_low = cfg.PREPROCESS.clip_low
        self.g_clip_high = cfg.PREPROCESS.clip_high
        self.is_norm = cfg.PREPROCESS.NORM
        self.valid = valid

        # Determine the file path to use (training or validation)
        current_path = self.valid_data_path if self.valid else self.data_path

        # Collect files ending with `.tif`
        self.files = [os.path.join(current_path, fname) for fname in os.listdir(current_path) if fname.endswith('.tif')]

        print(f"######init visor_3d_dataset#####")

    def __len__(self):
 
        return len(self.files) 


    def __getitem__(self,idx) :

        """
        randomly sample a 3d image cube, get the corresponding upsample maskl
        then decide whether use this cube to train 

        then preprocess it and transform it
        """
        roi = tif.imread(self.files[idx])
        roi = np.array(roi).astype(np.float32) 

        # if self.is_norm:
        #     roi = self.clip_norm(roi,clip_low=self.clip_low,clip_high=self.clip_high)
        # else:
        #     roi =self.clip(roi,clip_low=self.clip_low,clip_high=self.clip_high)
        roi=self.tran2tensor(roi)
        roi=torch.unsqueeze(roi,0)

        return roi, 1
    

    @staticmethod
    def clip_norm(img,clip_low = 0 ,clip_high=3000):
        """
        first clip the image to percentiles [clip_low, clip_high]
        second min_max normalize the image to [0,1]
        """
        # input img nparray [0,65535]
        # output img tensor [0,1]
        clipped_arr = np.clip(img,clip_low,clip_high) 
        min_value = clip_low 
        max_value = clip_high 

        img = (clipped_arr-min_value)/(max_value-min_value)

        img = img.astype(np.float32)

        return img
    
    @staticmethod
    def clip(img,clip_low = 0 ,clip_high=3000):
        """
        clip the image to percentiles [clip_low, clip_high]
        """
        img = np.clip(img,clip_low,clip_high) 
        img = img.astype(np.float32)
        return img

    @staticmethod 
    def tran2tensor(img):
        #using no augmentation at all
        trans=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale = False),
            ]
        )
        transed_img=trans(img)
        return transed_img




def get_dataset(args):

    # === Get Dataset === #
    train_dataset = FixedVisor(args)

    return train_dataset

def get_valid_dataset(args):

    # === Get Dataset === #
    train_dataset = FixedVisor(args,valid=True)

    return train_dataset

if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(tran_dataset.input_shape)
    print(f"success")
