import os
import numpy as np
import tifffile as tif
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder")

from lib.datasets.read_ims import Ims_Image
import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import transforms
import torch
from confettii.entropy_helper import entropy_filter
from confettii.rescale_helper import get_hr_mask
from lib.utils.augmentations import  apply_transforms,RandomRotation3D,RandomGaussianBlur3D,RandomBrightness3D,CenterCrop3D
###visor_dataset is the integration of random_visor and fixed_visor

class FixedVisor(Dataset):
    def __init__(self, cfg,return_mask = True,valid=False, label_in_name=False):
        """
        amount : control the amount of data for training
        """
               #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        
        cfg = cfg.DATASET
        self.e5 = cfg.e5
        self.return_mask = return_mask
        self.label_in_name = label_in_name
        self.input_shape = cfg.input_size

        # Set data paths based on configuration and mode
        self.data_path = cfg.e5_data_path_dir if self.e5 else cfg.data_path_dir
        self.valid_data_path = cfg.e5_valid_data_path_dir if self.e5 else cfg.valid_data_path_dir
        self.mask_data_path = cfg.e5_mask_path_dir if self.e5 else cfg.mask_path_dir

        # Determine the file path to use (training or validation)
        current_path = self.valid_data_path if valid else self.data_path

        # Collect files ending with `.tif`
        self.files = [os.path.join(current_path, fname) for fname in os.listdir(current_path) if fname.endswith('.tif')]
        self.mask_files = [os.path.join(self.mask_data_path, fname) for fname in os.listdir(self.mask_data_path) if fname.endswith('.tif')]

        print(f"######init Fixed visor_3d_dataset#####")
        print(f"data_dir:{self.data_path}")
        print(f"mask_dir {self.mask_data_path}")
 

    def __len__(self):
 
        return len(self.files) 


    def __getitem__(self,idx) :

        roi = tif.imread(self.files[idx])
        if self.return_mask:
            if self.label_in_name: 
                # one label for the entrie roi
                file_name = self.files[idx]
                label = int(file_name.split("/")[-1].split("_")[0])
                mask = label
            else:
                mask = tif.imread(self.mask_files[idx])

            return roi, mask

        else:
            return roi,-1




class RandomVisor(Dataset):
    def __init__(self,return_mask, cfg):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5

        cfg=cfg.DATASET
        self.cfg = cfg
        self.return_mask = return_mask
        self.data_path = cfg.e5_raw_img_path if cfg.e5 else cfg.raw_img_path
        self.mask_path = cfg.e5_raw_mask_path if cfg.e5 else cfg.raw_mask_path

        self.ims_vol=Ims_Image(self.data_path,channel=cfg.channel)
        self.return_mask = return_mask
        if return_mask:
            self.lr_mask = tif.imread(self.mask_path)

        self.amount = cfg.amount
        print(f"######init randomvisor dataset #####")
        print(f"amount: {cfg.amount}")


    def __len__(self):
 
        return self.amount


    def __getitem__(self,idx) :

        """
        randomly sample a 3d image cube, get the corresponding upsample maskl
        then decide whether use this cube to train 
        criteri: 1:foreground(entropy >1.8)  2.at leat two label

        then preprocess it and transform it
        """
        cfg= self.cfg
        input_shape = cfg.input_size
        level = cfg.level
        zoom_factor = cfg.zoom_factor
        
        #move self.ims_vol h5py objects (like those used to handle HDF5 files) cannot be pickled, which is required for torch.utils.data.DataLoader when it is configured to use multiple worker processes (num_workers > 0).
        valid_roi=False
        roi=None
        mask=None
        while not valid_roi:
            roi,indexs=self.ims_vol.get_random_roi(filter=entropy_filter(thres=1.8),roi_size=input_shape,level=level,skip_gap=True)
            if self.return_mask: 
                mask = get_hr_mask(self.lr_mask,indexs,input_shape,zoom_factor)
            else:
                mask = 1
            valid_roi= len(np.unique(mask)) >= 2
        return roi , mask 
    

class VisorDataset(Dataset):
    def __init__(self, cfg,valid=False):
        """
        Integrates RandomVisor and FixedVisor into a single dataset class.

        Args:
            cfg: Configuration object with dataset settings.
            use_random (bool): If True, use RandomVisor; otherwise, use FixedVisor.
            return_mask (bool): If True, return both ROI and mask; otherwise, return only ROI.
            preprocess_mode (str): Preprocessing method. Options:
                - "tensor": Transform to tensor only.
                - "norm": Global or instance normalization and transform to tensor.
                - "clip_norm": Global/instance clip, normalization, and transform to tensor.
        """
        self.cfg = cfg.DATASET
        self.preprocess_mode = self.cfg.preprocess_mode
        self.preprocess_instance = self.cfg.preprocess_instance
        self.is_clip = True if self.preprocess_mode == 'clip_norm' else False
        self.return_mask = self.cfg.return_mask
        self.use_random = self.cfg.random
        self.valid = valid

        if self.use_random:
            # Initialize RandomVisor settings
            self.dataset = RandomVisor(self.return_mask,cfg,valid)
        else:
            # Initialize FixedVisor settings
            self.dataset = FixedVisor(cfg,
                                      return_mask=self.return_mask,
                                      valid=valid,
                                      label_in_name=self.cfg.label_in_name)

        rotation, color,blur = cfg.DATASET.transform_type
        self.transformations = self.create_transforms(
                                                 rotation=rotation,
                                                 color=color,
                                                 blur=blur)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_random:
            roi, mask = self.dataset()
        else:
            roi, mask= self.dataset[idx]

        # Preprocessing
        roi =  apply_transforms(roi,self.transformations)
        roi = self.preprocess(roi)
                                                    

        if self.return_mask:
            return roi, mask
        
        return roi

    def preprocess(self, img):
        if self.preprocess_mode == "tensor":
            img = img
        elif self.preprocess_mode == "norm":
            img = self.normalize(img, 
                                 instance= self.cfg.preprocess_instance)

        elif self.preprocess_mode == "clip_norm":
            img = self.clip(img, 
                            clip_low=self.cfg.clip_low,
                            clip_high=self.cfg.clip_high,
                            instance= self.cfg.preprocess_instance)
            img = self.normalize(img, 
                                 instance= self.cfg.preprocess_instance)

        img = self.to_tensor(img)
        img = torch.unsqueeze(img,0)
        return img

    @staticmethod
    def to_tensor(img):
        return torch.tensor(img, dtype=torch.float32)

    @staticmethod
    def normalize(img, instance=False):
        if instance:
            min_val, max_val = img.min(), img.max()
        else:
            min_val, max_val = 0, 1  # Replace with global norm settings if needed
        return (img - min_val) / (max_val - min_val)

    @staticmethod
    def clip(img, instance=False, clip_low=0, clip_high=3000):
        if instance:
            clip_low, clip_high = np.percentile(img, [0.1, 99.9])
        return np.clip(img, clip_low, clip_high)
    

    def create_transforms(self, rotation=True, color=True, blur=True, s=1.0):
        #s=0.5: Brightness and contrast change by ±5% ([0.95, 1.05]).
	    #s=1.0: Brightness and contrast change by ±10% ([0.9, 1.1]).
        

        # Define the transformation list
        # do not support 3d crop
        transform_list = []
        if rotation:
            transform_list.append(RandomRotation3D(lower_limit=-np.pi/6,upper_limit=np.pi/6))
        if color:
            transform_list.append(RandomBrightness3D(brightness_factor_range=(0.9,1.2),clip=self.is_clip,
                                                       low_thres=self.cfg.clip_low,high_noise_threshold=self.cfg.clip_high))
        if blur:
            transform_list.append(RandomGaussianBlur3D(sigma_range=(0.1,1.5)))

        # Add the mandatory crop
        print(f'center crop to shape of {self.cfg.input_size}')
        transform_list.append(CenterCrop3D(crop_shape=self.cfg.input_size))

        # Compose the transformations
        return transform_list 

def get_dataset(cfg):
    dataset = VisorDataset(cfg)
    return dataset
def get_valid_dataset(cfg):
    dataset = VisorDataset(cfg,valid =True)
    return dataset