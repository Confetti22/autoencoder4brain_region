#%%
from lib.arch.cnn_classsify import build_classifier
from lib.utils.load_ckpt import modify_key
from lib.utils.compute_feature_map import TraverseDataset3d,contour_plt,get_feature_list
from lib.utils.augmentations import center_crop_3d

from scipy.ndimage import zoom
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as v2
from config import load_cfg
import numpy as np
from torchsummary import summary
from tqdm.auto import tqdm
import torch
import tifffile as tif
from glob import glob
import re
import os
import torch.nn.functional as F
import time

def load_last_ckpt_dict(model,exp_name,comment=None):
    # ckpts = sorted(glob(f'clf_out/weights/{exp_name}/Epoch_*.pth'))
    # ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+)', os.path.basename(x)).group(1)))
    # ckpt = torch.load(ckpts[-1])
    ckpt_path = "/home/confetti/e5_workspace/autoencoder/clf_out/weights/conv_k553_k33_avepool_hiddenlayer_pretrained_b32_re/Epoch_200.pth"
    ckpt = torch.load(ckpt_path)
    print("load the 200_pth")
    # ckpt = torch.load("out/weights/test_classifer3/Epoch_012best.pth")
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    model.load_state_dict(removed_module_dict,strict = False)
    print(f"ckpt at {exp_name} loaded")

cfg = load_cfg('config/classifier.yaml')
win_size =cfg.DATASET.input_size[0]
stride = 16 
batch_size = 48
exp_name = 'conv_k553_k33_avepool_hiddenlayer_pretrained_b32_re'
save_dir = './clf_out/'

#define model
device ='cuda'
model = build_classifier(cfg)
model.to(device)
model.eval()
print(model)
summary(model,(1,128,128,128))

#load parameters weights
load_last_ckpt_dict(model,exp_name,comment=None)


# iterate all the draw_border_test images
for img_id in range(3,6):

    draw_img_pth = f"/home/confetti/mnt/data/processed/t1779/draw_boder_test/{img_id:04d}.tif"
    image_name = os.path.basename(draw_img_pth).split('.')[0]
    draw_border_img = tif.imread(draw_img_pth)
    # make z depth the same as the win_size
    # draw_border_img = center_crop_3d(draw_border_img,crop_shape=(win_size,*draw_border_img.shape[1:]))

    draw_border_dataset = TraverseDataset3d(draw_border_img,stride=stride,win_size=win_size,net_input_shape=win_size)  
    border_draw_loader = DataLoader(draw_border_dataset,batch_size,shuffle=False,drop_last=False)
    print(f"len of dataset is {len(draw_border_dataset)}")

    current = time.time()
    feats_lst = get_feature_list('cuda',model,border_draw_loader,'fc',save_path=None,apply_softmax=True)
    print(f"extracting feature from image_{img_id} consume {time.time()-current} seconds")

    sample_shape = draw_border_dataset.sample_shape
    feats_map = feats_lst.reshape(*sample_shape,feats_lst.shape[-1])
    print(f"feats_map {feats_map.shape}")
    pro_map1=feats_map[:,:,:,0]
    # pro_map1= pro_map1.squeeze()
    zoomed = zoom(pro_map1,zoom= stride,order=0)
    shape_diff = [ l1 - l2 for l1,l2 in zip(draw_border_img.shape,zoomed.shape)]
    print(shape_diff)
    padding = [(diff // 2, diff - (diff // 2)) for diff in shape_diff]
    padded = np.pad(zoomed, pad_width=padding, mode='constant', constant_values=0)
    # Check the shape of the padded array
    print("Padded shape:", padded.shape)    
    tif.imwrite( f"{save_dir}/{exp_name}_200ckpt_{image_name}__S{stride}W{win_size}P1.tif",padded)
    contour_plt(pro_map1, 'P1', writer=None,save_path=f"{save_dir}/{exp_name}__200ckpt_{image_name}__S{stride}W{win_size}P1.png")




#%%

#%%



#%%

#%%




# %%
