#%%
from lib.arch.contrastive_net import build_contrastive_net
from config import load_cfg
import  matplotlib.pyplot  as plt
import numpy as np
from torchsummary import summary
from tqdm.auto import tqdm
import torch.nn as nn
import torch
import tifffile as tif
from glob import glob
import re
import os

activation = {}
def get_activation(name):
    def hook(model, input, output):
        #check for whether registered at last layer of classifier
        activation[name] = output.detach()
    return hook

#%%
#load cfg and load state_dict
cfg = load_cfg("config/contrastive_net.yaml")
device = 'cuda'

model = build_contrastive_net(cfg)
model.to(device)
model.eval()

summary(model,(1,128,128,128))
print(model)
#%%
import time
current= time.time()

def check_state_dict(state_dict):
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")

exp_name= 'testbigger_data6_autoencoder_3layers_64batch'
cpkg_pth = f"out/weights/{exp_name}"
ckpts = sorted(glob(f'out/weights/{exp_name}/Epoch_*.pth'))
ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1)))
print(ckpts[-1])
print(f"sort time:{time.time() - current}")
current = time.time()

#load the last ckpt
ckpt = torch.load(ckpts[-1])
print(f"load ckpt time:{time.time() - current}")
current = time.time()

weight_dict = ckpt['model']

def modify_key(weight_dict,source,target):
    new_weight_dict = {}
    for key, value in weight_dict.items():
        new_key = key.replace(source,target)
        new_weight_dict[new_key] = value
    return new_weight_dict
# Remove "module." from the key name, if it exists
removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
print("#####afeter remove module.")
check_state_dict(removed_module_dict)

#remove fc1 and fc2 from weight_dict
def delete_key(weight_dict,pattern_lst):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))
print("#####afeter remove unwanted layers.")
check_state_dict(deleted_unwanted_dict)


#check



#%%
model.encoder.load_state_dict(deleted_unwanted_dict,strict=False)

# %%
