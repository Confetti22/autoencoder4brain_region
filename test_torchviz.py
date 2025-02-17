#%%
from lib.arch.cnn_classsify import build_classifier
from lib.utils.load_ckpt import modify_key
from lib.utils.augmentations import center_crop_3d

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

import torch.nn as nn
import torch
from torchviz import make_dot
def load_ckpt_dict(model,exp_name,comment=None):


    ckpts = sorted(glob(f'clf_out/weights/{exp_name}/Epoch_*.pth'))
    ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+)', os.path.basename(x)).group(1)))
    ckpt = torch.load(ckpts[-1])
    # ckpt = torch.load("out/weights/test_classifer3/Epoch_012best.pth")
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    model.load_state_dict(removed_module_dict,strict = False)
    print(f"ckpt at {exp_name} loaded")

def delete_key(weight_dict,pattern_lst):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

def load_cnn_encoder_dict(model,exp_name):
    ckpts = sorted(glob(f'out/weights/{exp_name}/Epoch_*.pth'))
    ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1)))
    ckpt = torch.load(ckpts[-1])
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))

    model.cnn.load_state_dict(deleted_unwanted_dict,strict=False)
#%%


cfg = load_cfg('config/classifier.yaml')
model = build_classifier(cfg)
model.eval()

load_cnn_encoder_dict(model,'testbigger_data6_autoencoder_3layers_64batch')

x = torch.randn(1,1,128,128,128)

y = model(x)
print(y.shape)
print(y)
ret = y.mean()
print(ret.shape)
print(ret)
#%%
#visulaize the compute graph
# make_dot(y, params=dict(model.named_parameters()))
# %%


exp_name = 'test2_clf_pretrained_'

cfg = load_cfg('config/classifier.yaml')
model2 = build_classifier(cfg)
model2.eval()

load_ckpt_dict(model2,exp_name,comment=None)

print(model.cnn.conv_in[0].weight.shape)
print(model.cnn.conv_in[0].weight.mean())
print(model2.cnn.conv_in[0].weight.mean())

#%%
import matplotlib.pyplot as plt
import numpy as np
import math

# Calculate the grid size based on the number of filters
num_filters = model.cnn.conv_in[0].weight.shape[0]
grid_size = int(math.ceil(math.sqrt(num_filters)))

# Create a figure with a grid of subplots
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Loop through each filter and plot it
for idx in range(num_filters):
    z_length = model.cnn.conv_in[0].weight.shape[2]
    k = model.cnn.conv_in[0].weight[idx, :, int(z_length//2), :, :]
    k = k.cpu().detach().numpy().squeeze()
    axes[idx].imshow(k, cmap='viridis')
    axes[idx].axis('off')  # Turn off axes for better visualization

# Turn off unused subplots
for idx in range(num_filters, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# %%
