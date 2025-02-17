#%%
from lib.arch.autoencoder import build_autoencoder_model
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


#%%

#load cfg and load state_dict
cfg = load_cfg("config/autoencoder.yaml")
device = 'cuda'

model = build_autoencoder_model(cfg)
model.conv_out[0] = nn.ConvTranspose3d(32, 1, kernel_size=3,stride=2, padding=1,output_padding=1)
model.to(device)
model.eval()

print(model.conv_out[0])

summary(model,(1,128,128,128))
print(model)





#%%
exp_name= 'testbigger_data6_autoencoder_3layers_64batch'
cpkg_pth = f"out/weights/{exp_name}"
ckpts = sorted(glob(f'out/weights/{exp_name}/Epoch_*.pth'))
ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+)', os.path.basename(x)).group(1)))
print(ckpts[-1])
#load the last ckpt
ckpt = torch.load(ckpts[-1])
weight_dict = ckpt['model']


# Remove "module." from the key name, if it exists
new_weight_dict = {}
for key, value in weight_dict.items():
    new_key = key.replace('module.', '')
    new_weight_dict[new_key] = value

#remove fc1 and fc2 from weight_dict
new_weight_dict = {k: v for k, v in new_weight_dict.items() if not k.startswith(('fc1', 'fc2','contras'))}
for k, v in new_weight_dict.items():
    print(f"{k}, {v.shape}")
#%%
model.load_state_dict(new_weight_dict,strict=False)
#%%

from lib.datasets.visor_dataset import get_valid_dataset,get_dataset
from torch.utils.data import Dataset,DataLoader

#modify the input_size  128,96,64
in_size = 128 
cfg.DATASET.input_size =[in_size,in_size,in_size]
valid_dataset  = get_valid_dataset(cfg)
valid_loader = DataLoader(dataset=valid_dataset, 
                        batch_size= 6, 
                        num_workers=0, 
                        pin_memory = True,
                        drop_last = True
                        )
train_dataset  = get_dataset(cfg)
train_loader = DataLoader(dataset=train_dataset, 
                        batch_size= 6, 
                        num_workers=0, 
                        pin_memory = True,
                        drop_last = True
                        )
#%%
test_save_dir = f'./valid/{exp_name}'
os.makedirs(test_save_dir,exist_ok=True)
loss_fn = nn.L1Loss(reduction='mean')

valid_loss = []
input_images = []
pred_images = []

for input_data in tqdm(valid_loader):
    input_data = input_data.to('cuda')
    with torch.no_grad():
        preds = model(input_data)
    loss = loss_fn(preds, input_data)
    print(f"loss: {loss.item()}")
    valid_loss.append(loss.item())
    preds = preds.detach().cpu().numpy()
    preds = np.squeeze(preds)
    pred_images.append(preds)
    input_data = input_data.detach().cpu().numpy()
    input_data = np.squeeze(input_data)
    input_images.append(input_data)

valid_loss = sum(valid_loss) / len(valid_loss)
print(f" the average valid_loss is {valid_loss}")
input_images = np.concatenate(input_images,axis=0)
pred_images = np.concatenate(pred_images,axis=0)
#%%
#for each img in a batch
for idx in range(6):
    x = input_images[idx]
    re_x = pred_images[idx]
    residual = re_x - x
    print(f"shape is {x.shape},l1 loss is{np.mean(np.abs(residual))}")
    
    #compress to 2d
    x, re_x, residual = map(
                        lambda img: img[63], 
                        [x, re_x,residual]
                        )
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, (data, title) in enumerate(zip(
        [x, re_x,residual],
        ["x", "re_x", "residual"]
    )):
        img = axs[i].imshow(data, cmap='viridis')
        axs[i].set_title(title)
        fig.colorbar(img, ax=axs[i])

    plt.tight_layout()
    plt.show()


    


# %%
