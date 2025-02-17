#%%
from lib.arch.autoencoder import build_autoencoder_model
from config import load_cfg
import torch
import numpy as np
from torchsummary import summary


activation = {}
def get_activation(name):
    def hook(model, input, output):
        #check for whether registered at last layer of classifier
        activation[name] = output.detach()
    return hook




cfg = load_cfg("config/autoencoder.yaml")

device = 'cuda'

model = build_autoencoder_model(cfg)
model.to(device)
model.eval()

summary(model,(1,128,128,128))
print(model)


test_input = torch.zeros(size=(1,1,128,128,128))
test_input[:,:,64,64,64] = 1
test_input = test_input.to(device)

#register forward hook at the destination layer
extract_layer_name ='down_1'
hook1 = model.down_layers[1][0][0].register_forward_hook(get_activation(extract_layer_name))

out = model(test_input)
#check the shape of feats acquired by forward hook
feats = activation[extract_layer_name].cpu().detach().numpy()
feats=np.squeeze(feats)

#visulize the activation map via summary operation like std,mean
feats_std=np.std(feats,axis=0)
print(f"shape of feats is {feats.shape}")


import matplotlib.pyplot as plt

print(f"visualization of the model {cfg.EXP_NAME}")
test_input = np.squeeze(test_input.cpu().detach().numpy())
fig,axes=plt.subplots(1,2)
axes[0].imshow(test_input[int(test_input.shape[0]//2),:,:])
axes[0].set_title('input')

axes[1].imshow(feats_std[int(feats_std.shape[0]//2),:,:])
axes[1].set_title(f'{extract_layer_name}')


hook1.remove()






# %%

# %%

# %%