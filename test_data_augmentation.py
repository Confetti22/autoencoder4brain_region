from lib.utils.augmentations import generate_random_angles,generate_rotation_matrix,generate_3d_volume_with_line,rotate_volume,\
                                    random_rotation_3d,center_crop_3d,random_gaussian_blur_3d, random_brightness_3d
import numpy as np
import napari
import tifffile as tif
#%%
# Example of applying this function on a dataset of 3D images:
    
lower_limit = -np.pi/6
upper_limit =  np.pi/6
# Example usage:


angles = (np.pi/6 ,np.pi /4,-np.pi/16)  # Rotation angles: 45°, 30°, 60°
angles2 = generate_random_angles(lower_limit,upper_limit) 
order = 'zyx'  # First Z, then Y, then X rotation
rotation_matrix = generate_rotation_matrix(angles2, order)  # From the earlier function

# Parameters
volume_shape = (50, 50, 50)  # (depth, height, width)
thickness = 3
line_value = 150
background_value = 50

# Generate the volume
volume = generate_3d_volume_with_line(volume_shape, thickness, line_value, background_value)


rotated_volume = rotate_volume(volume, rotation_matrix)
volume = tif.imread('/home/confetti/mnt/data/processed/t1779/100roi/0080.tif')

blurred = random_gaussian_blur_3d(volume,P=1,sigma_range=(0.1,1.5))
adjusted1 = random_brightness_3d(volume,P=1,contrast_s=0,brightness_factor_range=(0.9,1.2))

adjusted2 = random_brightness_3d(volume,P=1,contrast_s=0.05,brightness_factor_range=(0.4,0.5))
adjusted3 = random_brightness_3d(volume,P=1,contrast_s=0.9,brightness_factor_range=(1.4,1.5))
mergered = np.concatenate((adjusted1,adjusted2,adjusted3),axis=1)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(volume,name='volume',colormap='viridis',opacity=0.68)
viewer.add_image(adjusted1,name ='adjusted1',colormap='viridis',opacity=0.68)
viewer.add_image(adjusted2,name ='adjusted2',colormap='viridis',opacity=0.68)
viewer.add_image(adjusted3,name ='adjusted3',colormap='viridis',opacity=0.68)
viewer.add_image(mergered,name ='merged',colormap='viridis',opacity=0.68)
# viewer.add_image(blurred,name='blurred',colormap='viridis',opacity=0.68)

# %%
from config import load_cfg
from torch.utils.data import Dataset,DataLoader
from lib.datasets.visor_dataset import get_dataset
import napari
import matplotlib.pyplot as plt
def hist_plot(img,commnet ='',xlim=None):
    plt.hist(img.flatten(), bins=960, range=(2000,6000 ), color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Histogram of {commnet}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    plt.show()


viewer = napari.Viewer(ndisplay=3)
cfg = load_cfg("config/classifier.yaml")
cfg2= load_cfg("config/no_aug.yaml")
dataset1 = get_dataset(cfg)
dataset2 = get_dataset(cfg2)

xlim =(2000,3500)
for idx in range(4):
    img,_= dataset1[idx]
    img = img.squeeze()
    img = img.numpy()
    hist_plot(img,f'transed_{idx}',xlim)
    viewer.add_image(img,name=f"transed_{idx}")

for idx in range(4):
    img,_ = dataset2[idx]
    viewer.add_image(img,name =f"ori_{idx}")
    hist_plot(img,f'ori_{idx}',xlim)
viewer.grid.enabled = True
viewer.grid.shape = (2,4)

# %%


hist_plot(dataset1[0][0],'ori',xlim)
hist_plot(dataset2[0][0],'transformed',xlim)


# %%
