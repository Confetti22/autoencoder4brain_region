#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Helper function to add pixel values to the plot
def annotate_pixel_values(ax, data):
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f'{data[i, j]:.2f}', 
                    ha='center', va='center', color='white' if data[i, j] < 0.5 else 'black')

# Function to visualize the effect of AvgPool2d with pixel values
def visualize_avgpool(kernel_size, stride, input_tensor):
    # Define AvgPool2d layer
    avgpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    
    # Apply the pooling operation
    output_tensor = avgpool(input_tensor)
    
    # Convert tensors to NumPy arrays for visualization
    input_np = input_tensor.squeeze().numpy()
    output_np = output_tensor.squeeze().numpy()
    
    # Plot input and output
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_np, cmap='viridis')
    axes[0].set_title(f'Input (Shape: {input_np.shape})')
    annotate_pixel_values(axes[0], input_np)  # Annotate pixel values
    
    axes[1].imshow(output_np, cmap='viridis')
    axes[1].set_title(f'Output (Shape: {output_np.shape})\n(kernel={kernel_size}, stride={stride})')
    annotate_pixel_values(axes[1], output_np)  # Annotate pixel values
    
    plt.tight_layout()
    plt.show()

# Generate a sample input tensor (random values)
input_tensor = torch.tensor(np.random.rand(8, 8), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Visualize for different kernel_size and stride combinations
visualize_avgpool(kernel_size=2, stride=2, input_tensor=input_tensor)
visualize_avgpool(kernel_size=3, stride=2, input_tensor=input_tensor)
visualize_avgpool(kernel_size=3, stride=2, input_tensor=input_tensor)
visualize_avgpool(kernel_size=4, stride=1, input_tensor=input_tensor)
# %%
