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


def configure_layer(layer):
    """
    Configures the ConvTranspose2d layer to have weight=1 and bias=80.
    """
    with torch.no_grad():
        layer.weight.fill_(1)  # Set all weights to 1
        layer.bias.fill_(0)  # Set all biases to 80

# Function to visualize the effect of AvgPool2d with pixel values
def visualize_transconv(kernel_size, stride,input_tensor, padding=0,output_padding=0):
    # Define AvgPool2d layer
    transconv = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=kernel_size,
                                   stride=stride,padding=padding,output_padding=output_padding)

    configure_layer(transconv)
    
    # Apply the pooling operation
    with torch.no_grad():
        output_tensor = transconv(input_tensor)
    
    # Convert tensors to NumPy arrays for visualization
    input_np = input_tensor.squeeze().numpy()
    output_np = output_tensor.squeeze().numpy()
    
    # Plot input and output
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_np, cmap='viridis')
    axes[0].set_title(f'Input (Shape: {input_np.shape})')
    annotate_pixel_values(axes[0], input_np)  # Annotate pixel values
    
    axes[1].imshow(output_np, cmap='viridis')
    axes[1].set_title(f'Output (Shape: {output_np.shape})\n(kernel={kernel_size}, stride={stride}, pad={padding},output_pad={output_padding})')
    annotate_pixel_values(axes[1], output_np)  # Annotate pixel values
    
    plt.tight_layout()
    plt.show()

# Generate a sample input tensor (random values)
input_tensor = torch.arange(1,17, dtype=torch.float32).view(4,4).unsqueeze(0).unsqueeze(0)
input_tensor.requires_grad = False

# Visualize for different kernel_size and stride combinations
visualize_transconv(kernel_size=5, stride=2, input_tensor=input_tensor)
visualize_transconv(kernel_size=5, stride=2, input_tensor=input_tensor,padding=2,output_padding=1)
visualize_transconv(kernel_size=5, stride=2, input_tensor=input_tensor,output_padding=1)
visualize_transconv(kernel_size=5, stride=2, input_tensor=input_tensor)
# %%
