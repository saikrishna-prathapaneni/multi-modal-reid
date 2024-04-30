import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def apply_sliding_pooling(matchmap, kernel_size=7, stride=1, pooling_type='max'):
    # Apply either max or average pooling
    if pooling_type == 'max':
        pool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=kernel_size//2)
    else:
        pool = torch.nn.AvgPool1d(kernel_size, stride=stride, padding=kernel_size//2)
    return pool(matchmap)

def normalize_and_threshold(matchmap, p=0.2):
    # Normalize so that values sum to 1
    matchmap = matchmap / matchmap.sum()
    
    # Threshold to keep top p percentage
    k = int(p * matchmap.numel())
    threshold_value = torch.kthvalue(matchmap.view(-1), matchmap.numel() - k).values
    matchmap[matchmap < threshold_value] = 0
    return matchmap

def overlay_map_on_image(map, original_image):
    map_image = Image.fromarray((map.numpy() * 255).astype('uint8'))
    map_image = map_image.convert('L')
    map_image_rgb = map_image.convert('RGB')
    alpha_mask = ImageOps.autocontrast(map_image)
    original_colored = Image.fromarray(original_image.numpy().transpose(1, 2, 0).astype('uint8'), 'RGB')
    blended_image = Image.composite(original_colored, map_image_rgb, alpha_mask)
    return blended_image

def process_and_visualize(matchmap, image, p=0.25):
    # Assume matchmap shape is (7, 7, 128)
    # Temporal dimension is the last one (128)
    matchmap = matchmap.permute(2, 0, 1)  # Change to (128, 7, 7)

    # Apply sliding window pooling
    pooled_maps = apply_sliding_pooling(matchmap, kernel_size=7, pooling_type='max')

    # Process each pooled map
    overlays = []
    for i in range(pooled_maps.shape[0]):
        processed_map = normalize_and_threshold(pooled_maps[i], p=p)
        
        # Resize and normalize for display
        resized_map = TF.resize(processed_map.unsqueeze(0), (224, 224), interpolation=Image.BILINEAR).squeeze(0)
        normalized_map = resized_map / resized_map.max()  # Normalize to [0, 1] for display purposes

        # Overlay on the original image
        overlay_image = overlay_map_on_image(normalized_map, image)
        overlays.append(overlay_image)

        # Display each overlay for demonstration purposes
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay_image)
        plt.title(f"Overlay for Frame {i+1}")
        plt.axis('off')
        plt.show()

# Example usage
image = torch.rand(3, 224, 224)  # Dummy image for demonstration
matchmap = torch.rand(7, 7, 128)  # Dummy matchmap for demonstration

process_and_visualize(matchmap, image)
