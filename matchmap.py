import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import pickle
import sys
import time

import dataloaders
import models
from steps import train, validate
import warnings
# [Your existing imports, model definitions, and initializations...]

warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError



def get_single_matchmap(val_loader, audio_model, image_model, device = "cuda:2"):

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    audio_model.eval()
    image_model.eval()

    with torch.no_grad():
        # Get a single batch from the validation loader
        image_input, audio_input, _ = next(iter(val_loader))
       
        image_input = image_input.to(device)
        audio_input = audio_input.to(device)

        # Generate embeddings
        audio_embedding = audio_model(audio_input).to('cpu').detach().squeeze()
        image_embedding = image_model(image_input).to('cpu').detach().squeeze()

        # Generate matchmap
        matchmap = computeMatchmap(image_embedding, audio_embedding )
       
        return matchmap

def tensor_to_np_normalized(tensor):
    return tensor.squeeze().cpu().detach().numpy()



def visualize_single_matchmap(val_loader, audio_model, image_model):
    # Compute a single matchmap
    matchmap = get_single_matchmap(val_loader, audio_model, image_model)
    match_misa = matchmapSim(matchmap,'SISA')
    # Convert the matchmap to a numpy array for visualization

    #matchmap_np = tensor_to_np_normalized(match_misa)

    # Load the original image
    image_input, audio_input, _ = next(iter(val_loader))

    #original_image = cv2.imread(image_input)
    image_np = np.array(image_input)
    original_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Create a figure to visualize the matchmap and the original image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show the original image
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')

    ax[0].axis('off')

    # Show the matchmap
    # Normalize the matchmap for better visualization
    #matchmap_viz = np.uint8(255 * (matchmap_np - np.min(matchmap_np)) / (np.max(matchmap_np) - np.min(matchmap_np)))
    # Assuming matchmap_viz is a numpy array
    # Convert matchmap_viz to the appropriate data type (CV_8UC1 or CV_8UC3)
    # matchmap_viz = cv2.normalize(matchmap_viz, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #matchmap_viz = cv2.applyColorMap(matchmap_viz, cv2.COLORMAP_JET)


    ax[1].imshow(match_misa)
    ax[1].set_title('Matchmap')
    ax[1].axis('off')

    plt.show()


if __name__ == "__main__":
    # [Your existing script initialization and model loading code...]
    
    audio_model = models.EnhancedDavenet()
    image_model = models.VGG16(pretrained=True)

    audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % ( "/home/dled/sai/DAVEnet/DAVEnet-pytorch/exp/Data-/AudioModel-Davenet_ImageModel-VGG16_Optim-sgd_LR-0.001_Epochs-50",141 )))
    image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % ( "/home/dled/sai/DAVEnet/DAVEnet-pytorch/exp/Data-/AudioModel-Davenet_ImageModel-VGG16_Optim-sgd_LR-0.001_Epochs-50",141 )))
    # Assuming args.data_val is the path to your validation dataset
    val_loader = torch.utils.data.DataLoader(
        dataloaders.ImageCaptionDataset("/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/", image_conf={'center_crop':True}, type="test"),
        batch_size=1, shuffle=True, num_workers=4)
    visualize_single_matchmap(val_loader, audio_model, image_model)

    
    # Now you can process the matchmap as needed, e.g., save it, visualize it, etc.
