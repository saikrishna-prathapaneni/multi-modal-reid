import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image,ImageOps
import argparse
import os
import pickle
import sys
import time
import soundfile as sf
import dataloaders
import models
from steps import train, validate
import warnings
import torchvision.transforms.functional as TF
import shutil
import matplotlib.cm as cm
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


def visualize_and_save(image, matchmap, audio, text):
    # print(image.shape)
    # original_image = Image.fromarray(image.squeeze().numpy().T, 'RGB')
    # original_image.save(f"original_image.png")

    image = image.squeeze(0)  # Assuming batch size of 1
    num_frames = matchmap.shape[2]
    frame_group_size = 2
  
    # Resize and normalize matchmap
    
    # Overlay and save frames
    for start in range(0, num_frames, frame_group_size):
        end = min(start + frame_group_size, num_frames)
        # Average the matchmaps in the current group
        current_group = matchmap[:, :, start:end]
        averaged_map = current_group.mean(dim=2)
        resized_map = TF.resize(averaged_map.unsqueeze(0), (224, 224), interpolation=Image.BILINEAR).squeeze(0)
        normalized_map = normalize_map(resized_map)

        overlay_image = overlay_map_on_image(normalized_map, image)
        plt.imsave(f"frames/frame{start}.png",overlay_image)

    # Save audio and text
    #save_audio(audio, "audio.wav")
    with open("text.txt", "w") as text_file:
        text_file.write(text[0])

def get_single_matchmap(val_loader, audio_model, image_model, device = "cuda:1"):

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    audio_model.eval()
    image_model.eval()

    with torch.no_grad():
        
        image, audio, _, image_id,_, _, text  = val_loader.dataset.__getitem__(100)
        print(text)
        plt.imsave("original_image.png",image.detach().squeeze().permute(1, 2, 0).numpy())
        image_input = image.unsqueeze(0).to(device)
        audio_input = audio.unsqueeze(0).to(device)
        

        # Generate embeddings
        audio_embedding = audio_model(audio_input).to('cpu').detach().squeeze()
        image_embedding = image_model(image_input).to('cpu').detach().squeeze()
        # Generate matchmap
        matchmap = computeMatchmap(image_embedding, audio_embedding)
        visualize_and_save(image, matchmap, audio, text)
        return matchmap


def normalize_map(map):
    min_val = map.min()
    max_val = map.max()
    normalized_map = (map - min_val) / (max_val - min_val)
    return normalized_map


def overlay_map_on_image(map, original_image):
    # Convert map to a colormap (heatmap)
    heatmap = cm.jet(map.numpy())  # Using 'jet' colormap for heatmap, you can change to other colormaps like 'viridis', 'plasma', etc.
    heatmap_image = np.uint8(heatmap * 255)
    original_image =  original_image *255
    
    # Extract RGB channels and convert to an image using cv2
    map_image_rgb = cv2.cvtColor(heatmap_image[:, :, :3], cv2.COLOR_RGB2BGR)

    # Create alpha mask using the intensity values in the heatmap
    alpha_mask = heatmap_image[:, :, 3]  # Alpha channel of the heatmap
    alpha_mask = cv2.normalize(alpha_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    alpha_mask = cv2.applyColorMap(alpha_mask, cv2.COLORMAP_JET)  # Enhance contrast

    # Convert the original image to BGR if it's a numpy array
    #plt.imsave("prev.jpg", original_image.numpy().transpose(1, 2, 0))
    original_colored = cv2.cvtColor(original_image.numpy().transpose(1, 2, 0).astype('uint8'), cv2.COLOR_RGB2BGR)
   
    # Blend the original image and the heatmap based on the alpha mask
    blended_image = cv2.addWeighted(original_colored, 0.7, map_image_rgb, 0.3, 0, dtype=cv2.CV_8UC3)  # Blend images

    return blended_image

def save_audio(audio, filename):
    sf.write(filename, audio.numpy(), 44100)  # Assuming 4410



def visualize_single_matchmap(val_loader, audio_model, image_model):
    # Compute a single matchmap
    matchmap = get_single_matchmap(val_loader, audio_model, image_model)
    #match_misa = matchmapSim(matchmap,'MISA')
    # Convert the matchmap to a numpy array for visualization



if __name__ == "__main__":
    # [Your existing script initialization and model loading code...]
    if os.path.exists("frames") and os.path.isdir("frames"):
    # Remove the folder and its contents
        shutil.rmtree("frames/")
        os.mkdir("frames")
    if not os.path.exists("frames"):
        os.mkdir("frames")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.different_text_prompt = False
    audio_model = models.EnhancedDavenet()
 
    image_model = models.Resnet50_MOCO()

    audio_model.load_state_dict(torch.load("/home/sp7238/sai/mulit-model-reid/multi-modal-reid/exp/Data-/AudioModel-Davenet_ImageModel-resnet_Optim-sgd_LR-0.001_Epochs-50_tweak_resnet50_final/models/best_audio_model.pth"))
    image_model.load_state_dict(torch.load("/home/sp7238/sai/mulit-model-reid/multi-modal-reid/exp/Data-/AudioModel-Davenet_ImageModel-resnet_Optim-sgd_LR-0.001_Epochs-50_tweak_resnet50_final/models/best_image_model.pth"))
    # Assuming args.data_val is the path to your validation dataset
    val_loader = torch.utils.data.DataLoader(
        dataloaders.ImageCaptionDataset(args, "/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/", image_conf={'center_crop':True}, type="test"),
        batch_size=1, shuffle=True, num_workers=4)
    visualize_single_matchmap(val_loader, audio_model, image_model)

    
    # Now you can process the matchmap as needed, e.g., save it, visualize it, etc.
