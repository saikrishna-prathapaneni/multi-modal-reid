import math
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def calc_recalls(image_outputs, audio_outputs, nframes,id_output, simtype='MISA'):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls


def save_top_images(audio_index, A2I_ind, val_loader, save_dir ="res"):
    """
    Saves the top 10 images for a given audio file, embedding the images with their actual IDs.
    
    Parameters:
    - audio_index: The index of the audio file in the batch.
    - A2I_ind: Indices of the top images for each audio from calc_recalls.
    - val_loader: The validation loader containing the dataset.
    - save_dir: Directory where the images will be saved.
    """
    # Ensure the save directory exists, create if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the indices of the top 10 images for the specific audio
    top_image_indices = A2I_ind[:, audio_index].tolist()

    # Plotting and saving
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for idx, ax in enumerate(axes.flatten()):
        # Use __getitem__ to load the image and its corresponding id
        image, _, _, image_id = val_loader.dataset.__getitem__(top_image_indices[idx])
        
        ax.imshow(image.permute(1, 2, 0))  # Adjust for channel order if necessary
        # Display the ID on the image
        ax.text(0.5, -0.15, f"ID: {image_id}", size=12, ha="center", transform=ax.transAxes, color='white')
        ax.set_title(f"ID: {image_id}", size=12)
        ax.axis('off')
    
    # Save the figure
    save_path = os.path.join(save_dir, f"top_images_for_audio_{audio_index}.png")
    plt.savefig(save_path)
    plt.close(fig) 


def calc_recalls_reid(image_outputs, audio_outputs, nframes, id_output, val_loader, visualise_id = 100, simtype='MISA'):
    """
    Computes recall at 1, 5, and 10 given encoded image and audio outputs.
    """
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(20, 0)
    I2A_scores, I2A_ind = S.topk(20, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    A_r20= AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    I_r20 = AverageMeter()

    # Create a mapping from index to id_output
    index_to_id = {index: int(id_val) for index, id_val in enumerate(id_output)}
    
    # Create a reverse mapping from id_output to all indices that share the same id
 
    id_to_indices = {}
    for index, id_val in enumerate(id_output):
    # Ensure the key is an integer
        key = int(id_val) if isinstance(id_val, torch.Tensor) else id_val
        if key not in id_to_indices:
            id_to_indices[key] = []
        id_to_indices[key].append(index)

    
    for i in range(n):
        # Get all target indices that share the same ID as the query index
        target_indices = id_to_indices[int(index_to_id[i])]
        #save_top_images(i, A2I_ind, val_loader)
        # Initialize variables to track if a match is found within top 1, 5, and 10
        I_match_at_1 = False
        I_match_at_5 = False
        I_match_at_10 = False
        I_match_at_20 = False

        A_match_at_1 = False
        A_match_at_5 = False
        A_match_at_10 = False
        A_match_at_20 = False

        for ind in range(20):
            if A2I_ind[ind, i] in target_indices:
                if ind == 0:
                    I_match_at_1 = True
                if ind < 5:
                    I_match_at_5 = True
                if ind < 10:
                    I_match_at_10 = True
                if ind < 20:
                    I_match_at_20 = True

            if I2A_ind[i, ind] in target_indices:
                if ind == 0:
                    A_match_at_1 = True
                if ind < 5:
                    A_match_at_5 = True
                if ind < 10:    
                    A_match_at_10 = True
                if ind < 20:
                    A_match_at_20 = True

        # Update recalls based on found matches
        A_r1.update(1 if A_match_at_1 else 0)
        A_r5.update(1 if A_match_at_5 else 0)
        A_r10.update(1 if A_match_at_10 else 0)
        A_r20.update(1 if A_match_at_20 else 0)

        I_r1.update(1 if I_match_at_1 else 0)
        I_r5.update(1 if I_match_at_5 else 0)
        I_r10.update(1 if I_match_at_10 else 0)
        I_r20.update(1 if I_match_at_20 else 0)

    recalls = {
        'A_r1': A_r1.avg, 'A_r5': A_r5.avg, 'A_r10': A_r10.avg,'A_r20': A_r20.avg,
        'I_r1': I_r1.avg, 'I_r5': I_r5.avg, 'I_r10': I_r10.avg, 'I_r20': I_r20.avg
    }

    return recalls


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

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes,id, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i or id[I_imp_ind] == id[i]: #tweak for reid
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i  or id[A_imp_ind] == id[i]: # tweak for reid
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
