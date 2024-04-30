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

def get_association(I, A, audio_index, val_loader, save_dir ="res"):
    pass


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
        image, _, _, image_id,_, _, _ = val_loader.dataset.__getitem__(top_image_indices[idx])
        #image_input, audio_input, nframes, id, input_ids, attention_mask, text
        ax.imshow(image.permute(1, 2, 0))  # Adjust for channel order if necessary
        # Display the ID on the image
        ax.text(0.5, -0.15, f"ID: {image_id}", size=12, ha="center", transform=ax.transAxes, color='white')
        ax.set_title(f"ID: {image_id}", size=12)
        ax.axis('off')
    
    # Save the figure
    save_path = os.path.join(save_dir, f"top_images_for_audio_{audio_index}.png")
    plt.savefig(save_path)
    plt.close(fig) 


def calc_recalls_reid(image_outputs, modal_outputs, nframes, id_output, val_loader,type = "text", visualise_id = 100, simtype='MISA'):
    """
    Computes recall at 1, 5, and 10 given encoded image and audio outputs.
    """
    S = compute_matchmap_similarity_matrix(image_outputs, modal_outputs, nframes,type, simtype=simtype)
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


def computeMatchmapAudio(I, A):
    """
    compute matchmap betweem Image features and Audio features

    """
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t() # flatten features  and transpose (1024,7,7) -> (49,1024)  and A would have the feature of (1024,44)
    matchmap = torch.mm(Ir, A) #(49,44)
    matchmap = matchmap.view(H, W, T) # (7,7,44)
    return matchmap


def computeMatchmapText(I, T):
    """
    compute matchmap betweem Image features and Text features

    """
    assert(I.dim() == 3)
    assert(T.dim() == 1)

    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    #T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t() # flatten features  and transpose (1024,7,7) -> (49,1024)  and T would have the feature of (1024,1)
    matchmap = torch.mm(Ir, T.unsqueeze(1)) #(49,1) since only CLS token/average of vectors is considered for the computation
    matchmap = matchmap.view(H, W, 1) # (7,7,1)
    return matchmap
    


def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0) #(7,44) 
        M_maxHW, _ = M_maxH.max(0) #(44)
        return M_maxHW.mean() #mean across 44 dim values
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(args, image_outputs, audio_outputs,
                              text_outputs, nframes, id, alpha, beta, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    if args.use_text_backbone:
        assert(text_outputs.dim() == 2)

    n = image_outputs.size(0)
    loss_total = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    
    
    if not args.use_text_backbone: # loss computation with audio and vision
        
        for i in range(n):
            I_imp_ind = i
            A_imp_ind = i
            attempt_counter_I = 0
            while I_imp_ind == i or id[I_imp_ind] == id[i] and attempt_counter_I < 1000: #tweak for reid
                I_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
            attempt_counter_I = 0
            while A_imp_ind == i  or id[A_imp_ind] == id[i] and attempt_counter_I < 1000: #tweak for reid
                A_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
            
            nF = nframes[i]
            nFimp = nframes[A_imp_ind]
            anchorsim = matchmapSim(computeMatchmapAudio(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
            Iimpsim = matchmapSim(computeMatchmapAudio(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
            Aimpsim = matchmapSim(computeMatchmapAudio(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
            A2I_simdif = margin + Iimpsim - anchorsim
            if (A2I_simdif.data > 0).all():
                loss_total = loss_total + A2I_simdif
            I2A_simdif = margin + Aimpsim - anchorsim
            if (I2A_simdif.data > 0).all():
                loss_total = loss_total + I2A_simdif
        loss_total = loss_total / n
        return loss_total
    else:  # loss computation with audio, vision and text
        
        # for audio data
        loss_audio = torch.zeros(1, device=image_outputs.device, requires_grad=True)
        loss_text = torch.zeros(1, device=image_outputs.device, requires_grad=True)
        
        for i in range(n):
            I_imp_ind = i
            A_imp_ind = i
            
            attempt_counter_I = 0
            while I_imp_ind == i or id[I_imp_ind] == id[i] and attempt_counter_I < 1000: #tweak for reid
                
                I_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
                if attempt_counter_I ==1000:
                    break

            
            attempt_counter_I = 0
            while A_imp_ind == i  or id[A_imp_ind] == id[i] and attempt_counter_I < 1000: # tweak for reid
                A_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
                if attempt_counter_I ==1000:
                    break

            
            # for audio
            nF = nframes[i]
            nFimp = nframes[A_imp_ind]
            anchorsim = matchmapSim(computeMatchmapAudio(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
            Iimpsim = matchmapSim(computeMatchmapAudio(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
            Aimpsim = matchmapSim(computeMatchmapAudio(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)

            A2I_simdif = margin + Iimpsim - anchorsim
            if (A2I_simdif.data > 0).all():
                loss_audio = loss_audio + A2I_simdif
            I2A_simdif = margin + Aimpsim - anchorsim
            if (I2A_simdif.data > 0).all():
                loss_audio = loss_audio + I2A_simdif        
        loss_audio = loss_audio / n
        # for text
        
        for i in range(n):
            I_imp_ind = i
            T_imp_ind = i
            attempt_counter_I = 0
            while I_imp_ind == i or id[I_imp_ind] == id[i] and attempt_counter_I < 1000: #tweak for reid
                I_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
                if attempt_counter_I ==1000:
                    break
            
            attempt_counter_I = 0
            while T_imp_ind == i  or id[T_imp_ind] == id[i] and attempt_counter_I < 1000: # tweak for reid
                T_imp_ind = np.random.randint(0, n)
                attempt_counter_I += 1
                if attempt_counter_I ==1000:
                    break
        

            anchorsim = matchmapSim(computeMatchmapText(image_outputs[i], text_outputs[i]), simtype)
            Iimpsim = matchmapSim(computeMatchmapText(image_outputs[I_imp_ind], text_outputs[i]), simtype)
            Timpsim = matchmapSim(computeMatchmapText(image_outputs[i], text_outputs[T_imp_ind]), simtype)

            T2I_simdif = margin + Iimpsim - anchorsim
            if (T2I_simdif.data > 0).all():
                loss_text = loss_text + T2I_simdif
            I2T_simdif = margin + Timpsim - anchorsim
            if (I2T_simdif.data > 0).all():
                loss_text = loss_text + I2T_simdif        
        loss_text = loss_text / n
        if args.use_alpha_beta:
            loss_total = args.alpha*loss_audio + args.beta*loss_text
        else:
            loss_total = loss_audio + loss_text
    return loss_total

def compute_matchmap_similarity_matrix(image_outputs, modal_outputs, nframes, type, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    if type =="audio":
        # for generating audio based retrieval system
        assert(image_outputs.dim() == 4)
        assert(modal_outputs.dim() == 3)
        n = image_outputs.size(0)
        S = torch.zeros(n, n, device=image_outputs.device)
        for image_idx in range(n):
            
                for audio_idx in range(n):
                    nF = max(1, nframes[audio_idx])
                    S[image_idx, audio_idx] = matchmapSim(computeMatchmapAudio(image_outputs[image_idx], modal_outputs[audio_idx][:, 0:nF]), simtype)
        return S
    else:
        # for generating text based retrieval system
        assert(image_outputs.dim() == 4)
        
        n = image_outputs.size(0)
        S = torch.zeros(n, n, device=image_outputs.device)
        for image_idx in range(n):
            
                for text_idx in range(n):
                    #nF = max(1, nframes[audio_idx])
                    S[image_idx, text_idx] = matchmapSim(computeMatchmapText(image_outputs[image_idx], modal_outputs[text_idx]), simtype)
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
    return prog, epoch-2, global_step, best_epoch, best_avg_r10
