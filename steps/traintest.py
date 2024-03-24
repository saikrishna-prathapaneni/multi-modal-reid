import os
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging

from .util import *

def train(audio_model, image_model, text_model, train_loader, test_loader, args, wandb_config= None):

    # args.lr = wandb_config.lr
    # args.optim = wandb_config.optim
    # args.lr-decay = wandb_config.lr-decay
    # args.momentum = wandb_config.momentum
    # args.weight-decay = wandb_config.momentum
    # args.different_text_prompt = wandb_config.different_text_prompt
    # args.use_alpha_beta = wandb_config.use_alpha_beta
    # args.alpha = wandb_config.alpha
    # args.beta = wandb_config.beta





    logging.info("Training started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)


    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    text_model = text_model.to(device)

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        if args.use_text_backbone:
            text_model.load_state_dict(torch.load("%s/models/text_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)
    
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     audio_model = nn.DataParallel(audio_model)
    #     image_model = nn.DataParallel(image_model)
    #     if args.use_text_backbone:
    #         text_model = nn.DataParallel(text_model)


    alpha = torch.tensor(.5, requires_grad= True)
    beta= torch.tensor(.5, requires_grad= True)

    # if args.use_text_backbone: # weight mulitpliers for final loss computation
    #     alpha = torch.nn.Parameter(alpha)
    #     beta = torch.nn.Parameter(beta)
    
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    text_trainables = [p for p in text_model.parameters() if p.requires_grad]
    
    other_trainables = audio_trainables + image_trainables + text_trainables

    # Learning rates
    # alpha_lr = args.lr/100  # learning rate for alpha
    # beta_lr = args.lr/100   # learning rate for beta
    default_lr = args.lr  # Default learning rate for other parameters

    # # Group parameters
    # if args.use_alpha_beta:
    #     param_groups = [
    #         {'params': [alpha], 'lr': alpha_lr},
    #         {'params': [beta], 'lr': beta_lr},
    #         {'params': other_trainables, 'lr': default_lr}
    #         ]
    # else:
    param_groups=[
                {'params': other_trainables, 'lr': default_lr}
        ]

    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(param_groups, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    elif args.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(param_groups, args.lr,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    text_model.train()
    while True:

        if epoch ==30:
            break
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        
        for i, (image_input, audio_input, nframes, id, input_ids, attention_mask) in enumerate(train_loader):
            #logging.info(f'Epoch: {epoch}, Step: {global_step}, Loss: {loss_meter.avg}')
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)
 
            audio_input = audio_input.to(device)
            image_input = image_input.to(device)
            if args.use_text_backbone:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if input_ids.dim() > 2:
                    input_ids = input_ids.view(-1, input_ids.size(-1))
                if attention_mask.dim() > 2:
                    attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            optimizer.zero_grad()

            audio_output = audio_model(audio_input)
            image_output = image_model(image_input)
            text_output = None
            if args.use_text_backbone:
                text_output = text_model([input_ids, attention_mask])
            
            # print("text feature shape: ",text_output.shape)
            # print("image feature shape: ",image_output.shape)
            # print("audio feature shape: ",audio_output.shape)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            pooling_ratio = torch.tensor(pooling_ratio, dtype=torch.float32)
            nframes = nframes.to(torch.float32)
            nframes = torch.round(nframes / pooling_ratio).to(torch.int64)
            loss = sampled_margin_rank_loss(args, image_output, audio_output, text_output,
                nframes, id,alpha, beta, margin=args.margin, simtype=args.simtype)

            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                if wandb_config:
                    wandb_config.log({"step": global_step, "train_loss": loss_meter.val, "batch_time": batch_time.avg})
                
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter))
                
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
       
        
        
        recalls = validate(audio_model, image_model, text_model, test_loader, args, epoch =epoch, wandb_config = wandb_config )
        
        avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

        if wandb_config:
            wandb_config.log({"epoch": epoch, "train_loss": loss_meter.avg, "batch_time": batch_time.avg, "ar10": avg_acc})

        
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            torch.save(audio_model.state_dict(),
                "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(image_model.state_dict(),
                    "%s/models/image_model.%d.pth" % (exp_dir, epoch))
            torch.save(text_model.state_dict(),
                    "%s/models/text_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

            shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_audio_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_image_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/text_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_text_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1

def validate(audio_model, image_model, text_model, val_loader, args, epoch, wandb_config= None, visualise_id = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    text_model = text_model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     audio_model = nn.DataParallel(audio_model)
    #     image_model = nn.DataParallel(image_model)
    # if not isinstance(audio_model, torch.nn.DataParallel):
    # audio_model = nn.DataParallel(audio_model)
    # # if not isinstance(image_model, torch.nn.DataParallel):
    # image_model = nn.DataParallel(image_model)
    
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()
    text_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    print(N_examples)
    I_embeddings = [] 
    A_embeddings = [] 
    T_embeddings = []
    frame_counts = []
    ids = []
    print("validation started")
    with torch.no_grad():
        for i, (image_input, audio_input, nframes, id, input_ids, attention_mask) in enumerate(val_loader):
            
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            if args.use_text_backbone:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if input_ids.dim() > 2:
                    input_ids = input_ids.view(-1, input_ids.size(-1))
                if attention_mask.dim() > 2:
                    attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            

            audio_output = audio_model(audio_input)
            image_output = image_model(image_input)
            text_output = None
            if args.use_text_backbone:
                text_output = text_model([input_ids, attention_mask])
            # compute output

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()
            text_output = text_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            T_embeddings.append(text_output)

            total_memory, used_memory, free_memory = map(
                int, os.popen('free -t -m').readlines()[-1].split()[1:])
            
            # Memory usage
            print("RAM memory % used:", total_memory, used_memory, free_memory, i)
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            pooling_ratio = torch.tensor(pooling_ratio, dtype=torch.float32)
            nframes = nframes.to(torch.float32)
            nframes = torch.round(nframes / pooling_ratio).to(torch.int64)
            ids.append(id)

            frame_counts.append(nframes.cpu())

            batch_time.update(time.time() - end)
            end = time.time()
        print("calculating recall")
        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        text_output = torch.cat(T_embeddings)
        id_output = torch.cat(ids)
        nframes = torch.cat(frame_counts)
        
        if args.recall_type =='topn':
            recalls = calc_recalls(image_output, audio_output, nframes, id_output, val_loader, visualise_id = 10, simtype=args.simtype)
            A_r10 = recalls['A_r10']
            I_r10 = recalls['I_r10']
            A_r5 = recalls['A_r5']
            I_r5 = recalls['I_r5']
            A_r1 = recalls['A_r1']
            I_r1 = recalls['I_r1']
            print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
            print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
            print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)
        else:
            recalls =calc_recalls_reid(image_output, audio_output, nframes, id_output, val_loader,type ="audio", visualise_id = 10, simtype=args.simtype)
            A_r20 = recalls['A_r20']
            I_r20 = recalls['I_r20']
            A_r10 = recalls['A_r10']
            I_r10 = recalls['I_r10']
            A_r5 = recalls['A_r5']
            I_r5 = recalls['I_r5']
            A_r1 = recalls['A_r1']
            I_r1 = recalls['I_r1']
            print(' * Audio R@20 {A_r20:.3f} Image R@20 {I_r20:.3f} over {N:d} validation pairs'
                .format(A_r20=A_r20, I_r20=I_r20, N=N_examples), flush=True)
            print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
            print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
            print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)
            logging.info(' * Audio R@20 {A_r20:.3f} Image R@20 {I_r20:.3f} over {N:d} validation pairs'
                .format(A_r20=A_r20, I_r20=I_r20, N=N_examples))
            logging.info(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
            logging.info(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
            logging.info(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))
            
            if wandb_config:
                wandb_config.log({"val_audio_recall@20": A_r20, "val_image_recall@20": I_r20, "epoch": epoch})
                wandb_config.log({"val_audio_recall@10": A_r10, "val_image_recall@10": I_r10, "epoch": epoch})
                wandb_config.log({"val_audio_recall@5": A_r5, "val_image_recall@5": I_r5, "epoch": epoch})
                wandb_config.log({"val_audio_recall@1": A_r1, "val_image_recall@5": I_r1, "epoch": epoch})

            #for text
            recalls =calc_recalls_reid(image_output, text_output, nframes, id_output, val_loader,type ="text", visualise_id = 10, simtype=args.simtype)
            A_r20 = recalls['A_r20']
            I_r20 = recalls['I_r20']
            A_r10 = recalls['A_r10']
            I_r10 = recalls['I_r10']
            A_r5 = recalls['A_r5']
            I_r5 = recalls['I_r5']
            A_r1 = recalls['A_r1']
            I_r1 = recalls['I_r1']
            print(' * Text R@20 {A_r20:.3f} Image R@20 {I_r20:.3f} over {N:d} validation pairs'
                .format(A_r20=A_r20, I_r20=I_r20, N=N_examples), flush=True)
            print(' * Text R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
            print(' * Text R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
            print(' * Text R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)
            
            logging.info(' * Text R@20 {A_r20:.3f} Image R@20 {I_r20:.3f} over {N:d} validation pairs'
                .format(A_r20=A_r20, I_r20=I_r20, N=N_examples))
            logging.info(' * Text R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
            logging.info(' * Text R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
            logging.info(' * Text R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))
            if wandb_config:
                wandb_config.log({"val_Text_recall@20": A_r20, "val_Text_recall@20": I_r20, "epoch": epoch})
                wandb_config.log({"val_Text_recall@10": A_r10, "val_Text_recall@10": I_r10, "epoch": epoch})
                wandb_config.log({"val_Text_recall@5": A_r5, "val_Text_recall@5": I_r5, "epoch": epoch})
                wandb_config.log({"val_Text_recall@1": A_r1, "val_Text_recall@5": I_r1, "epoch": epoch})



    return recalls


