# Author: Sai Krishna Prathapaneni
# reference from: https://github.com/dharwath/DAVEnet-pytorch
import argparse
import os
import pickle
import sys
import time
import torch
import wandb
import dataloaders
import models
from steps import train, validate
import warnings
from transformers import BertModel
# # Suppress all Deprecation warnings
import logging
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)


# Initialize wandb


print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/',
        help="training data json")
parser.add_argument("--data-val", type=str, default='/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/',
        help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
        help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adagrad",
        help="training optimizer", choices=["sgd", "adam","adagrad"])
parser.add_argument('-b', '--batch-size', default=64, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001,type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=50,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=10,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
        help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="resnet",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--text_model", type=str, default="bert",
        help="text model architecture", choices=["bert"])
parser.add_argument("--use_text_backbone", type=bool, default=True,
        help="text backbone architecture")
parser.add_argument("--different_text_prompt", type=bool, default=True,
        help="different text prompt for similarity match")
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
parser.add_argument("--use_alpha_beta", type=bool, default=False,
        help="using beta and alpha hyperparameter functions for loss computation")
parser.add_argument('--alpha', '--al', default=0.5, type=float,
    metavar='al', help='alpha weighing for audio')
parser.add_argument('--beta', '--be', default=0.5, type=float,
    metavar='bl', help='beta weighing for text')

parser.add_argument("--recall_type", type=str, default="topk",
        help="recall function type", choices=["topn", "topk"])

args = parser.parse_args()

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'),
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

wandb.init(project="multi-modal-reid", config=args)

#wandb agent dled/multi-modal-reid/6ba4z2f1


resume = args.resume


if args.resume:
        assert(bool(args.exp_dir))
        with open("%s/args.pkl" % args.exp_dir, "rb") as f:
                args = pickle.load(f)
args.resume = resume
        

train_loader = torch.utils.data.DataLoader(
dataloaders.ImageCaptionDataset(args,args.data_train),
batch_size=8, shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(
dataloaders.ImageCaptionDataset(args,args.data_val, image_conf={'center_crop':False},type="test"),
batch_size=args.batch_size//2,shuffle=False, num_workers=2)

audio_model = models.EnhancedDavenet()
#image_model = models.VGG16()

#image_model = models.Resnet50_Dino()
#image_model = models.Resnet50_Luperson()

image_model = models.Resnet50(pretrained=True)

#text_model = BertModel.from_pretrained('bert-base-uncased')
text_model= None
if args.use_text_backbone:
        text_model = models.BertEmbedding()

if not bool(args.exp_dir):
        print("exp_dir not specified, automatically creating one...")
        args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_TextModel-%s_Optim-%s_LR-%s_Epochs-%s_tweak_resnet50_I" % (
                os.path.basename(args.data_train), args.audio_model, args.image_model, args.text_model, args.optim,
                args.lr, args.n_epochs)

if not args.resume:
        print("\nexp_dir: %s" % args.exp_dir)
if not os.path.exists("%s/models" % args.exp_dir):
        os.makedirs("%s/models" % args.exp_dir)
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
                pickle.dump(args, f)


train(audio_model, image_model, text_model, train_loader, val_loader, args, wandb_config = wandb)


# if __name__ =="__main__":
        
#         def check_weights(model1, model2):
#                 for p1, p2 in zip(model1.parameters(), model2.parameters()):
#                         if p1.data.ne(p2.data).sum() > 0:
#                                 return False
#                 return True
#         def count_trainable_params(model):
#                 return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
#         audio_model1 = audio_model.to("cuda")
#         image_model1 = image_model.to("cuda")
#         text_model1 = text_model.to("cuda")
#         model_records = []  # To store model records
        
#         audio_model = audio_model.to("cuda")
#         image_model = image_model.to("cuda")
#         text_model = text_model.to("cuda")
#         i = 1
#         k = 10
#         args.exp_dir = "/home/sp7238/sai/mulit-model-reid/multi-modal-reid/exp/Data-/AudioModel-Davenet_ImageModel-resnet_Optim-sgd_LR-0.001_Epochs-50_tweak_resnet50_I/"
        
#         audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (args.exp_dir, i)))
#         image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (args.exp_dir, i)))
#         text_model.load_state_dict(torch.load("%s/models/text_model.%d.pth" % (args.exp_dir,i)))
        
#         audio_model1.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (args.exp_dir,k)))
#         image_model1.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (args.exp_dir,k)))
#         text_model1.load_state_dict(torch.load("%s/models/text_model.%d.pth" % (args.exp_dir,k)))
        
#         num_trainable_params_audio = count_trainable_params(audio_model)
#         num_trainable_params_image = count_trainable_params(image_model)
#         num_trainable_params_text = count_trainable_params(text_model)

#         print(f"Number of trainable parameters in audio model: {num_trainable_params_audio}")
#         print(f"Number of trainable parameters in image model: {num_trainable_params_image}")
#         print(f"Number of trainable parameters in text model: {num_trainable_params_text}")
                
#         # are_audio_weights_same = check_weights(audio_model, audio_model1)
#         # are_image_weights_same = check_weights(image_model, image_model1)
#         # are_text_weights_same = check_weights(text_model, text_model1)

#         # print(f"Audio model weights are the same: {are_audio_weights_same}")
#         # print(f"Image model weights are the same: {are_image_weights_same}")
#         # print(f"Text model weights are the same: {are_text_weights_same}")
        

        
#         #recalls = validate(audio_model, image_model, val_loader, args)
#         print("epcoch: ", i)

#   model_records.append((i, recalls))  # Record model number and its performance

# # Sort models based on recall
# sorted_models = sorted(model_records, key=lambda x: x[1]['A_r10'], reverse=True)

# # Keep the top 10 performing models
# models_to_keep = sorted_models[:10]

# # Delete models not in the top 10
# for model_num, _ in model_records:
#     if model_num not in [model[0] for model in models_to_keep]:
#         os.remove("%s/models/audio_model.%d.pth" % (args.exp_dir, model_num))
#         os.remove("%s/models/image_model.%d.pth" % (args.exp_dir, model_num))
