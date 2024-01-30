# Author: David Harwath
import argparse
import os
import pickle
import sys
import time
import torch

import dataloaders
import models
from steps import train, validate
import warnings

# # Suppress all Deprecation warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=64, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
parser.add_argument("--recall_type", type=str, default="topk",
        help="recall function type", choices=["topn", "topk"])

args = parser.parse_args()

resume = args.resume

if args.resume:
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume
        

train_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_train),
    batch_size=args.batch_size, shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_val, image_conf={'center_crop':False},type="test"),
    batch_size=args.batch_size//2,shuffle=False, num_workers=2)

audio_model = models.EnhancedDavenet()
#image_model = models.VGG16()

#image_model = models.Resnet50_Dino()
#image_model = models.Resnet50_Luperson()

image_model = models.Resnet50(pretrained=True)

if not bool(args.exp_dir):
    print("exp_dir not specified, automatically creating one...")
    args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_Optim-%s_LR-%s_Epochs-%s_tweak_resnet50_I" % (
        os.path.basename(args.data_train), args.audio_model, args.image_model, args.optim,
        args.lr, args.n_epochs)

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    if not os.path.exists("%s/models" % args.exp_dir):
        os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)


train(audio_model, image_model, train_loader, val_loader, args)
# if __name__ =="__main__":
    
#         audio_model = audio_model.to("cuda")
#         image_model = image_model.to("cuda")
#         model_records = []  # To store model records

#         i = 85
#         args.exp_dir = "/home/dled/sai/DAVEnet/DAVEnet-pytorch/exp/Data-/AudioModel-Davenet_ImageModel-resnet_Optim-sgd_LR-0.001_Epochs-50_LUP_resnet50"
        
#         #audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (args.exp_dir, i)))
#         #image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (args.exp_dir, i)))
#         audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (args.exp_dir)))
#         image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (args.exp_dir)))

#         recalls = validate(audio_model, image_model, val_loader, args)
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
