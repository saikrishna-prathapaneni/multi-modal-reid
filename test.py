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
from torchinfo import summary  # Library to summarize PyTorch models
import numpy as np
# Load ResNet50
resnet50 = models.Resnet50_Dino()


print("\nResNet50 Summary:")
summary(resnet50, input_size=(1, 3, 224, 224))



