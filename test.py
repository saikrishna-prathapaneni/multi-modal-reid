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
# from torchinfo import summary  # Library to summarize PyTorch models
from transformers import BertTokenizer, BertModel
import numpy as np
# Load ResNet50
# resnet50 = models.Resnet50_Dino()


# print("\nResNet50 Summary:")
# summary(resnet50, input_size=(1, 3, 224, 224))




# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample text
text = "Hello, my name is ChatGPT."

# Encode the text, adding the special tokens needed for BERT
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_tensor = torch.tensor([input_ids])

# Get the embeddings
with torch.no_grad():
    outputs = model(input_tensor)
    # The last hidden state is the sequence of hidden states of the last layer of the model
    last_hidden_states = outputs.last_hidden_state

# The `last_hidden_states` tensor contains the embeddings
# To get the embedding of a particular token, select the appropriate index
# For example, the embedding of the first token (CLS token in this case) can be accessed as follows:
embedding_of_first_token = last_hidden_states[0][0]

print(embedding_of_first_token.shape)
