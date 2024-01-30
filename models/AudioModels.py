import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDavenet(nn.Module):
    def __init__(self, embedding_dim=1024, dropout_rate=0.5):
        super(EnhancedDavenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.prelu = nn.PReLU()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.prelu(self.conv3(x))
        x = self.pool(x)
        x = self.prelu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.prelu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.squeeze(2)
        return x
