import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels



class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(imagemodels.resnet18(pretrained=True).state_dict())
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(imagemodels.resnet34(pretrained=True).state_dict())
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(imagemodels.resnet50(pretrained=True).state_dict())
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(VGG16, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x

    
class Resnet50_Dino(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Resnet50_Dino,self).__init__()
        self.dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.dino_resnet50 = nn.Sequential(*list(self.dino_resnet50.children())[:-2])  
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x = self.dino_resnet50(x)
        x = self.embedder(x)
        return x
        
class Resnet50_MOCO(nn.Module):
    def __init__(self, embedding_dim=1024, weights_path='lup'):
        super(Resnet50_MOCO, self).__init__()
        # Load ResNet50 model
        PATH = None
        if weights_path=='lup':
            PATH = '/home/sp7238/sai/mulit-model-reid/multi-modal-reid/weights/lup_moco_r50.pth'
        elif weights_path=="inet":
            PATH = '/home/sp7238/sai/mulit-model-reid/multi-modal-reid/weights/weights/moco_v1_200ep_pretrain.pth'
        else:
            raise "pass a suitable weights file"

        self.resnet50 = imagemodels.resnet50(pretrained=False)  # Get the ResNet50 architecture
        #self.resnet50.fc = nn.Identity()  # Remove the classification layer

        # Embedding layer
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

        # Load pre-trained weights for Luperson dataset
        if weights_path =='inet':

            # for name, param in self.resnet50.named_parameters():
            #     if name not in ['fc.weight', 'fc.bias']:
            #         param.requires_grad = False
            # init the fc layer
            
            checkpoint = torch.load(PATH, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            
            msg = self.resnet50.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        else:
            state_dict = torch.load(PATH)
            #self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
            self.resnet50.load_state_dict(state_dict, strict=False)

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2]) 

    def forward(self, x):
        x = self.resnet50(x)
        x = self.embedder(x)
        return x

