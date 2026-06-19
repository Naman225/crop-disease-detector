from torchvision import models
import torch.nn as nn

def get_resnet18(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def unfreeze_resnet_last_block(model):
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model
