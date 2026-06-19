from torchvision import models
import torch.nn as nn

def get_efficientnet(num_classes):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

def unfreeze_efficientnet_last_block(model):
    for param in model.features[-1].parameters():
        param.requires_grad = True

    return model
