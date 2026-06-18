import torch
import torch.nn as nn

from src.pipeline.dataset import prepare_data_splits, get_dataloader
from src.pipeline.train import train_model
from models.resnet18 import get_resnet18
from src.utils.logger import get_logger

def train():

    prepare_data_splits()
    train_loader, val_loader, test_loader, classes = get_dataloader()
    
    num_classes = len(classes)
    
    model = get_resnet18(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    model, best_acc= train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, model_name= 'resnet18' , lr=1e-3, epochs=20)


if __name__ == "__main__":
    train()