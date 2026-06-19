import torch
import torch.nn as nn
from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate_model
from src.config import CONFIG_1, CONFIG_2
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_model_pipeline(model_name, model_fn, unfreeze_fn, train_loader, test_loader, val_loader, classes):
    num_classes = len(classes)
    model = model_fn(num_classes)
    criterion = nn.CrossEntropyLoss()

    ## Phase 1
    logger.info(f"{model_name} |  Phase 1: frozen params: ")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = CONFIG_1['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = CONFIG_1['step_size'], gamma = CONFIG_1['gamma'])

    model, phase1_best_acc, phase1_best_id = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, model_name= f"{model_name}_phase1" , lr=CONFIG_1['lr'], epochs=CONFIG_1['epochs'])

    phase1_test_metrics = evaluate_model(
        model, test_loader, classes, model_name, run_id=phase1_best_id
    )


    ## Phase 2

    logger.info(f"{model_name} | Phase 2: fine-tuning params: ")
    model = unfreeze_fn(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = CONFIG_2['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = CONFIG_2['step_size'], gamma = CONFIG_2['gamma'])

    model, phase2_val_acc, phase2_run_id = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        model_name=f"{model_name}_phase2", lr=CONFIG_2["lr"],
        epochs=CONFIG_2["epochs"]
    )


    phase2_test_metrics = evaluate_model(
        model, test_loader, classes, model_name, run_id=phase2_run_id
    )

    return [phase1_test_metrics, phase2_test_metrics]


