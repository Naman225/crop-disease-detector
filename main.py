from src.pipeline.dataset import prepare_data_splits, get_dataloader
from src.pipeline.orchestrator import run_model_pipeline
from src.pipeline.evaluate import compare_models
from models.resnet import get_resnet18, unfreeze_resnet_last_block
from models.efficientnet import get_efficientnet, unfreeze_efficientnet_last_block



if __name__ == "__main__":
    prepare_data_splits()
    train_loader, val_loader, test_loader, classes = get_dataloader()

    all_results = []
    all_results += run_model_pipeline(
        "resnet18", get_resnet18, unfreeze_resnet_last_block,
        train_loader, val_loader, test_loader, classes
    )
    all_results += run_model_pipeline(
        "efficientnet", get_efficientnet, unfreeze_efficientnet_last_block,
        train_loader, val_loader, test_loader, classes
    )

    compare_models(all_results)
