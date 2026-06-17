import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
import os
from src.utils.logger import get_logger
import splitfolders

logger = get_logger(__name__)

def prepare_data_splits(raw_dir = "data/raw/PlantVillage", output_dir="data/processed/PlantVillage"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Splitting dataset in train, val, test sets...")
        splitfolders.ratio(
            raw_dir,
            output=output_dir,
            seed=42,
            ratio=(0.8, 0.1, 0.1), # 80% Train, 10% Val, 10% Test
            move=False
        )
        logger.info(f"Data splits successfully created at: {output_dir}")
    else:
        logger.info("Data splits already exist. Skipping split step.")

def get_transform():
    train_transform = v2.Compose([
        v2.Resize((224,224)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

    
def get_train_sampler(train_dataset):
    # count how many images per class in this subset
    targets = train_dataset.targets
    class_counts = Counter(targets)
    
    # weight = 1 / class_count so rare classes get sampled more
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

def get_dataloader(data_dir="data/processed/PlantVillage", batch_size = 64):
    logger.info("Splitting data and loading DataLoader ...")
    train_transform, val_transform = get_transform()

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_transform)

    train_sampler = get_train_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    
    logger.info("Successfully loaded...")
    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    prepare_data_splits()
    train_loader, val_loader, test_loader, classes = get_dataloader()
    print(f"Classes     : {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
    
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape  : {imgs.shape}")   
    print(f"Labels shape : {labels.shape}") 