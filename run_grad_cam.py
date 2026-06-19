import torch 
import random
from models.resnet import get_resnet18
from models.efficientnet import get_efficientnet
from src.pipeline.dataset import get_dataloader
from src.pipeline.grad_cam import generate_grad_cam_grid

import random
import os

train_loader, val_loader, test_loader, classes = get_dataloader()
num_classes = len(classes)

model1 = get_resnet18(num_classes)
model2 = get_efficientnet(num_classes)
model1.load_state_dict(torch.load("artifacts/models/resnet18_phase2_best.pth"))
model2.load_state_dict(torch.load("artifacts/models/efficientnet_phase2_best.pth"))

for param in model1.parameters():
    param.requires_grad = True

for param in model2.parameters():
    param.requires_grad = True
    
test_dir = "data/processed/PlantVillage/test"
sample_paths = []
for cls in random.sample(os.listdir(test_dir), 5):
    cls_dir = os.path.join(test_dir, cls)
    img_file = random.choice(os.listdir(cls_dir))
    sample_paths.append(os.path.join(cls_dir, img_file))

generate_grad_cam_grid(
    model1, "resnet", sample_paths, classes,
    save_path="artifacts/metrics/gradcam_resnet18.png"
)

generate_grad_cam_grid(
    model2, "efficientnet", sample_paths, classes,
    save_path="artifacts/metrics/gradcam_efficient_net.png"
)