import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import transforms
from src.utils.logger import get_logger

logger = get_logger(__name__)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return [model.layer4[-1]]
    elif "efficientnet" in model_name:
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model type for GradCAM: {model_name}")


def generate_grad_cam_grid(model, model_name, image_paths, class_names, save_path, device = None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=target_layer)

    n = len(image_paths)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB").resize((224,224))
        rgb_img = np.array(pil_img) / 255.0
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)
            pred_class = pred.argmax(1).item()
            confidence = torch.softmax(pred, dim=1)[0, pred_class].item()
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[0, i].imshow(pil_img)
        axes[0, i].set_title("Original", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(visualization)
        axes[1, i].set_title(
            f"{class_names[pred_class]}\n({confidence:.1%})", fontsize=9
        )
        axes[1, i].axis("off")

    plt.suptitle(f"GradCAM — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"GradCAM grid saved to {save_path}")