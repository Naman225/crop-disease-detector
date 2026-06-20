import torch
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return [model.layer4[-1]]
    elif "efficientnet" in model_name:
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model type for GradCAM: {model_name}")


def generate_gradcam_image(
    model,
    model_name,
    pil_img,
    device=None
):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])

    rgb_img = np.array(
        pil_img.resize((224, 224))
    ) / 255.0

    input_tensor = (
        transform(pil_img)
        .unsqueeze(0)
        .to(device)
    )

    target_layer = get_target_layer(
        model,
        model_name
    )

    cam = GradCAM(
        model=model,
        target_layers=target_layer
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
