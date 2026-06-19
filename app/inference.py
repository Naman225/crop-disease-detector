import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from src.pipeline.grad_cam import generate_gradcam_image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts" / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DISEASE_INFO = {
    "Bell Pepper - Bacterial Spot":
        "Bacterial infection causing dark leaf lesions. Remove infected leaves and avoid overhead watering.",

    "Bell Pepper - Healthy":
        "No disease detected. Continue normal crop management.",

    "Potato - Early Blight":
        "Fungal disease causing concentric brown spots. Use crop rotation and fungicides if necessary.",

    "Potato - Late Blight":
        "Serious disease caused by Phytophthora infestans. Remove infected plants and apply recommended fungicides.",

    "Potato - Healthy":
        "No disease detected. Plant appears healthy.",

    "Tomato - Bacterial Spot":
        "Bacterial disease causing leaf spots and fruit lesions. Improve air circulation and avoid leaf wetness.",

    "Tomato - Early Blight":
        "Fungal disease producing target-like lesions. Remove affected foliage and use fungicides when needed.",

    "Tomato - Late Blight":
        "Aggressive disease affecting leaves and fruit. Remove infected plants immediately and apply control measures.",

    "Tomato - Leaf Mold":
        "Common in humid conditions. Increase ventilation and reduce humidity.",

    "Tomato - Septoria Leaf Spot":
        "Fungal disease causing small circular spots. Remove infected leaves and avoid overhead irrigation.",

    "Tomato - Spider Mites":
        "Mite infestation causing yellowing and webbing. Use insecticidal soap or biological controls.",

    "Tomato - Target Spot":
        "Fungal disease producing circular lesions. Improve airflow and apply fungicide if necessary.",

    "Tomato - Yellow Leaf Curl Virus":
        "Viral disease spread by whiteflies. Remove infected plants and control insect vectors.",

    "Tomato - Mosaic Virus":
        "Viral infection causing mottled leaves. Remove infected plants and sanitize tools.",

    "Tomato - Healthy":
        "No disease detected. Plant appears healthy."
}

CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Bell Pepper - Bacterial Spot",
    "Pepper__bell___healthy": "Bell Pepper - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato_Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato_Early_blight": "Tomato - Early Blight",
    "Tomato_Late_blight": "Tomato - Late Blight",
    "Tomato_Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato - Spider Mites",
    "Tomato__Target_Spot": "Tomato - Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato - Mosaic Virus",
    "Tomato_healthy": "Tomato - Healthy",
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def _build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_model(arch, weights_path):
    if arch == "resnet18":
        model = _build_resnet18(len(CLASS_NAMES))
    elif arch == "efficientnet":
        model = _build_efficientnet(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

MODELS = {
    "ResNet18": load_model(
        "resnet18",
        MODEL_DIR / "resnet18_phase2_best.pth"
    ),
    "EfficientNet-B0": load_model(
        "efficientnet",
        MODEL_DIR / "efficientnet_phase2_best.pth"
    ),
}

def predict(image, model_choice):

    if image is None:
        return {}, None, ""

    model = MODELS[model_choice]

    img = image.convert("RGB")

    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top3_probs, top3_idx = probs.topk(3)

    
    predictions = {
        DISPLAY_NAMES.get(
            CLASS_NAMES[idx.item()],
            CLASS_NAMES[idx.item()]
        ): float(prob)
        for idx, prob in zip(
            top3_idx,
            top3_probs
        )
    }

    model_key = (
        "resnet18"
        if "ResNet" in model_choice
        else "efficientnet"
    )

    gradcam_img = generate_gradcam_image(
        model,
        model_key,
        img,
        DEVICE
    )

    top_class = DISPLAY_NAMES.get(
        CLASS_NAMES[top3_idx[0].item()],
        CLASS_NAMES[top3_idx[0].item()]
    )
    recommendation = f"""
    ## Diagnosis
    **{top_class}**

    ## Confidence
    {top3_probs[0].item():.2%}

    ## Recommendation
    {DISEASE_INFO.get(top_class, "No recommendation available.")}
    """

    return predictions, gradcam_img, recommendation