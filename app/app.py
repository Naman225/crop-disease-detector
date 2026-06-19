import gradio as gr
from app.inference import predict, MODELS
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
allow_flagging="never"
demo = gr.Interface(
    fn=predict,
    flagging_mode="never",
    cache_examples=True,
    theme=gr.themes.Soft(),
    inputs=[
        gr.Image(type="pil", label="Upload a leaf image"),
        gr.Dropdown(
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            label="Model"
        )
    ],
    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Prediction"
        ),
        gr.Image(
            label="Grad-CAM Explanation",
            height=350
        ),
        gr.Markdown(
            label="Disease Information"
        )
    ],
    title="🌿 Crop Disease Detector",
    description=(
        "Upload an image of a tomato, potato, or pepper leaf.\n\n"
        "Models:\n"
        "• ResNet18 (Fine-tuned)\n"
        "• EfficientNet-B0 (Fine-tuned)\n\n"
        "Dataset: PlantVillage (15 classes)"
    ),
    examples=[
    [str(BASE_DIR / "examples" / "tomato_blight.jpg"), list(MODELS.keys())[0]],
    [str(BASE_DIR / "examples" / "potato_healthy.jpg"), list(MODELS.keys())[1]],
    [str(BASE_DIR / "examples" / "pepper_bacterial_spot.jpg"), list(MODELS.keys())[0]],
    ],
    article="[GitHub Repo](your-link-here) | Built by Naman",
)

if __name__ == "__main__":
    demo.launch()