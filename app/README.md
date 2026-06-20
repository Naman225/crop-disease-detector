---
title: Crop Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.34.2"
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🌿 Crop Disease Detector

Upload an image of a **tomato, potato, or pepper leaf** and get an instant disease diagnosis with a **Grad-CAM heatmap** showing which regions influenced the model's decision.

## Models

| Model | Test Accuracy | F1 Score |
|:---|:---:|:---:|
| **ResNet18** (Fine-tuned) | 99.13% | 99.12% |
| EfficientNet-B0 (Fine-tuned) | 96.36% | 96.35% |

## Features

- **15-class classification** across tomato, potato, and pepper diseases
- **Real-time Grad-CAM heatmaps** on every prediction
- **Top-3 predictions** with confidence scores
- **Treatment recommendations** for each detected disease
- **Model comparison** — switch between ResNet18 and EfficientNet-B0

## Dataset

Trained on [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (~20,600 images) using two-phase transfer learning with MLflow experiment tracking.

## Links

- **[GitHub Repository](https://github.com/Naman225/crop-disease-detector)** — full training pipeline, metrics, and documentation

Built by **Naman**
