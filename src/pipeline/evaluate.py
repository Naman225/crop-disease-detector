import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    classification_report,confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, test_loader, class_names, model_name, run_id = None, device = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for X,y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average= 'weighted'
    )    

    report_dict = classification_report(
        test_labels, test_preds, target_names=class_names,
        output_dict=True, digits=3
    )
    report_str = classification_report(
        test_labels, test_preds, target_names=class_names, digits=3
    )
    logger.info(f"\n{model_name} Classification Report:\n{report_str}")

    os.makedirs("artifacts/metrics", exist_ok=True)


    cm = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap='mako',xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = f"artifacts/metrics/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    report_path = f"artifacts/metrics/{model_name}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    mlflow_ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"{model_name}_eval")
    with mlflow_ctx:
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

    logger.info(f"{model_name} — Acc: {acc:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | F1: {f1:.4f}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_report": report_dict,
        "confusion_matrix_path": cm_path
    }

def compare_models(results_list, save_path="artifacts/metrics/model_comparison.json"):
    comparison = [{
        "model": r["model_name"],
        "accuracy": round(r["accuracy"], 4),
        "precision": round(r["precision"], 4),
        "recall": round(r["recall"], 4),
        "f1": round(r["f1"], 4)
    } for r in results_list]
    with open(save_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    for c in comparison:
        logger.info(f"{c['model']:<25} {c['accuracy']:<10} {c['precision']:<10} "
                    f"{c['recall']:<10} {c['f1']:<10}")
    return comparison


        

