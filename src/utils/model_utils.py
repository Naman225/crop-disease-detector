import os
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_model(model, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), file_path)
        logger.info(f"Successfully model_params to {file_path}")
    except Exception as e:
        logger.error(f"Error while saving model_params to {file_path}: {e}")
        raise e

def load_model(model, file_path):
    try:
        model.load_state_dict(torch.load(file_path))
        logger.info(f"Successfully loaded params from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error while loading params from {file_path}: {e}")
        raise e
