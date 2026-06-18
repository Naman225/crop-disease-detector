import torch
import mlflow

from src.utils.logger import get_logger
from src.utils.model_utils import save_model

logger = get_logger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,model_name, lr, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")
    model = model.to(device)
    
    best_model_path = f"artifacts/{model_name}_best.pth"

    mlflow.set_experiment('crop-disease-detection')

    with mlflow.start_run(run_name = model_name):
        mlflow.log_params({
            "model" : model_name, 
            "epochs": epochs,
            "learning_rate" : lr,
            "batch_size" : train_loader.batch_size,
            "optimizer" : "Adam",
            "scheduler" : "step_scheduler"
        })
        best_val_acc=0
        for epoch in range(epochs):
            
            ## training 

            model.train()
            train_loss, correct, total = 0,0,0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad(set_to_none = True)
                pred = model(X)
                loss = criterion(pred,y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct += (pred.argmax(1)==y).sum().item()
                total += y.size(0)

            train_acc= correct/total
            train_loss = train_loss/len(train_loader)

            ## validation

            model.eval()
            val_loss, val_correct, val_total = 0,0,0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += criterion(pred, y).item()
                    val_correct += (pred.argmax(1) == y).sum().item()
                    val_total += y.size(0)

            val_acc = val_correct /val_total
            val_loss = val_loss / len(val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, best_model_path)
                logger.info(f" New best saved: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)
            
            logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
            if scheduler is not None:
                scheduler.step()

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(best_model_path)
        model.load_state_dict(torch.load(best_model_path))
        mlflow.pytorch.log_model(model, artifact_path="model")
        logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    
    return model,best_val_acc




    
            

         