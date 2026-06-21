import os
import yaml
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_model():
    # 1. Load configurations
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["paths"]["processed_dir"]
    model_output_path = config["paths"]["model_path"]
    
    # 2. Load processed training and validation data
    print("Loading processed data...")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    
    # Set the MLflow experiment name
    mlflow.set_experiment("Heart_Disease_Classification")
    
    # 3. Start an MLflow run
    with mlflow.start_run():
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=config["model_params"]["n_estimators"],
            max_depth=config["model_params"]["max_depth"],
            random_state=config["model_params"]["random_state"]
        )
        model.fit(X_train, y_train)
        
        # 4. Make predictions and compute required evaluation metrics
        predictions = model.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        
        # 5. Log hyperparameters to MLflow
        mlflow.log_params(config["model_params"])
        mlflow.log_param("test_size", config["data_params"]["test_size"])
        
        # 6. Log evaluation metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # 7. Log the model artifact itself
        mlflow.sklearn.log_model(model, "model")
        
        # Local save as well
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"Run Logged! Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    train_model()