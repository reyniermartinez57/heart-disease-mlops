import os
import yaml
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Import your new dedicated evaluation module
from src.evaluation import evaluate_model_performance, verify_performance_thresholds

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
        
        # 4. Use the dedicated evaluation module to compute metrics
        print("Evaluating model performance via modular framework...")
        metrics = evaluate_model_performance(model, X_val, y_val)
        
        # 5. Log hyperparameters to MLflow
        mlflow.log_params(config["model_params"])
        mlflow.log_param("test_size", config["data_params"]["test_size"])
        
        # 6. Log evaluation metrics dictionary dynamically to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        # 7. Run threshold check to verify model safety for deployment
        # Uses minimum 75% accuracy matching the Sprint 17 baseline check
        is_safe, message = verify_performance_thresholds(metrics, min_accuracy=0.75)
        print(message)
        if not is_safe:
            raise ValueError("Build Failed: Model metrics dropped below production quality minimums.")
        
        # 8. Log the model artifact itself
        mlflow.sklearn.log_model(model, "model")
        
        # Local save as well
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"Run Logged Successfully! Summary: {metrics}")

if __name__ == "__main__":
    train_model()