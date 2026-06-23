# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_performance(model, X_test, y_test):
    """
    Computes core classification performance metrics for a trained model.
    """
    predictions = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0))
    }
    
    return metrics

def verify_performance_thresholds(metrics, min_accuracy=0.75):
    """
    Validates if model metrics satisfy strict production release guardrails.
    """
    if metrics["accuracy"] < min_accuracy:
        return False, f"Model accuracy ({metrics['accuracy']:.2f}) fell below target ({min_accuracy:.2f})"
    return True, "Model satisfies all operational performance thresholds."